import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from difflib import SequenceMatcher
import os
import json
import copy
import random
import traceback
import numpy as np
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import faiss
from spider_agent.agent.agents import PromptAgent
from spider_agent.agent.planner_critique_agents import PlannerAgent, CritiqueAgent
from spider_agent.agent.action import Terminate, BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, LOCAL_DB_SQL
from spider_agent.agent.models import call_llm
from spider_agent.agent.agents import critique_needs_schema
from spider_agent.agent.schema_agent import SchemaAgent, SchemaAgentEnv
from spider_agent.agent.refinement_memory_bank_agent import RefinementLogRAG

logger = logging.getLogger("spider_agent") 



class SelfRefinementAgent(PromptAgent):

    def log_refinement_case(self, original_sql, refined_sql, error, success, description=None):
        if not success:
            return
        if any(msg in error for msg in [
            "Results saved to",
            "BigQuery Storage module not found",
            "This result was fetched from REST endpoint"  # 可擴充更多非錯誤訊息
        ]):
            return

        # 自動解析錯誤型別（加入 error_type）
        error_type = self.analyze_sql_error(error).split(":")[0] if error else "Unknown"

        record = {
            "original_sql": original_sql,
            "refined_sql": refined_sql,
            "error": error,
            "success": success,
            "dialect": self.dialect,
            "error_type": error_type,
            "description": description
        }
        try:
            with open(self.refinement_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log refinement case: {e}")


    def analyze_sql_error(self, error_msg: str) -> str:
        msg = error_msg.lower() if error_msg else ""

        # 注意：順序很重要，要先抓 AmbiguousColumn 再抓 ColumnNotFound
        if "ambiguous column" in msg or "is ambiguous" in msg:
            return "AmbiguousColumn"

        if any(x in msg for x in ["no such column", "column not found", "unknown column", "unrecognized name", "invalid column", "does not exist in"]):
            return "ColumnNotFound"

        if any(x in msg for x in ["no such table", "table not found", "unknown table", "relation does not exist"]):
            return "TableNotFound"

        if any(x in msg for x in ["syntax error", "parse error", "unexpected", "expected", "parsererror"]):
            return "SyntaxError"

        if any(x in msg for x in ["function not found", "no such function", "unknown function", "not a function"]):
            return "FunctionNotFound"

        if any(x in msg for x in ["invalid input syntax", "invalid type", "type mismatch", "cannot cast", "datatype mismatch"]):
            return "TypeMismatch"

        if any(x in msg for x in ["permission denied", "access denied", "not authorized", "insufficient privileges"]):
            return "PermissionDenied"

        if "timeout" in msg or "timed out" in msg or "query exceeded" in msg:
            return "Timeout"

        if "division by zero" in msg:
            return "DivisionByZero"

        if "duplicate column" in msg or "column name specified more than once" in msg:
            return "DuplicateColumn"

        if "invalid identifier" in msg or "invalid name" in msg:
            return "InvalidIdentifier"

        if "resources exceeded" in msg or "quota exceeded" in msg:
            return "ResourceExceeded"

        if "null value in column" in msg or "not null constraint" in msg:
            return "NotNullConstraint"

        if "foreign key constraint" in msg or "constraint failed" in msg:
            return "ConstraintError"

        if "no rows" in msg or "empty result" in msg or "no results" in msg:
            return "NoResult"

        return "OtherError"

    def construct_error_prompt(self, error_type):
        ERROR_HINT_TEMPLATES = {
            "SyntaxError": (
                "There is a syntax error in the SQL query. Please check for issues with "
                "parentheses, commas, quotes, reserved keywords, and clause order."
            ),
            "TableNotFound": (
                "The SQL query refers to a table that does not exist. "
                "Double-check the table name and whether it is present in the schema."
            ),
            "ColumnNotFound": (
                "The query uses a column that does not exist in the table. "
                "Please verify the column name spelling, casing, or use of aliases."
            ),
            "AmbiguousColumn": (
                "The query references a column that exists in multiple tables. "
                "Qualify the column name using the table prefix (e.g., table.column)."
            ),
            "FunctionNotFound": (
                "An undefined or mistyped function is being used. "
                "Please ensure all functions exist and their arguments are valid."
            ),
            "TypeMismatch": (
                "There is a data type mismatch in the query (e.g., comparing strings with numbers, or invalid casting). "
                "Ensure compatible types or apply proper CAST/CONVERT functions."
            ),
            "PermissionDenied": (
                "The query failed due to insufficient permissions. "
                "This may require access to the table or specific columns."
            ),
            "Timeout": (
                "The query execution timed out or exceeded resource limits. "
                "Try simplifying the query or limiting the data returned (e.g., add WHERE or LIMIT)."
            ),
            "DivisionByZero": (
                "The query attempts to divide by zero. "
                "Check the divisor column or add a conditional filter to avoid zero."
            ),
            "DuplicateColumn": (
                "The output of the query contains duplicate column names. "
                "Use column aliases (AS ...) to make them unique."
            ),
            "InvalidIdentifier": (
                "There is an invalid table, column, or alias name. "
                "Avoid using reserved keywords and ensure all identifiers are valid."
            ),
            "ResourceExceeded": (
                "The query exceeds available resources or quota. "
                "Try reducing the result size or simplifying aggregations and joins."
            ),
            "NotNullConstraint": (
                "The query inserts or updates a NULL into a column that does not allow NULL values. "
                "Ensure all NOT NULL columns are properly filled."
            ),
            "ConstraintError": (
                "The query violates a database constraint, such as a foreign key or unique constraint. "
                "Check the integrity and logic of the query joins or values."
            ),
            "NoResult": (
                "The query ran successfully but returned no rows. "
                "Consider loosening filter conditions, verifying column values, or using LIKE/ILIKE."
            ),
            "OtherError": (
                "An unknown error occurred. Please re-check the SQL syntax and verify database schema alignment."
            ),
        }

        return ERROR_HINT_TEMPLATES.get(error_type, error_type)
    """
    Extension of PromptAgent with self-refinement capabilities.
    This agent can iteratively refine SQL queries based on execution results
    until termination conditions are met.
    """
    
    def __init__(
        self,
        env,
        planner_agent,
        critique_agent,
        model="gpt-4",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.5,
        max_memory_length=10,
        max_steps=15,
        use_plan=False,
        max_refinement_iterations=5,  # Maximum number of refinement iterations
        base_top_k=2,                # 新增：RAG檢索案例的預設top_k
        rag_syntax=False,            # 新增：是否啟用 BigQuery syntax RAG
        self_refinement_enabled=True,  # 新增：是否啟用自我修正
        expected_csv_format=None,   # 新增：答案格式要求
        use_schema_linking=False  # 新增：是否啟用 schema linking
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            max_memory_length=max_memory_length,
            max_steps=max_steps,
            use_plan=use_plan
        )
        self.env = env
        self.planner_agent = planner_agent
        self.critique_agent = critique_agent
        self.rag_syntax = rag_syntax
        self.max_refinement_iterations = max_refinement_iterations
        self.base_top_k = base_top_k
        self.refinement_iterations = []
        self.consecutive_empty_results = 0
        self.previous_queries = set()
        self.refinement_log_path = "refinement_history.jsonl"  # for self-learning
        self.schema_link_agent = None
        self.schema_agent_env = None
        self.schema_agent = None
        self.self_refinement_enabled = self_refinement_enabled  # 新增：自我修正开关
        self.expected_csv_format = expected_csv_format  # 新增：答案格式要求
        self.use_schema_linking = use_schema_linking  # 新增：是否啟用 schema linking
        self.schema_retriever = None  # 兩階段 schema 檢索器（延遲初始化）
    
    def format_similar_cases(self, cases):
        """Format similar past refinement cases into a prompt-ready string."""
        if not cases:
            return ""
        lines = ["\nSimilar past refinements:"]
        for case in cases:
            lines.append(f"- Original SQL: {case.get('original_sql')}\n  Refined SQL: {case.get('refined_sql')}\n  Error Type: {case.get('error_type')}\n  Error: {case.get('error')}\n  Success: {case.get('success')}")
        return "\n".join(lines)
    def compose_refinement_prompt(self, original_sql, critique_msg, error_hint, similar_text, repeated_error, error_type, expected_csv_format):
        rewrite_hint = ""
        if repeated_error:
            rewrite_hint = (
                "\n\nNote: The previous fix did NOT resolve the problem. "
                "Please try a significantly different approach to rewrite the SQL."
            )

        extra_hint = ""
        if error_type.startswith("NoResult"):
            extra_hint = ("\n[Extra Hint] The previous query returned no data. "
                    "Please try the following:\n"
                    "- Double-check the database schema to ensure all table and column names are correct and exist.\n"
                    "- Verify the data types of columns used in filters (e.g., date formats, string vs. number).\n"
                    "- Consider possible differences in column names or nested structures in the schema.\n"
                    "- Check for possible mistakes in column values (e.g., typos, case sensitivity).\n"
            )

        format_instruction = ""
        if expected_csv_format:
            format_instruction = f"\nExpected CSV format:\n{expected_csv_format}\nEnsure the output matches this format."

        return (
            f"Original SQL:\n{original_sql}\n\n"
            f"{critique_msg or ''}\n"
            f"{error_hint}\n"
            f"{similar_text}\n"
            "Please refine the SQL query to resolve the issue."
            f"{rewrite_hint}{extra_hint}{format_instruction}"
        )
    def maybe_retrieve_syntax_reference(self):
        self.syntax_reference = ""
        question = self.env.task_config.get('question', '')
        schema = self.env.task_config.get('schema', '')
        plan = self.reference_plan
        critique = self.plan_critique if hasattr(self, 'plan_critique') else {}
        from spider_agent.agent.rag_syntax_agent import RAGSyntaxAgent
        prompt = (
            f"User Question: {question}\nSchema: {schema}\nPlan: {plan}\nCritique: {critique}\n"
            f"{RAGSyntaxAgent.get_syntax_request_prompt()}"
        )
        response = call_llm({
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        })
        if isinstance(response, tuple):
            _, content = response
        else:
            content = response
        topics = [t.strip() for t in content.split(',') if t.strip()] if content else []
        if topics:
            syntax_agent = RAGSyntaxAgent()
            syntax_results = syntax_agent.retrieve(topics)
            self.syntax_reference = syntax_agent.format_for_prompt(syntax_results)
            logger.info(f"Syntax Reference: {self.syntax_reference}")

    def generate_schema(self):
        schema_string = self.env.task_config.get('schema', '')
        question = self.env.task_config.get('question', '')
        if self.use_schema_linking:
            # === Schema Linking Agent (先做 schema linking) ===
            try:
                schema_linking_result = self.schema_agent.run(question, schema_string)
                logger.info(f"[MCP] Schema Linking: {schema_linking_result}")
                
            except Exception as e:
                logger.error(f"[MCP] Schema Linking failed: {e}")
    def _self_refine(self, original_sql, obs, error_msg=None, empty_result=False):
        """
        Try to refine the SQL query using LLM and RAG, based on the last error or empty result.
        Returns (success, refined_sql, result, error_type, action)
        success: bool - Whether the refinement process completed without errors
        refined_sql: str - The refined SQL query
        result: str - The observation from executing the query
        error_type: str - Type of error encountered if any
        action: Action - The last action taken (for external done determination)
        """
        refined_sql = original_sql
        last_error_type = None
        critique_msg = None
        for attempt in range(self.max_refinement_iterations):
            logger.info(f"[Self-Refinement] Attempt {attempt + 1}...")
            # 1. Analyze error or empty result
            if error_msg:
                error_type = self.analyze_sql_error(error_msg)
            elif empty_result:
                error_type = "NoResult"
            else:
                error_type = "Unknown"
            error_hint = self.construct_error_prompt(error_type)

            if not hasattr(self, "refinement_rag") or self.refinement_rag is None:
                self.refinement_rag = RefinementLogRAG(log_path=self.refinement_log_path, dialect=self.dialect)

            similar_text = self.format_similar_cases(
                self.refinement_rag.retrieve_similar(original_sql, error_msg or "", error_type, self.base_top_k)
            )
            prompt = self.compose_refinement_prompt(
                original_sql, critique_msg, error_hint, similar_text,
                last_error_type == error_type, error_type, self.expected_csv_format
            )
            if hasattr(self, 'plan_critique'):
                step_idx = self.env.task_config.get('current_step_idx', 0)
                step_critique = self.plan_critique.get('critique_agent', {}).get(str(step_idx))
                if step_critique:
                    prompt += f"\n[Step Critique {step_idx}] {step_critique}"

            if error_type in ["ColumnNotFound", "TableNotFound"] and self.use_schema_linking:
                linked_info = self.schema_agent.run(
                    self.env.task_config.get('question', ''),
                    self.env.task_config.get('schema', '')
                )
                if linked_info:
                    prompt += f"\nSchema Linking Info: {linked_info}"

            logger.info(f"[Self-Refinement] Prompt: {prompt}")
            response, action = self.predict(prompt)
            if not action or not getattr(action, 'sql_query', ''):
                continue

            refined_sql = action.sql_query
            critique_msg = self.critique_agent.critique_sql(
                refined_sql, self.reference_plan,
                self.env.task_config.get('question', ''),
                self.env.task_config.get('schema', ''),
                evidence=self.env.task_config.get('evidence', ''),
                execution_feedback=response
            )

            obs, _ = self.env.step(action)
            is_error = "error" in obs.lower() or "exception" in obs.lower()
            is_empty = ("no rows" in obs.lower() or "empty result" in obs.lower()) and not is_error

            self.log_refinement_case(original_sql, refined_sql, error=obs, success=(not is_error and not is_empty), description=response)

            if not is_error and not is_empty:
                return True, refined_sql, obs, error_type, action

            last_error_type = error_type

            return False, refined_sql, obs, error_type, action
    
    
    def run(self):
        """
        Override the run method to include MCP loop: Planning, Critique, and Multi-step Refinement with self-refinement on error or empty result.
        """
        assert self.env is not None, "Environment is not set."
        # --- 自動尋找 ddl.csv 並導入兩階段 Schema 檢索 Patch ---
        # if getattr(self, "use_schema_linking", False):
        #     logger.info(self.schema_string)
        #     if not self.schema_string:
        #         ddl_path = None
        #         ddl_path = self.find_ddl_csv(self.env.mnt_dir, self.env.task_config.get('db'))
        #         logger.info(f"[MCP] DDL Path: {ddl_path}")
        #         ddl_path = None if not ddl_path else ddl_path[0]
        #         if ddl_path:
        #             if self.schema_retriever is None:
        #                 self.schema_retriever = TwoStageSchemaRetriever(ddl_path)
        #             schema_string = self.schema_retriever.retrieve(
        #                 getattr(self.env, 'question', self.env.task_config.get('question', ''))
        #             )
        #         self.schema_string = schema_string
        #     else:
        #         logger.warning("No DDL/schema path found, skipping two-stage schema retrieval.")

        obs = "You are in the folder now."

        done, step_idx, result = False, 0, ""
        retry_count, last_action, repeat_action = 0, None, False
        sql_query, critique_msg = None, None

        # --- 初始化 schema_agent ---
        from spider_agent.agent.schema_link_agent import SchemaLinkAgent
        self.schema_agent_env = SchemaAgentEnv(base_dir=self.env.mnt_dir)
        self.schema_agent = SchemaAgent(self.schema_agent_env, llm_predict=self.predict)
        
        # 1. Always generate a plan at the start
        if not self.reference_plan:
            self.generate_schema()
            self.generate_reference_plan()
        logger.info(f"[MCP] Generated Plan in the Refinement Agent: {self.reference_plan}")
        

        
    # --- Plan Refinement Loop ---
        max_plan_refine, refine_count = 3, 0
        max_schema_refine, schema_refine_count = 2, 0
        while refine_count < max_plan_refine:
            success, critique_json_str = self.critique_agent.critique_plan(
                plan=self.reference_plan['plan'],
                question=self.env.task_config.get('question', ''),
                schema_string=self.env.task_config.get('schema', '')
            )
            if not success:
                logger.error("Failed to critique plan. Skipping plan refinement.")
                break
            self.plan_critique = json.loads(critique_json_str)
            logger.info(f"[MCP] Plan Critique: {self.plan_critique}")

            # --- 動態判斷是否需要 schema agent ---
            # if self.plan_critique['need_schema']:
            #     if schema_refine_count >= max_schema_refine:
            #         logger.error("Exceeded max schema refinement attempts! Stopping to avoid infinite loop.")
            #         break
                # schema_refine_count += 1
                # prev_schema = self.env.task_config.get('schema', '')
                # logger.info("Critique indicates schema is insufficient, invoking SchemaAgent...")
                # new_schema = self.schema_agent.run(
                #     user_question=self.env.task_config.get('question', ''),
                #     critique_note=self.plan_critique
                # )
                # logger.info(f"SchemaAgent generated new schema: {new_schema}")
                # if not new_schema or new_schema == prev_schema:
                #     logger.error("SchemaAgent did not generate a new schema. Stopping to avoid infinite loop.")
                #     break
                # self.env.task_config['schema'] = new_schema
                # self.plan_critique['need_schema'] = False
                # continue

            if self.plan_critique['update_plan']:
                self.reference_plan.setdefault('critique_notes', []).append(self.plan_critique['critique'])
                # === 自動修正 plan ===
                plan_prompt = (
                    f"Original Plan:\n{self.reference_plan['plan']}\n\n"
                    f"""Critique Notes:\n\n{self.reference_plan['critique_notes']}"""
                    "\n\nPlease revise the plan to address the above critique notes. Output a numbered step-by-step plan."
                )
                # 用 LLM 產生新 plan，可用 planner_agent 或 call_llm
                revised_plan_result = self.planner_agent.generate_plan(
                    question=self.env.task_config.get('question', ''),
                    schema_string=self.env.task_config.get('schema', ''),
                    evidence=self.env.task_config.get('evidence', ''),
                    prompt_prefix= plan_prompt
                )
                # 保留 critique_notes
                if isinstance(revised_plan_result, dict):
                    revised_plan_result['critique_notes'] = self.reference_plan['critique_notes']
                    self.reference_plan = revised_plan_result
                else:
                    self.reference_plan['plan'] = revised_plan_result
                refine_count += 1
            else:
                break

        # ====== 主動詢問 LLM 需要哪些 BigQuery syntax 補充 ======
        if self.rag_syntax and self.env.task_config.get('instance_id', '').startswith('bq'):
            self.maybe_retrieve_syntax_reference()
        # ======================================================

        # 2. MCP Loop: SQL generation, critique, refinement
        def get_plan_step(plan_obj, idx: int) -> str:
            import re
            plan = plan_obj['plan'] if isinstance(plan_obj, dict) else plan_obj
            steps = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|$)', plan, re.DOTALL)
            if not steps:
                step_text = plan.strip()
            elif idx < len(steps):
                step_text = steps[idx].strip()
            else:
                step_text = steps[-1].strip()
            critique_notes = plan_obj.get('critique_notes', []) if isinstance(plan_obj, dict) else []
            if critique_notes:
                notes = "\n".join([f"[Critique Note] {note['critique']}" for note in critique_notes if isinstance(note, dict) and 'critique' in note])
                step_text += f"\n\n{notes}"
            return step_text
        # obs += f"\n\nAfter the critique of plan, you are answering the following question:\n{self.env.task_config.get('question', '')}\n\nSchema:\n{self.env.task_config.get('schema', '')}"

        while not done and step_idx < self.max_steps:
            # On the first step, include the full reference plan; otherwise, only the current step
            # if step_idx == 0:
            #     prompt = f"Plan:\n{self.reference_plan['plan']}"
            # else:
            #     current_plan_step = get_plan_step(self.reference_plan, step_idx)
            #     prompt = f"Plan (current step):\n{current_plan_step}"
            prompt = (
                f"You are answering the question: {self.env.task_config.get('question', '')}\n"
                f"Based on the following plan:\n{self.reference_plan['plan']}\n\n"
            )
            if self.plan_critique:
                prompt += f"\n\nPlan Critique:\n{self.plan_critique}"
            if critique_msg:
                prompt += f"\n\nStep {step_idx} SQL Critique:\n{critique_msg}"
                if isinstance(critique_msg, dict) and 'reasoning' in critique_msg:
                    prompt += f"\n\n[Critique Reasoning] {critique_msg['reasoning']}"
            if self.rag_syntax and self.env.task_config.get('instance_id', '').startswith('bq'):
                if self.syntax_reference:
                    prompt += f"\n\n{self.syntax_reference}"

            prompt += f"\n\nObservation:\n{obs}"
            # Use PromptAgent's predict with the structured prompt
            response, action = self.predict(prompt)
            if action is None:
                logger.info("Failed to parse action from response, try again.")
                retry_count += 1
                if retry_count > 3:
                    logger.info("Failed to parse action from response, stop.")
                    break
                obs = "Failed to parse action from your response, make sure you provide a valid action."
                continue

            logger.info("Step %d: %s", step_idx + 1, action)
            # Extract SQL query string if action is SQL execution
            sql_query = getattr(action, 'sql_query', None)

            # 3. Critique the SQL after each generation
            critique_msg = None
            if sql_query:
                schema_string = self.env.task_config.get('schema', '')
                evidence = self.env.task_config.get('evidence', '')
                question = self.env.task_config.get('question', '')
                critique_msg = self.critique_agent.critique_sql(sql_query, self.reference_plan, question, schema_string, evidence, response = response, execution_feedback=obs)
                logger.info(f"[MCP] Critique: {critique_msg}")
                # 若為結構性問題，寫回 plan['critique_notes']
                if critique_msg and any(x in critique_msg['reasoning'].lower() for x in ["group by", "missing", "structure", "aggregate", "column", "select", "where", "join"]):
                    if isinstance(self.reference_plan, dict):
                        self.reference_plan.setdefault('critique_notes', []).append(critique_msg)

            if last_action is not None and last_action == action:
                if repeat_action:
                    return False, "ERROR: Repeated action"
                else:
                    obs = "The action is the same as the last one, you MUST provide a DIFFERENT SQL code or Python Code or different action."
                    repeat_action = True
                    continue

            obs, done = self.env.step(action)
            last_action = action
            repeat_action = False

            # --- Self-Refinement Trigger ---
            is_error = "error" in obs.lower() or "exception" in obs.lower()
            is_empty = ("no rows" in obs.lower() or "empty result" in obs.lower()) and not is_error

            if self.self_refinement_enabled and sql_query and (is_error or is_empty):
                logger.info("Triggering self-refinement due to error or empty result.")
                error_msg = obs if is_error else None
                empty_result = is_empty
                success, refined_sql, obs, error_type, refined_action = self._self_refine(sql_query, obs, error_msg=error_msg, empty_result=empty_result)
                if success:
                    logger.info("Self-refinement succeeded.")
                    result = obs
                    action = refined_action  # Use the refined action for done determination
                else:
                    logger.info("Self-refinement failed after maximum attempts.")
                    result = obs

            # Let done be determined by action
            if isinstance(action, Terminate):
                done = True
                result = action.output
                logger.info("The task is done.")
                break
            step_idx += 1

        return done, result