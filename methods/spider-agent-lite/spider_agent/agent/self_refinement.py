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
from spider_agent.agent.refinement_memory_bank_agent import RefinementLogRAG
import hashlib

logger = logging.getLogger("spider_agent") 



class SelfRefinementAgent(PromptAgent):
    def try_rag_fix_from_history(self, original_sql: str, error_type: str) -> Optional[str]:
        """從歷史修正記錄中根據 error_type 和相似 SQL 找到潛在修正版本"""
        if not hasattr(self, "refinement_rag"):
            self.refinement_rag = RefinementLogRAG()
        matched = self.refinement_rag.match(original_sql, error_type=error_type, top_k=3)
        if matched:
            _, record = matched[0]
            logger.info(f"[SelfRefinementAgent] Found similar fix from history for error {error_type}.")
            return record["refined_sql"]
        return None


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
        error_type = self.analyze_sql_error(error)
        historical_fix = self.try_rag_fix_from_history(refined_sql, error_type)
        if historical_fix:
            logger.info(f"[SelfRefinementAgent] Applying fix from historical RAG match.")
            return historical_fix.split(":")[0] if error else "Unknown"

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
                "Please verify if the table name and schema are accessible. "
                "If not, suggest an alternative table or schema."
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
        use_schema_linking=False,  # 新增：是否啟用 schema linking
        schema_link_mode='file',  # 新增：schema linking 模式
        validate_result=False  # 新增：是否啟用 validate_result
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
        self.validate_result = validate_result  # 新增：是否啟用 validate_result
        # self.schema_retriever = None  # 兩階段 schema 檢索器（延遲初始化）
        self.schema_link_mode = schema_link_mode  # 'file' or 'sql'
        self.plan_steps: List[str] = self.reference_plan.get("plan", [])
        self.expected_csv_format = self.reference_plan.get("expected_csv_format", "")

    
    def format_similar_cases(self, cases):
        """Format similar past refinement cases into a prompt-ready string."""
        if not cases:
            return ""
        lines = ["\nSimilar past refinements:"]
        for case in cases:
            lines.append(f"- Original SQL: {case.get('original_sql')}\n  Refined SQL: {case.get('refined_sql')}\n  Error Type: {case.get('error_type')}\n  Error: {case.get('error')}\n  Success: {case.get('success')}")
        return "\n".join(lines)
    def compose_refinement_prompt(self, original_sql, obs,critique_msg, error_hint, similar_text, repeated_error, error_type, expected_csv_format):
        dialect_hint = f"-- Target SQL Dialect: {self.dialect.upper()}\n\n"
        rewrite_hint = ""
        if repeated_error:
            rewrite_hint = (
                "\n\nNote: The previous fix did NOT resolve the issue. "
                f"\n\n[Warning] Your previous fix still had the same error type: {error_type}."
                "\n\nTry one of the following strategies:\n"
                "- Change the table joins or filters\n"
                "- Use different columns based on the schema\n"
                "- Try simplifying the query\n"
                "- Use different aggregation or grouping logic\n"
                "- Please change the core logic or try an alternative reasoning path.\n"
                "Avoid repeating the same structure as the previous query."
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
        from spider_agent.agent.refinement_strategies import refinement_strategy_selector
        strategy_prompt = refinement_strategy_selector(error_type)
        return (
            f"{dialect_hint}"
            f"Original SQL:\n{original_sql}\n\n"
            f"Previous Execution Summary:\n{obs}\n\n"
            f"{critique_msg or ''}\n"
            f"{error_hint}\n"
            # f"{similar_text}\n"
            f"{strategy_prompt}\n"
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

    def generate_schema(self, critique_note):
        schema_string = self.env.task_config.get('schema', '')
        question = self.env.task_config.get('question', '')
        if self.use_schema_linking:
            # === Schema Linking Agent (先做 schema linking) ===
            try:
                schema_linking_result = self.schema_agent.run(question, critique_note)
                logger.info(f"[MCP] Schema Linking: {schema_linking_result}")
                
            except Exception as e:
                logger.error(f"[MCP] Schema Linking failed: {e}")
    def validate_result_with_llm(self, question: str, result_csv_path: str) -> dict:
        import pandas as pd
        if not os.path.exists(result_csv_path):
            return {"valid_result": False, "columns_not_needed": [], "result_empty": True, "suggest_fix": "No output was generated and no result.csv file was generated."}
        df = pd.read_csv(result_csv_path)
        sample_rows = df.head(3).to_dict(orient="records")
        null_counts = df.isnull().sum().to_dict()
        summary = {
            "columns": list(df.columns),
            "null_counts": null_counts,
            "num_rows": len(df),
            "sample_rows": sample_rows
        }
        logger.info(f"Result Summary: {summary}")

        prompt = f"""
        Task: Based on the question and the table result, check if the output contains unnecessary columns, missing filters, or excessive empty rows.
        
        Question:
        {question}
        Result Summary:
            Columns: {summary['columns']}
            Empty values per column: {summary['null_counts']}
            Total rows: {summary['num_rows']}
            Example rows: {sample_rows}

        Please answer in this exact JSON format (no extra text or explanation)
        \n Do not include any explanations or Markdown. Only return the pure JSON object.:
        {{
        "valid_result": true or false,
        "columns_not_needed": [column names if any],
        "result_empty": true/false,
        "suggest_fix": "..."
        }}
        
        """
        _, response = call_llm({
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        })
        try:
            # 用非貪婪匹配方式提取包含 "valid_result" 的 JSON 區塊
            match = re.search(r"\{[^{}]*\"valid_result\"[^{}]*\}", response)
            if match:
                return json.loads(match.group())
            else:
                return {"valid_result": False, "columns_not_needed": [], "result_empty": True, "suggest_fix": "No valid JSON match in response"}
        except Exception as e:
            return {"valid_result": False, "columns_not_needed": [], "result_empty": True, "suggest_fix": f"Exception parsing response: {e}"}
    
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
        error_type = None
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
            # similar_text = ""
            prompt = self.compose_refinement_prompt(
                original_sql, obs,critique_msg, error_hint, similar_text,
                last_error_type == error_type, error_type, self.expected_csv_format
            )
            
            if hasattr(self, 'plan_critique'):
                step_idx = self.env.task_config.get('current_step_idx', 0)
                step_critique = self.plan_critique.get('critique_agent', {}).get(str(step_idx))
                if step_critique:
                    prompt += f"\n[Step Critique {step_idx}] {step_critique}"

            if error_type in ["ColumnNotFound", "TableNotFound"]:
                self.use_schema_linking = True
                if self.use_schema_linking:
                    if not critique_msg:
                        critique_msg = f"Error Type: {error_type}. [Error Description] {error_msg} \nThe SQL query might be using non-existent columns or tables. Please analyze the question and suggest the correct schema components."
                        linked_info = self.schema_agent.run(
                            self.env.task_config.get('question', ''),
                            critique_msg
                        )

                if linked_info:
                    self.env.task_config['schema'] = linked_info
                    prompt += f"\nSchema Linking Info: {linked_info}"
                    logger.info(f"[Self-Refinement] Schema Linking Info: {linked_info}")
                    logger.info(f"[Self-Refinement] Old Reference Plan: {self.reference_plan}")
                    logger.info("Generate New Reference Plan")
                    self.generate_reference_plan()
                    logger.info(f"[Self-Refinement] New Reference Plan: {self.reference_plan}")
            # if self.n_refine > 0:
            #     prompt += (
            #         f"Please based on the above information and generate {self.n_refine} correct SQL candidates."
            #         f"\n\nCandidate 1:\n<SQL1>\n\n"
            #         f"Candidate 2:\n<SQL2>\n\n"
            #         f"Candidate 3:\n<SQL3>\n"
            # )
            logger.info(f"[Self-Refinement] Prompt: {prompt}")
            response, action = self.predict(prompt)
            # if self.n_refine > 0:
            #     # 2. 解析出三条 SQL
            #     candidate_sqls = []
            #     for m in re.finditer(r"Candidate\s*\d+\s*:\s*(.*?)\n(?=Candidate|\Z)", response, re.S | re.I):
            #         sql = m.group(1).strip().strip('`')
            #         if sql:
            #             candidate_sqls.append(sql)
            #     # 保底：如果解析失败，就退回到原 logic
            #     if not candidate_sqls:
            #         candidate_sqls = [original_sql]
            #     # 放到类的某处，或直接在方法开头定义
            #     DIALECT_ACTION_MAP = {
            #         "bigquery": BIGQUERY_EXEC_SQL,
            #         "snowflake": SNOWFLAKE_EXEC_SQL,
            #         # 默认用本地执行
            #         "default": LOCAL_DB_SQL,
            #     }
            #     # 3. 并行执行（这里用 for loop，但根据 dialect 选择 action）
            #     results = []  # List of (sql, obs, action, is_error, is_empty)
            #     for sql in candidate_sqls:
            #         # 根据 dialect 拿到对应的 Action 类
            #         action_cls = DIALECT_ACTION_MAP.get(self.dialect.lower(), DIALECT_ACTION_MAP["default"])
                    
            #         if action_cls in (BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL):
            #             # 这两个只需要 sql_query
            #             action = action_cls(sql_query=sql, is_save=False)
            #         else:
            #             # LOCAL_DB_SQL 需要 file_path/command/output，根据你的 env 来填
            #             # 假设你把 sql 放到一个临时文件里，或者直接在 command 里执行
            #             tmp_path = "/tmp/query.sql"
            #             with open(tmp_path, "w") as f:
            #                 f.write(sql)
            #             action = action_cls(
            #                 file_path=tmp_path,
            #                 command=f"psql -d your_db -f {tmp_path}",
            #                 output="result.csv"
            #             )

            #         obs_i, _ = self.env.step(action)
            #         is_error = "error" in obs_i.lower() or "exception" in obs_i.lower()
            #         is_empty = ("no rows" in obs_i.lower() or "empty result" in obs_i.lower()) and not is_error
            #         results.append((sql, obs_i, action, is_error, is_empty))

            if not action or not getattr(action, 'sql_query', ''):
                continue
            prompt_prefix = f" It is the SQL for {self.dialect.upper()} dialect."
            refined_sql = action.sql_query
            

            obs, _ = self.env.step(action)
            critique_msg = self.critique_agent.critique_sql(
                refined_sql, self.reference_plan,
                self.env.task_config.get('question', ''),
                self.env.task_config.get('schema', ''),
                evidence=self.env.task_config.get('evidence', ''),
                response=response,
                execution_feedback=obs,
                prompt_prefix=prompt_prefix
            )
            is_error = "error" in obs.lower() or "exception" in obs.lower()
            is_empty = ("no rows" in obs.lower() or "empty result" in obs.lower()) and not is_error

            # self.log_refinement_case(original_sql, refined_sql, error=obs, success=(not is_error and not is_empty), description=response)

            if not is_error and not is_empty:
                return True, refined_sql, obs, error_type, action

            last_error_type = error_type

        return False, refined_sql, obs, error_type, action
    def build_prompt(self,question, plan_step, critique_msg=None, execution_summary=None, syntax_reference=None):
        prompt = f"[User Question]\n{question}\n\n"
        prompt += f"[Plan Step]\n{plan_step}\n\n"
        if critique_msg and isinstance(critique_msg, dict) and 'reasoning' in critique_msg:
            reasoning = critique_msg['reasoning'].strip().replace("\n", " ")
            prompt += f"[Critique]\n{reasoning}\n\n"
        if "UserWarning: BigQuery Storage module not found" not in execution_summary:
            prompt += f"[Execution Result]\n{execution_summary}\n\n"
        if syntax_reference:
            prompt += f"[Reference Syntax]\n{syntax_reference}\n\n"
        prompt += "Generate your next action."
        return prompt
    def generate_critique_msg(self, sql_query, obs, response) -> Optional[Dict]:
        schema_string = self.env.task_config.get('schema', '')
        evidence = self.env.task_config.get('evidence', '')
        question = self.env.task_config.get('question', '')
        critique_msg = self.critique_agent.critique_sql(
            sql_query, self.reference_plan, question, schema_string, evidence,
            response=response, execution_feedback=obs,
            prompt_prefix=f" It is the SQL for {self.dialect.upper()} dialect."
        )
        logger.info(f"[MCP] Critique: {critique_msg}")
        # if critique_msg and any(
        #     x in critique_msg.get('reasoning', '').lower()
        #     for x in ["group by", "missing", "structure", "aggregate", "column", "select", "where", "join"]):
        if critique_msg:
            self.reference_plan['critique_notes'].append(critique_msg)
        return critique_msg
    def run(self):
        """
        Override the run method to include MCP loop: Planning, Critique, and Multi-step Refinement with self-refinement on error or empty result.
        """
        assert self.env is not None, "Environment is not set."
        obs = "You are in the folder now."

        done, step_idx, result = False, 0, ""
        retry_count, last_action_hash, repeat_action = 0, None, False
        sql_query, critique_msg = None, None

        # --- 初始化 schema_agent ---
        
        if self.schema_link_mode == "file":
            from spider_agent.agent.schema_agent import SchemaAgent, SchemaAgentEnv
            self.schema_agent_env = SchemaAgentEnv(base_dir=self.env.mnt_dir)
            self.schema_agent = SchemaAgent(self.schema_agent_env, llm_predict=self.predict)
        else:
            from spider_agent.agent.database_schema_agent import DBSchemaAgentEnv, SQLSchemaLinkingAgent
            folder_name = self.find_ddl_folder_name(self.env.mnt_dir, self.env.task_config.get('db'))
            self.schema_agent_env = DBSchemaAgentEnv(base_dir=self.env.mnt_dir,schema_path=folder_name)
            self.schema_agent = SQLSchemaLinkingAgent(self.schema_agent_env, llm_predict=self.predict,llm_step=self.env.step,db_type=self.dialect,db_name=self.env.task_config.get('db'))
            

        
        # 1. Always generate a plan at the start
        if not self.reference_plan:
            self.generate_schema(critique_msg)
            self.generate_reference_plan()
        logger.info(f"[MCP] Generated Plan in the Refinement Agent: {self.reference_plan}")
        

        
    # --- Plan Refinement Loop ---
        max_plan_refine, refine_count = 3, 0
        max_schema_refine, schema_refine_count = 2, 0
        self.last_validation_critique = None  # <--- 初始化
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

        # obs += f"\n\nAfter the critique of plan, you are answering the following question:\n{self.env.task_config.get('question', '')}\n\nSchema:\n{self.env.task_config.get('schema', '')}"
        while not done and step_idx < self.max_steps:
            
            question = self.env.task_config.get('question', '')
            plan_step = self.reference_plan['plan']
            plan_step += f"\n\n{self.reference_plan['expected_csv_format']}"
            # 簡化 obs（僅摘要錯誤）
            execution_summary = ""
            if "error" in obs.lower() or "exception" in obs.lower():
                execution_summary = "Error detected in previous query."
                execution_summary += f"\nError Message: {obs}"
            elif "no rows" in obs.lower() or "empty result" in obs.lower():
                execution_summary = "Previous query returned empty result."
                execution_summary += f"\nResult: {obs}"
            else:
                execution_summary =  obs
            syntax_ref = self.syntax_reference if self.rag_syntax and self.env.task_config.get('instance_id', '').startswith('bq') else None
            prompt = self.build_prompt(question, plan_step, critique_msg, execution_summary, syntax_reference=syntax_ref)
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
            # --- judge if action is repeated ---
            # action_hash = hashlib.md5(str(action).encode()).hexdigest()
            action_hash = action
            if last_action_hash == action_hash:
                if repeat_action:
                    return False, "ERROR: Repeated action detected twice."
                else:
                    obs = "Same action repeated. Please generate a different one."
                    repeat_action = True
                    continue

            obs, done = self.env.step(action)
            last_action_hash = action_hash
            repeat_action = False

            # --- Self-Refinement Trigger ---
            is_error = "error" in obs.lower() or "exception" in obs.lower()
            is_empty = ("no rows" in obs.lower() or "empty result" in obs.lower()) and not is_error

            if self.self_refinement_enabled and sql_query and (is_error or is_empty):
                logger.info("Triggering self-refinement due to error or empty result.")
                error_msg = obs if is_error else None
                empty_result = is_empty
                result = self._self_refine(sql_query, obs, error_msg=error_msg, empty_result=empty_result)
                if result is None:
                    logger.warning("[Self-Refinement] Returned None. Skipping.")
                    continue
                success, refined_sql, obs, error_type, refined_action = result

                if success:
                    logger.info("Self-refinement succeeded.")
                    result = obs
                    sql_query = refined_sql
                    action = refined_action  # Use the refined action for done determination
                else:
                    logger.info("Self-refinement failed after maximum attempts.")
                    result = obs

            # 3. Critique the SQL after each generation
            critique_msg = self.generate_critique_msg(sql_query, obs, response)


            # === Validation block ===
            if self.validate_result and isinstance(action, Terminate):
                validate_passed = True
                question = self.env.task_config.get("question", "")
                result_csv_path = os.path.join(self.env.mnt_dir, getattr(action, "output", "result.csv").split("/")[-1])
                logger.info(f"[Validation] Result CSV Path: {result_csv_path}")

                validation_feedback = self.validate_result_with_llm(question, result_csv_path)
                logger.info(f"[Validation Feedback] {validation_feedback}")

                if isinstance(validation_feedback, dict) and (
                    validation_feedback.get("result_empty") or validation_feedback.get("columns_not_needed") or not validation_feedback.get("valid_result")
                ):
                    validate_passed = False
                    critique_msg = {"reasoning": validation_feedback.get("suggest_fix", "Output validation failed.")}

                if validate_passed and validation_feedback.get("valid_result"):
                    done = True
                    result = action.output
                    logger.info("The task is done.")
                    break
                else:
                    done = False
                    logger.info("Validation failed. Continuing to next refinement step.")
                    # step_idx += 1
                    continue

            elif isinstance(action, Terminate) and not self.validate_result:
                done = True
                result = action.output
                logger.info("The task is done.")
                break

            step_idx += 1

        return done, result