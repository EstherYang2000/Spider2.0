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

def extract_eq_columns_from_where(sql_query: str) -> List[str]:
    """
    從 SQL 的 WHERE 子句自動抓取所有等值過濾的欄位名（如 a.xxx = 'yyy'）。
    支援多個條件、AND/OR、別名。回傳所有欄位名（含 table/alias 前綴）。
    """
    # 只抓 WHERE ... 之後到 GROUP BY/ORDER BY/結尾
    match = re.search(r"where(.+?)(group by|order by|limit|$)", sql_query, re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    where_clause = match.group(1)
    # 找所有 xxx = 'yyy' 或 xxx = N
    pattern = re.compile(r"([\w\.]+)\s*=\s*('[^']*'|\d+|\?)", re.IGNORECASE)
    columns = [m.group(1).strip() for m in pattern.finditer(where_clause)]
    # 去重
    return list(sorted(set(columns)))


def generate_distinct_sqls(sql_query: str, target_columns: List[str]) -> List[Tuple[str, str]]:
    """
    根據原始 SQL query 和目標欄位，產生 [(col, distinct_sql)] 列表。
    每個 distinct_sql 會保留 FROM/JOIN 結構，只查詢該欄位的 distinct 值。
    """
    # 抓 FROM ... 之後到 GROUP BY/ORDER BY/結尾
    match = re.search(r"from(.+?)(group by|order by|limit|$)", sql_query, re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    from_clause = match.group(0).strip()
    # 產生每個欄位的 distinct SQL
    sqls = []
    for col in target_columns:
        dsql = f"SELECT DISTINCT {col} {from_clause} LIMIT 20"
        sqls.append((col, dsql))
    return sqls


from spider_agent.agent.agents import PromptAgent
from spider_agent.agent.planner_critique_agents import PlannerAgent, CritiqueAgent
from spider_agent.agent.action import Terminate, BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, LOCAL_DB_SQL
from spider_agent.agent.models import call_llm

logger = logging.getLogger("spider_agent")

@dataclass
class RefinementResult:
    """Class to store the result of a refinement iteration"""
    sql_query: str
    result: str
    is_empty: bool = False
    error: bool = False
    error_type: str = ""

    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss

    def _should_probe_column(self, col: str) -> bool:
        """
        決定某個欄位是否適合做 distinct probing（過濾 id/time/date 等常見不適合 probing 的欄位）
        """
        ignore_keywords = ["id", "time", "date", "created", "updated"]
        return not any(kw in col.lower() for kw in ignore_keywords)

    def _parse_distinct_values(self, obs: str) -> str:
        """
        解析 SQL distinct 結果，回傳值列表字串。
        假設 obs 是表格格式，第一行是欄位名，後面是值。
        """
        lines = obs.strip().splitlines()
        if len(lines) < 2:
            return "[no values]"
        # 過濾掉分隔線與欄位名
        values = []
        for line in lines[1:]:
            if line.strip() and not set(line.strip()) <= set("-+| "):
                # 只取第一個欄位值
                cell = line.split("|")[0].strip() if "|" in line else line.strip()
                values.append(cell)
        return "[" + ", ".join(values) + "]"

class RefinementLogRAG:
    def __init__(self, log_path="refinement_history.jsonl", model_name="all-MiniLM-L6-v2"):
        self.log_path = log_path
        self.model = SentenceTransformer(model_name)
        self.logs = []
        self.embeddings = None
        self._load_and_embed_logs()

    def _load_and_embed_logs(self):
        import os, json
        self.logs = []
        if not os.path.exists(self.log_path):
            self.embeddings = None
            return
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    case = json.loads(line)
                    self.logs.append(case)
                except Exception:
                    continue
        if self.logs:
            texts = [case.get("original_sql", "") + " " + case.get("refined_sql", "") for case in self.logs]
            self.embeddings = self.model.encode(texts, convert_to_numpy=True)
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        else:
            self.embeddings = None

    def retrieve_similar(self, sql_query, top_k=2):
        if not self.logs or self.embeddings is None:
            return []
        query_emb = self.model.encode([sql_query], convert_to_numpy=True)
        _, indices = self.index.search(query_emb, top_k)
        return [self.logs[idx] for idx in indices[0] if idx < len(self.logs)]


class SelfRefinementAgent(PromptAgent):
    def log_refinement_case(self, original_sql, refined_sql, error, success):
        import json
        record = {
            "original_sql": original_sql,
            "refined_sql": refined_sql,
            "error": error,
            "success": success
        }
        try:
            with open(self.refinement_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log refinement case: {e}")

    def retrieve_similar_refinements(self, sql_query, top_k=2):
        """Retrieve similar past refinements using semantic vector search (RAG)."""
        if not hasattr(self, "_refinement_rag") or self._refinement_rag is None:
            self._refinement_rag = RefinementLogRAG(self.refinement_log_path)
        # Always reload in case log updated (can optimize if needed)
        self._refinement_rag._load_and_embed_logs()
        return self._refinement_rag.retrieve_similar(sql_query, top_k)

    def analyze_sql_error(self, error_msg):
        """
        分析 SQL 執行錯誤訊息，回傳更細緻的錯誤型態與建議。
        會回傳格式：<error_type>: <suggestion>
        """
        msg = error_msg.lower() if error_msg else ""

        # Syntax / Parse
        if any(x in msg for x in ["syntax error", "parse error", "parsererror", "unexpected", "expected"]):
            return "SyntaxError: Please check the SQL syntax, parentheses, commas, single quotes,double quotes, and reserved words."

        # Table not found / does not exist
        if any(x in msg for x in ["no such table", "table not found", "does not exist", "unknown table", "relation does not exist"]):
            return "TableNotFound: Please check the table name, schema, or database."

        # Column not found / does not exist
        if any(x in msg for x in ["no such column", "column not found", "unknown column", "does not exist in", "unrecognized name", "ambiguous column", "invalid column"]):
            return "ColumnNotFound: Please check the column names in SELECT/FROM/WHERE clauses, and ensure all columns exist."

        # Ambiguous column
        if "ambiguous column" in msg or "is ambiguous" in msg:
            return "AmbiguousColumn: Column name is ambiguous, please add table or alias prefix."

        # Function not found
        if any(x in msg for x in ["function not found", "no such function", "unknown function", "not a function"]):
            return "FunctionNotFound: Please check the function name and arguments."

        # Invalid data type
        if any(x in msg for x in ["invalid input syntax", "invalid type", "type mismatch", "cannot cast", "datatype mismatch"]):
            return "TypeError: Please check the data types of columns and literals."

        # Permission
        if any(x in msg for x in ["permission denied", "access denied", "not authorized", "insufficient privileges"]):
            return "PermissionDenied: Please check the database access permissions."

        # Timeout
        if "timeout" in msg or "timed out" in msg or "query exceeded" in msg:
            return "Timeout: Query execution exceeded time limit, please optimize the SQL or check the data volume."

        # Division by zero
        if "division by zero" in msg:
            return "DivisionByZero: Please check the calculation expressions and avoid dividing by zero."

        # Duplicate column
        if "duplicate column" in msg or "column name specified more than once" in msg:
            return "DuplicateColumn: Please ensure all output columns have unique names."

        # Invalid identifier
        if "invalid identifier" in msg or "invalid name" in msg:
            return "InvalidIdentifier: Please check table/column/alias names for typos or reserved words."

        # Resource exceeded (BigQuery/Cloud)
        if "resources exceeded" in msg or "quota exceeded" in msg:
            return "ResourceExceeded: Query exceeded resource or quota limits, try to simplify the query."

        # Not null constraint
        if "null value in column" in msg or "not null constraint" in msg:
            return "NotNullConstraint: Column does not allow NULL values, please check your data or query."

        # Foreign key / constraint error
        if "foreign key constraint" in msg or "constraint failed" in msg:
            return "ConstraintError: Check foreign key or other constraints in your query."

        # No result
        if "no rows" in msg or "empty result" in msg or "no results" in msg:
            return "NoResult: The query returned no data, please check your filtering conditions."

        # Default fallback
        return "OtherError: Please review the error message and fix accordingly."

    """
    Extension of PromptAgent with self-refinement capabilities.
    This agent can iteratively refine SQL queries based on execution results
    until termination conditions are met.
    """
    
    def __init__(
        self,
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
        self_refinement_enabled=True  # 新增：是否啟用自我修正
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
        self.rag_syntax = rag_syntax
        self.max_refinement_iterations = max_refinement_iterations
        self.base_top_k = base_top_k
        self.refinement_iterations = []
        self.consecutive_empty_results = 0
        self.previous_queries = set()
        self.refinement_log_path = "refinement_history.jsonl"  # for self-learning
        self.refinement_rag = RefinementLogRAG(self.refinement_log_path)
        self.planner_agent = PlannerAgent(model=model)
        self.critique_agent = CritiqueAgent(model=model)
        self.plan_critique = None
        self.self_refinement_enabled = self_refinement_enabled  # 新增：自我修正开关

    def run(self):
        """
        Override the run method to include MCP loop: Planning, Critique, and Multi-step Refinement.
        """
        assert self.env is not None, "Environment is not set."
        result = ""
        done = False
        step_idx = 0
        obs = "You are in the folder now."
        retry_count = 0
        last_action = None
        repeat_action = False
        sql_query = None
        critique_msg = None

        # 1. Always generate a plan at the start
        if not self.reference_plan:
            schema_string = self.env.task_config.get('schema', '')
            evidence = self.env.task_config.get('evidence', '')
            question = self.env.task_config.get('question', '')
            self.reference_plan = self.planner_agent.generate_plan(question, schema_string, evidence)
            logger.info(f"[MCP] Generated Plan: {self.reference_plan}") 
        # Plan critique (only once)
        self.plan_critique = self.critique_agent.critique_sql(
            self.reference_plan,  # 直接用 reference_plan 當作 "sql_query" 參數
            self.reference_plan,  # 也可傳 plan 當作 plan
            self.env.task_config.get('question', ''),
            self.env.task_config.get('schema', ''),
            self.env.task_config.get('evidence', '')
        )
        logger.info(f"[MCP] Plan Critique: {self.plan_critique}")

        # ====== 主動詢問 LLM 需要哪些 BigQuery syntax 補充 ======
        syntax_reference = ""
        if self.rag_syntax and self.env.task_config.get('instance_id', '').startswith('bq'):
            from spider_agent.agent.rag_syntax_agent import RAGSyntaxAgent
            question = self.env.task_config.get('question', '')
            schema = self.env.task_config.get('schema', '')
            plan = self.reference_plan
            critique = self.plan_critique
            syntax_query_prompt = (
                f"User Question: {question}\n"
                f"Schema: {schema}\n"
                f"Plan: {plan}\n"
                f"Critique: {critique}\n"
                f"{RAGSyntaxAgent.get_syntax_request_prompt()}"
            )
            syntax_response = call_llm({
                "model": self.model,
                "messages": [{"role": "user", "content": [{"type": "text", "text": syntax_query_prompt}]}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            })
            logger.info(f"Syntax Response: {syntax_response}")
            def parse_syntax_needs(llm_response):
                if not llm_response or "none" in llm_response.lower():
                    return []
                return [item.strip() for item in llm_response.split(",") if item.strip()]
            # 解包 call_llm 回傳 tuple
            if isinstance(syntax_response, tuple):
                syntax_success, syntax_content = syntax_response
            else:
                syntax_success, syntax_content = True, syntax_response
            if not syntax_success or not syntax_content or syntax_content == "None":
                syntax_topics = []
            else:
                syntax_topics = parse_syntax_needs(syntax_content)
            syntax_reference = ""
            if syntax_topics:
                rag_syntax_agent = RAGSyntaxAgent()
                syntax_results = rag_syntax_agent.retrieve(syntax_topics)
                syntax_reference = rag_syntax_agent.format_for_prompt(syntax_results)
                logger.info(f"Syntax Reference: {syntax_reference}")
        # ======================================================

        # 2. MCP Loop: SQL generation, critique, refinement
        def get_plan_step(plan: str, idx: int) -> str:
            """Extract the idx-th step from a numbered plan string."""
            import re
            steps = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|$)', plan, re.DOTALL)
            if not steps:
                return plan.strip()  # fallback: whole plan if not numbered
            if idx < len(steps):
                return steps[idx].strip()
            else:
                return steps[-1].strip()  # fallback: last step

        while not done and step_idx < self.max_steps:
            # On the first step, include the full reference plan; otherwise, only the current step
            if step_idx == 0:
                prompt = f"Plan:\n{self.reference_plan}"
            else:
                current_plan_step = get_plan_step(self.reference_plan, step_idx)
                prompt = f"Plan (current step):\n{current_plan_step}"

            if self.plan_critique:
                prompt += f"\n\nPlan Critique:\n{self.plan_critique}"
            if critique_msg:
                prompt += f"\n\nPrevious Critique:\n{critique_msg}"
            # 融入 syntax_reference
            if syntax_reference and self.env.task_config.get('instance_id', '').startswith('bq') and syntax_reference:
                prompt += f"\n\n{syntax_reference}"
            prompt += f"\n\nObservation:\n{obs}"
            # Use PromptAgent's predict with the structured prompt
            _, action = self.predict(prompt)

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
                critique_msg = self.critique_agent.critique_sql(sql_query, self.reference_plan, question, schema_string, evidence)
                logger.info(f"[MCP] Critique: {critique_msg}")

            if last_action is not None and last_action == action:
                if repeat_action:
                    return False, "ERROR: Repeated action"
                else:
                    obs = "The action is the same as the last one, you MUST provide a DIFFERENT SQL code or Python Code or different action."
                    repeat_action = True
            else:
                # === 擴大 obs 錯誤關鍵字判斷，自動啟用 self-refinement ===
                obs, done = self.env.step(action)
                last_action = action
                repeat_action = False

            # Optionally: Use critique to refine plan (advanced, not default)
            # (You can add logic here to update self.current_plan based on repeated critique)

            if done:
                if isinstance(action, Terminate):
                    result = action.output
                logger.info("The task is done.")
                break
            step_idx += 1

        return done, result

    
    def perform_self_refinement(self, action) -> Tuple[Optional[Any], str]:
        """
        Perform self-refinement on a SQL query action.
        
        Args:
            action: The SQL action to refine
            
        Returns:
            Tuple of (refined_action, observation)
        """
        logger.info("Starting self-refinement process")
        
        # Extract the SQL query from the action
        if isinstance(action, BIGQUERY_EXEC_SQL):
            sql_query = action.sql_query
            action_type = "BIGQUERY_EXEC_SQL"
            is_save = action.is_save
            save_path = action.save_path
        elif isinstance(action, SNOWFLAKE_EXEC_SQL):
            sql_query = action.sql_query
            action_type = "SNOWFLAKE_EXEC_SQL"
            is_save = action.is_save
            save_path = action.save_path
        elif isinstance(action, LOCAL_DB_SQL):
            sql_query = action.code
            action_type = "LOCAL_DB_SQL"
            file_path = action.file_path
            output = action.output
        else:
            # Not a SQL action, skip refinement
            return None, ""
        
        # Initialize refinement variables
        refinement_iterations = 0
        previous_results = []
        self.consecutive_empty_results = 0
        self.consecutive_error_count = 0  # 每次進入 refinement 都歸零
        
        # Add the initial query to previous queries
        self.previous_queries.add(sql_query)
        
        # Execute the initial query
        obs, _ = self.env.step(action)
        
        # Parse the result
        current_result = self._parse_sql_result(obs, sql_query)
        previous_results.append(current_result)
        
        # Check if we need to refine
        if self._should_terminate_refinement(current_result, previous_results):
            logger.info("No refinement needed, terminating refinement process")
            return action, obs
        
        # Start refinement loop
        while refinement_iterations < self.max_refinement_iterations:
            refinement_iterations += 1
            logger.info(f"Refinement iteration {refinement_iterations}")
            

            # Generate refinement prompt
            refinement_prompt = self._generate_refinement_prompt(sql_query, obs, previous_results, refinement_iterations)
            logger.info(f"Refinement prompt: {refinement_prompt}")
            # Call LLM for refinement
            status, refined_sql = self._call_llm_for_refinement(refinement_prompt)
            logger.info(f"Refinement status: {status}")
            logger.info(f"Refined SQL: {refined_sql}")
            
            if not status:
                logger.error(f"Failed to call LLM for refinement: {refined_sql}")
                break
            
            # Extract SQL query from the response
            refined_sql = self._extract_sql_from_response(refined_sql)
            
            if not refined_sql or refined_sql in self.previous_queries:
                logger.info("Refined SQL is empty or duplicate, recording as failed iteration")
                self.refinement_iterations.append({
                    "iteration": refinement_iterations,
                    "sql_query": refined_sql if refined_sql else "",
                    "result": "Empty or duplicate SQL, refinement skipped.",
                    "is_empty": False,
                    "error": True,
                    "skipped": True
                })
                continue
            
            # Check for repeated SQL to avoid infinite loops
            if refined_sql.strip() in [r.sql_query.strip() for r in previous_results]:
                logger.warning("Refined SQL is the same as a previous attempt, terminating refinement to avoid infinite loop.")
                break
            
            # Add to previous queries
            self.previous_queries.add(refined_sql)
            
            # Create a new action with the refined SQL
            if action_type == "BIGQUERY_EXEC_SQL":
                refined_action = BIGQUERY_EXEC_SQL(sql_query=refined_sql, is_save=is_save, save_path=save_path)
            elif action_type == "SNOWFLAKE_EXEC_SQL":
                refined_action = SNOWFLAKE_EXEC_SQL(sql_query=refined_sql, is_save=is_save, save_path=save_path)
            elif action_type == "LOCAL_DB_SQL":
                refined_action = LOCAL_DB_SQL(code=refined_sql, file_path=file_path, output=output)
            
            # Execute the refined query
            refined_obs, _ = self.env.step(refined_action)
            
            # Parse the result
            current_result = self._parse_sql_result(refined_obs, refined_sql)
            previous_results.append(current_result)
            
            # Store refinement iteration
            self.refinement_iterations.append({
                "iteration": refinement_iterations,
                "sql_query": refined_sql,
                "result": refined_obs,
                "is_empty": current_result.is_empty,
                "error": current_result.error
            })
            
            # Check if we should terminate refinement
            if self._should_terminate_refinement(current_result, previous_results):
                logger.info(f"Terminating refinement after {refinement_iterations} iterations")
                
                # If we have a successful result, create a Terminate action
                if not current_result.error and not current_result.is_empty:
                    if action_type == "BIGQUERY_EXEC_SQL" or action_type == "SNOWFLAKE_EXEC_SQL":
                        if is_save and save_path:
                            return Terminate(output=save_path), f"Self-refinement complete. Final result saved to {save_path}"
                        else:
                            return refined_action, refined_obs
                    else:
                        return refined_action, refined_obs
                else:
                    # Return the best result from previous iterations
                    best_result_idx = self._find_best_result_index(previous_results)
                    if best_result_idx == 0:
                        return action, obs
                    else:
                        best_result = previous_results[best_result_idx]
                        if action_type == "BIGQUERY_EXEC_SQL":
                            best_action = BIGQUERY_EXEC_SQL(sql_query=best_result.sql_query, is_save=is_save, save_path=save_path)
                        elif action_type == "SNOWFLAKE_EXEC_SQL":
                            best_action = SNOWFLAKE_EXEC_SQL(sql_query=best_result.sql_query, is_save=is_save, save_path=save_path)
                        elif action_type == "LOCAL_DB_SQL":
                            best_action = LOCAL_DB_SQL(code=best_result.sql_query, file_path=file_path, output=output)
                        
                        # Re-execute the best action to get its observation
                        best_obs, _ = self.env.step(best_action)
                        return best_action, best_obs
            
            # Update for next iteration
            sql_query = refined_sql
            obs = refined_obs
            action = refined_action
        
        # If we reach max iterations without termination, return the best result
        logger.info(f"Reached maximum refinement iterations ({self.max_refinement_iterations})")
        best_result_idx = self._find_best_result_index(previous_results)
        
        if best_result_idx == 0:
            return action, obs
        else:
            best_result = previous_results[best_result_idx]
            if action_type == "BIGQUERY_EXEC_SQL":
                best_action = BIGQUERY_EXEC_SQL(sql_query=best_result.sql_query, is_save=is_save, save_path=save_path)
            elif action_type == "SNOWFLAKE_EXEC_SQL":
                best_action = SNOWFLAKE_EXEC_SQL(sql_query=best_result.sql_query, is_save=is_save, save_path=save_path)
            elif action_type == "LOCAL_DB_SQL":
                best_action = LOCAL_DB_SQL(code=best_result.sql_query, file_path=file_path, output=output)
            
            # Re-execute the best action to get its observation
            best_obs, _ = self.env.step(best_action)
            return best_action, best_obs
    
    def _parse_sql_result(self, observation: str, sql_query: str) -> RefinementResult:
        """
        Parse the SQL execution result from the observation.

        Args:
            observation: The observation from executing the SQL query
            sql_query: The SQL query that was executed

        Returns:
            RefinementResult object containing the parsed result
        """
        result = RefinementResult(sql_query=sql_query, result=observation)

        # 強化錯誤訊息捕捉
        error_keywords = [
            "syntax error", "exception", "traceback", "invalid", "error", "failed", "not found", "no such"
        ]
        for kw in error_keywords:
            if kw in observation.lower():
                result.error = True
                break
        if result.error:
            # 自動分析錯誤型態
            result.error_type = self.analyze_sql_error(observation)
            return result

        # Check for empty results
        empty_patterns = [
            r"0 rows? affected",
            r"no rows? returned",
            r"empty result",
            r"no results?",
            r"returned 0 rows?"
        ]

        for pattern in empty_patterns:
            if re.search(pattern, observation.lower()):
                result.is_empty = True
                self.consecutive_empty_results += 1
                break
        else:
            # Reset consecutive empty results if this result is not empty
            self.consecutive_empty_results = 0

        return result
    
    def _should_terminate_refinement(self, current_result: RefinementResult, previous_results: List[RefinementResult]) -> bool:
        """
        Determine if refinement should be terminated based on the termination conditions.
        Enhanced: Handles consecutive errors, logs error types, and avoids infinite error loops.
        """
        # Track consecutive errors
        if not hasattr(self, 'consecutive_error_count'):
            self.consecutive_error_count = 0
        
        # Condition 1: Error in the current result
        if current_result.error:
            self.consecutive_error_count += 1
            logger.warning(f"[Refinement] Error detected in result (count={self.consecutive_error_count}): {current_result.result}")
            # If too many consecutive errors, terminate refinement
            if self.consecutive_error_count >= 3:
                logger.error("Too many consecutive errors, terminating refinement.")
                return True
            return False
        else:
            self.consecutive_error_count = 0
        
        # Condition 2: Self-consistency - same result twice
        if len(previous_results) >= 2:
            for i in range(len(previous_results) - 1):
                prev_result = previous_results[i]
                if (not prev_result.error and not prev_result.is_empty and 
                    not current_result.error and not current_result.is_empty and
                    self._results_are_equivalent(prev_result.result, current_result.result)):
                    # Only terminate if the result is not 0
                    lines = current_result.result.split("\n")
                    if len(lines) > 1:
                        target_line = lines[1]
                    else:
                        target_line = lines[0] if lines else ""
                    if "0" not in target_line:
                        logger.info("Self-consistency achieved: same result obtained twice")
                        return True
                    else:
                        # Don't terminate if the result is 0, as we know it's incorrect
                        return False
        
        # Condition 3: Consecutive empty results
        if self.consecutive_empty_results >= 2:
            logger.info("Terminating refinement due to consecutive empty results")
            return True
        
        return False
    
    import pandas as pd
    from io import StringIO
    from difflib import SequenceMatcher

    def _results_are_equivalent(self, result1: str, result2: str) -> bool:
        """
        Compare two SQL results using DataFrame equality, with fallback to string similarity.
        """
        def to_df(result):
            lines = [line for line in result.strip().split('\n') if '|' in line and not line.startswith('+')]
            if not lines:
                return None
            columns = [col.strip() for col in lines[0].split('|') if col.strip()]
            data = []
            for line in lines[1:]:
                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                if row:
                    data.append(row)
            try:
                df = pd.DataFrame(data, columns=columns)
                df = df.sort_values(by=columns).reset_index(drop=True)
                return df
            except Exception:
                return None
        df1 = to_df(result1)
        df2 = to_df(result2)
        if df1 is not None and df2 is not None:
            return df1.equals(df2)
        # fallback: string similarity
        similarity = SequenceMatcher(None, result1.strip(), result2.strip()).ratio()
        return similarity > 0.95
    def _extract_all_columns_from_schema(self, schema):
        """
        Extract all column names from schema string or dict.
        """
        # If schema is a dict: {table: [(col, type), ...]}
        if isinstance(schema, dict):
            columns = set()
            for table, cols in schema.items():
                for col, _ in cols:
                    columns.add(col)
            return sorted(list(columns))
        # If schema is a string: parse lines like 'table.column type'
        columns = set()
        import re
        for line in str(schema).splitlines():
            m = re.match(r"([\w\.]+)\s+(\w+)", line.strip())
            if m:
                col = m.group(1)
                columns.add(col)
        return sorted(list(columns))

    def _extract_all_tables_from_schema(self, schema):
        """
        Extract all table names from schema string or dict.
        """
        if isinstance(schema, dict):
            return sorted(list(schema.keys()))
        tables = set()
        import re
        for line in str(schema).splitlines():
            m = re.match(r"([\w]+)\.([\w]+)\s+(\w+)", line.strip())
            if m:
                tables.add(m.group(1))
        return sorted(list(tables))

    def _extract_all_columns_types_from_schema(self, schema):
        """
        Extract column:type mapping from schema string or dict.
        """
        result = {}
        if isinstance(schema, dict):
            for table, cols in schema.items():
                for col, typ in cols:
                    result[col] = typ
            return result
        import re
        for line in str(schema).splitlines():
            m = re.match(r"([\w\.]+)\s+(\w+)", line.strip())
            if m:
                result[m.group(1)] = m.group(2)
        return result
    def _should_probe_column(self, col):
        """
        決定某欄位是否適合 probing。優先只允許 string/text 型欄位。
        """
        # 嘗試根據 schema 型態過濾
        schema = self.env.task_config.get('schema', None)
        if schema is not None:
            # 取得所有欄位型態
            col_types = self._extract_all_columns_types_from_schema(schema)
            col_type = col_types.get(col)
            # 常見 string 型態判斷
            if col_type is not None:
                if col_type.lower() in ['string', 'text', 'varchar', 'char']:
                    return True
                else:
                    return False
            # 若無 schema/type，預設允許所有欄位
        return True

    def _parse_distinct_values(self, obs):
        """
        解析 BigQuery 查詢結果，回傳 distinct value list 字串。
        """
        # 假設 obs 是 list of dict 或 str，根據實際格式調整
        if isinstance(obs, list):
            vals = []
            for row in obs:
                if isinstance(row, dict):
                    vals.extend([str(v) for v in row.values()])
                else:
                    vals.append(str(row))
            return ', '.join(vals)
        elif isinstance(obs, str):
            # 嘗試從字串中提取 value
            import re
            vals = re.findall(r"'([^']+)'", obs)
            if vals:
                return ', '.join(vals)
            return obs
        return str(obs)
    def _generate_refinement_prompt(self, sql_query: str, observation: str, previous_results: List[RefinementResult], refinement_iterations: int) -> str:
        """
        Generate a prompt for the LLM to refine the SQL query.
        
        Args:
            sql_query: The current SQL query
            observation: The observation from executing the SQL query
            previous_results: List of previous refinement results
            refinement_iterations: The number of refinement iterations
            
        Returns:
            A prompt string for the LLM
        """
        # 動態調整 top_k：可根據 refinement 次數，或直接用 self.base_top_k
        if refinement_iterations < 2:
            top_k = self.base_top_k
        elif refinement_iterations < 5:
            top_k = max(self.base_top_k, 3)
        else:
            top_k = max(self.base_top_k, 6)
        similar_cases = self.retrieve_similar_refinements(sql_query, top_k=top_k)

        # 判斷 error 或 no data 狀態
        error = False
        no_data = False
        if previous_results:
            last_result = previous_results[-1]
            error = getattr(last_result, "error", False)
            no_data = getattr(last_result, "is_empty", False)
        
        prompt = f"You are an expert SQL developer. I need your help to refine the following SQL query based on the execution results.\n\n"
        prompt += f"Task: {self.env.task_config['question']}\n\n"
        prompt += f"Current SQL Query:\n```sql\n{sql_query}\n```\n\n"
        prompt += f"Execution Result:\n```\n{observation}\n```\n\n"

        # 根據 error/no_data 狀態插入不同指示語
        if error:
            # 插入 error_type
            error_type = getattr(last_result, "error_type", "") if previous_results else ""
            if error_type:
                prompt += f"\nError Type: {error_type}\n"
            prompt += (
                "The SQL execution returned an ERROR. You MUST carefully read the error message and propose a concrete fix to the SQL."
                " Do not simply rephrase the query—identify and directly address the root cause of the error, such as syntax mistakes, missing or misspelled columns/tables, type mismatches, or logic errors."
                " Your refinement MUST demonstrate a substantial correction that resolves the specific error."
                " In your critique, explicitly state what was wrong and how your fix addresses it."
            )
        elif no_data:
            prompt += (
                "The SQL executed successfully but returned NO DATA. Please check if the filtering conditions are too strict, if the target value exists in the data, or if there are any logic errors that may cause empty results."
                " Try to debug by querying the distinct values of key columns to identify potential mismatches."
                " Your refinement should adjust filters or logic to retrieve relevant data, and your critique must explain your reasoning."
            )
            # ====== 智能 distinct value probing（結構化 & 避免重複） ======
            if not hasattr(self, '_probed_columns'):
                self._probed_columns = set()
            target_columns = [col for col in extract_eq_columns_from_where(sql_query) if self._should_probe_column(col)]
            probing_results = []
            for col in target_columns:
                if col in self._probed_columns:
                    continue
                dsql = f"SELECT DISTINCT {col} FROM (" + sql_query + ") AS sub LIMIT 20"
                try:
                    action = self._make_sql_action(dsql) if hasattr(self, '_make_sql_action') else dsql
                    obs, _ = self.env.step(action)
                    values = self._parse_distinct_values(obs)
                    probing_results.append(f"{col}: {values}")
                    self._probed_columns.add(col)
                except Exception as e:
                    probing_results.append(f"{col}: [distinct probing error: {e}]")
            if probing_results:
                # 美化 probing 結果為 markdown table
                prompt += "\nBelow are the possible values for key columns (based on current data). Please use these values to adjust your SQL if needed.\n"
                prompt += "\n| Column | Distinct Values |\n|--------|----------------|\n"
                for pr in probing_results:
                    if ":" in pr:
                        col, vals = pr.split(":", 1)
                        prompt += f"| `{col.strip()}` | {vals.strip()} |\n"
                prompt += "\n"

        # 錯誤型態導向自動 probing
        error_type = getattr(last_result, "error_type", "") if previous_results else ""
        # 1. SyntaxError/NoResult: 自動 probing 目標欄位
        if error_type and (error_type.startswith("SyntaxError") or error_type.startswith("NoResult")):
            # 自動 distinct probing
            if not hasattr(self, '_probed_columns'):
                self._probed_columns = set()
            target_columns = [col for col in extract_eq_columns_from_where(sql_query) if self._should_probe_column(col)]
            probing_results = []
            for col in target_columns:
                if col in self._probed_columns:
                    continue
                dsql = f"SELECT DISTINCT {col} FROM (" + sql_query + ") AS sub LIMIT 20"
                try:
                    action = self._make_sql_action(dsql) if hasattr(self, '_make_sql_action') else dsql
                    obs, _ = self.env.step(action)
                    values = self._parse_distinct_values(obs)
                    probing_results.append(f"{col}: {values}")
                    self._probed_columns.add(col)
                except Exception as e:
                    probing_results.append(f"{col}: [distinct probing error: {e}]")
            if probing_results:
                prompt += "\nBelow are the possible values for key columns (based on current data). Please use these values to adjust your SQL if needed.\n"
                prompt += "\n| Column | Distinct Values |\n|--------|----------------|\n"
                for pr in probing_results:
                    if ":" in pr:
                        col, vals = pr.split(":", 1)
                        prompt += f"| `{col.strip()}` | {vals.strip()} |\n"
                prompt += "\n"



        # 2. 其他 error_type 對應 schema probing
        if error_type:
            if error_type.startswith("ColumnNotFound"):
                schema = self.env.task_config.get('schema', '')
                all_columns = self._extract_all_columns_from_schema(schema)
                if all_columns:
                    prompt += "\nThe following are all valid columns in the database. Please ensure your SQL only uses these columns.\n"
                    prompt += ", ".join(f'`{c}`' for c in all_columns) + "\n"
            elif error_type.startswith("TableNotFound"):
                schema = self.env.task_config.get('schema', '')
                all_tables = self._extract_all_tables_from_schema(schema)
                if all_tables:
                    prompt += "\nThe following are all valid tables in the database. Please ensure your SQL only uses these tables.\n"
                    prompt += ", ".join(f'`{t}`' for t in all_tables) + "\n"
            elif error_type.startswith("TypeError"):
                schema = self.env.task_config.get('schema', '')
                all_columns_types = self._extract_all_columns_types_from_schema(schema)
                if all_columns_types:
                    prompt += "\nColumn types for reference (please match types in your SQL):\n"
                    prompt += "\n| Column | Type |\n|--------|------|\n"
                    for col, typ in all_columns_types.items():
                        prompt += f"| `{col}` | {typ} |\n"
                    prompt += "\n"

        # 3. 若 SQL 字串 literal 含單引號，提醒 LLM 用雙引號/三重引號包覆
        import re
        string_literals = re.findall(r"'(.*?)'", sql_query)
        if any("'" in val for val in string_literals):
            prompt += ("\n[Notice] Your SQL string literal contains apostrophes ('). "
                       "In BigQuery, use two single quotes ('') to escape an apostrophe inside a string. "
                       "If you are passing SQL as a string in Python or another language, consider using double quotes (\"...\") or triple quotes (\"\"\"...\"\"\") to wrap the SQL, and avoid mixing escape styles.\n")
        else:
            prompt += (
                "Please analyze the result and refine the SQL if necessary to better match the user intent."
                " Your critique must be actionable and specific—do not just restate the result, but explain what you changed and why."
            )

        prompt += (
            "\nSQL Refinement Guidelines:\n"
            "1. You MUST directly address and fix any error messages."
            "2. Critique should include actionable, specific suggestions for resolving the error or improving the query."
            "3. Carefully analyze the error messages or empty results."
            "4. Check for syntax errors, incorrect table names, or missing joins."
            "5. Ensure column names are correct and properly referenced."
            "6. Verify that filtering conditions are appropriate for the task."
            "7. Consider using Common Table Expressions (CTEs) to break down complex logic."
            "8. Make sure aggregation functions are used correctly."
            "9. Ensure the output column names match what's expected in the task."
            "10. For date-based queries, verify the date format and filtering approach."
            "11. For BigQuery tables with date suffixes, use _TABLE_SUFFIX for filtering when appropriate."
        )
        if similar_cases:
            prompt += "\nRelevant Past Refinements:\n"
            for case in similar_cases:
                prompt += f"- Original SQL: {case.get('original_sql','')[:80]}...\n"
                prompt += f"  Refined SQL: {case.get('refined_sql','')[:80]}...\n"
                prompt += f"  Error: {case.get('error','')}\n"
        if previous_results and len(previous_results) > 1:
            prompt += "\nPrevious Refinement Attempts:\n"
            for idx, res in enumerate(previous_results[:-1]):
                prompt += f"- Iteration {idx+1}: SQL: {res.sql_query}\n  | Result: {res.result}\n"
                prompt += f"  Error: {res.error}\n"
        else:
            prompt += "\nNo previous refinement results.\n"
        
        prompt += (
            "\nIMPORTANT: Do NOT repeat any of the previous SQL queries listed above. "
            "If you encounter the same error as before, you MUST try a different approach or significantly modify the SQL structure. "
            "If you cannot fix the issue, clearly state the reason.\n"
            "Please provide a refined SQL query that addresses the issues. Return ONLY the SQL query without any explanations or markdown formatting.\n"
        )
        return prompt

    def _make_sql_action(self, sql: str):
        """
        Helper: 將 SQL 字串包裝成對應的 action。
        根據 self.action_type 屬性自動判斷 BigQuery、Snowflake、Local DB。
        """
        from spider_agent.agent.action import BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, LOCAL_DB_SQL
        action_type = getattr(self, 'action_type', None)
        if action_type is None and hasattr(self, 'env') and hasattr(self.env, 'action_type'):
            action_type = getattr(self.env, 'action_type', None)
        # 預設 BigQuery
        if action_type == 'SNOWFLAKE_EXEC_SQL':
            return SNOWFLAKE_EXEC_SQL(sql_query=sql, is_save=False, save_path=None)
        elif action_type == 'LOCAL_DB_SQL':
            return LOCAL_DB_SQL(sql_query=sql, is_save=False, save_path=None)
        else:
            return BIGQUERY_EXEC_SQL(sql_query=sql, is_save=False, save_path=None)

    def _call_llm_for_refinement(self, prompt: str) -> Tuple[bool, str]:
        """
        Call the LLM to refine the SQL query.
        
        Args:
            prompt: The refinement prompt
            
        Returns:
            Tuple of (success, response)
        """
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an expert SQL developer. Your task is to refine SQL queries based on execution results."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        return call_llm({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature
        })
    
    def _extract_sql_from_response(self, response: str) -> str:
        """
        Extract the SQL query from the LLM response.
        
        Args:
            response: The LLM response
            
        Returns:
            The extracted SQL query
        """
        # Try to extract SQL from code blocks
        sql_matches = re.findall(r'```(?:sql)?\s*(.*?)\s*```', response, re.DOTALL)
        if sql_matches:
            return sql_matches[0].strip()
        
        # If no code blocks, return the entire response
        return response.strip()
    
    def _find_best_result_index(self, results: List[RefinementResult]) -> int:
        """
        Find the index of the best result from the list of results.
        
        Args:
            results: List of refinement results
            
        Returns:
            Index of the best result
        """
        # Prioritize non-error, non-empty results
        valid_indices = [i for i, result in enumerate(results) if not result.error and not result.is_empty]
        if valid_indices:
            return valid_indices[-1]  # Return the latest valid result
        
        # If no valid results, prioritize non-error results
        non_error_indices = [i for i, result in enumerate(results) if not result.error]
        if non_error_indices:
            return non_error_indices[-1]
        # If all have errors, return the first result (original query)
        return 0
    def get_trajectory(self):
        """
        Override get_trajectory to include refinement information.
        """
        trajectory = super().get_trajectory()
        
        # Add refinement information
        trajectory["refinement_iterations"] = self.refinement_iterations
        
        return trajectory
