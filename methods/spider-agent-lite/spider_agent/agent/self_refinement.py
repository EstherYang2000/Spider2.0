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
    # Predefined common SQL syntax error correction examples
    PREDEFINED_CASES = [
        # MySQL/SQLite style (backslash escape)
        {
            "original_sql": "SELECT * FROM orders WHERE product_name = 'Youtube Men's Vintage Henley'",
            "refined_sql": "SELECT * FROM orders WHERE product_name = 'Youtube Men\'s Vintage Henley'",
            "error": "Syntax error: Unterminated string literal",
            "success": True,
            "error_type": "StringLiteral",
            "description": "MySQL/SQLite: Fix single quotes by escaping with backslash",
            "dialect": "sqlite"
        },
        # BigQuery/PostgreSQL style (double single quotes)
        {
            "original_sql": "SELECT * FROM orders WHERE product_name = 'Youtube Men's Vintage Henley'",
            "refined_sql": "SELECT * FROM orders WHERE product_name = 'Youtube Men''s Vintage Henley'",
            "error": "Syntax error: Unterminated string literal",
            "success": True,
            "error_type": "StringLiteral",
            "description": "BigQuery/PostgreSQL: Fix single quotes by doubling them",
            "dialect": "bigquery"
        },
        # Snowflake: String literal with single quotes (double single quotes)
        {
            "original_sql": "SELECT * FROM orders WHERE product_name = 'Youtube Men's Vintage Henley'",
            "refined_sql": "SELECT * FROM orders WHERE product_name = 'Youtube Men''s Vintage Henley'",
            "error": "SQL compilation error: syntax error line 1 at position ...",
            "success": True,
            "error_type": "StringLiteral",
            "description": "Snowflake: Fix single quotes in string literals by doubling them",
            "dialect": "snowflake"
        },
        # Snowflake: Quoting identifiers (case sensitivity)
        {
            "original_sql": "SELECT ProductName FROM Orders",
            "refined_sql": "SELECT \"PRODUCTNAME\" FROM \"ORDERS\"",
            "error": "SQL compilation error: Unknown column 'ProductName'",
            "success": True,
            "error_type": "IdentifierCase",
            "description": "Snowflake: Use double quotes for case-sensitive identifiers",
            "dialect": "snowflake"
        },
        # Snowflake: Data type conversion
        {
            "original_sql": "SELECT order_id + '123' FROM orders",
            "refined_sql": "SELECT CAST(order_id AS STRING) || '123' FROM orders",
            "error": "SQL compilation error: Cannot apply '+' to arguments of type ...",
            "success": True,
            "error_type": "TypeMismatch",
            "description": "Snowflake: Use CAST and '||' for string concatenation",
            "dialect": "snowflake"
        },
        # Snowflake: Date/time parsing
        {
            "original_sql": "SELECT * FROM orders WHERE order_date = '2023-01-01'",
            "refined_sql": "SELECT * FROM orders WHERE TO_DATE(order_date) = '2023-01-01'",
            "error": "SQL compilation error: Can't compare date with string",
            "success": True,
            "error_type": "DateTime",
            "description": "Snowflake: Use TO_DATE or TO_TIMESTAMP for date/time comparisons",
            "dialect": "snowflake"
        },
        # Snowflake: Ambiguous column reference in JOIN
        {
            "original_sql": "SELECT * FROM orders JOIN customers ON id = customer_id",
            "refined_sql": "SELECT * FROM orders JOIN customers ON orders.id = customers.customer_id",
            "error": "SQL compilation error: ambiguous column reference 'id'",
            "success": True,
            "error_type": "JoinCondition",
            "description": "Snowflake: Fully qualify column names in JOIN conditions",
            "dialect": "snowflake"
        },
        {
            "original_sql": "SELECT COUNT(*) products GROUP BY category",
            "refined_sql": "SELECT COUNT(*) as count FROM products GROUP BY category",
            "error": "Syntax error: FROM keyword not found",
            "success": True,
            "error_type": "MissingKeyword",
            "description": "Add missing FROM clause in the query",
            "dialect": "sqlite"
        },
        {
            "original_sql": "SELECT * FROM orders INNER JOIN users ON user_id",
            "refined_sql": "SELECT * FROM orders INNER JOIN users ON orders.user_id = users.id",
            "error": "Syntax error: ON clause must be a boolean expression",
            "success": True,
            "error_type": "JoinCondition",
            "description": "Specify complete JOIN condition with table names and fields",
            "dialect": "sqlite"
        },
        {
            "original_sql": "SELECT product_name, SUM(quantity), AVG(price) FROM sales",
            "refined_sql": "SELECT product_name, SUM(quantity), AVG(price) FROM sales GROUP BY product_name",
            "error": "Syntax error: Column must appear in the GROUP BY clause or be used in an aggregate function",
            "success": True,
            "error_type": "GroupBy",
            "description": "Add missing GROUP BY clause for non-aggregated columns",
            "dialect": "sqlite"
        },
        {
            "original_sql": "SELECT * FROM orders WHERE date = '2023-01-01' AND time BETWEEN '09:00' AND '17:00'",
            "refined_sql": "SELECT * FROM orders WHERE DATE(date) = '2023-01-01' AND TIME(time) BETWEEN '09:00:00' AND '17:00:00'",
            "error": "Syntax error: Invalid timestamp format",
            "success": True,
            "error_type": "DateTime",
            "description": "Use correct date/time functions and standard format for timestamps",
            "dialect": "sqlite"
        }
    ]

    def __init__(self, log_path="refinement_history.jsonl", model_name="all-MiniLM-L6-v2", dialect=None):
        self.log_path = log_path
        self.model = SentenceTransformer(model_name)
        self.logs = []
        self.embeddings = None
        self.dialect = dialect
        self._load_and_embed_logs()

    def _load_and_embed_logs(self):
        import os, json
        # 首先加载预定义的案例
        self.logs = [case for case in self.PREDEFINED_CASES if case.get("dialect") == self.dialect]
        
        # 然后加载历史记录
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        case = json.loads(line)
                        if case.get("dialect") == self.dialect:
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
        """
        只將成功（success==True）的 refinement case 寫入 refinement_log_path。
        """
        if not success:
            return
        if "Results saved to /workspace/result.csv" in error:
            return
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
            self._refinement_rag = RefinementLogRAG(self.refinement_log_path, dialect=self.dialect)
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
        self.expected_csv_format = expected_csv_format  # 新增：答案格式要求
        self.use_schema_linking = use_schema_linking  # 新增：是否啟用 schema linking
        self.schema_retriever = None  # 兩階段 schema 檢索器（延遲初始化）
 

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
        for attempt in range(self.max_refinement_iterations):
            # 1. Analyze error or empty result
            if error_msg:
                error_type = self.analyze_sql_error(error_msg)
                critique_msg = f"Error encountered: {error_type}\nError message: {error_msg}"
            elif empty_result:
                error_type = "NoResult: The query returned no data, please check your filtering conditions."
                critique_msg = (
                    "Previous SQL returned empty result. "
                    "Please consider if the filtering conditions are too strict or if there is a mistake in the WHERE clause. "
                    "Try relaxing some conditions or checking for possible column/value mismatches."
                )
            else:
                error_type = "Unknown"
                critique_msg = "Unknown issue encountered."

            # 2. Retrieve similar refinement cases
            similar_cases = self.retrieve_similar_refinements(original_sql, top_k=self.base_top_k)
            similar_text = ""
            if similar_cases:
                similar_text = "\n\nSimilar past refinements:\n"
                for case in similar_cases:
                    similar_text += f"- Original SQL: {case.get('original_sql')}\n  Refined SQL: {case.get('refined_sql')}\n  Error: {case.get('error')}\n  Success: {case.get('success')}\n"

            # 3. Construct prompt for LLM
            rewrite_hint = ""
            if last_error_type is not None and error_type == last_error_type:
                rewrite_hint = (
                    "\n\nNote: The previous fix did NOT resolve the problem. "
                    "Please try a significantly different approach to rewrite the SQL, "
                    "not just minor changes."
                )
            # 若遇到 NoResult 類型，額外強化提示 LLM
            extra_noresult_hint = ""
            if error_type.startswith("NoResult"):
                extra_noresult_hint = (
                    "\n[Extra Hint] The previous query returned no data. "
                    "Please try the following:\n"
                    "- Double-check the database schema to ensure all table and column names are correct and exist.\n"
                    "- Verify the data types of columns used in filters (e.g., date formats, string vs. number).\n"
                    "- Consider possible differences in column names or nested structures in the schema.\n"
                    "- Check for possible mistakes in column values (e.g., typos, case sensitivity).\n"
                    "- If possible, try a different approach or alternative query to retrieve some data."
                )
            # 新增：插入格式要求
            format_instruction = ""
            if self.expected_csv_format:
                format_instruction = (
                    f"\nExpected CSV format:\n{self.expected_csv_format}\n"
                    "The output must strictly follow this CSV format."
                )

            prompt = (
                f"Original SQL:\n{original_sql}\n\n"
                f"{critique_msg}\n"
                f"{similar_text}\n"
                "Please refine the SQL query to fix the above issue."
                f"{rewrite_hint}"
                f"{extra_noresult_hint}"
            )
            logger.info(f"[Self-Refinement] Prompt: {prompt}")
            # 4. Generate new SQL via LLM
            _, action = self.predict(prompt)
            if action is None or not hasattr(action, 'sql_query') or not action.sql_query:
                continue  # Try again

            refined_sql = action.sql_query

            # 5. Critique and execute
            schema_string = self.env.task_config.get('schema', '')
            evidence = self.env.task_config.get('evidence', '')
            question = self.env.task_config.get('question', '')
            critique_msg = self.critique_agent.critique_sql(refined_sql, self.reference_plan, question, schema_string, evidence)

            # update last_error_type for next round
            last_error_type = error_type

            obs, _ = self.env.step(action)  # Let external code determine done
            is_error = "error" in obs.lower() or "exception" in obs.lower()
            is_empty = ("no rows" in obs.lower() or "empty result" in obs.lower()) and not is_error

            # Log this refinement attempt
            self.log_refinement_case(original_sql, refined_sql, error=obs, success=(not is_error and not is_empty))

            if not is_error and not is_empty:
                return True, refined_sql, obs, error_type, action
            # Otherwise, try again with the new SQL

        # If all attempts fail, return last result
        return False, refined_sql, obs, error_type, action
    
    def run(self):
        """
        Override the run method to include MCP loop: Planning, Critique, and Multi-step Refinement with self-refinement on error or empty result.
        """
        assert self.env is not None, "Environment is not set."
        # --- 自動尋找 ddl.csv 並導入兩階段 Schema 檢索 Patch ---
        if getattr(self, "use_schema_linking", False):
            logger.info(self.schema_string)
            if not self.schema_string:
                ddl_path = None
                ddl_path = self.find_ddl_csv(self.env.mnt_dir, self.env.task_config.get('db'))
                logger.info(f"[MCP] DDL Path: {ddl_path}")
                ddl_path = None if not ddl_path else ddl_path[0]
                if ddl_path:
                    if self.schema_retriever is None:
                        self.schema_retriever = TwoStageSchemaRetriever(ddl_path)
                    schema_string = self.schema_retriever.retrieve(
                        getattr(self.env, 'question', self.env.task_config.get('question', ''))
                    )
                self.schema_string = schema_string
            else:
                logger.warning("No DDL/schema path found, skipping two-stage schema retrieval.")
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
            if self.use_schema_linking:
                # === Schema Linking Agent (先做 schema linking) ===
                try:
                    from spider_agent.agent.schema_link_agent import SchemaLinkAgent
                    if not hasattr(self, 'schema_link_agent'):
                        self.schema_link_agent = SchemaLinkAgent()
                    schema_linking_result = self.schema_link_agent.link(question, schema_string)
                    logger.info(f"[MCP] Schema Linking: {schema_linking_result}")
                    self.env.task_config['linked_tables'] = schema_linking_result['linked_tables']
                    self.env.task_config['linked_columns'] = schema_linking_result['linked_columns']
                except Exception as e:
                    logger.error(f"[MCP] Schema Linking failed: {e}")
                # 再產生 plan，可將 linking 結果傳給 planner（如 planner 支援）
                self.reference_plan = self.planner_agent.generate_plan(
                    question, schema_string, evidence,
                )
            else:
            # 不啟用 schema linking，直接產生 plan
                self.reference_plan = self.planner_agent.generate_plan(
                question, schema_string, evidence
            )
        logger.info(f"[MCP] Generated Plan in the Refinement Agent: {self.reference_plan}")

        max_plan_refine = 2
        refine_count = 0
        while refine_count < max_plan_refine:
            self.plan_critique = self.critique_agent.critique_sql(
                self.reference_plan,
                self.reference_plan,
                self.env.task_config.get('question', ''),
                self.env.task_config.get('schema', ''),
                self.env.task_config.get('evidence', '')
            )[1]
            logger.info(f"[MCP] Plan Critique: {self.plan_critique}")
            if self.plan_critique and "no issue" not in self.plan_critique.lower():
                self.reference_plan.setdefault('critique_notes', []).append(self.plan_critique)
                # === 自動修正 plan ===
                plan_prompt = (
                    f"Original Plan:\n{self.reference_plan['plan']}\n\n"
                    f"Critique Notes:\n" + "\n".join(self.reference_plan['critique_notes']) +
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
        def get_plan_step(plan_obj, idx: int) -> str:
            """Extract the idx-th step from a numbered plan string, and append critique notes if any."""
            import re
            plan = plan_obj['plan'] if isinstance(plan_obj, dict) else plan_obj
            steps = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|$)', plan, re.DOTALL)
            if not steps:
                step_text = plan.strip()  # fallback: whole plan if not numbered
            elif idx < len(steps):
                step_text = steps[idx].strip()
            else:
                step_text = steps[-1].strip()  # fallback: last step
            # 動態補入 critique_notes
            critique_notes = plan_obj.get('critique_notes', []) if isinstance(plan_obj, dict) else []
            if critique_notes:
                notes = "\n".join([f"[Critique Note] {note}" for note in critique_notes])
                step_text += f"\n\n{notes}"
            return step_text

        while not done and step_idx < self.max_steps:
            # On the first step, include the full reference plan; otherwise, only the current step
            if step_idx == 0:
                prompt = f"Plan:\n{self.reference_plan['plan']}"
            else:
                current_plan_step = get_plan_step(self.reference_plan['plan'], step_idx)
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
                critique_msg = self.critique_agent.critique_sql(sql_query, self.reference_plan, question, schema_string, evidence,execution_feedback=obs)[1]
                logger.info(f"[MCP] Critique: {critique_msg}")
                # 若為結構性問題，寫回 plan['critique_notes']
                if critique_msg and any(x in critique_msg.lower() for x in ["group by", "missing", "structure", "aggregate", "column", "select", "where", "join"]):
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