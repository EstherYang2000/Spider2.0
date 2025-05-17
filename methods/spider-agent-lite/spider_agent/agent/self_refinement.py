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
        
        # 增加相似度阈值
        similarity_threshold = 0.8
        
        # 获取更多候选修复
        matched = self.refinement_rag.match(
            original_sql, 
            error_type=error_type, 
            top_k=5,  # 增加候选数量
            min_similarity=similarity_threshold
        )
        
        if matched:
            # 选择最佳匹配
            best_match = matched[0]
            if best_match[0] >= similarity_threshold:
                logger.info(f"[SelfRefinementAgent] Found high-confidence fix from history.")
                return best_match[1]["refined_sql"]
        
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

        # Snowflake-specific error patterns
        if "numeric value" in msg and "not recognized" in msg:
            return "NumericFormatError"
        if "timestamp" in msg and "not recognized" in msg:
            return "TimestampFormatError"
        if "variant" in msg and "cannot be cast" in msg:
            return "VariantCastError"
        if "array_contains" in msg and "invalid argument" in msg:
            return "ArrayContainsError"
        if "like" in msg and "invalid argument" in msg:
            return "LikePatternError"
        if "object does not exist" in msg:
            return "ObjectNotFound"
        if "cannot convert" in msg:
            return "TypeConversionError"
        if "not a valid" in msg and "type" in msg:
            return "InvalidTypeError"

        return "OtherError"

    def construct_error_prompt(self, error_type):
        ERROR_HINT_TEMPLATES = {
            # Snowflake-specific error hints
            "NumericFormatError": (
                "The query contains numeric values in an incorrect format. "
                "For Snowflake, ensure numbers are not quoted and use proper decimal points. "
                "Example: Use 123.45 instead of '123.45' or '123,45'."
            ),
            "TimestampFormatError": (
                "The timestamp format is not recognized. "
                "For Snowflake, use TO_TIMESTAMP() or proper timestamp literals. "
                "Example: TO_TIMESTAMP('2023-01-18 00:00:00') or TIMESTAMP '2023-01-18 00:00:00'."
            ),
            "VariantCastError": (
                "There is an issue with the VARIANT type column. "
                "Use proper type casting (::TYPE) or JSON functions to handle VARIANT data. "
                "Example: column::STRING or PARSE_JSON(column)."
            ),
            "ArrayContainsError": (
                "The ARRAY_CONTAINS function is not available in Snowflake. "
                "Use ARRAY_CONTAINS() or proper array functions. "
                "Example: ARRAY_CONTAINS(array_column, value) or value IN (SELECT value FROM TABLE(FLATTEN(input => array_column)))."
            ),
            "LikePatternError": (
                "The LIKE pattern is invalid. "
                "For Snowflake, ensure proper escaping of special characters and use correct wildcards. "
                "Example: Use % for any sequence of characters and _ for a single character."
            ),
            "ObjectNotFound": (
                "The object (table, view, etc.) does not exist. "
                "Verify the object name, schema, and database. "
                "Use SHOW TABLES or DESCRIBE TABLE to list available objects."
            ),
            "TypeConversionError": (
                "Cannot convert between data types. "
                "Use explicit CAST or :: operator for type conversion. "
                "Example: CAST(column AS STRING) or column::STRING."
            ),
            "InvalidTypeError": (
                "The data type is not valid for the operation. "
                "Check the data types of columns and ensure they are compatible. "
                "Use DESCRIBE TABLE to verify column types."
            ),

            # General error hints
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
                "The identifier (table name, column name, etc.) is invalid. "
                "For Snowflake, identifiers are case-sensitive by default. "
                "Use double quotes for case-sensitive identifiers."
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
        self.plan_steps: List[str] = None
        self.expected_csv_format = None

    
    def format_similar_cases(self, cases):
        """Format similar past refinement cases into a prompt-ready string."""
        if not cases:
            return ""
        lines = ["\nSimilar past refinements:"]
        for case in cases:
            lines.append(f"- Original SQL: {case.get('original_sql')}\n  Refined SQL: {case.get('refined_sql')}\n  Error Type: {case.get('error_type')}\n  Error: {case.get('error')}\n  Success: {case.get('success')}")
        return "\n".join(lines)
    def compose_refinement_prompt(self, original_sql, obs, critique_msg, error_hint, similar_text, repeated_error, error_type, expected_csv_format, db):
        dialect_hint = f"-- Target SQL Dialect: {self.dialect.upper()}\n\n"
        
        # Add project and dataset information for BigQuery
        project_info = ""
        if self.dialect.lower() == "bigquery":
            project_id = "bigquery-public-data"
            if project_id:
                project_info = (
                    f"[Project Information]\n"
                    f"Current Project ID: {project_id}\n"
                    "Important: Use these IDs in your queries.\n"
                    "Example: `project_id.dataset_id.table_id`\n"
                    "For public datasets, you can use: `bigquery-public-data.dataset_id.table_id`\n\n"
                )
        
        # Add database context and verification instructions
        db_context = (
            f"[Database Context]\n"
            f"Current Database: {db}\n\n"
            "[Database Verification Steps]\n"
            "1. First, check the database structure:\n"
            "   - Use 'ls' to see all files and directories\n"
            "   - Look for files ending in .sql, .ddl, or .schema\n"
            "   - Check for any README.md or documentation files\n"
            "   - If directory access fails, try alternative paths\n"
            "   - If a directory is not accessible, try parent directory\n\n"
            "2. Verify table names and schema:\n"
            "   - Open any .sql or .ddl files to see table definitions\n"
            "   - Look for CREATE TABLE statements\n"
            "   - Note the exact database and schema names\n"
            "   - Check for any table aliases or views\n"
            "   - If schema file not found, try other directories\n\n"
            "3. Important: Use ONLY the tables and schemas you find in these files\n"
            "   - Do not assume table names\n"
            "   - Do not use tables that aren't explicitly defined\n"
            "   - If unsure, use 'ls' to check again\n"
            "   - If access denied, try alternative paths or files\n\n"
            "4. Error Handling:\n"
            "   - If directory access fails, try parent directory\n"
            "   - If file not found, check for alternative locations\n"
            "   - If permission denied, try different paths\n"
            "   - Document any access issues for troubleshooting\n\n"
        )

        rewrite_hint = ""
        if repeated_error:
            rewrite_hint = (
                "\n[Previous Fix Failed]\n"
                f"The previous fix still had the same error type: {error_type}.\n"
                "Try these alternative strategies:\n"
                "1. Check the database structure and table names again\n"
                "2. Verify the schema and column names in the files\n"
                "3. Try a different approach to solve the problem\n"
                "4. Simplify the query structure\n"
                "5. Use different table joins or filters\n"
                "6. Consider using subqueries or CTEs\n"
                "7. Check for any syntax specific to the database dialect\n"
                "8. Verify directory and file access permissions\n"
            )

        extra_hint = ""
        if error_type.startswith("NoResult"):
            extra_hint = (
                "\n[Empty Result Analysis]\n"
                "The previous query returned no data. Please check:\n"
                "1. Database and table names in the files\n"
                "2. Column names and data types\n"
                "3. Filter conditions and their values\n"
                "4. Join conditions and table relationships\n"
                "5. Date/time formats if used\n"
                "6. Case sensitivity in string comparisons\n"
                "7. Directory and file access permissions\n"
            )

        format_instruction = ""
        if expected_csv_format:
            format_instruction = (
                f"\n[Expected Output Format]\n"
                f"CSV Format:\n{expected_csv_format}\n"
                "Ensure the output matches this format exactly.\n"
            )

        from spider_agent.agent.refinement_strategies import refinement_strategy_selector
        strategy_prompt = refinement_strategy_selector(error_type)

        # Build the base prompt
        prompt = (
            f"{dialect_hint}"
            f"{project_info}\n"
            f"{db_context}\n"
            f"[Original SQL]\n{original_sql}\n\n"
            f"[Previous Execution Result]\n{obs}\n\n"
            f"{critique_msg or ''}\n"
            f"{error_hint}\n"
            f"{similar_text}\n"
            f"{strategy_prompt}\n"
            f"{rewrite_hint}\n"
            f"{extra_hint}\n"
            f"{format_instruction}\n"
            "[Next Steps]\n"
            "1. First verify the database structure and table names\n"
            "2. If directory access fails, try alternative paths\n"
            "3. Document any access issues encountered\n"
            "4. Then refine the SQL query to resolve the issue\n"
            "5. Ensure the output format matches the requirements\n"
            "\n[Required Action Format]\n"
            "You must output exactly one of the following actions (no other text):\n"
            "Important: When dealing with numerical results, DO NOT round the numbers. Keep the full precision of the results.\n"
        )

        # Add dialect-specific action formats
        if self.dialect == "bigquery":
            prompt += (
                f"- Action: {BIGQUERY_EXEC_SQL.__name__}(sql_query=\"...\",is_save=..., save_path=\".../result.csv\")\n"
                f"- Action: {Terminate.__name__}(output=\".../result.csv\")\n\n"
            )
        elif self.dialect == "snowflake":
            prompt += (
                f"- Action: {SNOWFLAKE_EXEC_SQL.__name__}(sql_query=\"...\",is_save=..., save_path=\".../result.csv\")\n"
                f"- Action: {Terminate.__name__}(output=\".../result.csv\")\n\n"
            )
        elif self.dialect == "sqlite":
            prompt += (
                f"- Action: {LOCAL_DB_SQL.__name__}(file_path=\"...\", command=\"...\", output=\".../result.csv\")\n"
                f"- Action: {Terminate.__name__}(output=\".../result.csv\")\n\n"
            )
        else:
            # Default action format if dialect is not recognized
            prompt += (
                f"- Action: {Terminate.__name__}(output=\".../result.csv\")\n\n"
            )

        return prompt
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
                last_error_type == error_type, error_type, self.expected_csv_format, self.env.task_config.get('db', '')
            )
            logger.info(f"[Self-Refinement] Attempt {attempt + 1} Prompt: {prompt}")
            
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
            logger.info(f"[Self-Refinement] Prompt: {prompt}")
            response, action = self.predict(prompt)
            logger.info(f"[Self-Refinement] Attempt {attempt + 1} Response: {response}")
            logger.info(f"[Self-Refinement] Attempt {attempt + 1} Action: {action}")
            if not action or not getattr(action, 'sql_query', ''):
                continue
            prompt_prefix = f" It is the SQL for {self.dialect.upper()} dialect."
            refined_sql = action.sql_query
            

            obs, _ = self.env.step(action)
            critique_msg = self.critique_agent.critique_sql(
                self.dialect,
                refined_sql, self.reference_plan,
                self.env.task_config.get('question', ''),
                self.env.task_config.get('schema', ''),
                self.env.task_config.get('db', ''),
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
    def build_prompt(self, question, database, plan_step, critique_msg=None, execution_summary=None, syntax_reference=None, step_idx=None):
        prompt = f"[User Question]\n{question}\n\n"
        project_info = ""
        if self.dialect.lower() == "bigquery":
            project_id = "bigquery-public-data"
            if project_id:
                project_info = (
                    f"[Project Information]\n"
                    f"Current Project ID: {project_id}\n"
                    "Important: Use these IDs in your queries.\n"
                    "Example: `project_id.dataset_id.table_id`\n"
                    "For public datasets, you can use: `bigquery-public-data.dataset_id.table_id`\n\n"
                )
        # Add clear database context and checking instructions
        prompt += (
            f"{project_info}\n"
            f"[Database Context]\n"
            f"Current Database: {database}\n\n"
            "[Database Verification Steps]\n"
            "1. First, check the database structure:\n"
            "   - Use 'ls' to see all files and directories\n"
            "   - Look for files ending in .sql, .ddl, or .schema\n"
            "   - Check for any README.md or documentation files\n"
            "   - If directory access fails, try alternative paths\n"
            "   - If a directory is not accessible, try parent directory\n\n"
            "2. Verify table names and schema:\n"
            "   - Open any .sql or .ddl files to see table definitions\n"
            "   - Look for CREATE TABLE statements\n"
            "   - Note the exact database and schema names\n"
            "   - Check for any table aliases or views\n"
            "   - If schema file not found, try other directories\n\n"
            "3. Important: Use ONLY the tables and schemas you find in these files\n"
            "   - Do not assume table names\n"
            "   - Do not use tables that aren't explicitly defined\n"
            "   - If unsure, use 'ls' to check again\n"
            "   - If access denied, try alternative paths or files\n\n"
            "4. Error Handling:\n"
            "   - If directory access fails, try parent directory\n"
            "   - If file not found, check for alternative locations\n"
            "   - If permission denied, try different paths\n"
            "   - Document any access issues for troubleshooting\n\n"
        )
        
        prompt += f"[Current Plan]\n{plan_step}\n\n"
        
        if critique_msg and isinstance(critique_msg, dict) and 'reasoning' in critique_msg:
            reasoning = critique_msg['reasoning'].strip().replace("\n", " ")
            prompt += f"[Critique]\n{reasoning}\n\n"
            
        if execution_summary and "UserWarning: BigQuery Storage module not found" not in execution_summary:
            prompt += f"[Execution Result]\n{execution_summary}\n\n"
            
        if syntax_reference:
            prompt += f"[Reference Syntax]\n{syntax_reference}\n\n"
            
        # Add clear action constraints
        prompt += (
            "[Required Action Format]\n"
            "You must output exactly one of the following actions (no other text):\n"
            "Important: When dealing with numerical results, DO NOT round the numbers. Keep the full precision of the results.\n"
        )
        if self.dialect == "bigquery":
            prompt += (
                f"- Action: {BIGQUERY_EXEC_SQL.__name__}(sql_query=\"...\",is_save=..., save_path=\".../result.csv\")\n"
                f"- Action: {Terminate.__name__}(output=\".../result.csv\")\n\n"
            )
        elif self.dialect == "snowflake":
            prompt += (
                f"- Action: {SNOWFLAKE_EXEC_SQL.__name__}(sql_query=\"...\",is_save=..., save_path=\".../result.csv\")\n"
                f"- Action: {Terminate.__name__}(output=\".../result.csv\")\n\n"
            )
        elif self.dialect == "sqlite":
            prompt += (
                f"- Action: {LOCAL_DB_SQL.__name__}(file_path=\"...\", command=\"...\", output=\".../result.csv\")\n"
                f"- Action: {Terminate.__name__}(output=\".../result.csv\")\n\n"
            )
        
        # # Add step-specific instructions
        # if step_idx is not None and self.reference_plan and 'plan' in self.reference_plan:
        #     total_steps = len(self.reference_plan['plan'])
        #     if step_idx == total_steps - 1:
        #         prompt += (
        #             "[Final Step]\n"
        #             "This is the final step. You can use Terminate action ONLY if:\n"
        #             "1. All previous steps have been completed successfully\n"
        #             "2. The result matches the expected format\n"
        #             "3. The data has been properly saved to result.csv\n"
        #         )
        #     else:
        #         prompt += (
        #             f"[Intermediate Step {step_idx + 1}/{total_steps}]\n"
        #             "This is NOT the final step. DO NOT use Terminate action.\n"
        #             f"Remaining steps: {total_steps - (step_idx + 1)}\n"
        #         )
        prompt += (
            "- The result matches the expected format\n"
            "- The data has been properly saved to result.csv\n"
         )
            
        prompt += "\nGenerate your next action."
        return prompt
    def generate_critique_msg(self, current_plan,sql_query, obs, response) -> Optional[Dict]:
        schema_string = self.env.task_config.get('schema', '')
        evidence = self.env.task_config.get('evidence', '')
        question = self.env.task_config.get('question', '')
        critique_msg = self.critique_agent.critique_sql(
            self.dialect,
            sql_query, current_plan, question, schema_string, evidence,
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
            # Initialize plan_steps and expected_csv_format after generating the plan
            self.plan_steps = self.reference_plan.get("plan", [])
            self.expected_csv_format = self.reference_plan.get("expected_csv_format", "")
            
        logger.info(f"[MCP] Generated Plan in the Refinement Agent: {self.reference_plan}")

        # --- Plan Refinement Loop ---
        max_plan_refine, refine_count = 3, 0
        max_schema_refine, schema_refine_count = 2, 0
        self.last_validation_critique = None

        while refine_count < max_plan_refine:
            success, critique_json_str = self.critique_agent.critique_plan(
                plan=self.reference_plan['plan'],
                expected_csv_format=self.reference_plan.get('expected_csv_format', ''),
                question=self.env.task_config.get('question', ''),
                schema_string=self.env.task_config.get('schema', '')
            )
            if not success:
                logger.error("Failed to critique plan. Skipping plan refinement.")
                break
            self.plan_critique = json.loads(critique_json_str)
            logger.info(f"[MCP] Plan Critique: {self.plan_critique}")

            if self.plan_critique['update_plan']:
                self.reference_plan.setdefault('critique_notes', []).append(self.plan_critique['critique'])
                plan_prompt = (
                    f"Original Plan:\n{self.reference_plan['plan']}\n\n"
                    f"""Critique Notes:\n\n{self.reference_plan['critique_notes']}"""
                    "\n\nPlease revise the plan to address the above critique notes. Output a numbered step-by-step plan."
                )
                revised_plan_result = self.planner_agent.generate_plan(
                    database_id=self.env.task_config.get('db', ''),
                    question=self.env.task_config.get('question', ''),
                    schema_string=self.env.task_config.get('schema', ''),
                    evidence=self.env.task_config.get('evidence', ''),
                    prompt_prefix=plan_prompt,
                    dialect=self.dialect
                )
                if isinstance(revised_plan_result, dict):
                    revised_plan_result['critique_notes'] = self.reference_plan['critique_notes']
                    self.reference_plan = revised_plan_result
                    # Update plan_steps and expected_csv_format after plan refinement
                    self.plan_steps = self.reference_plan.get("plan", [])
                    self.expected_csv_format = self.reference_plan.get("expected_csv_format", "")
                else:
                    self.reference_plan['plan'] = revised_plan_result
                    self.plan_steps = revised_plan_result
                refine_count += 1
            else:
                break

        # ====== 主動詢問 LLM 需要哪些 BigQuery syntax 補充 ======
        if self.rag_syntax and self.env.task_config.get('instance_id', '').startswith('bq'):
            self.maybe_retrieve_syntax_reference()

        # 2. MCP Loop: SQL generation, critique, refinement
        while not done and step_idx < self.max_steps:
            question = self.env.task_config.get('question', '')
            # if step_idx >= len(self.reference_plan['plan']):
            plan_step = self.reference_plan['plan']  # 只取整個計畫
            # else:
            #     plan_step = self.reference_plan['plan'][step_idx]  # 只取當前步驟
            
            # 簡化 obs（僅摘要錯誤）
            execution_summary = ""
            if "error" in obs.lower() or "exception" in obs.lower():
                execution_summary = "Error detected in previous query."
                execution_summary += f"\nError Message: {obs}"
            elif "no rows" in obs.lower() or "empty result" in obs.lower():
                execution_summary = "Previous query returned empty result."
                execution_summary += f"\nResult: {obs}"
            else:
                execution_summary = obs

            syntax_ref = self.syntax_reference if self.rag_syntax and self.env.task_config.get('instance_id', '').startswith('bq') else None
            prompt = self.build_prompt(
                question, 
                self.env.task_config.get('db', ''),
                plan_step, 
                critique_msg, 
                execution_summary, 
                syntax_reference=syntax_ref,
                step_idx=step_idx
            )

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
            logger.info(f"Step {step_idx + 1} Prompt : {prompt}")
            logger.info(f"Step {step_idx + 1} Response : {response}")
            logger.info("Step %d: %s", step_idx + 1, action)
            sql_query = getattr(action, 'sql_query', None)

            # --- judge if action is repeated ---
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

            # 檢查是否為 bash 命令
            is_bash_command = False
            if hasattr(action, 'command'):
                bash_commands = ['ls', 'cat', 'head', 'tail', 'grep', 'find', 'wc', 'sort', 'uniq', 'cut', 'awk', 'sed']
                cmd = action.command.lower() if action.command else ''
                is_bash_command = any(cmd.startswith(b) for b in bash_commands)

            # --- Self-Refinement Trigger ---
            is_error = "error" in obs.lower() or "exception" in obs.lower()
            is_empty = ("no rows" in obs.lower() or "empty result" in obs.lower()) and not is_error

            if self.x and sql_query and (is_error or is_empty):
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
                    action = refined_action
                else:
                    logger.info("Self-refinement failed after maximum attempts.")
                    result = obs
                    # continue  # 如果自我修正失敗，留在同一步重試
                    step_idx += 1

            # 只有在沒有錯誤且非空結果，且不是 bash 命令，且是 SQL 執行或終止操作時才前進到下一步
            if not is_error and not is_empty and not is_bash_command:
                if isinstance(action, Terminate):
                    # 檢查是否所有步驟都已完成
                    # if step_idx < len(self.reference_plan['plan']) - 1:
                    #     logger.warning("Cannot terminate: Not all steps are completed.")
                    #     obs = "Cannot terminate yet: Not all steps are completed. Please continue with the next step."
                    #     done = False  # 確保不會終止
                    #     continue
                    step_idx += 1
                elif isinstance(action, (BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, LOCAL_DB_SQL)):
                    step_idx += 1
                    critique_msg = self.generate_critique_msg(plan_step, sql_query, obs, response)

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
                    critique_msg = {"reasoning": validation_feedback.get("analysis", "Output validation failed.")}

                if validate_passed and validation_feedback.get("valid_result"):
                    # 再次檢查是否所有步驟都已完成
                    # if step_idx < len(self.reference_plan['plan']) - 1:
                    #     logger.warning("Cannot terminate: Not all steps are completed.")
                    #     obs = "Cannot terminate yet: Not all steps are completed. Please continue with the next step."
                    #     done = False  # 確保不會終止
                    #     continue
                    done = True
                    result = action.output
                    logger.info("The task is done.")
                    break
                else:
                    done = False
                    logger.info("Validation failed. Continuing to next refinement step.")
                    continue

            elif isinstance(action, Terminate) and not self.validate_result:
                # 檢查是否所有步驟都已完成
                # if step_idx < len(self.reference_plan['plan']) - 1:
                #     logger.warning("Cannot terminate: Not all steps are completed.")
                #     obs = "Cannot terminate yet: Not all steps are completed. Please continue with the next step."
                #     done = False  # 確保不會終止
                #     continue
                done = True
                result = action.output
                logger.info("The task is done.")
                break

        return done, result