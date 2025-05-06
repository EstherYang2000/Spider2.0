import logging
from typing import Callable, Optional, Tuple, Any
# from spider_agent.agent.prompts import BIGQUERY_SYSTEM, LOCAL_SYSTEM, DBT_SYSTEM, SNOWFLAKE_SYSTEM, REFERENCE_PLAN_SYSTEM,EXTERNAL_KNOWLEDGE_SYSTEM
from spider_agent.agent.action import Action, Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL, BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, BQ_GET_TABLES, BQ_GET_TABLE_INFO, BQ_SAMPLE_ROWS,SF_GET_TABLES, SF_GET_TABLE_INFO, SF_SAMPLE_ROWS,LOCAL_GET_TABLES, LOCAL_GET_TABLE_INFO, LOCAL_SAMPLE_ROWS


logger = logging.getLogger("spider_agent.schema_linking_agent")

class DBSchemaAgentEnv:
    def __init__(self, base_dir, schema_path):
        self.base_dir = base_dir
        self.schema_path = schema_path

    # def step(self, action):
    #     if hasattr(action, "output"):
    #         return action.output, True
    #     if hasattr(action, "code"):
    #         command = action.code
    #     else:
    #         command = action
    #     import subprocess
    #     try:
    #         output = subprocess.check_output(command, shell=True, cwd=self.base_dir, text=True, stderr=subprocess.STDOUT)
    #     except subprocess.CalledProcessError as e:
    #         output = e.output
    #     return output, False

class SQLSchemaLinkingAgent:
    def __init__(
        self,
        env: DBSchemaAgentEnv,
        llm_predict: Callable[[str], Tuple[str, Optional[Any]]],
        llm_step: Callable[[Action], Tuple[str, bool]],
        db_type: str,
        db_name: str,
        max_steps: int = 20,
    ):
        self.env = env
        self.llm_predict = llm_predict
        self.llm_step = llm_step
        self.db_type = db_type
        self.db_name = db_name
        self.max_steps = max_steps
        # self._AVAILABLE_ACTION_CLASSES = [
        #     Terminate, BQ_GET_TABLES, BQ_GET_TABLE_INFO,
        #     BQ_SAMPLE_ROWS, BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, LOCAL_DB_SQL,
        #     SF_GET_TABLES, SF_GET_TABLE_INFO, SF_SAMPLE_ROWS,
        #     LOCAL_GET_TABLES, LOCAL_GET_TABLE_INFO, LOCAL_SAMPLE_ROWS
        # ]
    # def parse_action(self, output) -> Action:
    #     """ Parse action from text or dict (robust for LLM output) """
    #     import re
    #     if self.db_type.lower() == 'bigquery':
    #         self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, BIGQUERY_EXEC_SQL, CreateFile, EditFile]
    #         # action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
    #     elif self.db_type.lower() == 'snowflake':
    #         self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, SNOWFLAKE_EXEC_SQL, CreateFile, EditFile]
    #         # action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
    #     elif self.db_type.lower() == 'local':
    #         self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL]
    #         # action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
    #     elif self.db_type.lower() == 'dbt':
    #         self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL]
    #         # action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        
    #     action_string = None
    #     # Multi-line robust action extraction
    #     multiline_patterns = [
    #         r'Action\s*:\s*((?:.|\n)*?)(?=^Thought:|^Observation:|\Z)',  # up to next block or end
    #         r'Action\s*:\s*((?:.|\n)*)' , # fallback: grab everything after Action:
    #         r'Action\s*:\s*((?:.|\n)*?)(?=^Thought:|^Observation:|^\[\d{4}-|\Z)',
    #         r'Action\s*:\s*([A-Z_]+\s*\(.*)',
    #         r'Action\s*:\s*((?:[A-Z_]+\s*\(.*?^\))|(?:[A-Z_]+\s*\(.*\Z))',
    #         r'Action\s*:\s*([A-Z_]+\s*\((?:.|\n)*?\))'


    #     ]
        

    #     # If output is a dict (e.g., {'thought':..., 'action':..., 'response':...})
    #     if isinstance(output, dict):
    #         action_string = output.get('action')
    #         # If action string is None or empty, try to extract from response
    #         if not action_string or action_string == "None":
    #             resp = output.get('response', '')
    #             for p in multiline_patterns:
    #                 match = re.search(p, resp, flags=re.DOTALL | re.MULTILINE)
    #                 if match:
    #                     action_string = match.group(1).strip()
    #                     break
    #             if not action_string:
    #                 action_string = resp.strip()
    #     else:
    #         # output is a string
    #         action_string = ""
    #         for p in multiline_patterns:
    #             match = re.search(p, output, flags=re.DOTALL | re.MULTILINE)
    #             if match:
    #                 action_string = match.group(1).strip()
    #                 break
    #         if not action_string:
    #             action_string = output.strip()
    #         # logger.info("Parsed action string: %s", action_string)
    #     output_action = None
    #     for action_cls in self._AVAILABLE_ACTION_CLASSES:
    #         action = action_cls.parse_action_from_text(action_string)
    #         if action is not None:
    #             output_action = action
    #             break
    #     # logger.info("Parsed action: %s", output_action)
    #     if output_action is None:
    #         action_string = action_string.replace("\_", "_").replace("'''","```")
    #         for action_cls in self._AVAILABLE_ACTION_CLASSES:
    #             action = action_cls.parse_action_from_text(action_string)
    #             if action is not None:
    #                 output_action = action
    #                 break
    #     logger.info("Parsed action: %s", output_action)
        
    def run(self, user_question: str, critique_note: str):
        obs = self._get_initial_prompt(user_question, critique_note)
        step = 0
        done = False
        result = ""

        while not done and step < self.max_steps:
            step += 1
            logger.info(f"[Step {step}] LLM Input:\n{obs}")
            _, action = self.llm_predict(obs)
            logger.info(f"[Step {step}] LLM Output:\n{action}")
            # action = self.parse_action(action)
            if not action:
                obs = "Could not parse valid action from LLM response."
                continue

            logger.info(f"[Step {step}] Parsed Action: {action}")
            obs, done = self.llm_step(action)
            logger.info(f"[Step {step}] LLM Output:\n{obs}")
            if done:
                result = action.output if isinstance(action, Terminate) else None
        logger.info(f"[Final Output]\n{result}")
        return result

    def _get_initial_prompt(self, user_question: str, critique_note: str) -> str:
        action_str = ""
        table_str = ""
        if self.db_type.lower() == 'local':
            action_str = (
                "- LOCAL_GET_TABLES(file_path=..., save_path=...): List tables from a local SQLite or DuckDB file.\n"
                "- LOCAL_GET_TABLE_INFO(file_path=..., table=..., save_path=...): Get schema from local table.\n"
                "- LOCAL_SAMPLE_ROWS(file_path=..., table=..., row_number=..., save_path=...): Sample rows from local DB.\n"
                "- LOCAL_DB_SQL(file_path=..., command=..., output=...): Run custom SQL locally.\n"
            )
            table_str = "LOCAL_GET_TABLES"
        elif self.db_type.lower() == 'bigquery':
            action_str =(
            "- BQ_GET_TABLES(database_name=..., dataset_name=..., save_path=...): Get all table names and their DDL.\n"
            "- BQ_GET_TABLE_INFO(database_name=..., dataset_name=..., table=..., save_path=...): Inspect columns of a specific table.\n"
            "- BQ_SAMPLE_ROWS(database_name=..., dataset_name=..., table=..., row_number=3, save_path=...): Sample rows to inspect real data values.\n"
            "- BIGQUERY_EXEC_SQL(sql_query=..., is_save=True/False, save_path=...): Run custom SQL queries for advanced insights.\n"

            )
            table_str = "BQ_GET_TABLES"
        elif self.db_type.lower() == 'dbt':
            action_str = (

            )
        elif self.db_type.lower() == 'postgres':
            action_str = (

            )
        elif self.db_type.lower() == 'snowflake':
            action_str = (
                "- SF_GET_TABLES(database_name=..., schema_name=..., save_path=...): List all tables in Snowflake.\n"
                "- SF_GET_TABLE_INFO(database_name=..., schema_name=..., table=..., save_path=...): Inspect Snowflake table schema.\n"
                "- SF_SAMPLE_ROWS(database_name=..., schema_name=..., table=..., row_number=..., save_path=...): Sample Snowflake table rows.\n"
                "- SNOWFLAKE_EXEC_SQL(sql_query=..., is_save=True/False, save_path=...): Run custom SQL on Snowflake.\n"
            )
            table_str = "SF_GET_TABLES"

        prompt =  (
            "You are a SQL exploration agent designed to interact with a structured database system to extract\n"
            "a schema linking string that will help an LLM understand the database context for SQL generation.\n\n"
            "Your task is to explore the database step by step, issuing one command per step, in order to gather:\n"
            "1. Table names and their relationships.\n"
            "2. Column names and data types.\n"
            "3. Sample cell values for important columns (especially foreign keys, entity IDs, categorical values).\n\n"
            f"---\nUser Question:\n{user_question}\n\n"
            f"---\nSchema Path:\n{self.env.schema_path}\n\n"
            f"---\nCritique Note (optional guidance from the prior system's feedback):\n{critique_note}\n\n"
            "You can use the following actions to explore the schema:\n"
            f"{action_str}"
            "- Terminate(output=...): When you've gathered enough information, stop and return a single schema linking string.\n\n"
            "Each response MUST return exactly ONE Action object. Do NOT wrap it in markdown/code blocks or explanations.\n"
            "Use ONLY valid Action types as listed above.\n\n"
            "Example Schema Linking Format:\n"
            '  Action: Terminate(output="singer(Singer_ID[1,2], Name[Joe,Timbaland]); song(Song_Name[You], Year[1992])")\n\n'
            "Include at most 1–2 sample values per column. Omit unimportant columns if needed for brevity.\n\n"
                # f"Begin by listing tables in the dataset using {table_str}."
        )
        return prompt
