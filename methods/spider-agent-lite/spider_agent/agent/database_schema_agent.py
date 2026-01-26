"""
Database Schema Agent Module

This module contains agents for exploring database schemas and generating schema linking strings
to assist in SQL query generation. The agents interact with various database types (local, BigQuery, Snowflake)
to gather table information, column details, and sample data.
"""

import logging
from typing import Callable, Optional, Tuple, Any
from spider_agent.agent.action import (
    Action,
    Terminate,
    
)


logger = logging.getLogger("spider_agent.schema_linking_agent")


class DBSchemaAgentEnv:
    """
    Environment class for database schema agent operations.

    Provides configuration for base directory and schema path.
    """

    def __init__(self, base_dir, schema_path):
        self.base_dir = base_dir
        self.schema_path = schema_path


class SQLSchemaLinkingAgent:
    """
    Agent for exploring database schemas and generating schema linking strings.

    This agent interacts with databases to gather schema information and create
    compact representations that help LLMs understand database structure for SQL generation.
    """

    def __init__(
        self,
        env: DBSchemaAgentEnv,
        llm_predict: Callable[[str], Tuple[str, Optional[Any]]],
        llm_step: Callable[[Action], Tuple[str, bool]],
        db_type: str,
        db_name: str,
        max_steps: int = 20,
    ):
        """
        Initialize the SQL Schema Linking Agent.

        Args:
            env: Environment configuration.
            llm_predict: Function to predict actions from observations.
            llm_step: Function to execute actions and get next observation.
            db_type: Type of database (local, bigquery, snowflake, etc.).
            db_name: Name of the database.
            max_steps: Maximum number of exploration steps.
        """
        self.env = env
        self.llm_predict = llm_predict
        self.llm_step = llm_step
        self.db_type = db_type
        self.db_name = db_name
        self.max_steps = max_steps

    def run(self, user_question: str, critique_note: str):
        """
        Run the schema exploration process.

        Args:
            user_question: The user's query that requires schema understanding.
            critique_note: Optional feedback from previous attempts.

        Returns:
            str: The generated schema linking string.
        """
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
        """
        Generate the initial prompt for the schema exploration task.

        Args:
            user_question: The user's query.
            critique_note: Feedback from previous attempts.

        Returns:
            str: The formatted prompt for the LLM.
        """
        # Define available actions based on database type
        action_str = ""
        table_str = ""
        if self.db_type.lower() == "local":
            action_str = (
                "- LOCAL_GET_TABLES(file_path=..., save_path=...): List tables from a local SQLite or DuckDB file.\n"
                "- LOCAL_GET_TABLE_INFO(file_path=..., table=..., save_path=...): Get schema from local table.\n"
                "- LOCAL_SAMPLE_ROWS(file_path=..., table=..., row_number=..., save_path=...): Sample rows from local DB.\n"
                "- LOCAL_DB_SQL(file_path=..., command=..., output=...): Run custom SQL locally.\n"
            )
            table_str = "LOCAL_GET_TABLES"
        elif self.db_type.lower() == "bigquery":
            action_str = (
                "- BQ_GET_TABLES(database_name=..., dataset_name=..., save_path=...): Get all table names and their DDL.\n"
                "- BQ_GET_TABLE_INFO(database_name=..., dataset_name=..., table=..., save_path=...): Inspect columns of a specific table.\n"
                "- BQ_SAMPLE_ROWS(database_name=..., dataset_name=..., table=..., row_number=3, save_path=...): Sample rows to inspect real data values.\n"
                "- BIGQUERY_EXEC_SQL(sql_query=..., is_save=True/False, save_path=...): Run custom SQL queries for advanced insights.\n"
            )
            table_str = "BQ_GET_TABLES"
        elif self.db_type.lower() == "dbt":
            action_str = ()
        elif self.db_type.lower() == "postgres":
            action_str = ()
        elif self.db_type.lower() == "snowflake":
            action_str = (
                "- SF_GET_TABLES(database_name=..., schema_name=..., save_path=...): List all tables in Snowflake.\n"
                "- SF_GET_TABLE_INFO(database_name=..., schema_name=..., table=..., save_path=...): Inspect Snowflake table schema.\n"
                "- SF_SAMPLE_ROWS(database_name=..., schema_name=..., table=..., row_number=..., save_path=...): Sample Snowflake table rows.\n"
                "- SNOWFLAKE_EXEC_SQL(sql_query=..., is_save=True/False, save_path=...): Run custom SQL on Snowflake.\n"
            )
            table_str = "SF_GET_TABLES"

        # Construct the main prompt
        prompt = (
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
            # Optional: Suggest starting with table listing
            # f"Begin by listing tables in the dataset using {table_str}."
        )
        return prompt
