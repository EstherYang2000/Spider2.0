"""
Critique Agent Module

This module contains the CritiqueAgent class, which is responsible for critiquing
and revising SQL queries and execution plans in the Spider Agent system.
"""

import logging
from spider_agent.agent.models import call_llm
import re
import json
from spider_agent.agent.action import (
    BIGQUERY_EXEC_SQL,
    SNOWFLAKE_EXEC_SQL,
    LOCAL_DB_SQL,
)

logger = logging.getLogger("spider_agent")


class CritiqueAgent:
    """
    Agent for critiquing either a step-by-step plan or an SQL query.

    This agent uses a language model to analyze and provide feedback on SQL plans
    and queries, suggesting improvements and corrections.
    """

    def format_correct(self, response: str) -> bool:
        """
        Check if the response contains a valid SQL action string.

        Args:
            response: The response string to validate.

        Returns:
            bool: True if a valid SQL action is found, False otherwise.
        """
        pattern_list = [
            r"""
                BIGQUERY_EXEC_SQL\(
                    \s*sql_query\s*=\s*
                    (?P<quote_sql>\"\"\"|\"|\'\'\'|\')  # Match opening quote for sql_query
                    (?P<sql_query>.*?)
                    (?<!\\)(?P=quote_sql)              # Match closing quote for sql_query
                    ,\s*is_save\s*=\s*
                    (?P<is_save>True|False)
                    (?:,\s*save_path\s*=\s*
                        (?P<quote_path>\"\"\"|\"|\'\'\'|\')  # Match opening quote for save_path
                        (?P<save_path>.*?)
                        (?<!\\)(?P=quote_path)              # Match closing quote for save_path
                    )?
                    \s*\)
            """,
            r"""
            SNOWFLAKE_EXEC_SQL\(
                \s*sql_query\s*=\s*
                (?P<quote_sql>\"\"\"|\"|\'\'\'|\'|\"\"\")  # Match opening quote for sql_query
                (?P<sql_query>.*?)
                (?<!\\)(?P=quote_sql)                      # Match closing quote for sql_query
                ,\s*is_save\s*=\s*
                (?P<is_save>True|False)
                (?:,\s*save_path\s*=\s*
                    (?P<quote_path>\"\"\"|\"|\'\'\'|\'|\"\"\")  # Match opening quote for save_path
                    (?P<save_path>.*?)
                    (?<!\\)(?P=quote_path)                     # Match closing quote for save_path
                )?
                \s*\)
            """,
            r"LOCAL_DB_SQL\(file_path=(.*?), command=(.*?), output=(.*?)\)",
        ]
        for pattern in pattern_list:
            if re.search(pattern, response, re.DOTALL | re.VERBOSE):
                return True
        return False

    def __init__(self, model="gpt-4", max_tokens=800, temperature=0.3):
        """
        Initialize the CritiqueAgent.

        Args:
            model: The language model to use for critiques.
            max_tokens: Maximum tokens for the model's response.
            temperature: Temperature setting for response generation.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def critique_plan(
        self,
        plan: str,
        question: str,
        expected_csv_format: str,
        schema_string: str = "",
        prompt_prefix: str = "",
    ) -> tuple:
        """
        Critique a step-by-step plan for answering a user's question.

        Args:
            plan: The plan to critique.
            question: The user's question.
            expected_csv_format: Expected format of the output CSV.
            schema_string: Database schema information.
            prompt_prefix: Optional prefix for the prompt.

        Returns:
            tuple: (success: bool, critique_json: str)
        """
        prompt = (
            f"{prompt_prefix}\n"
            "You are an expert SQL planner. Critique the following step-by-step plan for answering the user's question.\n"
            "1. Does the plan cover all necessary steps?\n"
            "2. Is the logic sound and clear?\n"
            "Return your response in the following JSON format:\n"
            "{\n"
            '  "critique": "...",\n'
            '  "need_schema": true/false,\n'
            '  "update_plan": true/false\n'
            "}\n\n"
            f"User Question:\n{question}\n"
            f"Schema:\n{schema_string}\n"
            f"Plan:\n{plan}\n"
            f"Expected CSV format:\n{expected_csv_format}\n"
            "---"
        )
        # Prepare payload for LLM call
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        # Call the language model
        response = call_llm(payload)
        if not response:
            return False, json.dumps(
                {"critique": "", "need_schema": False, "update_plan": False}
            )

        try:
            # Extract JSON from the response
            match = re.search(r"\{[\s\S]+?\}", response[1])
            if match:
                critique_obj = json.loads(match.group())
            else:
                raise ValueError("No JSON found in critique_plan output.")
            # Ensure all required keys are present
            for key in ["critique", "need_schema", "update_plan"]:
                critique_obj.setdefault(key, False if key != "critique" else "")
            return True, json.dumps(critique_obj)
        except Exception as e:
            logger.warning(
                f"[critique_plan] Failed to parse JSON: {e}, raw: {response}"
            )
            fallback = {"critique": None, "need_schema": True, "update_plan": True}
            return True, json.dumps(fallback)

    def critique_sql(
        self,
        dialect: str,
        sql_query: str,
        plan: str,
        question: str,
        schema_string: str,
        db: str,
        evidence: str = "",
        response: str = "",
        execution_feedback: str = "",
        prompt_prefix: str = "",
        rewrite_only: bool = False,
    ) -> dict:
        """
        Critique and revise an SQL query.

        Args:
            dialect: SQL dialect (bigquery, snowflake, local).
            sql_query: The SQL query to critique.
            plan: The execution plan.
            question: The user's question.
            schema_string: Database schema.
            db: Database name.
            evidence: Additional evidence.
            response: Previous response.
            execution_feedback: Feedback from execution.
            prompt_prefix: Optional prompt prefix.
            rewrite_only: If True, only rewrite the SQL without full critique.

        Returns:
            dict: Critique response from the model.
        """
        # Construct prompt based on rewrite_only flag
        prompt = None
        if rewrite_only:
            prompt = (
                f"{prompt_prefix}\n"
                "Rewrite the following SQL to fix potential errors. Output ONLY the corrected SQL.\n"
                f"SQL Query:\n{sql_query}"
            )
        else:
            # Determine action string based on dialect
            if dialect == "bigquery":
                action_str = f'- Action: {BIGQUERY_EXEC_SQL.__name__}(sql_query="...",is_save=..., save_path=".../result.csv")\n'
            elif dialect == "snowflake":
                action_str = f'- Action: {SNOWFLAKE_EXEC_SQL.__name__}(sql_query="...",is_save=..., save_path=".../result.csv")\n'
            else:
                action_str = f'- Action: {LOCAL_DB_SQL.__name__}(file_path=..., command="...", output=".../result.csv")\n'
            prompt = (
                f"{prompt_prefix}\n"
                "You are an expert SQL reviewer. Critique and revise the following SQL query.\n"
                "Respond in two sections:\n"
                "[Reasoning]\nExplain your critique and what needs fixing.\n"
                "[SQL]\nOutput the corrected SQL using this format:\n"
                f"{action_str}\n"
                f"User Question: {question}\n"
                f"Database: {db}\n"
                f"Schema:\n{schema_string}\n"
                f"Current Plan:\n{plan}\n"
                f"SQL Query:\n{sql_query}\n"
            )
            # Add optional sections
            if evidence:
                prompt += f"Evidence:\n{evidence}\n"
            if response:
                prompt += f"Response:\n{response}\n"
            if execution_feedback:
                prompt += f"Execution Feedback:\n{execution_feedback}\n"

        # Prepare payload for LLM call
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        # Retry up to 3 times
        max_retry = 3
        for attempt in range(max_retry):
            critique = call_llm(payload)
            logger.info(
                f"[CritiqueAgent] Attempt {attempt + 1} to critique SQL: {sql_query}"
            )
            if not critique:
                return {"reasoning": "", "revised_sql": ""}
            else:
                # Note: Parsing logic is commented out; returning raw response
                return critique[1]

    def _parse_critique_sql_response(self, response: str) -> dict:
        """
        Parse the LLM response into reasoning and revised SQL sections.

        Args:
            response: The raw response from the language model.

        Returns:
            dict: Dictionary with 'reasoning' and 'revised_sql' keys.
        """
        reasoning = ""
        revised_sql = ""

        reasoning_match = re.search(
            r"\[Reasoning\](.*?)(?=\[SQL\])", response, re.DOTALL | re.IGNORECASE
        )
        sql_match = re.search(r"\[SQL\](.*)", response, re.DOTALL | re.IGNORECASE)

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        if sql_match:
            sql_block = sql_match.group(1).strip()
            # Handle different SQL dialects
            # sql_regex = [
            #     r'BIGQUERY_EXEC_SQL\(sql_query=(?P<quote>\"\"\"|\"|\'|\"\"|\'\')(.*?)(?P=quote), is_save=(True|False)(, save_path=(?P<quote2>\"|\'|\"\"|\'\')(.*?)(?P=quote2))?\)',
            #     r'''
            #                 SNOWFLAKE_EXEC_SQL\(
            #                     \s*sql_query\s*=\s*
            #                     (?P<quote_sql>\"\"\"|\"|\'\'\'|\'|\"\"\")  # Match opening quote for sql_query
            #                     (?P<sql_query>.*?)
            #                     (?<!\\)(?P=quote_sql)                      # Match closing quote for sql_query
            #                     ,\s*is_save\s*=\s*
            #                     (?P<is_save>True|False)
            #                     (?:,\s*save_path\s*=\s*
            #                         (?P<quote_path>\"\"\"|\"|\'\'\'|\'|\"\"\")  # Match opening quote for save_path
            #                         (?P<save_path>.*?)
            #                         (?<!\\)(?P=quote_path)                     # Match closing quote for save_path
            #                     )?
            #                     \s*\)
            #             ''',
            #     r'LOCAL_DB_SQL\(file_path=(.*?), command=(.*?), output=(.*?)\)'
            # ]
            revised_sql = sql_block
            # for pattern in sql_regex:
            #     action_match = re.search(pattern, sql_block)
            #     if action_match:
            #         revised_sql = action_match.group(1).strip()
            #         break

        return {
            "reasoning": reasoning,
            "revised_sql": revised_sql,
            # "raw": response.strip()
        }


# Test the format_correct method
if __name__ == "__main__":
    test = """BIGQUERY_EXEC_SQL(sql_query="WITH TopSource AS (SELECT trafficSource.source AS top_source FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` WHERE _TABLE_SUFFIX BETWEEN '20170101' AND '20171231' GROUP BY trafficSource.source ORDER BY SUM(totals.totalTransactionRevenue) DESC LIMIT 1) SELECT t.top_source, EXTRACT(MONTH FROM PARSE_DATE('%Y%m%d', _TABLE_SUFFIX)) AS month, SUM(totals.totalTransactionRevenue) / 1000000.0 AS monthly_revenue_millions FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`, TopSource t WHERE _TABLE_SUFFIX BETWEEN '20170101' AND '20171231' AND trafficSource.source = t.top_source GROUP BY t.top_source, month ORDER BY month", is_save=True, save_path="/workspace/monthly_revenue_top_source.csv")"""
    agent = CritiqueAgent()
    result = agent.format_correct(test)
    print(result)
