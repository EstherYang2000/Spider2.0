"""
Planner Agent Module

This module contains the PlannerAgent class, which is responsible for generating
detailed step-by-step plans for SQL query construction based on user questions,
database schemas, and available evidence.
"""

import logging
from spider_agent.agent.models import call_llm
import json
import re

logger = logging.getLogger("spider_agent")


class PlannerAgent:
    """
    Agent for generating detailed plans for SQL query construction.

    This agent analyzes user questions, database schemas, and evidence to create
    structured plans that guide the SQL generation process.
    """

    def __init__(self, model="gpt-4", max_tokens=800, temperature=0.3, dialect=None):
        """
        Initialize the PlannerAgent.

        Args:
            model: The language model to use for planning.
            max_tokens: Maximum tokens for the model's response.
            temperature: Temperature setting for response generation.
            dialect: Default SQL dialect (e.g., 'bigquery', 'snowflake').
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.dialect = dialect  # Default dialect

    def generate_plan(
        self,
        database_id: str,
        question: str,
        schema_string: str,
        evidence: str = "",
        prompt_prefix: str = "",
        linked_tables=None,
        linked_columns=None,
        dialect=None,
    ) -> dict:
        """
        Generate a detailed plan for SQL query construction.

        Args:
            database_id: Identifier of the database/dataset.
            question: The user's query question.
            schema_string: Database schema information.
            evidence: Additional evidence or context.
            prompt_prefix: Optional prefix for the prompt.
            linked_tables: List of relevant table names.
            linked_columns: List of relevant column names.
            dialect: SQL dialect to use.

        Returns:
            dict: Plan containing steps, expected CSV format, and critique notes.
        """
        # Filter schema_string to only include linked tables/columns if specified
        if linked_tables is not None or linked_columns is not None:
            filtered_schema = []
            for table_def in schema_string.split(";"):
                table_def = table_def.strip()
                if not table_def:
                    continue
                tname = table_def.split("(")[0].strip()
                if linked_tables and tname not in linked_tables:
                    continue
                if "(" in table_def and ")" in table_def:
                    col_str = table_def.split("(")[1].split(")")[0]
                    col_list = [c.strip() for c in col_str.split(",")]
                    if linked_columns:
                        col_list = [
                            col
                            for col in col_list
                            if f"{tname}.{col}" in linked_columns
                        ]
                    filtered_schema.append(f"{tname}({', '.join(col_list)})")
            schema_string = "; ".join(filtered_schema)

        # Add project information for BigQuery
        project_info = ""
        if dialect.lower() == "bigquery":
            project_id = "bigquery-public-data"
            if project_id:
                project_info = (
                    f"[Project Information]\n"
                    f"Current Project ID: {project_id}\n"
                    "Important: Use these IDs in your queries.\n"
                    "Example: `project_id.dataset_id.table_id`\n"
                    "For public datasets, you can use: `bigquery-public-data.dataset_id.table_id`\n\n"
                )

        # Construct the planning prompt
        prompt = (
            f"You are generating a SQL plan for {dialect}\n"
            f"{prompt_prefix}\n"
            "Task: Generate a step-by-step plan for constructing an accurate SQL query based on the user question, schema, and evidence.\n"
            "Only output the step-by-step plan. Do NOT generate any SQL or code.\n"
            f"User Question: {question}\n"
            f"{project_info}\n"
            f"Current Dataset: {database_id}\n"
            f"Schema (linked by question, possible relevant tables/columns, may be incomplete or noisy):\n{schema_string}\nIf there is an error, please recheck the table and schema in the database.\n"
        )
        if evidence:
            prompt += f"External Knowledge: {evidence}\n"
        prompt += (
            "Example of the exact JSON format:\n"
            "{\n"
            '  "plan": [\n'
            '    "Step 1 description",\n'
            '    "Step 2 description",\n'
            '    "..."\n'
            "  ],\n"
            '  "expected_csv_format": "<column1> (type)\\n[notes]"\n'
            "}\n"
            "IMPORTANT JSON FORMATTING RULES:\n"
            "1. All property names MUST be enclosed in double quotes\n"
            "2. All string values MUST be enclosed in double quotes\n"
            "3. Do not use trailing commas\n"
            "4. Do not use single quotes\n"
            "5. Do not use line breaks within string values\n"
            "6. The expected_csv_format should be a single line string\n"
            "7. Make sure all quotes are properly closed\n\n"
            "After the plan, define the expected CSV output format for the answer, strictly matching the user question's requirements. "
            "Do NOT add extra columns or information unless explicitly required by the question. "
            "The expected_csv_format should be a single line string describing the column names and their types.\n"
        )

        logger.info(f"[MCP] Generate Reference Plan Prompt: {prompt}")

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
        _, response = call_llm(payload)
        logger.info(f"[MCP] Generate Reference Plan Response: {response}")

        # Parse the response as JSON
        try:
            # Clean up the response string
            cleaned_response = response.strip()
            # Remove markdown code block markers if present
            cleaned_response = cleaned_response.strip("```json").strip("```").strip()
            result = json.loads(cleaned_response)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            # Fallback to string parsing
            plan_text = response
            plan_steps = re.findall(r"\d+\.\s*(.*?)(?=\n\d+\.|$)", plan_text, re.S)
            expected_csv_format = ""
            return {
                "plan": [s.strip() for s in plan_steps],
                "expected_csv_format": expected_csv_format,
                "critique_notes": [],
            }

        return {
            "plan": result.get("plan", []),
            "expected_csv_format": result.get("expected_csv_format", ""),
            "critique_notes": [],
        }
