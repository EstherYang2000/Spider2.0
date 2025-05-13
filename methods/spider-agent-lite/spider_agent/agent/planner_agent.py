import logging
from spider_agent.agent.models import call_llm
import json
import re
logger = logging.getLogger("spider_agent")

class PlannerAgent:
    """
    Generates a detailed plan for SQL query construction.
    """
    def __init__(self, model="gpt-4", max_tokens=800, temperature=0.3, dialect=None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.dialect = dialect  # Default dialect

    def generate_plan(self,database_id: str, question: str, schema_string: str, evidence: str = "", prompt_prefix: str = "", linked_tables=None, linked_columns=None,dialect=None) -> dict:
        # 過濾 schema_string，只保留 linked_tables/linked_columns
        if linked_tables is not None or linked_columns is not None:
            filtered_schema = []
            for table_def in schema_string.split(';'):
                table_def = table_def.strip()
                if not table_def: continue
                tname = table_def.split('(')[0].strip()
                if linked_tables and tname not in linked_tables:
                    continue
                if '(' in table_def and ')' in table_def:
                    col_str = table_def.split('(')[1].split(')')[0]
                    col_list = [c.strip() for c in col_str.split(',')]
                    if linked_columns:
                        col_list = [col for col in col_list if f"{tname}.{col}" in linked_columns]
                    filtered_schema.append(f"{tname}({', '.join(col_list)})")
            schema_string = '; '.join(filtered_schema)
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
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        _,response = call_llm(payload)
        logger.info(f"[MCP] Generate Reference Plan Response: {response}")
        # 直接解析成 dict
        try:
            # Clean up the response string
            cleaned_response = response.strip()
            # Remove markdown code block markers if present
            cleaned_response = cleaned_response.strip("```json").strip("```").strip()
            result = json.loads(cleaned_response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            # 退回到原本的字串拆分 fallback
            plan_text = response
            plan_steps = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|$)', plan_text, re.S)
            expected_csv_format = ""
            return {
                "plan": [s.strip() for s in plan_steps],
                "expected_csv_format": expected_csv_format,
                "critique_notes": []
            }

        return {
            "plan": result.get("plan", []),
            "expected_csv_format": result.get("expected_csv_format", ""),
            "critique_notes": []
        }