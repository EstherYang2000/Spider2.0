import logging
from spider_agent.agent.models import call_llm

logger = logging.getLogger("spider_agent")

class CritiqueAgent:
    """
    Critiques a generated SQL query based on the provided plan and context.
    """
    def __init__(self, model="gpt-4", max_tokens=800, temperature=0.3):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def critique_sql(self, sql_query: str, plan: str, question: str, schema_string: str, evidence: str = "", execution_feedback: str = "", prompt_prefix: str = "") -> str:
        prompt = (
            f"{prompt_prefix}\n"
            "You are an expert SQL reviewer. Your task is to critique and, if needed, revise the provided SQL query based on the plan, user question, schema, and evidence.\n\n"
            "Follow this process:\n"
            "1. Carefully read the user question, plan, schema, and evidence.\n"
            "2. Analyze the SQL query and, if provided, the execution result or error message.\n"
            "3. Identify any issues with the SQL (logic, syntax, schema mismatch, etc.).\n"
            "4. Clearly explain your reasoning and what needs to be corrected.\n"
            "5. Output a revised SQL query that fully solves the user's question.\n\n"
            f"User Question: {question}\n"
            f"Schema:\n{schema_string}\n"
            f"Plan:\n{plan}\n"
            f"SQL Query:\n{sql_query}\n"
        )
        if evidence:
            prompt += f"Evidence:\n{evidence}\n"
        if execution_feedback:
            prompt += f"Execution Feedback (error message or result):\n{execution_feedback}\n"
        prompt += (
            "\n---\n"
            "First, explain your reasoning and identify any problems with the SQL above.\n"
            "Then, output your revised SQL in the following format (and nothing else):\n"
            "Action: BIGQUERY_EXEC_SQL(sql_query=\"...your revised SQL...\")\n"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        logger.info("Calling LLM for critique generation...")
        response = call_llm(payload)
        return response if response else ""
