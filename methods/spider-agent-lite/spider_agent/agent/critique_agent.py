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

    def critique_sql(self, sql_query: str, plan: str, question: str, schema_string: str, evidence: str = "", prompt_prefix: str = "") -> str:
        prompt = (
            f"{prompt_prefix}\n"
            "Task: Critique the provided SQL query based on the plan, user question, schema, and evidence.\n"
            f"User Question: {question}\n"
            f"Schema: {schema_string}\n"
            f"Plan: {plan}\n"
            f"SQL Query: {sql_query}\n"
        )
        if evidence:
            prompt += f"Evidence: {evidence}\n"
        prompt += "Critique:"
        prompt += "\n\nAfter your reasoning, you must output a valid action in the following format, and nothing else:\nAction: Bash(code=\"...\")\nor\nAction: BIGQUERY_EXEC_SQL(sql_query=\"...\", ...)\n"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        logger.info("Calling LLM for critique generation...")
        response = call_llm(payload)
        return response["choices"][0]["message"]["content"] if response and "choices" in response else ""
