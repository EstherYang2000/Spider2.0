import logging
from spider_agent.agent.models import call_llm

logger = logging.getLogger("spider_agent")

class PlannerAgent:
    """
    Generates a detailed plan for SQL query construction.
    """
    def __init__(self, model="gpt-4", max_tokens=800, temperature=0.3):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate_plan(self, question: str, schema_string: str, evidence: str = "", prompt_prefix: str = "") -> str:
        prompt = (
            f"{prompt_prefix}\n"
            "Task: Generate a step-by-step plan for constructing an accurate SQL query based on the user question, schema, and evidence.\n"
            "Only output the step-by-step plan. Do NOT generate any SQL or code.\n"
            f"User Question: {question}\n"
            f"Schema: {schema_string}\n"
        )
        if evidence:
            prompt += f"Evidence: {evidence}\n"
        prompt += "Plan:"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        logger.info("Calling LLM for plan generation...")
        _,response = call_llm(payload)
        return response