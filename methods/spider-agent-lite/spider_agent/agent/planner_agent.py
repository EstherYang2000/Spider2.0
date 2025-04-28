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

    def generate_plan(self, question: str, schema_string: str, evidence: str = "", prompt_prefix: str = "") -> dict:
        prompt = (
            f"{prompt_prefix}\n"
            "Task: Generate a step-by-step plan for constructing an accurate SQL query based on the user question, schema, and evidence.\n"
            "Only output the step-by-step plan. Do NOT generate any SQL or code.\n"
            f"User Question: {question}\n"
            f"Schema: {schema_string}\n"
        )
        if evidence:
            prompt += f"Evidence: {evidence}\n"
        prompt += (
            "Plan:\n"
            "After the plan, define the expected CSV output format for the answer, strictly matching the user question's requirements. "
            "Do NOT add extra columns or information unless explicitly required by the question. "
            "Use the following format:\n"
            "Expected CSV format:\n"
            "<column1> (type)\n"
            "[Add notes such as \"answer in one row\" if applicable.]\n"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        logger.info("Calling LLM for plan and expected CSV format generation...")
        _, response = call_llm(payload)
        # 解析 LLM 回傳，把 Plan 跟 Expected CSV format 分開
        if "Expected CSV format:" in response:
            plan, expected_csv_format = response.split("Expected CSV format:", 1)
        else:
            plan, expected_csv_format = response, ""
        return {
            "plan": plan.strip(),
            "expected_csv_format": expected_csv_format.strip()
        }