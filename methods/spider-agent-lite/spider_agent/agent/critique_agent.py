import logging
from spider_agent.agent.models import call_llm
import re
import json

logger = logging.getLogger("spider_agent")

class CritiqueAgent:
    """
    Agent for critiquing either a step-by-step plan or an SQL query.
    """
    def format_correct(self, response: str) -> bool:
        """確認回傳內容中包含有效 SQL action 字串"""
        pattern_list = [
            r'''
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
            ''',
            r'''
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
            ''',
            r'LOCAL_DB_SQL\(file_path=(.*?), command=(.*?), output=(.*?)\)'
            ]
        for pattern in pattern_list:
            if re.search(pattern, response, re.DOTALL):
                return True
        return False
    def __init__(self, model="gpt-4", max_tokens=800, temperature=0.3):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def critique_plan(self, plan: str, question: str, schema_string: str = "", prompt_prefix: str = "") -> tuple:
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
            "---"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        response = call_llm(payload)
        if not response:
            return False, json.dumps({"critique": "", "need_schema": False, "update_plan": False})

        try:
            match = re.search(r"\{[\s\S]+?\}", response[1])
            if match:
                critique_obj = json.loads(match.group())
            else:
                raise ValueError("No JSON found in critique_plan output.")
            for key in ["critique", "need_schema", "update_plan"]:
                critique_obj.setdefault(key, False if key != "critique" else "")
            return True, json.dumps(critique_obj)
        except Exception as e:
            logger.warning(f"[critique_plan] Failed to parse JSON: {e}, raw: {response}")
            fallback = {"critique": None, "need_schema": True, "update_plan": True}
            return True, json.dumps(fallback)

    def critique_sql(
        self,
        sql_query: str,
        plan: str,
        question: str,
        schema_string: str,
        evidence: str = "",
        response: str = "",
        execution_feedback: str = "",
        prompt_prefix: str = "",
        rewrite_only: bool = False
    ) -> dict:
        if rewrite_only:
            prompt = (
                f"{prompt_prefix}\n"
                "Rewrite the following SQL to fix potential errors. Output ONLY the corrected SQL.\n"
                f"SQL Query:\n{sql_query}"
            )
        else:
            prompt = (
                f"{prompt_prefix}\n"
                "You are an expert SQL reviewer. Critique and revise the following SQL query.\n"
                "Respond in two sections:\n"
                "[Reasoning]\nExplain your critique and what needs fixing.\n"
                "[SQL]\nOutput the corrected SQL using this format:\n"
                "Action: BIGQUERY_EXEC_SQL(sql_query=\"...\")\n\n"
                f"User Question: {question}\n"
                f"Schema:\n{schema_string}\n"
                f"Plan:\n{plan}\n"
                f"SQL Query:\n{sql_query}\n"
            )
            if evidence:
                prompt += f"Evidence:\n{evidence}\n"
            if response:
                prompt += f"Response:\n{response}\n"
            if execution_feedback:
                prompt += f"Execution Feedback:\n{execution_feedback}\n"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        max_retry = 3
        for attempt in range(max_retry):
            response = call_llm(payload)
            if not response:
                return {"reasoning": "", "revised_sql": "", "raw": ""}
            else:
                result = self._parse_critique_sql_response(response[1])
                if not self.format_correct(result["revised_sql"]):
                    return {"reasoning": "", "revised_sql": "", "raw": ""}
                return result

    def _parse_critique_sql_response(self, response: str) -> dict:
        """
        將 LLM 的 response 拆為 reasoning + revised SQL
        """
        reasoning = ""
        revised_sql = ""

        reasoning_match = re.search(r"\[Reasoning\](.*?)(?=\[SQL\])", response, re.DOTALL | re.IGNORECASE)
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

# if __name__ == "__main__":
#     test ="""[Reasoning]\nThe provided SQL query has a solid foundation but requires several critiques and improvements for correctness, efficiency, and alignment with the schema and user requirements. Below are the key points of critique and areas for improvement:\n\n1. **Schema Assumptions**: The query assumes the existence of fields like `hits.product.v2ProductName` and `hits.product.productQuantity` in the `bigquery-public-data.google_analytics_sample.ga_sessions_*` dataset. While the evidence provided in the user input lists fields like `hits.product.productQuantity` and `hits.product.productSKU`, it’s unclear if `v2ProductName` is the correct field for identifying the product name. If this field does not exist or is named differently (e.g., `productName`), the query will fail. I will adjust to use a more generic or confirmed field if necessary, but for now, I\'ll retain it based on the user\'s query.\n\n2. **Date Filtering with `_TABLE_SUFFIX`**: The query uses `_TABLE_SUFFIX BETWEEN \'20170701\' AND \'20170731\'` to filter for July 2017 data. This is correct for BigQuery\'s Google Analytics sample dataset, as it partitions data by date in the table suffix. However, it’s worth noting that this assumes the dataset structure matches the expected format. I will retain this approach as it aligns with the provided schema snippet.\n\n3. **Nested UNNEST Operations**: The query uses `CROSS JOIN UNNEST(hits) AS h` and `CROSS JOIN UNNEST(h.product) AS p` to access nested data. This is appropriate for the Google Analytics dataset structure, where `hits` and `hits.product` are nested arrays. However, it can lead to performance issues with large datasets due to the explosion of rows. I will keep this structure but note that optimization (e.g., filtering before UNNEST) could be considered if performance is a concern.\n\n4. **Exclusion of Target Product**: The query correctly excludes \'Youtube Men’s Vintage Henley\' using `p.v2ProductName != \'Youtube Men’s Vintage Henley\'` in the `OtherProducts` CTE. This is logically sound and aligns with the user’s requirement.\n\n5. **Aggregation and Sorting**: The query aggregates the total quantity sold using `SUM(p.productQuantity)` and sorts by `total_quantity DESC` to find the top-selling product, limiting to 1 result. This is correct, but it does not handle ties (e.g., if two products have the same total quantity). I will adjust the query to include a secondary sort criterion (e.g., alphabetical order of product name) to break ties deterministically.\n\n6. **Edge Cases**: The query does not account for scenarios where no other products are purchased by the target customers. While the `LIMIT 1` ensures a result is returned if data exists, it could return an empty result set if no qualifying data is found. I will not modify this behavior as it aligns with the expected output format, but it’s worth noting.\n\n7. **Readability and Maintainability**: The query is structured with CTEs (`HenleyBuyers` and `OtherProducts`), which improves readability. However, adding comments and consistent aliasing can further enhance clarity.\n\nIn summary, the query is mostly correct but relies on unverified schema assumptions (e.g., `v2ProductName`), lacks handling for ties in top-selling products, and could be optimized for performance. I will revise the query to address the tie-breaking issue and improve readability while retaining the core logic.\n\n[SQL]\nAction: BIGQUERY_EXEC_SQL(sql_query="\nWITH HenleyBuyers AS (\n  -- Identify unique customers who bought \'Youtube Men’s Vintage Henley\' in July 2017\n  SELECT DISTINCT fullVisitorId\n  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`\n  CROSS JOIN UNNEST(hits) AS h\n  CROSS JOIN UNNEST(h.product) AS p'"""
#     agent = CritiqueAgent()
#     result = agent._parse_critique_sql_response(test)
#     print(result)