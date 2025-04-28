# rag_syntax_agent.py
"""
RAGSyntaxAgent: 負責根據 LLM 輸入主題，檢索 syntax 文件夾下最相關的語法說明，並回傳給主流程。
"""
import os
from spider_agent.agent.rag_action import RAG_QUERY

class RAGSyntaxAgent:
    SYNTAX_REQUEST_PROMPT = (
        "Before generating the SQL, please list any BigQuery-specific syntax, functions, or usage examples you need clarification or documentation for. "
        "If none, reply 'None'."
    )

    @classmethod
    def get_syntax_request_prompt(cls):
        return cls.SYNTAX_REQUEST_PROMPT

    def __init__(self, syntax_dir="../../spider2-lite/resource/syntax", top_k=1):
        self.syntax_dir = syntax_dir
        self.top_k = top_k

    def _retrieve_doc(self, topic):
        """
        檢索單一主題，回傳語法說明文字。
        """
        rag_query = RAG_QUERY(query=topic, top_k=self.top_k)
        return rag_query.execute(knowledge_base_path=self.syntax_dir)

    def retrieve(self, topics):
        """
        topics: List[str]  # 需要檢索的語法主題
        return: Dict[str, str]  # {主題: 語法說明文字}
        """
        results = {}
        for topic in topics:
            doc = self._retrieve_doc(topic)
            if doc and "No relevant document found" not in doc:
                results[topic] = doc
        return results

    def format_for_prompt(self, results):
        """
        將檢索結果格式化為 prompt 可插入的區塊
        """
        if not results:
            return ""
        out = ["# BigQuery Syntax Reference"]
        for topic, doc in results.items():
            out.append(f"\n## {topic}\n{doc}")
        return "\n".join(out)
