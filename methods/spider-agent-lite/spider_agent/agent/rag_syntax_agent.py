# rag_syntax_agent.py
"""
RAG Syntax Agent Module

This module implements the RAGSyntaxAgent class, which is responsible for retrieving
BigQuery-specific syntax documentation and usage examples based on topics requested
by the language model. It uses RAG (Retrieval-Augmented Generation) to find and
return relevant syntax information to enhance SQL query generation.
"""
from spider_agent.agent.rag_action import RAG_QUERY


class RAGSyntaxAgent:
    """
    Agent for retrieving BigQuery syntax documentation using RAG.

    This agent searches through syntax documentation files to find relevant
    BigQuery-specific syntax, functions, and usage examples based on topics
    requested by the language model. It helps ensure accurate SQL generation
    by providing necessary syntax reference information.
    """

    SYNTAX_REQUEST_PROMPT = (
        "Before generating the SQL, please list any BigQuery-specific syntax, functions, or usage examples you need clarification or documentation for. "
        "If none, reply 'None'."
    )

    @classmethod
    def get_syntax_request_prompt(cls):
        """
        Get the prompt used to request syntax clarification from the language model.

        Returns:
            str: The prompt text that asks the LLM to identify any BigQuery-specific
                 syntax or functions it needs clarification on.
        """
        return cls.SYNTAX_REQUEST_PROMPT

    def __init__(self, syntax_dir="../../spider2-lite/resource/syntax", top_k=1):
        """
        Initialize the RAG Syntax Agent.

        Args:
            syntax_dir: Path to the directory containing syntax documentation files.
            top_k: Number of top relevant document snippets to retrieve for each topic.
        """
        self.syntax_dir = syntax_dir
        self.top_k = top_k

    def _retrieve_doc(self, topic):
        """
        Retrieve syntax documentation for a single topic.

        Uses RAG to search through the syntax documentation directory
        and find the most relevant information for the given topic.

        Args:
            topic: The syntax topic to search for (e.g., "ARRAY functions", "WINDOW functions").

        Returns:
            str: The retrieved syntax documentation text, or error message if not found.
        """
        rag_query = RAG_QUERY(query=topic, top_k=self.top_k)
        return rag_query.execute(knowledge_base_path=self.syntax_dir)

    def retrieve(self, topics):
        """
        Retrieve syntax documentation for multiple topics.

        Processes a list of syntax topics and retrieves relevant documentation
        for each one, filtering out topics that don't have relevant documentation.

        Args:
            topics: List of syntax topics to search for.

        Returns:
            dict: Dictionary mapping topics to their retrieved documentation,
                  excluding topics with no relevant documents found.
        """
        results = {}
        for topic in topics:
            doc = self._retrieve_doc(topic)
            if doc and "No relevant document found" not in doc:
                results[topic] = doc
        return results

    def format_for_prompt(self, results):
        """
        Format retrieved syntax results for insertion into prompts.

        Converts the dictionary of retrieved syntax documentation into
        a formatted string that can be easily inserted into language model prompts,
        with clear section headers for each topic.

        Args:
            results: Dictionary of topic-documentation pairs from retrieve() method.

        Returns:
            str: Formatted string with syntax reference sections, or empty string if no results.
        """
        if not results:
            return ""
        out = ["# BigQuery Syntax Reference"]
        for topic, doc in results.items():
            out.append(f"\n## {topic}\n{doc}")
        return "\n".join(out)
