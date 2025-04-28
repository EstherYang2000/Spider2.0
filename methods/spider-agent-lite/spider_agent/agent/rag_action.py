from dataclasses import dataclass, field
from typing import Optional, List
import os
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from spider_agent.agent.action import Action, remove_quote
@dataclass
class RAG_QUERY(Action):
    """Represents an action to query external documents using RAG (Retrieval-Augmented Generation)."""

    action_type: str = field(
        default="rag_query",
        init=False,
        repr=False,
        metadata={"help": 'Type of action, c.f., "rag_query"'}
    )

    query: str = field(
        metadata={"help": "Query string for document retrieval"}
    )

    top_k: int = field(
        default=3,
        metadata={"help": "Number of top relevant document snippets to retrieve"}
    )

    model_name: str = field(
        default="all-MiniLM-L6-v2",
        metadata={"help": "Embedding model for vector retrieval"}
    )

    def __post_init__(self):
        """Load the embedding model on initialization."""
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.documents = []

    @classmethod
    def get_action_description(cls) -> str:
        """Returns the action format description and examples."""
        return """
## RAG_QUERY Action
* Signature: RAG_QUERY(query="your search query", top_k=3)
* Description: Searches external documents for information related to the query and retrieves the top-k most relevant snippets.
* Examples:
  - Example 1: RAG_QUERY(query="What are the sales figures for Q1 2024?", top_k=3)
  - Example 2: RAG_QUERY(query="employee retention policies", top_k=5)
"""

    def load_documents(self, file_path: str):
        """Load and encode a document for vector search."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: File {file_path} not found.")

        with open(file_path, 'r', encoding='utf-8') as f:
            self.documents = [line.strip() for line in f.readlines() if line.strip()]

        embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve_relevant_knowledge(self, file_path: str) -> str:
        """Retrieve the most relevant snippets using vector search."""
        self.load_documents(file_path)

        # Perform vector search
        query_embedding = self.model.encode([self.query], convert_to_numpy=True)
        _, indices = self.index.search(query_embedding, self.top_k)

        relevant_texts = [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]
        return "\n\n".join(relevant_texts) if relevant_texts else "No relevant external knowledge found."

    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional["RAG_QUERY"]:
        """
        Parses a text input to extract and create a RAG_QUERY action instance.

        Supports various query formats, ensuring robustness in parsing.
        """
        pattern = r'RAG_QUERY\s*\(\s*query\s*=\s*(?P<quote>["\']?)(?P<query>.*?)(?P=quote)\s*,\s*top_k\s*=\s*(?P<top_k>\d+)\s*\)'
        match = re.search(pattern, text, flags=re.DOTALL)

        if match:
            query = remove_quote(match.group("query").strip())
            try:
                top_k = int(match.group("top_k"))
            except ValueError:
                top_k = 3  # Default fallback

            return cls(query=query, top_k=top_k)
        
        return None

    def execute(self, knowledge_base_path: str) -> str:
        """
        Executes the RAG query and retrieves relevant knowledge from the specified knowledge base.
        - `knowledge_base_path` should be the directory containing knowledge files (can have subfolders).
        """
        # 遞迴抓所有檔案（只加 isfile 的 path）
        knowledge_files = []
        for root, dirs, files in os.walk(knowledge_base_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    knowledge_files.append(file_path)

        best_match = None
        max_overlap = 0
        query_terms = set(self.query.lower().split())

        for file_path in knowledge_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
            except Exception:
                continue  # skip unreadable files

            overlap = sum(1 for word in query_terms if word in content)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = file_path

        if best_match:
            return self.retrieve_relevant_knowledge(best_match)
        else:
            return "No relevant document found."

    def __repr__(self) -> str:
        """Returns a structured string representation of the instance."""
        return f'RAG_QUERY(query="{self.query}", top_k={self.top_k})'




if __name__ == "__main__":
    # Example usage of the RAG_QUERY action
    knowledge_file = "ga4_obfuscated_sample_ecommerce.events.md"
    knowledge_path = os.path.join("../../spider2-lite/resource/documents", knowledge_file)
    instruction = "How many pseudo users were active in the last 7 days but inactive in the last 2 days as of January 7, 2021?"

    if os.path.exists(knowledge_path):
        rag_query_action = RAG_QUERY(query=instruction, top_k=3)
        external_knowledge_content = rag_query_action.retrieve_relevant_knowledge(knowledge_path)
        print(external_knowledge_content)
    else:
        print(f"Knowledge file '{knowledge_file}' not found.")