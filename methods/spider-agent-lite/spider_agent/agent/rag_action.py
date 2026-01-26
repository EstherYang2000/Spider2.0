"""
RAG Action Module

This module implements the RAG_QUERY action for retrieval-augmented generation.
It provides functionality to search external documents using semantic similarity
and retrieve relevant knowledge snippets to enhance SQL query generation.
"""

from dataclasses import dataclass, field
from typing import Optional
import os
import faiss
import re
from sentence_transformers import SentenceTransformer
from spider_agent.agent.action import Action, remove_quote


@dataclass
class RAG_QUERY(Action):
    """
    Action for querying external documents using Retrieval-Augmented Generation (RAG).

    This action performs semantic search on external knowledge bases to retrieve
    relevant information that can help in SQL query generation. It uses sentence
    transformers for embedding generation and FAISS for efficient vector search.
    """

    action_type: str = field(
        default="rag_query",
        init=False,
        repr=False,
        metadata={"help": 'Type of action, c.f., "rag_query"'},
    )

    query: str = field(metadata={"help": "Query string for document retrieval"})

    top_k: int = field(
        default=3,
        metadata={"help": "Number of top relevant document snippets to retrieve"},
    )

    model_name: str = field(
        default="all-MiniLM-L6-v2",
        metadata={"help": "Embedding model for vector retrieval"},
    )

    def __post_init__(self):
        """
        Initialize the embedding model and prepare data structures.

        Loads the sentence transformer model and initializes empty FAISS index
        and document list for later use in retrieval operations.
        """
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.documents = []

    @classmethod
    def get_action_description(cls) -> str:
        """
        Get the action format description and usage examples.

        Returns:
            str: Detailed description of the RAG_QUERY action format,
                 including signature, description, and practical examples.
        """
        return """
## RAG_QUERY Action
* Signature: RAG_QUERY(query="your search query", top_k=3)
* Description: Searches external documents for information related to the query and retrieves the top-k most relevant snippets.
* Examples:
  - Example 1: RAG_QUERY(query="What are the sales figures for Q1 2024?", top_k=3)
  - Example 2: RAG_QUERY(query="employee retention policies", top_k=5)
"""

    def load_documents(self, file_path: str):
        """
        Load and encode documents for vector search.

        Reads a text file, splits it into lines, creates embeddings using the
        sentence transformer model, and builds a FAISS index for efficient search.

        Args:
            file_path: Path to the text file containing documents to index.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: File {file_path} not found.")

        # Read file and split into document lines
        with open(file_path, "r", encoding="utf-8") as f:
            self.documents = [line.strip() for line in f.readlines() if line.strip()]

        # Generate embeddings for all documents
        embeddings = self.model.encode(self.documents, convert_to_numpy=True)

        # Create FAISS index and add embeddings
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve_relevant_knowledge(self, file_path: str) -> str:
        """
        Retrieve the most relevant document snippets using vector search.

        Loads the document, encodes the query, performs similarity search,
        and returns the top-k most relevant text snippets.

        Args:
            file_path: Path to the document file to search.

        Returns:
            str: Concatenated relevant text snippets, or a message if none found.
        """
        self.load_documents(file_path)

        # Encode the query and perform vector search
        query_embedding = self.model.encode([self.query], convert_to_numpy=True)
        _, indices = self.index.search(query_embedding, self.top_k)

        # Extract relevant documents based on search results
        relevant_texts = [
            self.documents[idx] for idx in indices[0] if idx < len(self.documents)
        ]
        return (
            "\n\n".join(relevant_texts)
            if relevant_texts
            else "No relevant external knowledge found."
        )

    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional["RAG_QUERY"]:
        """
        Parse text input to extract and create a RAG_QUERY action instance.

        Uses regex pattern matching to extract query and top_k parameters from
        text containing RAG_QUERY action calls. Supports various quote formats.

        Args:
            text: Input text containing the action call to parse.

        Returns:
            Optional[RAG_QUERY]: Parsed action instance, or None if parsing fails.
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
        Execute the RAG query and retrieve relevant knowledge from the knowledge base.

        Searches through all files in the knowledge base directory (including subfolders),
        finds the most relevant document based on term overlap with the query,
        and performs vector search to retrieve top-k relevant snippets.

        Args:
            knowledge_base_path: Directory path containing knowledge files (can have subfolders).

        Returns:
            str: Retrieved relevant knowledge snippets or error message.
        """
        # Recursively collect all files from the knowledge base directory
        knowledge_files = []
        for root, dirs, files in os.walk(knowledge_base_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    knowledge_files.append(file_path)

        # Find the best matching file based on term overlap
        best_match = None
        max_overlap = 0
        query_terms = set(self.query.lower().split())

        for file_path in knowledge_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().lower()
            except Exception:
                continue  # Skip unreadable files

            # Count overlapping terms between query and file content
            overlap = sum(1 for word in query_terms if word in content)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = file_path

        # Retrieve knowledge from the best matching file
        if best_match:
            return self.retrieve_relevant_knowledge(best_match)
        else:
            return "No relevant document found."

    def __repr__(self) -> str:
        """
        Return a structured string representation of the RAG_QUERY instance.

        Returns:
            str: String representation showing the query and top_k parameters.
        """
        return f'RAG_QUERY(query="{self.query}", top_k={self.top_k})'


if __name__ == "__main__":
    # Example usage of the RAG_QUERY action
    knowledge_file = "ga4_obfuscated_sample_ecommerce.events.md"
    knowledge_path = os.path.join(
        "../../spider2-lite/resource/documents", knowledge_file
    )
    instruction = "How many pseudo users were active in the last 7 days but inactive in the last 2 days as of January 7, 2021?"

    if os.path.exists(knowledge_path):
        # Create RAG query action and retrieve relevant knowledge
        rag_query_action = RAG_QUERY(query=instruction, top_k=3)
        external_knowledge_content = rag_query_action.retrieve_relevant_knowledge(
            knowledge_path
        )
        print(external_knowledge_content)
    else:
        print(f"Knowledge file '{knowledge_file}' not found.")
