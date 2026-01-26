"""
Test Script for RAG Query Action

This module provides a test script for the RAG_QUERY action, demonstrating
how to use Retrieval-Augmented Generation to retrieve relevant knowledge
from external documentation files.

The test demonstrates:
- Loading a specific knowledge document
- Creating a RAG query with a user question
- Retrieving relevant snippets using semantic similarity
- Displaying the retrieved knowledge content

This is typically used as part of a larger agent system where external
knowledge enhances SQL query generation capabilities.
"""

from rag_action import RAG_QUERY
import os

# Initialize external knowledge content variable
external_knowledge_content = None

# Check if external knowledge is available (typically done in agent context)
# if 'external_knowledge' in self.env.task_config:

# Specify the knowledge document to use for testing
knowledge_file = "ga4_obfuscated_sample_ecommerce.events.md"
knowledge_path = os.path.join("../../spider2-lite/resource/documents", knowledge_file)

# Define the user query for which we want to retrieve relevant knowledge
instruction = "How many pseudo users were active in the last 7 days but inactive in the last 2 days as of January 7, 2021?"

if os.path.exists(knowledge_path):
    # Initialize RAG_QUERY instance with the user query and retrieval parameters
    rag_query_action = RAG_QUERY(query=instruction, top_k=9)

    # Retrieve relevant knowledge snippets from the specific document
    external_knowledge_content = rag_query_action.retrieve_relevant_knowledge(
        knowledge_path
    )
    print(external_knowledge_content)

    # In a full agent implementation, the RAG_QUERY action would be added to the agent's memory
    # self.actions.append(rag_query_action)
