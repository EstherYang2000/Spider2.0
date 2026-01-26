# test_rag_syntax_agent.py
"""
Test Script for RAG Syntax Agent

This module provides a simple test script for the RAGSyntaxAgent class.
It demonstrates how to use the agent to retrieve BigQuery syntax documentation
for specific topics and format the results for use in prompts.

The test covers:
- Retrieving syntax documentation for multiple topics
- Handling topics with available documentation
- Handling topics without available documentation
- Formatting retrieved results into prompt-compatible blocks

Prerequisites:
- Ensure the syntax_dir directory contains BigQuery syntax documentation files
- The RAGSyntaxAgent class must be properly implemented and available
"""
from rag_syntax_agent import RAGSyntaxAgent


def main():
    """
    Main test function demonstrating RAG Syntax Agent functionality.

    Tests retrieval of syntax documentation for various BigQuery topics
    and demonstrates the formatting of results for prompt usage.
    """
    # Test topics - adjust based on available syntax documentation
    topics = ["ARRAY_AGG", "WITH clause", "partition by", "nonexistent_topic"]

    # Initialize the RAG Syntax Agent
    agent = RAGSyntaxAgent(syntax_dir="../../spider2-lite/resource/syntax", top_k=1)

    # Retrieve syntax documentation for the specified topics
    results = agent.retrieve(topics)

    print("=== Retrieval Results ===")
    for topic, documentation in results.items():
        print(f"[Topic] {topic}\n{documentation}\n")

    # Format the results into a prompt-compatible block
    prompt_block = agent.format_for_prompt(results)
    print("=== Prompt Block ===")
    print(prompt_block)


if __name__ == "__main__":
    main()
