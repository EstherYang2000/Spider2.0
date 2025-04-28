# test_rag_syntax_agent.py
"""
簡單測試 RAGSyntaxAgent 的 retrieve 與 format_for_prompt 方法。
請確保 syntax_dir 目錄下有 BigQuery 相關語法文件。
"""
from rag_syntax_agent import RAGSyntaxAgent

def main():
    # 測試主題可根據你 syntax 目錄內容調整
    topics = ["ARRAY_AGG", "WITH clause", "partition by", "nonexistent_topic"]
    agent = RAGSyntaxAgent(syntax_dir="../../spider2-lite/resource/syntax", top_k=1)
    results = agent.retrieve(topics)
    print("=== 檢索結果 ===")
    for k, v in results.items():
        print(f"[Topic] {k}\n{v}\n")
    prompt_block = agent.format_for_prompt(results)
    print("=== Prompt Block ===")
    print(prompt_block)

if __name__ == "__main__":
    main()
