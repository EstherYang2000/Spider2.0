from rag_action import RAG_QUERY
import os 

external_knowledge_content = None
# Check if external knowledge is available
# if 'external_knowledge' in self.env.task_config:

knowledge_file = "ga4_obfuscated_sample_ecommerce.events.md"
knowledge_path = os.path.join("../../spider2-lite/resource/documents", knowledge_file)
instruction = "How many pseudo users were active in the last 7 days but inactive in the last 2 days as of January 7, 2021?"
if os.path.exists(knowledge_path):
    # 🔹 Initialize RAG_QUERY instance with the exact document
    rag_query_action = RAG_QUERY(query=instruction, top_k=9)

    # 🔹 Retrieve relevant knowledge **only from the specific file**
    external_knowledge_content = rag_query_action.retrieve_relevant_knowledge(knowledge_path)
    print(external_knowledge_content)
    # 🔹 Add the RAG_QUERY action to the agent's memory
    # self.actions.append(rag_query_action)
