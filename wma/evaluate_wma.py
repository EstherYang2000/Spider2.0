import argparse
import json
import os
import sys
import re
# Ensure project root is on path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wma.wma import WeightedMajorityAlgorithm, auto_select_epsilon
from utils import get_bigquery_sql_result, get_snowflake_sql_result, get_sqlite_result,append_json,pre_evaluate_spider2sql


def extract_sql_text(response: str):
    """
    Extract SQL query or command from the response string.
    Returns a tuple of (sql_query, is_save, save_path), (file_path, command, output), or (None, None, None).
    """
    if response.startswith("BIGQUERY_EXEC_SQL("):
        pattern = r'''BIGQUERY_EXEC_SQL\(sql_query=(?P<quote>"""|"|'|""|'')(.*?)(?P=quote), is_save=(True|False)(, save_path=(?P<quote2>"|'|""|'')(.*?)(?P=quote2))?\)'''
        match = re.search(pattern, response, flags=re.DOTALL)
        if match:
            sql_query = match.group(2).strip()  # Capturing the SQL query part
            is_save = match.group(3).strip().lower() == 'true'  # Determining is_save
            save_path = match.group(6) if match.group(6) else ""  # Optional save_path handling
            return (sql_query, is_save, save_path)
    elif response.startswith("SNOWFLAKE_EXEC_SQL("):
        # Use only valid quote regex (no triple quotes in pattern string)
        pattern = r'SNOWFLAKE_EXEC_SQL\(\s*sql_query\s*=\s*(?P<quote_sql>"|\'|\"\"\"|\'\'\')(?P<sql_query>.*?)(?<!\\)(?P=quote_sql),\s*is_save\s*=\s*(?P<is_save>True|False)(?:,\s*save_path\s*=\s*(?P<quote_path>"|\'|\"\"\"|\'\'\')(?P<save_path>.*?)(?<!\\)(?P=quote_path))?\s*\)'
        match = re.search(pattern, response, flags=re.DOTALL)
        if match:
            sql_query_raw = match.group('sql_query')
            sql_query = sql_query_raw.replace(r'\"', '"').replace(r"\'", "'").replace('\\\\', '\\')
            is_save_str = match.group('is_save')
            is_save = is_save_str.strip().lower() == 'true'
            save_path = ""
            if match.group('save_path'):
                save_path_raw = match.group('save_path')
                save_path = save_path_raw.replace(r'\"', '"').replace(r"\'", "'").replace('\\\\', '\\')
            return (sql_query, is_save, save_path)
        return (None, None, None)
    elif response.startswith("LOCAL_DB_SQL("):
        matches = re.findall(r'LOCAL_DB_SQL\(file_path=(.*?), command=(.*?), output=(.*?)\)', response, flags=re.DOTALL)
        if matches:
            file_path, command, output = (item.strip() for item in matches[-1])
            return (file_path, command, output)
    return (None, None, None)
def execute_sql(sql_query,database_id=None,db_path=None):
    data_df = None
    if database_id.startswith("bq"):
        data_df = get_bigquery_sql_result(sql_query)
    elif database_id.startswith("sf"):
        data_df = get_snowflake_sql_result(sql_query,database_id)
    elif database_id.startswith("local"):
        data_df = get_sqlite_result(sql_query,db_path)
    return data_df

def run_sql_generation_wma(path_generate:str,questions:list, gold_dir:str, start_num_prompts:int, end_num_prompts:int, strategy:str="wma",auto_epsilon:bool=False):
    if args.experts:
        expert_list = []
        if "llamaapi_3_3" in args.experts:
            expert_list.append({"name": "llamaapi_3.3", "model": "llamaapi", "version": "3.3", "path_generate": "path_generate"})
        if "gpt-4o" in args.experts:
            expert_list.append({"name": "gpt-4o", "model": "gptapi", "version": "chatgpt-4o-latest", "path_generate": "chatgpt-4o-latest-mcp_rag_log-plan-self-refinement"})
        if "qwen_api_32b-instruct-fp16" in args.experts:
            expert_list.append({"name": "qwen_api_32b-instruct-fp16", "model": "qwen_api", "version": "32b-instruct-fp16", "path_generate": "path_generate"})
        if "qwen_api_2_5_72b" in args.experts:
            expert_list.append({"name": "qwen_api_2_5_72b", "model": "qwen_api", "version": "2_5_72b" ,"path_generate": "path_generate"})
        if "gemini" in args.experts:
            expert_list.append({"name": "gemini", "model": "googlegeminiapi", "version": "gemini-2.5-pro-exp-03-25", "path_generate": "path_generate"})
        if "grok3" in args.experts:
            expert_list.append({"name": "grok3", "model": "grokapi", "version": "grok-3-beta", "path_generate": "grok-3-beta-rag_log-plan-self-refinement"})
    print(expert_list)
    wma = WeightedMajorityAlgorithm(experts=[expert["name"] for expert in expert_list])
    if auto_epsilon:
        epsilon = auto_select_epsilon(len(expert_list), end_num_prompts - start_num_prompts)
        print(f"[auto_epsilon] epsilon selected: {epsilon:.6f}")
    else:
        epsilon = 0.005
    wma = WeightedMajorityAlgorithm(epsilon=epsilon)

    for expert in expert_list:
        wma.add_expert(expert["name"], init_weight=1.0)
        expert['raw_sql_outputs'] = {}
        for i in range(start_num_prompts, end_num_prompts):
            instance_id = questions[i]["instance_id"]
            sql_path = os.path.join(path_generate,expert['path_generate'], instance_id,"spider", "result.json")
            with open(sql_path) as f:
                instance_data = json.load(f)
            final_sql = instance_data.get('final_sql', '')
            expert['raw_sql_outputs'][instance_id] = final_sql  
    # gold sql path
    gold_sql_dir = os.path.join(gold_dir, "sql")
    results, final_results = [], []
    for i in range(start_num_prompts, end_num_prompts):
        instance_id = questions[i]["instance_id"]
        predictions_dict = {}
        for expert in expert_list:
            predictions_dict[expert["name"]] = expert["raw_sql_outputs"][instance_id]
        print(predictions_dict)
        if strategy == "wma":
            final_sql, chosen_experts, best_weight = wma.weighted_majority_vote(predictions_dict)
        elif strategy == "rwma":
            final_sql, chosen_experts, best_weight = wma.randomized_weighted_majority_vote(predictions_dict)
        
        elif strategy == "naive":
            pass
        elif strategy == "rl":
            pass
        # print(final_sql)
        # print(chosen_experts)
        # print(best_weight)
        db_path = os.path.join("spider2-lite/resource/databases/spider2-localdb",f"{instance_id}.sqlite")
        gold_sql_path = os.path.join(gold_sql_dir,f"{instance_id}.sql")
        with open(gold_sql_path) as f:
            gold_sql = f.read()
        print(gold_sql)
        gold_data = execute_sql(gold_sql,instance_id,db_path)
        if gold_sql:
            is_correct_any = False
            for expert, sql in predictions_dict.items():
                sql_query, _, _ = extract_sql_text(sql)
                print(sql_query)
                pred_data = execute_sql(sql_query,instance_id,db_path)
                print(pred_data)
                score = pre_evaluate_spider2sql("exec_result", gold_data, pred_data, instance_id)
                # if score > 0:
                #     is_correct_any = True
                # if strategy != "naive" and strategy != "rl":
                #     wma.update_weights(expert, is_correct_any, strategy=strategy)
        # if final_sql:
        #     sql_query, _, _ = extract_sql_text(final_sql)
        #     print(sql_query)
        #     final_data = execute_sql(sql_query,instance_id,db_path)
        #     save_dir = os.path.join("spider2-lite/evaluation_suite/vote", "vote")
        #     final_data.to_csv(os.path.join(save_dir, f"{instance_id}.csv"), index=False)
        # if auto_epsilon and index > 0:
        #     mistake_counts = wma.get_mistake_counts()
        #     best_expert_name, best_mistake_count = min(mistake_counts.items(), key=lambda x: x[1])

        #     print(f"[Round {index}] epsilon updated to {epsilon:.6f} using best_expert: {best_expert_name} (mistakes: {best_mistake_count})")
        # else:
        #     best_expert_name, best_mistake_count = "-", 0
        # print(f"✅ Overall best expert: {best_expert_name} with {best_mistake_count} mistakes.")
        # # results.append({
        #     "index": index + start_num_prompts,
        #     "question": questions[i]["question"],
        #     "gold_sql": gold_sql,
        #     "final_sql": final_sql,
        #     "chosen_experts": chosen_experts,
        #     # "is_correct": is_correct_any,
        #     "current_weights": wma.get_weights(),
        #     "current_epsilon": epsilon,
        #     "currenrt_mistakes": wma.get_mistake_counts(),
        #     "best_expert": best_expert_name,
        #     "best_expert_mistakes": best_mistake_count
            
        # })
        # final_results.append({
        #     "index": index + start_num_prompts,
        #     "final_sql": final_sql,
        #     "chosen_expert": chosen_experts,
        #     "best_weight": best_weight,
        #     "current_epsilon": epsilon,
        #     "currenrt_mistakes": wma.get_mistake_counts(),
        #     "best_expert": best_expert_name,
        #     "best_expert_mistakes": best_mistake_count
        # })
        # append_json(os.path.join(path_generate,"vote", f"final_result_{round}.json"), final_results)
        # append_json(os.path.join(path_generate,"vote", f"results_{round}.json"), results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call LLM on prompts and output results.")
    parser.add_argument("--path_generate", type=str, required=True,default="methods/spider-agent-lite/output")
    parser.add_argument("--dataset_path", type=str, required=True,default="spider2-lite/evaluation_suite/gold")
    parser.add_argument("--gold_dir", type=str, required=True,default="methods/spider-agent-lite/output")
    parser.add_argument("--start_num_prompts", type=int, default=0)
    parser.add_argument("--end_num_prompts", type=int, default=1534,
                        help="Number of prompts to process from the prompt file (if not specified, take all).")
    parser.add_argument('--experts',
                    nargs='+',  # <--- 接受一個或多個值
                    required=True,
                    help="List of experts...")
    parser.add_argument('--strategy', type=str, default="wma", choices=["wma", "rwma","naive","rl"],
                        help="Strategy to use for SQL generation")
    parser.add_argument('--auto_epsilon', action='store_true',
                        help="Use auto epsilon selection")
    args = parser.parse_args()
    # Load JSONL dataset
    with open(args.dataset_path) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    run_sql_generation_wma(args.path_generate,questions, args.gold_dir, args.start_num_prompts, args.end_num_prompts, args.strategy, args.auto_epsilon)    
    """
    python wma/evaluate_wma.py \
        --path_generate methods/spider-agent-lite/output \
        --dataset_path methods/spider-agent-lite/examples/spider2-lite.jsonl \
        --gold_dir spider2-lite/evaluation_suite/gold \
        --start_num_prompts 0 \
        --end_num_prompts 1 \
        --experts gpt-4o \
        --strategy wma \
        --auto_epsilon
    """