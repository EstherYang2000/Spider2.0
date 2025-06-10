import argparse
import json
import os
import sys
import re
import pandas as pd
import shutil
# Ensure project root is on path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wma.wma import WeightedMajorityAlgorithm, auto_select_epsilon
from utils import get_bigquery_sql_result, get_snowflake_sql_result, get_sqlite_result,append_json,pre_evaluate_spider2sql
import importlib.util
spec = importlib.util.spec_from_file_location("evaluate", os.path.join(os.path.dirname(__file__), "..", "spider2-lite", "evaluation_suite", "evaluate.py"))
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)
load_jsonl_to_dict = evaluate.load_jsonl_to_dict
compare_pandas_table = evaluate.compare_pandas_table
compare_multi_pandas_table = evaluate.compare_multi_pandas_table

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

def run_sql_generation_wma(path_generate:str,questions:list, gold_dir:str, start_num_prompts:int, end_num_prompts:int, strategy:str="wma",auto_epsilon:bool=False,vote_folder:str="vote"):
    if args.experts:
        expert_list = []
        if "llamaapi_3_3" in args.experts:
            expert_list.append({"name": "llamaapi_3.3", "model": "llamaapi", "version": "3.3", "path_generate": "path_generate"})
        if "gpt-4.1-2025-04-14" in args.experts:
            expert_list.append({"name": "gpt-4.1-2025-04-14", "model": "gptapi", "version": "gpt-4.1-2025-04-14", "path_generate": "spider2-lite/evaluation_suite/gpt-4.1-2025-04-14-test8-plan-self-refinement"})
        if "qwen_api_32b-instruct-fp16" in args.experts:
            expert_list.append({"name": "qwen_api_32b-instruct-fp16", "model": "qwen_api", "version": "32b-instruct-fp16", "path_generate": "path_generate"})
        if "qwen_api_2_5_72b" in args.experts:
            expert_list.append({"name": "qwen_api_2_5_72b", "model": "qwen_api", "version": "2_5_72b" ,"path_generate": "path_generate"})
        if "gemini" in args.experts:
            expert_list.append({"name": "gemini", "model": "googlegeminiapi", "version": "gemini-2.5-pro-exp-03-25", "path_generate": "path_generate"})
        if "grok3" in args.experts:
            expert_list.append({"name": "grok3", "model": "grokapi", "version": "grok-3-beta", "path_generate": "spider2-lite/evaluation_suite/grok-3-beta-test24-plan-self-refinement"})
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
            pred_data_path = os.path.join(expert['path_generate'], f"{instance_id}.csv",)
            expert['raw_sql_outputs'][instance_id] = pred_data_path  
    # gold sql path
    results, final_results = [], []
    for index in range(start_num_prompts, end_num_prompts):
        instance_id = questions[index]["instance_id"]
        predictions_dict = {}
        for expert in expert_list:
            predictions_dict[expert["name"]] = expert["raw_sql_outputs"][instance_id]
        print(predictions_dict)
        final_ans_path = ""
        chosen_experts = []
        best_weight = 0.0
        if strategy == "wma" or strategy == "naive":
            final_ans_path, chosen_experts, best_weight = wma.weighted_majority_vote(predictions_dict)
            probs = []
            expected_error_rate = 0.0
        elif strategy == "rwma":
            final_ans_path, chosen_experts, best_weight,probs = wma.randomized_weighted_majority_vote(predictions_dict)
            expected_error_rate = 0.0
            for expert in predictions_dict:
                historical_error_rate = wma.get_mistake_counts().get(expert, 0) / index if index > 0 else 0.0
                expected_error_rate += probs[expert] * historical_error_rate
        print("final_ans_path:", final_ans_path)
        print("chosen_experts:", chosen_experts)
        print("best_weight:", best_weight)
        
        # load eval standard
        eval_standard_dict = load_jsonl_to_dict(os.path.join(args.gold_dir, "spider2lite_eval.jsonl"))
        gold_result_dir = os.path.join(args.gold_dir, "exec_result")

        if gold_result_dir:
            
            expert_correctness = {}  # Track correctness for each expert
            for expert, pred_path in predictions_dict.items():
                is_correct_any = False
                pred_pd = pd.read_csv(os.path.join(pred_path))
                print("instance_id:",instance_id)
                if '_' in instance_id:
                    pattern = re.compile(rf'^{re.escape(instance_id)}(_[a-z])?\.csv$')
                else:
                    pattern = re.compile(rf'^{re.escape(instance_id)}(_[a-z])?\.csv$')
                all_files = os.listdir(gold_result_dir)
                csv_files = [file for file in all_files if pattern.match(file)]
                csv_files = sorted(csv_files)
                if len(csv_files) == 1:
                    gold_pd = pd.read_csv(os.path.join(args.gold_dir, "exec_result",f"{instance_id}.csv"))
                    score = compare_pandas_table(pred_pd, gold_pd, eval_standard_dict.get(instance_id)['condition_cols'], eval_standard_dict.get(instance_id)['ignore_order'])
                elif len(csv_files) > 1:
                    gold_pds = [pd.read_csv(os.path.join(gold_result_dir, file)) for file in csv_files]
                    score = compare_multi_pandas_table(pred_pd, gold_pds, eval_standard_dict.get(instance_id)['condition_cols'], eval_standard_dict.get(instance_id)['ignore_order'])
                print("score:",score)
                if score == 1:
                    is_correct_any = True
                if strategy in ["wma","rwma"]:
                    wma.update_weights(expert, is_correct_any, strategy=strategy)
        if final_ans_path and os.path.exists(final_ans_path):
            save_dir = os.path.join(path_generate, "data", vote_folder)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # Copy the file from final_ans_path to the destination
            dest_path = os.path.join(save_dir, f"{instance_id}.csv")
            print("dest_path:", dest_path)
            print("final_ans_path:", final_ans_path)
            shutil.copy2(final_ans_path, dest_path)
        else:
            print(f"Warning: final_ans_path is invalid or file does not exist: {final_ans_path}")
        if auto_epsilon and i > 0:
            mistake_counts = wma.get_mistake_counts()
            best_expert_name, best_mistake_count = min(mistake_counts.items(), key=lambda x: x[1])

            print(f"[Round {i}] epsilon updated to {epsilon:.6f} using best_expert: {best_expert_name} (mistakes: {best_mistake_count})")
        else:
            best_expert_name, best_mistake_count = "-", 0
        print(f"✅ Overall best expert: {best_expert_name} with {best_mistake_count} mistakes.")
        results.append({
            "index": index,
            "question": questions[index]["question"],
            "final_path": final_ans_path,
            "chosen_experts": chosen_experts,
            "is_correct": is_correct_any,
            "current_weights": wma.get_weights(),
            "current_epsilon": epsilon,
            "expert_probabilities": probs , # 新增專家期望值
            "expected_error_rate": expected_error_rate,
            "current_mistakes": wma.get_mistake_counts(),
            "best_expert": best_expert_name,
            "best_expert_mistakes": best_mistake_count
            
        })
        final_results.append({
            "index": index ,
            "final_path": final_ans_path,
            "chosen_expert": chosen_experts,
            "best_weight": best_weight,
            "current_epsilon": epsilon,
            "current_mistakes": wma.get_mistake_counts(),
            "best_expert": best_expert_name,
            "best_expert_mistakes": best_mistake_count
        })
        save_log_dir = os.path.join(path_generate, "log", vote_folder)
        if not os.path.exists(save_log_dir):
            os.makedirs(save_log_dir)
    append_json(os.path.join(save_log_dir, f"final_result_{vote_folder}.json"), final_results)
    append_json(os.path.join(save_log_dir, f"results_{vote_folder}.json"), results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call LLM on prompts and output results.")
    parser.add_argument("--path_generate", type=str, required=True,default="methods/spider-agent-lite/output")
    parser.add_argument("--dataset_path", type=str, required=True,default="spider2-lite/evaluation_suite/gold")
    parser.add_argument("--gold_dir", type=str, required=True,default="spider2-lite/evaluation_suite/gold ")
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
    parser.add_argument('--vote_folder', type=str, default="vote",
                        help="Vote folder name")
    args = parser.parse_args()
    # Load JSONL dataset
    with open(args.dataset_path) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    run_sql_generation_wma(args.path_generate,questions, args.gold_dir, args.start_num_prompts, args.end_num_prompts, args.strategy, args.auto_epsilon,args.vote_folder)    
    """
    python wma/evaluate_wma.py \
        --path_generate spider2-lite/evaluation_suite/vote \
        --dataset_path methods/spider-agent-lite/examples/spider2-lite.jsonl \
        --gold_dir spider2-lite/evaluation_suite/gold \
        --start_num_prompts 0 \
        --end_num_prompts 100 \
        --experts gpt-4.1-2025-04-14 grok3 \
        --strategy rwma \
        --auto_epsilon \
        --vote_folder gpt_4.1_grok3_vote_100_rwma
    """