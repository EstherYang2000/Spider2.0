import argparse
import datetime
import json
import logging
import os
import random
import sys
import glob

from tqdm import tqdm

from spider_agent.envs.spider_agent import Spider_Agent_Env
from spider_agent.agent.agents import PromptAgent
from spider_agent.agent.self_refinement import SelfRefinementAgent

# Ensure the logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")
#  Logger Configs {{{ #
logger = logging.getLogger("spider_agent")
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8")
debug_handler = logging.FileHandler(os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8")
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8")

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("spider_agent"))
sdebug_handler.addFilter(logging.Filter("spider_agent"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)
#  }}} Logger Configs # 



def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    
    parser.add_argument("--max_steps", type=int, default=20)
    
    parser.add_argument("--max_memory_length", type=int, default=25)
    parser.add_argument("--suffix", '-s', type=str, default="gpt-4-try1")
    
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--stop_token", type=str, default=None)
    
    # example config
    parser.add_argument("--test_path","-t", type=str, default="./examples/spider2-lite.jsonl")
    parser.add_argument("--example_index", "-i", type=str, default="all", help="index range of the examples to run, e.g., '0-10', '2,3', 'all'")
    parser.add_argument("--example_name", "-n", type=str, default="", help="name of the example to run")
    parser.add_argument("--overwriting", action="store_true", default=False)
    parser.add_argument("--retry_failed", action="store_true", default=False)

    # output related
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--bq_only", action="store_true")
    parser.add_argument("--local_only", action="store_true")
    parser.add_argument("--dbt_only", action="store_true")
    parser.add_argument("--sf_only", action="store_true")
    
    # Self-refinement related
    parser.add_argument("--self_refinement", action="store_true", help="Enable self-refinement for SQL queries")
    parser.add_argument("--max_refinement_iterations", type=int, default=5, help="Maximum number of refinement iterations")
    parser.add_argument("--rag_syntax", action="store_true", help="Enable RAG syntax for self-refinement")

    # Schema linking toggle
    parser.add_argument('--use_schema_linking', action='store_true', default=False, help='是否啟用 schema linking (False by default)')
    parser.add_argument('--schema_link_mode', choices=['file', 'sql'], default='file', help='Choose schema linking mode: "file" (use DDL or schema files) or "sql" (use database exploration).')
    parser.add_argument('--validate_result', action='store_true', default=False, help='use validate_result to check answer (False by default)')
    args = parser.parse_args()

    return args



def test(
    args: argparse.Namespace,
    test_all_meta: dict = None
) -> None:
    scores = []
    
    # log args
    logger.info("Args: %s", args)

    if args.suffix == "":
        logger.warning("No suffix is provided, the experiment id will be the model name.")
        experiment_id = args.model.split("/")[-1]
    else:
        experiment_id = args.model.split("/")[-1] + "-" + args.suffix
        
    if args.plan:
        experiment_id = f"{experiment_id}-plan"
        
    if args.self_refinement:
        experiment_id = f"{experiment_id}-self-refinement"

    env_config = \
    {
        "image_name": "spider_agent-image",
        "init_args": {
            "name": experiment_id,
            "work_dir": "/workspace",
        }
    }
    
    # Create the appropriate agent based on arguments
    if args.self_refinement:
        from spider_agent.agent.planner_critique_agents import PlannerAgent, CritiqueAgent
        # 初始化 env、planner_agent、critique_agent
        env = None  # 先設為 None，稍後 set_env_and_task 會補上
        planner_agent = PlannerAgent(model=args.model)
        critique_agent = CritiqueAgent(model=args.model)
        agent = SelfRefinementAgent(
            env=env,
            planner_agent=planner_agent,
            critique_agent=critique_agent,
            model=args.model,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            max_memory_length=args.max_memory_length,
            max_steps=args.max_steps,
            use_plan=args.plan,
            max_refinement_iterations=args.max_refinement_iterations,
            rag_syntax=args.rag_syntax,
            use_schema_linking=args.use_schema_linking,
            schema_link_mode=args.schema_link_mode,
            validate_result=args.validate_result
        )
    else:
        agent = PromptAgent(
            model=args.model,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            max_memory_length=args.max_memory_length,
            max_steps=args.max_steps,
            use_plan=args.plan,
        )
    valid_ids = []
    ## load task configs
    assert os.path.exists(args.test_path) and args.test_path.endswith(".jsonl"), f"Invalid test_path, must be a valid jsonl file: {args.test_path}"
    with open(args.test_path, "r") as f:
        task_configs = [json.loads(line) for line in f]

        
    if args.example_name != "":
        task_configs = [task for task in task_configs if args.example_name in task["id"]]
    else:
        if args.example_index != "all":
            if "-" in args.example_index:
                start, end = map(int, args.example_index.split("-"))
                task_configs = task_configs[start:end]
            else:
                indices = list(map(int, args.example_index.split(",")))
                task_configs = [task_configs[i] for i in indices]
    for task_config in task_configs:
        instance_id = experiment_id +"/"+ task_config["instance_id"]
        output_dir = os.path.join(args.output_dir, instance_id)
        result_json_path =os.path.join(output_dir, "spider/result.json")


        task_type = None
        if task_config["instance_id"].startswith("bq") or task_config["instance_id"].startswith("ga"):
            task_type = 'bq'
            task_config['type'] = 'Bigquery'
        elif task_config["instance_id"].startswith("local"):
            task_type = 'local'
            task_config['type'] = 'Local'
        elif task_config["instance_id"].startswith("sf"):
            task_type = 'sf'
            task_config['type'] = 'Snowflake'
        else:
            task_type = 'dbt'

        valid_types = set()
        if args.local_only: valid_types.add('local')
        if args.bq_only: valid_types.add('bq')
        if args.sf_only: valid_types.add('sf')
        if args.dbt_only: valid_types.add('dbt')
        
        if  (args.local_only or args.bq_only or args.sf_only or args.dbt_only):
            if task_type not in valid_types: continue
        else:
            pass

        valid_ids.append(task_config["instance_id"])
        
        if not args.overwriting and os.path.exists(result_json_path):
            logger.info("Skipping %s", instance_id)
            continue
        elif os.path.exists(result_json_path):
            logger.info("Overwriting %s", instance_id)
        else:
            logger.info("Running %s", instance_id)
        if args.retry_failed and os.path.exists(result_json_path):
            with open(result_json_path, "r") as f:
                result = json.load(f)
                if result["finished"] and (not "FAIL" in result["result"]) and (not "error" in result["result"].lower()):
                    logger.info("Skipping %s", instance_id)
                    continue
            logger.info("Retrying %s", instance_id)

        if os.path.exists(output_dir):
            os.system(f"rm -rf {output_dir}")
            logger.info("Removed existing %s", output_dir)

        os.makedirs(output_dir, exist_ok=True)

        env_config["init_args"]["name"] = experiment_id +"-"+ task_config["instance_id"]

        
        source_data_dir = os.path.dirname(args.test_path)        
        task_config['config'] = [{"type": "copy_all_subfiles", "parameters": {"dirs": [os.path.join(source_data_dir, task_config["instance_id"])]}}]

        env = Spider_Agent_Env(
            env_config=env_config,
            task_config=task_config,
            cache_dir="./cache",
            mnt_dir=output_dir
        )
    
        agent.set_env_and_task(env)
    
        logger.info('Task input:' + task_config['question'])
        done, result_output = agent.run()
        trajectory = agent.get_trajectory()

        os.makedirs(os.path.join(output_dir, "spider"), exist_ok=True)
        result_files = env.post_process()
        spider_result = {"finished": done, "steps": len(trajectory["trajectory"]),
                           "result": result_output,"result_files": result_files, **trajectory}
        # Extract last SQL action as final SQL and store it
        final_sql = None
        for step in spider_result.get("trajectory", []):
            action_str = step.get("action", "")
            if action_str.startswith("BIGQUERY_EXEC_SQL") or action_str.startswith("SNOWFLAKE_EXEC_SQL") or action_str.startswith("LOCAL_DB_SQL"):
                final_sql = action_str
        spider_result["final_sql"] = final_sql
        with open(os.path.join(output_dir, "spider/result.json"), "w") as f:
            json.dump(spider_result, f, indent=2)
            
            
        
        # Delete sqlite files
        if task_type == 'local':
            sqlite_files = glob.glob(os.path.join(output_dir, '*.sqlite')) + glob.glob(os.path.join(output_dir, '*.duckdb'))

            for file_path in sqlite_files:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        
        logger.info("Finished %s", instance_id)
        env.close()

if __name__ == '__main__':
    args = config()
    
    test(args)

"""
python run.py --model mistral-saba-24b -s test1 --example_index 0-1
python run.py --model gemini-2.5-pro-preview-03-25 -s mcp --example_index 0-1
python run.py --model qwen_api_32b-instruct-fp16 -s mcp --example_index 0-1 
python run.py --model llamaapi_3.3 -s test1 --example_index 0-1 --self_refinement
python run.py --model chatgpt-4o-latest -s mcp_rag_log --example_index 0-1 --self_refinement --plan
python run.py --model grok-3-beta -s rag_log --example_index 4-5 --self_refinement --plan
python run.py --model grok-3-beta -s base --example_index 1-20
python run.py --model grok-3-beta -s test7 --example_index 11-20 --self_refinement --plan --rag_syntax
python run.py --model o4-mini-2025-04-16 -s test3_wo_sl --example_index 11-20 --self_refinement --plan --rag_syntax --use_schema_linking
python run.py --model grok-3-beta -s test12 --example_index 0-10 --self_refinement --plan --use_schema_linking
python run.py --model gemini-2.5-pro-preview-03-25 -s test8 --example_index 0-5 --self_refinement --plan --use_schema_linking
python run.py --model gpt-4.1-2025-04-14 -s test2 --example_index 10-20 --self_refinement --plan --use_schema_linking --overwriting --validate_result
python run.py --model grok-3-beta -s test17 --example_index 0-1 --self_refinement --plan --rag_syntax --validate_result
python run.py --model gemini-2.5-pro-preview-03-25 -s test10 --example_index 0-1 --self_refinement --plan --use_schema_linking --overwriting --validate_result --schema_link_mode sql
python run.py --model o4-mini-2025-04-16 -s test7 --example_index 10-15 --self_refinement --plan --rag_syntax --validate_result

"""