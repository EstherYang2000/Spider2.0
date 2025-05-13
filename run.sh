source methods/spider-agent/spider2/bin/activate
cd methods/spider-agent-lite
python run.py --model grok-3-beta -s rag_log_syntax \
	--example_index 0-11 --self_refinement --plan --rag_syntax \

# python run.py --model qwen_api_32b-instruct-fp16 -s test1 --example_index 0-11 \
# 	--self_refinement \
# 	--plan \
# 	--rag_syntax \

python get_spider2lite_submission_data.py \
	--experiment_suffix grok-3-beta-test21-plan-self-refinement \
	--results_folder_name ../../spider2-lite/evaluation_suite/grok-3-beta-test21-plan-self-refinement \
	

cd ../..

cd spider2-lite/evaluation_suite
python evaluate.py \
	--result_dir grok-3-beta-test21-plan-self-refinement --mode exec_result --mode exec_result --max_evaluate_num 30

git pull --no-rebase origin main


python get_spider2lite_submission_data.py \
	--experiment_suffix  gpt-4.1-2025-04-14-test5-plan-self-refinement \
	--results_folder_name ../../spider2-lite/evaluation_suite/gpt-4.1-2025-04-14-test5-plan-self-refinement

python evaluate.py \
	--result_dir gpt-4.1-2025-04-14-test5-plan-self-refinement --mode exec_result --mode exec_result --max_evaluate_num 20

python run.py --model grok-3-beta -s test21 --example_index 1-2 --self_refinement --plan --rag_syntax --validate_result --use_schema_linking --schema_link_mode file --overwriting
python run.py --model o4-mini-2025-04-16 -s test10 --example_index 0-1 --self_refinement --plan --rag_syntax --validate_result --use_schema_linking --schema_link_mode file --overwriting
python run.py --model gemini-2.5-pro-preview-05-06 -s test2 --example_index 0-10 --self_refinement --plan --rag_syntax --validate_result --use_schema_linking --schema_link_mode file --overwriting
