source methods/spider-agent/spider2/bin/activate
cd methods/spider-agent-lite

python run.py --model qwen_api_32b-instruct-fp16 -s test1 --example_index 0-11

python get_spider2lite_submission_data.py \
	--experiment_suffix qwen_api_2_5_72b-test1    \ 
	--results_folder_name ../../spider2-lite/evaluation_suite/qwen_api_2_5_72b-test1 \

python evaluate.py \
	--result_dir qwen_api_2_5_72b-test1 --mode exec_result --mode exec_result


git pull --no-rebase origin main