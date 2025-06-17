#!/bin/bash
# Update the repository with latest changes
git pull --no-rebase origin main

# Activate the Spider2 virtual environment
source methods/spider-agent/spider2/bin/activate

# Change to the spider-agent-lite directory
cd methods/spider-agent-lite

# Run the initial experiment with GPT-4.1 model
# Parameters:
#   --model: Specifies the language model to use
#   -s [test8:suffix]: Specifies the suffix to use
#   --example_index [32-33:index]: Processes examples 32 and 33
#   --self_refinement: Enables self-refinement mechanism
#   --plan: Enables planning
#   --rag_syntax: Enables RAG syntax
#   --validate_result: Enables result validation
#   --overwriting: Allows overwriting existing results
python run.py --model gpt-4.1-2025-04-14 -s test8 --example_index 32-33 --self_refinement --plan --rag_syntax --validate_result --overwriting

# Process results for GPT-4.1 model
# This generates submission data for evaluation
python get_spider2lite_submission_data.py \
	--experiment_suffix gpt-4.1-2025-04-14-test8-plan-self-refinement \
	--results_folder_name ../../spider2-lite/evaluation_suite/gpt-4.1-2025-04-14-test8-plan-self-refinement \
	
# Return to the root directory
cd ../..
# Change to the evaluation suite directory
cd spider2-lite/evaluation_suite
# Evaluate the Grok-3-beta results
# Parameters:
#   --result_dir: Directory containing the results
#   --mode exec_result: Evaluates execution results
#   --max_evaluate_num 30: Maximum number of examples to evaluate
python evaluate.py \
	--result_dir gpt-4.1-2025-04-14-test8-plan-self-refinement --mode exec_result --mode exec_result --max_evaluate_num 30


python run.py --model grok-3-beta -s test24 --example_index 60-80 --self_refinement --plan --rag_syntax --validate_result --overwriting

# Process results for GPT-4.1 model
# This generates submission data for evaluation
python get_spider2lite_submission_data.py \
	--experiment_suffix  grok-3-beta-test24-plan-self-refinement\
	--results_folder_name ../../spider2-lite/evaluation_suite/grok-3-beta-test24-plan-self-refinement

# Return to the root directory
cd ../..
# Change to the evaluation suite directory
cd spider2-lite/evaluation_suite
# Evaluate the grok-3-beta results
# Parameters:
#   --result_dir: Directory containing the results
#   --mode exec_result: Evaluates execution results
#   --max_evaluate_num 20: Maximum number of examples to evaluate

python evaluate.py \
	--result_dir grok-3-beta-test24-plan-self-refinement --mode exec_result --mode exec_result --max_evaluate_num 20

python evaluate.py \
	--result_dir vote/data/gpt_4.1_grok3_vote_100_rwma --mode exec_result --mode exec_result

python run.py --model gemini-2.5-pro-preview-05-06 -s test6 --example_index 0-1 --self_refinement --plan --rag_syntax --validate_result --overwriting


python get_spider2lite_submission_data.py \
	--experiment_suffix  gemini-2.5-pro-preview-05-06-test6-plan-self-refinement\
	--results_folder_name ../../spider2-lite/evaluation_suite/gemini-2.5-pro-preview-05-06-test6-plan-self-refinement

python evaluate.py \
	--result_dir gemini-2.5-pro-preview-05-06-test6-plan-self-refinement --mode exec_result --mode exec_result


python evaluate.py \
	--result_dir vote/data/gpt_4.1_grok3_vote_100_rwma --mode exec_result --mode exec_result

