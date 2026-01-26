#!/usr/bin/env bash
set -e  # Exit immediately if a command fails
set -u  # Treat unset variables as errors

#######################################
# Global configuration
#######################################
REPO_BRANCH="main"
VENV_PATH="methods/spider-agent/spider2/bin/activate"
AGENT_DIR="methods/spider-agent-lite"
EVAL_DIR="spider2-lite/evaluation_suite"

#######################################
# Helper functions
#######################################

activate_env() {
  echo ">> Activating Spider2 virtual environment"
  source "$VENV_PATH"
}

update_repo() {
  echo ">> Pulling latest code from $REPO_BRANCH"
  git pull --no-rebase origin "$REPO_BRANCH"
}

run_agent() {
  local model=$1
  local suffix=$2
  local index_range=$3

  echo ">> Running Spider-Agent-lite"
  echo "   Model: $model"
  echo "   Index: $index_range"
  echo "   Suffix: $suffix"

  python run.py \
    --model "$model" \
    -s "$suffix" \
    --example_index "$index_range" \
    --self_refinement \
    --plan \
    --rag_syntax \
    --validate_result \
    --overwriting
}

generate_submission() {
  local exp_suffix=$1
  local result_dir=$2

  echo ">> Generating submission data: $exp_suffix"

  python get_spider2lite_submission_data.py \
    --experiment_suffix "$exp_suffix" \
    --results_folder_name "$result_dir"
}

evaluate_results() {
  local result_dir=$1
  local max_num=${2:-""}

  echo ">> Evaluating results: $result_dir"

  if [[ -n "$max_num" ]]; then
    python evaluate.py \
      --result_dir "$result_dir" \
      --mode exec_result \
      --max_evaluate_num "$max_num"
  else
    python evaluate.py \
      --result_dir "$result_dir" \
      --mode exec_result
  fi
}

#######################################
# Main pipeline
#######################################

update_repo
activate_env

cd "$AGENT_DIR"

# ---------- GPT-4.1 ----------
run_agent "gpt-4.1-2025-04-14" "test8" "32-33"

generate_submission \
  "gpt-4.1-2025-04-14-test8-plan-self-refinement" \
  "../../$EVAL_DIR/gpt-4.1-2025-04-14-test8-plan-self-refinement"

cd ../../"$EVAL_DIR"
evaluate_results "gpt-4.1-2025-04-14-test8-plan-self-refinement" 30

# ---------- Grok-3-beta ----------
cd ../../"$AGENT_DIR"

run_agent "grok-3-beta" "test24" "60-80"

generate_submission \
  "grok-3-beta-test24-plan-self-refinement" \
  "../../$EVAL_DIR/grok-3-beta-test24-plan-self-refinement"

cd ../../"$EVAL_DIR"
evaluate_results "grok-3-beta-test24-plan-self-refinement" 20

# ---------- Gemini 2.5 ----------
cd ../../"$AGENT_DIR"

run_agent "gemini-2.5-pro-preview-05-06" "test6" "0-1"

generate_submission \
  "gemini-2.5-pro-preview-05-06-test6-plan-self-refinement" \
  "../../$EVAL_DIR/gemini-2.5-pro-preview-05-06-test6-plan-self-refinement"

cd ../../"$EVAL_DIR"
evaluate_results "gemini-2.5-pro-preview-05-06-test6-plan-self-refinement"

# ---------- Voting Evaluation ----------
echo ">> Evaluating ensemble voting results"
evaluate_results "vote/data/gpt_4.1_grok3_vote_100_rwma"

echo "✅ All experiments finished successfully."
