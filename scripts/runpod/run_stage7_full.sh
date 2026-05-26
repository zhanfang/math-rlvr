#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/math-rlvr}"
cd "$PROJECT_ROOT"

export HF_HOME="${HF_HOME:-/workspace/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/data/hf_datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_DIR="outputs/stage-7-full-gsm8k-training/logs"
mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "[$(date -Is)] START ${name}" | tee -a "$LOG_DIR/full_pipeline.log"
  "$@" 2>&1 | tee "$log"
  echo "[$(date -Is)] END ${name}" | tee -a "$LOG_DIR/full_pipeline.log"
}

nvidia-smi | tee "$LOG_DIR/nvidia_start.log"

run_step sft_full \
  "$PYTHON_BIN" scripts/run/gsm8k_sft.py \
    --model-name Qwen/Qwen2.5-0.5B-Instruct \
    --allow-dataset-download \
    --output-dir outputs/stage-7-full-gsm8k-training/sft

run_step grpo_full \
  "$PYTHON_BIN" scripts/run/minimal_grpo_training.py \
    --experiment-profile stage7-full-gsm8k-training \
    --model-name Qwen/Qwen2.5-0.5B-Instruct \
    --allow-dataset-download \
    --initial-adapter-path outputs/stage-7-full-gsm8k-training/sft/adapter \
    --use-vllm \
    --vllm-mode colocate \
    --vllm-gpu-memory-utilization 0.35 \
    --vllm-max-model-length 1024

run_step eval_200 \
  "$PYTHON_BIN" scripts/run/gsm8k_eval.py \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter-path outputs/stage-7-full-gsm8k-training/train/adapter \
    --mode both \
    --limit 200 \
    --k 1 \
    --max-new-tokens 192 \
    --allow-dataset-download \
    --output-dir outputs/stage-7-full-gsm8k-training/eval \
    --train-run-config outputs/stage-7-full-gsm8k-training/train/run_config.json \
    --train-metrics outputs/stage-7-full-gsm8k-training/train/train_metrics.json

nvidia-smi | tee "$LOG_DIR/nvidia_end.log"
echo "[$(date -Is)] FULL PIPELINE FINISHED" | tee -a "$LOG_DIR/full_pipeline.log"
