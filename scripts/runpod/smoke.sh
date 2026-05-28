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

source "$PROJECT_ROOT/scripts/runpod/common.sh"
resolve_runpod_python

mkdir -p outputs/stage-7-full-gsm8k-training/logs

require_runpod_python_env

"$PYTHON_BIN" scripts/run/gsm8k_sft.py \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --allow-dataset-download \
  --output-dir outputs/stage-7-full-gsm8k-training/sft-smoke \
  --train-limit 8 \
  --max-steps 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --logging-steps 1 \
  --save-steps 1 \
  2>&1 | tee outputs/stage-7-full-gsm8k-training/logs/sft_smoke.log

"$PYTHON_BIN" scripts/run/minimal_grpo_training.py \
  --experiment-profile stage7-full-gsm8k-training \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --allow-dataset-download \
  --output-dir outputs/stage-7-full-gsm8k-training/grpo-vllm-smoke \
  --train-limit 2 \
  --max-steps 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --steps-per-generation 2 \
  --num-generations 2 \
  --max-completion-length 32 \
  --logging-steps 1 \
  --save-steps 1 \
  --use-vllm \
  --vllm-mode colocate \
  --vllm-gpu-memory-utilization 0.35 \
  --vllm-max-model-length 768 \
  2>&1 | tee outputs/stage-7-full-gsm8k-training/logs/grpo_vllm_smoke.log
