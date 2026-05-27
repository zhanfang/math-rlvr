#!/usr/bin/env bash
# Stage 7 alternative pipeline: Qwen2.5-1.5B-Instruct, no SFT warm-start,
# GRPO with num_generations=4 / temperature=0.7 / lr=5e-7 / 400 steps,
# LoRA on q,k,v,o,gate,up,down, vLLM colocate on a single L40S 48GB.
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

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
RUN_DIR="${RUN_DIR:-outputs/stage-7-full-gsm8k-training-qwen15b}"
LOG_DIR="$RUN_DIR/logs"
TRAIN_DIR="$RUN_DIR/train"
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$LOG_DIR" "$TRAIN_DIR" "$EVAL_DIR"

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "[$(date -Is)] START ${name}" | tee -a "$LOG_DIR/full_pipeline.log"
  "$@" 2>&1 | tee "$log"
  echo "[$(date -Is)] END ${name}" | tee -a "$LOG_DIR/full_pipeline.log"
}

nvidia-smi | tee "$LOG_DIR/nvidia_start.log"

# GRPO directly from base, no SFT warm-start.
# pdbs * ga * num_processes must be a multiple of num_generations:
#   pdbs=4, ga=1, num_generations=4 -> 4 * 1 = 4 (ok).
run_step grpo_qwen15b \
  "$PYTHON_BIN" scripts/run/minimal_grpo_training.py \
    --experiment-profile stage7-full-gsm8k-training \
    --model-name "$MODEL_NAME" \
    --allow-dataset-download \
    --output-dir "$TRAIN_DIR" \
    --train-limit 7473 \
    --max-steps 400 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 1 \
    --steps-per-generation 4 \
    --num-generations 4 \
    --max-prompt-length 512 \
    --max-completion-length 192 \
    --learning-rate 5e-7 \
    --correctness-reward-weight 1.0 \
    --format-reward-weight 0.1 \
    --beta 0.02 \
    --temperature 0.7 \
    --top-p 0.95 \
    --top-k 50 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging-steps 25 \
    --save-steps 100 \
    --use-vllm \
    --vllm-mode colocate \
    --vllm-gpu-memory-utilization 0.4 \
    --vllm-max-model-length 1024

run_step eval_200 \
  "$PYTHON_BIN" scripts/run/gsm8k_eval.py \
    --base-model "$MODEL_NAME" \
    --adapter-path "$TRAIN_DIR/adapter" \
    --mode both \
    --limit 200 \
    --k 1 \
    --max-new-tokens 192 \
    --allow-dataset-download \
    --output-dir "$EVAL_DIR" \
    --train-run-config "$TRAIN_DIR/run_config.json" \
    --train-metrics "$TRAIN_DIR/train_metrics.json"

nvidia-smi | tee "$LOG_DIR/nvidia_end.log"
echo "[$(date -Is)] FULL PIPELINE FINISHED" | tee -a "$LOG_DIR/full_pipeline.log"
