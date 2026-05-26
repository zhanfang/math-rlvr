#!/usr/bin/env bash
# Local full pipeline: SFT warm-start -> GRPO RLVR -> held-out eval.
# Outputs are kept under outputs/stage-6-gsm8k-experiment/local-full/.
set -euo pipefail

REPO_ROOT="/Users/bytedance/Documents/code/github/math-rlvr"
PYTHON="${REPO_ROOT}/.venv/bin/python"
MODEL="/Users/bytedance/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"

ROOT="${REPO_ROOT}/outputs/stage-6-gsm8k-experiment/local-full"
SFT_DIR="${ROOT}/sft"
TRAIN_DIR="${ROOT}/train"
EVAL_DIR="${ROOT}/eval"
LOG_DIR="${ROOT}/logs"

mkdir -p "${SFT_DIR}" "${TRAIN_DIR}" "${EVAL_DIR}" "${LOG_DIR}"

cd "${REPO_ROOT}"

echo "[pipeline] $(date '+%F %T') stage = sft"
"${PYTHON}" scripts/run/gsm8k_sft.py \
  --model-name "${MODEL}" \
  --split train \
  --train-limit 7473 \
  --output-dir "${SFT_DIR}" \
  --max-steps 600 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --max-length 768 \
  --learning-rate 1e-5 \
  --logging-steps 25 \
  --save-steps 200 \
  > "${LOG_DIR}/sft.log" 2>&1

echo "[pipeline] $(date '+%F %T') stage = grpo"
"${PYTHON}" scripts/run/minimal_grpo_training.py \
  --experiment-profile stage6-gsm8k-experiment \
  --model-name "${MODEL}" \
  --output-dir "${TRAIN_DIR}" \
  --initial-adapter-path "${SFT_DIR}/adapter" \
  --train-limit 7473 \
  --max-steps 1500 \
  --per-device-train-batch-size 4 \
  --steps-per-generation 4 \
  --num-generations 4 \
  --max-completion-length 192 \
  --learning-rate 5e-7 \
  --correctness-reward-weight 1 \
  --format-reward-weight 0 \
  --beta 0.02 \
  --logging-steps 25 \
  --save-steps 250 \
  > "${LOG_DIR}/grpo.log" 2>&1

echo "[pipeline] $(date '+%F %T') stage = eval"
"${PYTHON}" scripts/run/gsm8k_eval.py \
  --base-model "${MODEL}" \
  --adapter-path "${TRAIN_DIR}/adapter" \
  --mode both \
  --limit 200 \
  --k 1 \
  --train-run-config "${TRAIN_DIR}/run_config.json" \
  --train-metrics "${TRAIN_DIR}/train_metrics.json" \
  --output-dir "${EVAL_DIR}" \
  > "${LOG_DIR}/eval.log" 2>&1

echo "[pipeline] $(date '+%F %T') stage = done"
