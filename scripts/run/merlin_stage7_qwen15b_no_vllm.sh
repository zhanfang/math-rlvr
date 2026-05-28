#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PLAYGROUND_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

cd "${PROJECT_ROOT}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PIP_USER="${PIP_USER:-no}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-${PLAYGROUND_ROOT}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_ROOT}/data/hf_datasets}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
MODEL_NAME="${MODEL_NAME:-${PLAYGROUND_ROOT}/models/Qwen2.5-1.5B-Instruct}"
RUN_NAME="${RUN_NAME:-train}"
RUN_DIR="${RUN_DIR:-outputs/stage-7-merlin-a10-qwen15b/${RUN_NAME}}"
TRAIN_DIR="${RUN_DIR}"
LOG_DIR="${RUN_DIR}/logs"
EVAL_DIR="${RUN_DIR}/eval"
ALLOW_DATASET_DOWNLOAD="${ALLOW_DATASET_DOWNLOAD:-0}"
TRAIN_LIMIT="${TRAIN_LIMIT:-7473}"
MAX_STEPS="${MAX_STEPS:-400}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"

if [[ "${ALLOW_DATASET_DOWNLOAD}" == "1" ]]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
  DATASET_MODE="online"
else
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  DATASET_MODE="offline"
fi

mkdir -p "${TRAIN_DIR}" "${LOG_DIR}" "${EVAL_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"
echo "[merlin-train] dataset_mode=${DATASET_MODE} cache_dir=${HF_DATASETS_CACHE}" | tee -a "${LOG_DIR}/pipeline.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[merlin-train] python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "[merlin-train] model directory not found: ${MODEL_NAME}" >&2
  exit 1
fi

dataset_args=()
if [[ "${ALLOW_DATASET_DOWNLOAD}" == "1" ]]; then
  dataset_args+=(--allow-dataset-download)
fi

run_step() {
  local name="$1"
  shift
  local log="${LOG_DIR}/${name}.log"
  echo "[$(date -Is)] START ${name}" | tee -a "${LOG_DIR}/pipeline.log"
  "$@" 2>&1 | tee "${log}"
  echo "[$(date -Is)] END ${name}" | tee -a "${LOG_DIR}/pipeline.log"
}

nvidia-smi | tee "${LOG_DIR}/nvidia_start.log"

run_step train \
  "${PYTHON_BIN}" scripts/run/minimal_grpo_training.py \
    --experiment-profile stage7-merlin-a10-qwen15b-no-vllm \
    --model-name "${MODEL_NAME}" \
    --output-dir "${TRAIN_DIR}" \
    --train-limit "${TRAIN_LIMIT}" \
    --max-steps "${MAX_STEPS}" \
    --logging-steps "${LOGGING_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    "${dataset_args[@]}"

run_step eval \
  "${PYTHON_BIN}" scripts/run/gsm8k_eval.py \
    --base-model "${MODEL_NAME}" \
    --adapter-path "${TRAIN_DIR}/adapter" \
    --mode both \
    --limit "${EVAL_LIMIT}" \
    --k 1 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --output-dir "${EVAL_DIR}" \
    --train-run-config "${TRAIN_DIR}/run_config.json" \
    --train-metrics "${TRAIN_DIR}/train_metrics.json" \
    "${dataset_args[@]}"

nvidia-smi | tee "${LOG_DIR}/nvidia_end.log"
echo "[$(date -Is)] FULL PIPELINE FINISHED" | tee -a "${LOG_DIR}/pipeline.log"
