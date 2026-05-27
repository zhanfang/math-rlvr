#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PLAYGROUND_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PIP_USER="${PIP_USER:-no}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-${PLAYGROUND_ROOT}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_ROOT}/data/hf_datasets}"

MODEL_NAME="${MODEL_NAME:-${PLAYGROUND_ROOT}/models/Qwen2.5-1.5B-Instruct}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
ALLOW_DATASET_DOWNLOAD="${ALLOW_DATASET_DOWNLOAD:-1}"
DRY_RUN_OUTPUT_DIR="${DRY_RUN_OUTPUT_DIR:-outputs/stage-7-merlin-a10-qwen15b/preflight-dry-run}"

cd "${PROJECT_ROOT}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[preflight] creating project-local .venv"
  python3 -m venv .venv || {
    echo "[preflight] failed to create .venv; install python3.11-venv first" >&2
    exit 1
  }
fi

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_BIN}" -m pip install -r requirements-merlin-stage7.txt
fi

if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "[preflight] model directory not found: ${MODEL_NAME}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/check/smoke_check_env.py
"${PYTHON_BIN}" scripts/check/verify_gsm8k_answers.py
"${PYTHON_BIN}" scripts/check/verify_answer_rewards.py

cache_args=()
if [[ "${ALLOW_DATASET_DOWNLOAD}" == "1" ]]; then
  cache_args+=(--splits train test)
else
  cache_args+=(--splits train test --local-files-only)
fi
"${PYTHON_BIN}" scripts/data/cache_gsm8k_dataset.py "${cache_args[@]}"

"${PYTHON_BIN}" scripts/run/minimal_grpo_training.py \
  --experiment-profile stage7-merlin-a10-qwen15b-no-vllm \
  --model-name "${MODEL_NAME}" \
  --output-dir "${DRY_RUN_OUTPUT_DIR}" \
  --train-limit 8 \
  --max-steps 2 \
  --dry-run

echo "[preflight] Merlin stage7 no-vLLM chain is ready."
