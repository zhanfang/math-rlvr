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

source "$PROJECT_ROOT/scripts/runpod/common.sh"
resolve_runpod_python

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" outputs/stage-7-full-gsm8k-training/logs

require_runpod_python_env

"$PYTHON_BIN" - <<'PY'
import inspect
import importlib.metadata as md
from packaging.version import Version

import torch
import vllm
from trl import GRPOConfig

hub_version = Version(md.version("huggingface-hub"))
if not (Version("0.34.0") <= hub_version < Version("1.0.0")):
    raise SystemExit(f"incompatible huggingface-hub={hub_version}; expected >=0.34,<1.0")

params = inspect.signature(GRPOConfig).parameters
required = {
    "use_vllm",
    "vllm_mode",
    "vllm_gpu_memory_utilization",
    "vllm_max_model_length",
    "vllm_tensor_parallel_size",
}
missing = sorted(required - set(params))
if missing:
    raise SystemExit(f"TRL GRPOConfig missing vLLM params: {missing}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

print("preflight versions")
print("python ok")
print("torch", torch.__version__)
print("cuda", torch.cuda.get_device_name(0))
print("vllm", vllm.__version__)
print("huggingface-hub", hub_version)
PY

"$PYTHON_BIN" scripts/check/smoke_check_env.py
"$PYTHON_BIN" scripts/check/verify_gsm8k_answers.py
"$PYTHON_BIN" scripts/check/verify_answer_rewards.py
"$PYTHON_BIN" scripts/data/cache_gsm8k_dataset.py
"$PYTHON_BIN" scripts/run/minimal_grpo_training.py \
  --experiment-profile stage7-full-gsm8k-training \
  --dry-run \
  --train-limit 8 \
  --use-vllm \
  --vllm-mode colocate
"$PYTHON_BIN" scripts/run/gsm8k_sft.py --dry-run --train-limit 8
