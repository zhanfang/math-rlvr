#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

cd "$PROJECT_ROOT"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

echo "RunPod stage 7 local validation"
echo "================================"
echo "project_root=$PROJECT_ROOT"
echo "python=$PYTHON_BIN"
echo

echo "[1/6] shell syntax"
bash -n \
  scripts/runpod/common.sh \
  scripts/runpod/preflight.sh \
  scripts/runpod/smoke.sh \
  scripts/runpod/run_stage7_full.sh

echo "[2/6] Python entrypoint syntax"
"$PYTHON_BIN" -m py_compile \
  scripts/check/smoke_check_env.py \
  scripts/check/verify_gsm8k_answers.py \
  scripts/check/verify_answer_rewards.py \
  scripts/data/cache_gsm8k_dataset.py \
  scripts/run/gsm8k_sft.py \
  scripts/run/minimal_grpo_training.py \
  scripts/run/gsm8k_eval.py

echo "[3/6] pinned RunPod requirements sanity"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

path = Path("requirements-runpod-stage7.txt")
required = {
    "torch==2.10.0",
    "transformers==4.57.6",
    "huggingface-hub==0.36.2",
    "trl[vllm]==1.5.0",
    "vllm==0.18.0",
    "hf_transfer==0.1.9",
}
lines = {
    line.strip()
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
}
missing = sorted(required - lines)
if missing:
    raise SystemExit(f"missing pinned requirements: {missing}")
print("requirements pins ok")
PY

echo "[4/6] Dockerfile sanity"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

dockerfile = Path("docker/runpod-stage7/Dockerfile")
text = dockerfile.read_text(encoding="utf-8")
required = [
    "FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "requirements-runpod-stage7.txt",
    "huggingface-hub",
    "use_vllm",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"Dockerfile missing expected text: {missing}")
print("Dockerfile references ok")
PY

echo "[5/6] local offline checks"
"$PYTHON_BIN" scripts/check/verify_gsm8k_answers.py
"$PYTHON_BIN" scripts/check/verify_answer_rewards.py

echo "[6/6] local dry-runs"
"$PYTHON_BIN" scripts/run/minimal_grpo_training.py \
  --experiment-profile stage7-full-gsm8k-training \
  --dry-run \
  --train-limit 8
"$PYTHON_BIN" scripts/run/gsm8k_sft.py --dry-run --train-limit 8

echo
echo "Local validation passed."
echo "Remaining proof must run on a CUDA Pod: scripts/runpod/preflight.sh and scripts/runpod/smoke.sh"
