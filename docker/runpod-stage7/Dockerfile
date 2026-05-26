FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/opt/hf_home \
    HF_HUB_CACHE=/opt/hf_home/hub \
    TRANSFORMERS_CACHE=/opt/hf_home/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/math-rlvr

COPY requirements.txt requirements-runpod-stage7.txt ./

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r requirements-runpod-stage7.txt

COPY . .

RUN python3 - <<'PY'
import inspect
import importlib.metadata as md
from packaging.version import Version

import torch
import vllm
from trl import GRPOConfig

hub_version = Version(md.version("huggingface-hub"))
if not (Version("0.34.0") <= hub_version < Version("1.0.0")):
    raise SystemExit(f"incompatible huggingface-hub: {hub_version}")
if "use_vllm" not in inspect.signature(GRPOConfig).parameters:
    raise SystemExit("TRL GRPOConfig does not expose use_vllm")

print("stage7 image check")
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("vllm", vllm.__version__)
print("huggingface-hub", hub_version)
PY

RUN python3 scripts/check/verify_gsm8k_answers.py && \
    python3 scripts/check/verify_answer_rewards.py

WORKDIR /workspace/math-rlvr
