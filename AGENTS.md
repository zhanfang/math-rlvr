# Agent Notes

## Environment

Use the local `.venv` first. This repository has already been validated with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For non-interactive commands, prefer the explicit interpreter:

```bash
.venv/bin/python scripts/smoke_check_env.py
```

Do not assume `conda` is installed. `environment.yml` is kept as an optional conda path, but `.venv` is the current working setup.

## Stage Boundary

The active stage is environment installation and validation only.

Do not download model weights or datasets during stage 1. Do not add GRPO training, evaluation, Math-Verify, vLLM, Ray, DeepSpeed, FSDP, verl, OpenRLHF, or multi-GPU requirements until a later OpenSpec change.

## Verification

Before reporting stage-1 environment changes as complete, run:

```bash
.venv/bin/python scripts/smoke_check_env.py
openspec validate single-machine-learning-rlvr --strict
```

Expected smoke check imports:

```text
torch
transformers
datasets
accelerate
peft
trl
```
