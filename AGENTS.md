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
.venv/bin/python scripts/check_gsm8k_answer_extraction.py
.venv/bin/python scripts/init_gsm8k_dataset.py
.venv/bin/python scripts/inspect_gsm8k_data.py --split train --limit 3
```

Do not assume `conda` is installed. `environment.yml` is kept as an optional conda path, but `.venv` is the current working setup.

## Stage Boundary

Stage 1 environment installation and validation is complete. The active stage is stage 2: GSM8K data loading, sample inspection, and standard answer extraction.

The stage 2 GSM8K initialization command stores the dataset cache under `data/hf_datasets/`, which is ignored by git. It may download the GSM8K dataset cache on first run. Do not download model weights. Do not add model generation, GRPO training, evaluation scoring, reward functions, Math-Verify, vLLM, Ray, DeepSpeed, FSDP, verl, OpenRLHF, or multi-GPU requirements until a later OpenSpec change.

## Verification

Before reporting stage-2 data changes as complete, run:

```bash
.venv/bin/python scripts/smoke_check_env.py
.venv/bin/python scripts/check_gsm8k_answer_extraction.py
.venv/bin/python scripts/init_gsm8k_dataset.py
.venv/bin/python scripts/inspect_gsm8k_data.py --split train --limit 3
openspec validate stage-2-gsm8k-data-baseline --strict
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
