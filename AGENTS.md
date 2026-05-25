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
.venv/bin/python scripts/verify_gsm8k_answers.py
.venv/bin/python scripts/cache_gsm8k_dataset.py
.venv/bin/python scripts/inspect_gsm8k_dataset.py --split train --limit 3
.venv/bin/python scripts/verify_answer_rewards.py
.venv/bin/python scripts/run_minimal_grpo_training.py --dry-run
```

Do not assume `conda` is installed. `environment.yml` is kept as an optional conda path, but `.venv` is the current working setup.

## Stage Boundary

Stage 1 environment installation and validation is complete. Stage 2 GSM8K data loading, sample inspection, and standard answer extraction is complete. Stage 3 reward-function prototyping is complete. The active stage is stage 4: minimal single-machine GRPO training.

The stage 2 GSM8K initialization command stores the dataset cache under `data/hf_datasets/`, which is ignored by git. It may download the GSM8K dataset cache on first run.

Stage 3 reward checks use local hard-coded examples through `.venv/bin/python scripts/verify_answer_rewards.py`.

Stage 4 may download a small model and run a tiny GRPO training job through `.venv/bin/python scripts/run_minimal_grpo_training.py`, but first run `--dry-run`. Keep stage 4 single-machine and minimal: no vLLM, Ray, DeepSpeed, FSDP, verl, OpenRLHF, multi-GPU requirements, wandb, Math-Verify, or formal evaluation scoring. Outputs belong under `outputs/stage-4-minimal-grpo/`, which is ignored by git.

## Verification

Before reporting stage-4 training changes as complete, run:

```bash
.venv/bin/python -m py_compile src/gsm8k_dataset.py src/answer_rewards.py src/minimal_grpo_training.py scripts/verify_gsm8k_answers.py scripts/verify_answer_rewards.py scripts/run_minimal_grpo_training.py
.venv/bin/python scripts/verify_gsm8k_answers.py
.venv/bin/python scripts/verify_answer_rewards.py
.venv/bin/python scripts/run_minimal_grpo_training.py --dry-run
openspec validate stage-4-minimal-grpo-training --strict
```

Expected stage-4 checks:

```text
GSM8K extraction check passed.
Reward check passed.
Dry-run passed.
Change 'stage-4-minimal-grpo-training' is valid
```

If network and resources allow, also run a tiny real training command:

```bash
.venv/bin/python scripts/run_minimal_grpo_training.py --max-steps 1 --train-limit 2
```
