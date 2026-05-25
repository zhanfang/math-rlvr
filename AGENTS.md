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

Stage 1 environment installation and validation is complete. Stage 2 GSM8K data loading, sample inspection, and standard answer extraction is complete. Stage 3 reward-function prototyping is complete. Stage 4 minimal single-machine GRPO training is complete. The active stage is stage 5: real single-machine RLVR training.

The stage 2 GSM8K initialization command stores the dataset cache under `data/hf_datasets/`, which is ignored by git. It may download the GSM8K dataset cache on first run.

Stage 3 reward checks use local hard-coded examples through `.venv/bin/python scripts/verify_answer_rewards.py`.

Stage 5 reuses `.venv/bin/python scripts/run_minimal_grpo_training.py` for a real single-machine RLVR run. Start with `--dry-run`, then use the stage-5 defaults or explicit overrides for a medium-scale run. Keep stage 5 single-machine: no vLLM, Ray, DeepSpeed, FSDP, verl, OpenRLHF, multi-GPU requirements, wandb, Math-Verify, or formal evaluation scoring. Outputs belong under `outputs/stage-5-real-rlvr/`, which is ignored by git.

## Verification

Before reporting stage-5 training changes as complete, run:

```bash
.venv/bin/python -m py_compile src/gsm8k_dataset.py src/answer_rewards.py src/minimal_grpo_training.py scripts/verify_gsm8k_answers.py scripts/verify_answer_rewards.py scripts/run_minimal_grpo_training.py
.venv/bin/python scripts/verify_gsm8k_answers.py
.venv/bin/python scripts/verify_answer_rewards.py
.venv/bin/python scripts/run_minimal_grpo_training.py --dry-run
.venv/bin/python scripts/run_minimal_grpo_training.py --model-name /Users/bytedance/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
openspec validate stage-5-real-rlvr-training --strict
```

Expected stage-5 checks:

```text
GSM8K extraction check passed.
Reward check passed.
Dry-run passed.
Change 'stage-5-real-rlvr-training' is valid
```

The default real-training command now serves as the stage-5 medium-scale run:

```bash
.venv/bin/python scripts/run_minimal_grpo_training.py --model-name /Users/bytedance/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
```
