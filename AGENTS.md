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
.venv/bin/python scripts/check_reward_functions.py
```

Do not assume `conda` is installed. `environment.yml` is kept as an optional conda path, but `.venv` is the current working setup.

## Stage Boundary

Stage 1 environment installation and validation is complete. Stage 2 GSM8K data loading, sample inspection, and standard answer extraction is complete. Stage 3 reward-function prototyping is complete. The next stage should be stage 4: minimal single-machine GRPO training.

The stage 2 GSM8K initialization command stores the dataset cache under `data/hf_datasets/`, which is ignored by git. It may download the GSM8K dataset cache on first run.

Stage 3 reward checks use local hard-coded examples through `.venv/bin/python scripts/check_reward_functions.py`. They should not download datasets or model weights, run model generation, execute GRPO training, save LoRA adapters, score an evaluation set, or add Math-Verify, vLLM, Ray, DeepSpeed, FSDP, verl, OpenRLHF, or multi-GPU requirements. Stage 4 may add model download and minimal training only through a new OpenSpec change.

## Verification

Before reporting stage-3 reward changes as complete, run:

```bash
.venv/bin/python -m py_compile src/gsm8k_data.py src/reward_math.py scripts/check_gsm8k_answer_extraction.py scripts/check_reward_functions.py
.venv/bin/python scripts/check_gsm8k_answer_extraction.py
.venv/bin/python scripts/check_reward_functions.py
openspec validate stage-3-reward-functions --strict
```

Expected stage-3 checks:

```text
GSM8K extraction check passed.
Reward check passed.
Change 'stage-3-reward-functions' is valid
```
