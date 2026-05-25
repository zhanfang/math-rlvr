# math-rlvr

Single-machine learning project for understanding Math RLVR step by step.

Stage 1 environment validation, stage 2 GSM8K data inspection, and stage 3 reward-function prototyping are complete. The next milestone is a minimal single-machine GRPO training loop. Math-Verify, vLLM, and multi-card frameworks are later milestones.

## Stage 1: Environment

This workspace currently uses a local `.venv` environment. Use this path first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the smoke check from the activated environment:

```bash
python scripts/smoke_check_env.py
```

Or run it directly without activation:

```bash
.venv/bin/python scripts/smoke_check_env.py
```

Conda remains available as an optional alternative:

```bash
conda env create -f environment.yml
conda activate math-rlvr
```

If the conda environment already exists, update it:

```bash
conda env update -f environment.yml --prune
conda activate math-rlvr
```

Expected result:

- Python is 3.10 or newer.
- `torch`, `transformers`, `datasets`, `accelerate`, `peft`, and `trl` import successfully.
- The script reports available PyTorch backends: CPU, CUDA, and/or MPS.

Stage 1 does not download model weights or datasets and does not require vLLM, Ray, DeepSpeed, FSDP, verl, OpenRLHF, or multi-GPU hardware.

## Stage 2: GSM8K Data

Run the offline answer extraction check:

```bash
.venv/bin/python scripts/check_gsm8k_answer_extraction.py
```

Initialize GSM8K into the project-local dataset cache:

```bash
.venv/bin/python scripts/init_gsm8k_dataset.py
```

Inspect a small GSM8K subset:

```bash
.venv/bin/python scripts/inspect_gsm8k_data.py --split train --limit 3
```

Expected result:

- The offline check passes without downloading datasets or model weights.
- The dataset cache is stored under `data/hf_datasets/`, which is ignored by git.
- The GSM8K inspection command prints field names, question text, raw answer text, and the extracted final answer.
- The inspection command is read-only and does not run model generation, training, reward calculation, or scoring.

The first initialization run may need network access to download the dataset cache from Hugging Face. After the dataset is cached, later inspections can reuse `data/hf_datasets/`.

## Stage 3: Reward Functions

Run the offline reward-function check:

```bash
.venv/bin/python scripts/check_reward_functions.py
```

Expected result:

- The check uses local hard-coded examples only.
- Model completions use `<reasoning>...</reasoning><answer>...</answer>`.
- The script prints each sample's extracted model answer, extracted GSM8K-style expected answer, correctness reward, and format reward.
- Correctness reward returns `1.0` for lightweight numeric equivalence and `0.0` otherwise.
- Format reward returns `1.0` only when both reasoning and answer tags are present and non-empty.
- The check does not download datasets, download model weights, run model generation, train, or score an evaluation set.

## Next Stage

After reward functions work on local examples, stage 4 can add a minimal single-machine GRPO training loop.
