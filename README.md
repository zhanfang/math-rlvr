# math-rlvr

Single-machine learning project for understanding Math RLVR step by step.

Stage 1 environment validation is complete. The current active stage is GSM8K data loading and baseline inspection. Training, reward functions, model download, Math-Verify, vLLM, and multi-card frameworks are later milestones.

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

## Next Stage

After GSM8K loading and answer extraction are stable, stage 3 can add reward-function experiments with local hard-coded examples.
