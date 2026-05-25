# math-rlvr

Single-machine learning project for understanding Math RLVR step by step.

The current active stage is environment installation and validation. Training, reward functions, dataset loading, model download, Math-Verify, vLLM, and multi-card frameworks are later milestones.

## Stage 1: Environment

Recommended path with conda:

```bash
conda env create -f environment.yml
conda activate math-rlvr
```

If the environment already exists, update it:

```bash
conda env update -f environment.yml --prune
conda activate math-rlvr
```

Alternative path with Python 3.10 or 3.11 and venv:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the smoke check:

```bash
python scripts/smoke_check_env.py
```

If your shell only has `python3`, use:

```bash
python3 scripts/smoke_check_env.py
```

Expected result:

- Python is 3.10 or newer.
- `torch`, `transformers`, `datasets`, `accelerate`, `peft`, and `trl` import successfully.
- The script reports available PyTorch backends: CPU, CUDA, and/or MPS.

Stage 1 does not download model weights or datasets and does not require vLLM, Ray, DeepSpeed, FSDP, verl, OpenRLHF, or multi-GPU hardware.

## Next Stage

After the smoke check passes, stage 2 can add a small GSM8K data loading and baseline inspection script.
# math-rlvr
