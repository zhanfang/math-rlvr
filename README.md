# math-rlvr

Single-machine learning project for understanding Math RLVR step by step.

The current active stage is environment installation and validation. Training, reward functions, dataset loading, model download, Math-Verify, vLLM, and multi-card frameworks are later milestones.

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

## Next Stage

After the smoke check passes, stage 2 can add a small GSM8K data loading and baseline inspection script.
