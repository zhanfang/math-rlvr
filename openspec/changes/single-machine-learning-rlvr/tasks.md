## 1. Planning Documentation

- [x] 1.1 Rewrite `plan.md` as a compact staged roadmap for the single-machine learning version.
- [x] 1.2 Mark environment installation and validation as the active first stage.
- [x] 1.3 Move GRPO training, reward functions, evaluation, Math-Verify, vLLM, and multi-card frameworks into later milestones.

## 2. Environment Setup

- [x] 2.1 Add a reproducible Python environment file for Python 3.10 or newer.
- [x] 2.2 Include core stage-1 dependencies for PyTorch, Transformers, Datasets, Accelerate, PEFT, and TRL.
- [x] 2.3 Document the local setup command for creating and activating the environment.

## 3. Smoke Validation

- [x] 3.1 Add a local smoke check script that imports the core dependencies.
- [x] 3.2 Report Python version, package import status, and available PyTorch backends.
- [x] 3.3 Document the smoke check command and the expected success criteria.

## 4. Stage Boundary

- [x] 4.1 Confirm stage 1 does not download model weights or datasets.
- [x] 4.2 Confirm stage 1 does not require vLLM, Ray, DeepSpeed, FSDP, verl, OpenRLHF, or multi-GPU hardware.
- [x] 4.3 Leave stage 2 entry criteria for dataset loading and baseline evaluation.
