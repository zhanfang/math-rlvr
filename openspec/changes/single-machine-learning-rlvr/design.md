## Context

This repository is currently a lightweight learning workspace with planning notes, empty `src/`, `scripts/`, `outputs/`, and `reports/` directories. The existing `plan.md` describes a full Math RLVR path with TRL, GRPO, GSM8K, reward functions, evaluation, Math-Verify, vLLM, and later multi-card frameworks.

For the first learning pass, that scope is too broad. The project should first make the local environment reproducible and verifiable, then add the minimum code needed for a single-machine RLVR loop in later stages.

## Goals / Non-Goals

**Goals:**

- Define a staged, single-machine learning roadmap for Math RLVR.
- Make stage 1 limited to environment installation and dependency validation.
- Keep the first implementation small enough that each step can be run and understood independently.
- Leave clear later milestones for dataset loading, answer verification, GRPO training, and evaluation.

**Non-Goals:**

- No distributed training, Ray, DeepSpeed, FSDP, verl, or OpenRLHF in the first implementation.
- No vLLM rollout acceleration in the first implementation.
- No immediate training script or evaluation script until the environment is validated.
- No requirement to reach benchmark performance during the setup stage.

## Decisions

1. Use a local Python environment as the first deliverable.

   The first useful artifact should be an installable environment file plus a smoke check. This lets the learner verify Python, PyTorch, Transformers, Datasets, PEFT, and TRL before touching RLVR logic.

   Alternative considered: write the GRPO script immediately. That would make failures harder to diagnose because environment, dataset, model download, reward logic, and training would all fail in the same step.

2. Keep the project single-machine by default.

   The learning workflow will assume one local machine with CPU, CUDA, or Apple MPS support depending on hardware. Scripts must not require a cluster, scheduler, remote inference server, or multi-GPU setup.

   Alternative considered: plan directly for vLLM or verl. Those are useful later, but they add deployment and debugging complexity before the learner understands the core loop.

3. Separate roadmap from active implementation scope.

   `plan.md` should become a compact staged roadmap. OpenSpec tasks should identify only the next implementable step, while future RLVR features remain visible as later milestones.

   Alternative considered: keep all generated training code in `plan.md`. That makes the plan look complete but turns it into a large copy-paste target instead of a learning sequence.

4. Use smoke checks before model/data downloads.

   Stage 1 validation should import core packages and report available acceleration backends. It should not require downloading Qwen weights or GSM8K.

   Alternative considered: validate with a full base-model evaluation. That is a better stage 2 check after dependency installation is proven.

## Risks / Trade-offs

- Environment packages may change quickly -> Pin major learning dependencies enough to keep the setup understandable, and document that exact versions can be adjusted if hardware requires it.
- Local hardware may not support fast training -> Start with smoke checks and small dry runs before adding expensive training commands.
- The roadmap may feel slower than direct implementation -> Keep each stage concrete and verifiable so progress is visible.
- PyTorch installation differs across CPU, CUDA, and Apple Silicon -> Document the default path and leave hardware-specific overrides explicit.
