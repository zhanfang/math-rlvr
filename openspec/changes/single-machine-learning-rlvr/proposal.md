## Why

The current `plan.md` describes a useful RLVR destination, but it reads like a full implementation plan and makes the first learning step too large. This change narrows the project into a single-machine learning workflow so the project can progress one verified step at a time.

## What Changes

- Reframe the project as a local, single-machine Math RLVR learning project.
- Replace the "build everything" path with staged milestones that begin with environment setup and a smoke check.
- Keep distributed training, vLLM rollout acceleration, verl, OpenRLHF, and large-scale datasets out of the first implementation scope.
- Define the first implementation target as installing a reproducible environment and verifying that the core Python ML dependencies can import locally.
- Preserve later RLVR goals as roadmap items rather than immediate tasks.

## Capabilities

### New Capabilities
- `single-machine-learning-workflow`: A staged local workflow for learning Math RLVR, starting with environment installation and validation before adding datasets, rewards, training, and evaluation.

### Modified Capabilities
- None.

## Impact

- Affects planning documentation and initial project scaffolding.
- Adds OpenSpec artifacts under `openspec/changes/single-machine-learning-rlvr/`.
- Future implementation may add environment files, setup scripts, dependency pins, and lightweight smoke tests.
- No distributed infrastructure, remote services, or multi-GPU assumptions are introduced in this change.
