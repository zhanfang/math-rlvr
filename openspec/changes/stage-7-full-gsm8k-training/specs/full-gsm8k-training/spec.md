## ADDED Requirements

### Requirement: Stage 7 Full GSM8K Profile
The system SHALL provide a `stage7-full-gsm8k-training` training profile for full GSM8K train-split RLVR experiments without changing the stage 5 or stage 6 defaults.

#### Scenario: Stage 7 profile is selected
- **WHEN** the training entrypoint is invoked with `--experiment-profile stage7-full-gsm8k-training --dry-run`
- **THEN** the command SHALL load the stage 7 defaults and print a run summary whose `training_profile` is `stage7-full-gsm8k-training`

#### Scenario: Full training output path is isolated
- **WHEN** the stage 7 profile is used without an explicit `--output-dir`
- **THEN** the run summary SHALL set the output directory under `outputs/stage-7-full-gsm8k-training/train`

#### Scenario: RunPod L40S target is recorded
- **WHEN** the stage 7 profile is used
- **THEN** the run summary SHALL include a cloud target for RunPod with an NVIDIA L40S 48GB GPU, 16 vCPUs, 64GB RAM, 200GB disk, and colocate vLLM as the recommended single-GPU mode

#### Scenario: Earlier profiles remain available
- **WHEN** the training entrypoint is invoked with `stage5-real-rlvr` or `stage6-gsm8k-experiment`
- **THEN** those profiles SHALL preserve their existing output directories and default training parameters

### Requirement: vLLM Configuration For GRPO
The system SHALL expose vLLM rollout configuration through the GRPO training entrypoint and record the effective vLLM request in the run summary.

#### Scenario: vLLM colocate mode is requested
- **WHEN** the user passes `--use-vllm --vllm-mode colocate`
- **THEN** the GRPO configuration SHALL receive supported vLLM fields for the installed TRL version and the run summary SHALL record that vLLM was requested in colocate mode

#### Scenario: vLLM server mode is requested
- **WHEN** the user passes `--use-vllm --vllm-mode server --vllm-server-host <host> --vllm-server-port <port>`
- **THEN** the GRPO configuration SHALL receive supported server connection fields and the run summary SHALL record the server host and port

#### Scenario: vLLM is not requested
- **WHEN** the user does not pass `--use-vllm`
- **THEN** the training command SHALL keep the existing non-vLLM GRPO path available for local smoke runs and ordinary single-device training

### Requirement: SFT Warm-Start Entry Point
The system SHALL provide an optional GSM8K SFT warm-start entrypoint that trains a LoRA adapter using GSM8K supervised solutions formatted in the same XML answer style used by RLVR.

#### Scenario: SFT dry-run validates data formatting
- **WHEN** the SFT entrypoint is invoked with `--dry-run`
- **THEN** it SHALL load GSM8K examples, render supervised prompt/completion records, validate non-empty reasoning and final answers, and exit without loading model weights or saving an adapter

#### Scenario: SFT training saves adapter artifacts
- **WHEN** the SFT entrypoint completes a real training run
- **THEN** it SHALL save a LoRA adapter directory, an SFT config summary, and train metrics under `outputs/stage-7-full-gsm8k-training/sft` or the user-provided output directory

#### Scenario: SFT uses existing dataset cache behavior
- **WHEN** the SFT entrypoint loads GSM8K
- **THEN** it SHALL use the project-local dataset cache by default and require an explicit download flag before fetching missing data

### Requirement: GRPO Can Start From Existing Adapter
The GRPO training entrypoint SHALL support starting from an existing LoRA adapter while preserving trainer checkpoint resume semantics.

#### Scenario: Initial adapter is provided
- **WHEN** the user passes `--initial-adapter-path <adapter>`
- **THEN** GRPO training SHALL initialize from the base model plus that trainable adapter and save the resulting adapter to the configured stage 7 output directory

#### Scenario: Trainer checkpoint is provided
- **WHEN** the user passes `--resume-from-checkpoint <checkpoint>`
- **THEN** GRPO training SHALL pass that checkpoint path to the trainer resume mechanism and record it in the run summary

#### Scenario: No initial adapter is provided
- **WHEN** the user omits `--initial-adapter-path`
- **THEN** GRPO training SHALL preserve the existing behavior of creating a fresh LoRA adapter from the configured LoRA parameters

### Requirement: Full Training Evaluation Artifacts
The system SHALL support independent evaluation of stage 7 full-training outputs using the existing base vs adapter comparison workflow.

#### Scenario: Stage 7 adapter is evaluated
- **WHEN** `scripts/run/gsm8k_eval.py` is invoked with a stage 7 adapter path and `--mode both`
- **THEN** it SHALL generate base cases, adapter cases, comparison summary, failure reports, and experiment summary under the requested stage 7 evaluation directory

#### Scenario: Training metadata is attached
- **WHEN** the evaluation command receives `--train-run-config` and `--train-metrics`
- **THEN** the experiment summary SHALL include those training metadata files alongside base and adapter metrics

#### Scenario: Evaluation remains configurable
- **WHEN** the user passes `--limit`, `--k`, or `--max-new-tokens`
- **THEN** the evaluation command SHALL use those values and record the requested sample count and pass@k setting in its summaries

### Requirement: Documentation Separates Local And Cloud Workflows
The project documentation SHALL describe local validation and cloud full-training workflows as separate paths.

#### Scenario: User wants to validate locally
- **WHEN** the user reads the stage 7 documentation
- **THEN** it SHALL provide commands for local dry-run, SFT dry-run, and small GRPO verification that do not require vLLM or CUDA

#### Scenario: User wants to train full GSM8K remotely
- **WHEN** the user reads the stage 7 documentation
- **THEN** it SHALL provide cloud CUDA setup guidance, optional vLLM installation guidance, full training commands, SFT warm-start commands, resume commands, and evaluation commands

#### Scenario: Agent continues the project later
- **WHEN** an agent reads `AGENTS.md`
- **THEN** it SHALL see that stage 7 permits cloud CUDA and vLLM while stage 6 remains the last validated local GSM8K experiment
