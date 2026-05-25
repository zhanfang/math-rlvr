## ADDED Requirements

### Requirement: Single-machine scope
The workflow SHALL target a single local machine for the initial learning version and MUST NOT require distributed training infrastructure.

#### Scenario: Learner starts the project locally
- **WHEN** the learner follows the initial setup path
- **THEN** the workflow uses only local commands and local project files
- **AND** it does not require Ray, DeepSpeed, FSDP, verl, OpenRLHF, or a vLLM server

#### Scenario: Future scaling topics are referenced
- **WHEN** the roadmap mentions vLLM, verl, OpenRLHF, or multi-GPU training
- **THEN** those topics are clearly marked as later milestones outside the first implementation scope

### Requirement: Stage-based learning progression
The workflow SHALL divide Math RLVR work into small, ordered stages that can be completed and verified independently.

#### Scenario: Stage 1 is selected
- **WHEN** the learner begins implementation
- **THEN** the active stage is environment installation and validation
- **AND** training, evaluation, reward design, dataset loading, and model download remain later stages

#### Scenario: A stage is completed
- **WHEN** a stage finishes successfully
- **THEN** the next stage has a clear entry condition and a small concrete deliverable

### Requirement: Reproducible environment setup
The workflow SHALL provide a reproducible Python environment setup for the single-machine learning version.

#### Scenario: Environment files are added
- **WHEN** the first implementation task is applied
- **THEN** the repository includes environment setup instructions or files for Python 3.10 or newer
- **AND** the setup includes the core learning dependencies for PyTorch, Transformers, Datasets, Accelerate, PEFT, and TRL

#### Scenario: Optional dependencies are deferred
- **WHEN** dependencies for Math-Verify, vLLM, verl, OpenRLHF, or experiment tracking are considered
- **THEN** they are marked optional or later-stage unless needed by the current stage

### Requirement: Environment smoke validation
The workflow SHALL include a local smoke check that verifies the installed environment before RLVR code is implemented.

#### Scenario: Smoke check runs successfully
- **WHEN** the learner runs the environment validation command
- **THEN** it reports Python, PyTorch, Transformers, Datasets, PEFT, and TRL import status
- **AND** it reports available acceleration backends such as CPU, CUDA, or MPS

#### Scenario: Smoke check fails
- **WHEN** a required package cannot be imported
- **THEN** the validation output identifies the missing package
- **AND** the learner can fix setup before moving to dataset loading or training

### Requirement: Roadmap remains visible
The workflow SHALL keep the broader RLVR destination visible without making all future work part of the first implementation.

#### Scenario: Learner reads the plan
- **WHEN** the learner opens the project plan
- **THEN** they see the final destination of a Math RLVR loop
- **AND** they can distinguish the immediate environment setup task from later GRPO, reward, evaluation, and scaling milestones
