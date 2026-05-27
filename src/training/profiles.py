"""Training profiles and defaults for local RLVR stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = Path("outputs/stage-5-real-rlvr")
DEFAULT_STAGE4_SMOKE_OUTPUT_DIR = Path("outputs/stage-4-dual-reward-smoke-test")
DEFAULT_STAGE7_FULL_OUTPUT_DIR = Path("outputs/stage-7-full-gsm8k-training/train")
DEFAULT_TRAIN_LIMIT = 200
DEFAULT_MAX_STEPS = 100
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 8
DEFAULT_MAX_COMPLETION_LENGTH = 48
DEFAULT_LOGGING_STEPS = 10
DEFAULT_SAVE_STEPS = 100
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = ("q_proj", "v_proj")
DEFAULT_TRAINING_PROFILE = "stage5-real-rlvr"
STAGE7_FULL_GSM8K_TRAIN_LIMIT = 7473

RUNPOD_L40S_CLOUD_TARGET: dict[str, Any] = {
    "provider": "RunPod",
    "pod_type": "on-demand Pod",
    "cloud_preference": "Secure Cloud preferred; Community Cloud is acceptable for budget runs.",
    "gpu": "NVIDIA L40S",
    "gpu_memory_gb": 48,
    "gpu_count": 1,
    "server_mode_gpu_count": 2,
    "system_memory_gb": 64,
    "vcpus": 16,
    "disk_gb": 200,
    "os": "Ubuntu 22.04 or 24.04 with NVIDIA driver and CUDA 12.x",
    "container": "PyTorch/CUDA image",
    "recommended_vllm_mode": "colocate",
    "server_mode_note": "Use GPU 0 for GRPO training and GPU 1 for trl vllm-serve when renting 2 GPUs.",
}

MERLIN_A10_CLOUD_TARGET: dict[str, Any] = {
    "provider": "Merlin Devbox Worker",
    "resource_type": "workspace",
    "gpu": "NVIDIA A10",
    "gpu_memory_gb": 24,
    "gpu_count": 1,
    "system_memory_gb": 124,
    "cpu_cores": 31,
    "runtime_note": "Validated on a Merlin devbox worker with a project-local virtualenv and no vLLM.",
}

STAGE4_SMOKE_CASES: tuple[tuple[str, str], ...] = (
    ("What is 6 multiplied by 7?", "42"),
    ("A basket has 8 apples and then gets 5 more apples. How many apples are in the basket now?", "13"),
    ("A pack has 3 pencils and costs 2 dollars. How many dollars do 6 pencils cost?", "4"),
)


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    description: str
    output_dir: Path
    train_limit: int
    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    steps_per_generation: int
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    learning_rate: float
    correctness_reward_weight: float
    format_reward_weight: float
    beta: float
    temperature: float
    top_p: float
    top_k: int
    logging_steps: int
    save_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    cloud_target: dict[str, Any] | None = None

    def to_summary(self) -> dict[str, Any]:
        summary = asdict(self)
        summary["output_dir"] = str(self.output_dir)
        summary["lora_target_modules"] = list(self.lora_target_modules)
        return summary


TRAINING_PROFILES: dict[str, TrainingProfile] = {
    DEFAULT_TRAINING_PROFILE: TrainingProfile(
        name=DEFAULT_TRAINING_PROFILE,
        description="Stage 5 real RLVR baseline tuned for local MPS throughput.",
        output_dir=DEFAULT_OUTPUT_DIR,
        train_limit=DEFAULT_TRAIN_LIMIT,
        max_steps=DEFAULT_MAX_STEPS,
        per_device_train_batch_size=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        steps_per_generation=2,
        num_generations=2,
        max_prompt_length=512,
        max_completion_length=DEFAULT_MAX_COMPLETION_LENGTH,
        learning_rate=1e-6,
        correctness_reward_weight=1.0,
        format_reward_weight=1.0,
        beta=0.0,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        logging_steps=DEFAULT_LOGGING_STEPS,
        save_steps=DEFAULT_SAVE_STEPS,
        lora_r=DEFAULT_LORA_R,
        lora_alpha=DEFAULT_LORA_ALPHA,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
    ),
    "stage4-dual-reward-smoke-test": TrainingProfile(
        name="stage4-dual-reward-smoke-test",
        description="Stage 4 smoke test for verifying both correctness and format rewards online.",
        output_dir=DEFAULT_STAGE4_SMOKE_OUTPUT_DIR,
        train_limit=len(STAGE4_SMOKE_CASES),
        max_steps=30,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        steps_per_generation=2,
        num_generations=2,
        max_prompt_length=256,
        max_completion_length=96,
        learning_rate=1e-6,
        correctness_reward_weight=1.0,
        format_reward_weight=1.0,
        beta=0.0,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        logging_steps=5,
        save_steps=30,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
    ),
    "stage6-gsm8k-experiment": TrainingProfile(
        name="stage6-gsm8k-experiment",
        description="Stage 6 complete GSM8K experiment profile aligned with the best verified local run.",
        output_dir=Path("outputs/stage-6-gsm8k-experiment/train"),
        train_limit=2000,
        max_steps=300,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        steps_per_generation=4,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=192,
        learning_rate=5e-7,
        correctness_reward_weight=1.0,
        format_reward_weight=0.0,
        beta=0.02,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        logging_steps=25,
        save_steps=120,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
    ),
    "stage7-full-gsm8k-training": TrainingProfile(
        name="stage7-full-gsm8k-training",
        description="Stage 7 full GSM8K GRPO profile tuned for a RunPod L40S 48GB CUDA/vLLM run.",
        output_dir=DEFAULT_STAGE7_FULL_OUTPUT_DIR,
        train_limit=STAGE7_FULL_GSM8K_TRAIN_LIMIT,
        max_steps=1200,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        steps_per_generation=4,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=192,
        learning_rate=5e-7,
        correctness_reward_weight=1.0,
        format_reward_weight=0.0,
        beta=0.02,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        logging_steps=25,
        save_steps=200,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
        cloud_target=RUNPOD_L40S_CLOUD_TARGET,
    ),
    "stage7-merlin-a10-qwen15b-no-vllm": TrainingProfile(
        name="stage7-merlin-a10-qwen15b-no-vllm",
        description="Stage 7 Merlin devbox A10 profile validated with Qwen2.5-1.5B-Instruct and no vLLM.",
        output_dir=Path("outputs/stage-7-merlin-a10-qwen15b/train"),
        train_limit=STAGE7_FULL_GSM8K_TRAIN_LIMIT,
        max_steps=400,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        steps_per_generation=4,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=128,
        learning_rate=5e-7,
        correctness_reward_weight=1.0,
        format_reward_weight=0.1,
        beta=0.02,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        logging_steps=10,
        save_steps=100,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        cloud_target=MERLIN_A10_CLOUD_TARGET,
    ),
}


def get_training_profile(profile_name: str) -> TrainingProfile:
    try:
        return TRAINING_PROFILES[profile_name]
    except KeyError as error:
        known_profiles = ", ".join(sorted(TRAINING_PROFILES))
        raise ValueError(f"unknown training profile '{profile_name}'. Known profiles: {known_profiles}") from error
