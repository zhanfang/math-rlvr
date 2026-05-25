#!/usr/bin/env python
"""Single-machine RLVR training entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gsm8k_dataset import DEFAULT_DATA_CACHE_DIR
from src.minimal_grpo_training import (
    DEFAULT_LOGGING_STEPS,
    DEFAULT_MAX_COMPLETION_LENGTH,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    DEFAULT_SAVE_STEPS,
    DEFAULT_TRAIN_LIMIT,
    score_grpo_correctness_rewards,
    score_grpo_format_rewards,
    load_grpo_training_examples,
    training_examples_to_dataset,
    validate_grpo_training_examples,
    self_check_reward_adapters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-machine RLVR training. Use --dry-run before real training.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Check dependencies, data, rewards, and config only.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name or local model path.")
    parser.add_argument("--split", default="train", help="GSM8K split to use.")
    parser.add_argument("--train-limit", type=int, default=DEFAULT_TRAIN_LIMIT, help="Maximum GSM8K examples to use.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum GRPO training steps.")
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / DEFAULT_DATA_CACHE_DIR),
        help="Project-local Hugging Face dataset cache directory.",
    )
    parser.add_argument(
        "--allow-dataset-download",
        action="store_true",
        help="Allow downloading GSM8K data if it is not already cached. Dry-run still stays local-only.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / DEFAULT_OUTPUT_DIR),
        help="Directory for training output and LoRA adapter.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
        help="Per-device batch size tuned for local single-machine training.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument(
        "--steps-per-generation",
        type=int,
        default=2,
        help="Generation refresh interval; default keeps generation batch divisible by num generations.",
    )
    parser.add_argument("--num-generations", type=int, default=2, help="Completions sampled per prompt.")
    parser.add_argument("--max-prompt-length", type=int, default=512, help="Maximum prompt tokens.")
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=DEFAULT_MAX_COMPLETION_LENGTH,
        help="Maximum completion tokens.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="GRPO learning rate.")
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS, help="Training log interval.")
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS, help="Trainer save interval.")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU mode for GRPOConfig.")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj",
        help="Comma-separated LoRA target modules. Defaults work for Qwen-style models.",
    )
    return parser.parse_args()


def collect_dependency_versions() -> dict[str, str | bool]:
    import datasets
    import peft
    import torch
    import transformers
    import trl

    return {
        "trl": trl.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }


def build_grpo_config(args: argparse.Namespace):
    from trl import GRPOConfig
    import inspect
    import torch

    auto_use_cpu = not torch.cuda.is_available() and not torch.backends.mps.is_available()

    config_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_generation": args.steps_per_generation,
        "num_generations": args.num_generations,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "report_to": "none",
        "use_cpu": args.use_cpu or auto_use_cpu,
    }
    supported = inspect.signature(GRPOConfig).parameters
    return GRPOConfig(**{key: value for key, value in config_kwargs.items() if key in supported})


def build_lora_config(args: argparse.Namespace):
    from peft import LoraConfig, TaskType

    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )


def build_run_summary(
    args: argparse.Namespace,
    training_examples: list[dict[str, str]],
    versions: dict[str, Any],
) -> dict[str, Any]:
    effective_use_cpu = args.use_cpu or (
        versions.get("cuda_available") is False and versions.get("mps_available") is False
    )
    return {
        "training_profile": "stage-5-real-rlvr",
        "model_name": args.model_name,
        "split": args.split,
        "train_limit": args.train_limit,
        "training_examples_loaded": len(training_examples),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_generation": args.steps_per_generation,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "learning_rate": args.learning_rate,
        "use_cpu": effective_use_cpu,
        "auto_cpu_fallback": effective_use_cpu and not args.use_cpu,
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": [item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
        },
        "versions": versions,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\nConfiguration summary")
    print("=" * 21)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def load_training_examples(args: argparse.Namespace) -> list[dict[str, str]]:
    return load_grpo_training_examples(
        split=args.split,
        train_limit=args.train_limit,
        cache_dir=args.cache_dir,
        local_files_only=args.dry_run or not args.allow_dataset_download,
    )


def run_dry_run(args: argparse.Namespace) -> int:
    print("Single-machine RLVR dry-run")
    print("=" * 28)
    print("mode: no model loading, no model download, no training, no LoRA adapter save")

    versions = collect_dependency_versions()
    training_examples = load_training_examples(args)
    validate_grpo_training_examples(training_examples)
    self_check_reward_adapters()
    build_grpo_config(args)
    build_lora_config(args)

    print(f"\nLoaded {len(training_examples)} training examples.")
    print("\nFirst prompt preview:")
    print(training_examples[0]["prompt"][:600])
    print(f"\nFirst expected answer: {training_examples[0]['answer']}")
    print_summary(build_run_summary(args, training_examples, versions))
    print("\nDry-run passed.")
    return 0


def write_run_summary_json(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def run_training(args: argparse.Namespace) -> int:
    print("Stage 5 real RLVR training")
    print("=" * 26)
    print("mode: real training; model weights may be downloaded if not cached")

    from trl import GRPOTrainer

    versions = collect_dependency_versions()
    training_examples = load_training_examples(args)
    validate_grpo_training_examples(training_examples)
    self_check_reward_adapters()

    training_args = build_grpo_config(args)
    lora_config = build_lora_config(args)
    train_dataset = training_examples_to_dataset(training_examples)
    output_dir = Path(args.output_dir)
    summary = build_run_summary(args, training_examples, versions)

    print_summary(summary)
    write_run_summary_json(output_dir, summary)

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=[score_grpo_correctness_rewards, score_grpo_format_rewards],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )

    trainer.train()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    print(f"\nSaved LoRA adapter to: {adapter_dir}")
    return 0


def build_failure_hint(error: BaseException) -> str:
    text = str(error)
    lowered = text.lower()
    if "connection" in lowered or "network" in lowered or "not found" in lowered or "401" in lowered:
        return "模型或数据下载/访问可能失败。先运行 --dry-run；真实训练需要模型已缓存或网络可访问。"
    if "mps" in lowered or "cuda" in lowered or "out of memory" in lowered or "memory" in lowered:
        return "设备或内存可能不足。减小 --train-limit、--max-steps、--max-completion-length，或换更小模型。"
    if "dataset" in lowered or "gsm8k" in lowered or "local_files_only" in lowered:
        return "GSM8K 数据可能未初始化。先运行 scripts/cache_gsm8k_dataset.py。"
    if "grpo" in lowered or "trainer" in lowered or "num_generations" in lowered:
        return "TRL GRPO 配置可能不兼容。检查 batch size、steps-per-generation 和 num-generations 是否匹配。"
    return "先运行 --dry-run 区分准备问题和真实训练资源问题。"


def main() -> int:
    args = parse_args()

    try:
        if args.dry_run:
            return run_dry_run(args)
        return run_training(args)
    except KeyboardInterrupt:
        raise
    except Exception as error:
        print("\nStage 4 command failed.", file=sys.stderr)
        print(f"error_type: {type(error).__name__}", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        print(f"hint: {build_failure_hint(error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
