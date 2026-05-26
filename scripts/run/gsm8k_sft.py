#!/usr/bin/env python
"""Optional GSM8K SFT warm-start for stage 7."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.gsm8k_dataset import DEFAULT_DATA_CACHE_DIR
from src.training import (
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_MODEL_NAME,
    STAGE7_FULL_GSM8K_TRAIN_LIMIT,
    build_failure_hint,
    build_lora_config,
    collect_dependency_versions,
    load_sft_training_examples,
    print_summary,
    sft_examples_to_dataset,
    validate_sft_training_examples,
    write_training_metrics_json,
)


DEFAULT_STAGE7_SFT_OUTPUT_DIR = ROOT / "outputs/stage-7-full-gsm8k-training/sft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optional GSM8K supervised warm-start before stage 7 GRPO.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate data and config only.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name or local model path.")
    parser.add_argument("--split", default="train", help="GSM8K split to use.")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=STAGE7_FULL_GSM8K_TRAIN_LIMIT,
        help="Maximum GSM8K examples to use for SFT.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / DEFAULT_DATA_CACHE_DIR),
        help="Project-local Hugging Face dataset cache directory.",
    )
    parser.add_argument(
        "--allow-dataset-download",
        action="store_true",
        help="Allow downloading GSM8K data if it is not already cached. Dry-run stays local-only.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_STAGE7_SFT_OUTPUT_DIR), help="SFT output directory.")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum SFT training steps.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4, help="Per-device SFT batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--max-length", type=int, default=768, help="Maximum tokenized SFT sequence length.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="SFT learning rate.")
    parser.add_argument("--logging-steps", type=int, default=25, help="Training log interval.")
    parser.add_argument("--save-steps", type=int, default=150, help="Trainer save interval.")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU mode for SFTConfig.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT, help="LoRA dropout.")
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(DEFAULT_LORA_TARGET_MODULES),
        help="Comma-separated LoRA target modules.",
    )
    return parser.parse_args()


def build_sft_config(args: argparse.Namespace):
    from trl import SFTConfig
    import torch

    auto_use_cpu = not torch.cuda.is_available() and not torch.backends.mps.is_available()
    config_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "max_length": args.max_length,
        "dataset_text_field": "text",
        "packing": False,
        "report_to": "none",
        "use_cpu": args.use_cpu or auto_use_cpu,
    }
    supported = inspect.signature(SFTConfig).parameters
    return SFTConfig(**{key: value for key, value in config_kwargs.items() if key in supported})


def build_sft_summary(
    args: argparse.Namespace,
    training_examples: list[dict[str, str]],
    versions: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "stage7-full-gsm8k-training",
        "mode": "sft-warm-start",
        "model_name": args.model_name,
        "split": args.split,
        "train_limit": args.train_limit,
        "training_examples_loaded": len(training_examples),
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": [
                item.strip()
                for item in args.lora_target_modules.split(",")
                if item.strip()
            ],
        },
        "versions": versions,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_training_examples(args: argparse.Namespace) -> list[dict[str, str]]:
    return load_sft_training_examples(
        split=args.split,
        train_limit=args.train_limit,
        cache_dir=args.cache_dir,
        local_files_only=args.dry_run or not args.allow_dataset_download,
        model_name=args.model_name,
        model_local_files_only=args.dry_run,
    )


def run_dry_run(args: argparse.Namespace) -> int:
    print("Stage 7 GSM8K SFT dry-run")
    print("==========================")
    print("mode: no model loading, no model download, no training, no adapter save")

    versions = collect_dependency_versions()
    training_examples = load_training_examples(args)
    validate_sft_training_examples(training_examples)
    build_sft_config(args)
    build_lora_config(args)

    print(f"\nLoaded {len(training_examples)} SFT examples.")
    print("\nFirst supervised text preview:")
    print(training_examples[0]["text"][:900])
    print_summary(build_sft_summary(args, training_examples, versions))
    print("\nSFT dry-run passed.")
    return 0


def run_training(args: argparse.Namespace) -> int:
    print("Stage 7 GSM8K SFT warm-start")
    print("============================")
    print("mode: real SFT training; model weights may be downloaded if not cached")

    from transformers import AutoTokenizer
    from trl import SFTTrainer

    versions = collect_dependency_versions()
    training_examples = load_training_examples(args)
    validate_sft_training_examples(training_examples)

    output_dir = Path(args.output_dir)
    sft_args = build_sft_config(args)
    lora_config = build_lora_config(args)
    train_dataset = sft_examples_to_dataset(training_examples)
    summary = build_sft_summary(args, training_examples, versions)

    print_summary(summary)
    write_json(output_dir / "sft_config.json", summary)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = SFTTrainer(
        model=args.model_name,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    train_output = trainer.train()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    metrics = dict(train_output.metrics)
    metrics["adapter_dir"] = str(adapter_dir)
    write_training_metrics_json(output_dir, metrics)
    print(f"\nSaved SFT LoRA adapter to: {adapter_dir}")
    print(f"Saved SFT metrics to: {output_dir / 'train_metrics.json'}")
    return 0


def main() -> int:
    args = parse_args()

    try:
        if args.dry_run:
            return run_dry_run(args)
        return run_training(args)
    except KeyboardInterrupt:
        raise
    except Exception as error:
        print("\nSFT command failed.", file=sys.stderr)
        print(f"error_type: {type(error).__name__}", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        print(f"hint: {build_failure_hint(error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
