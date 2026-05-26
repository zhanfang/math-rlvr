#!/usr/bin/env python
"""Single-machine RLVR training entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.gsm8k_dataset import DEFAULT_DATA_CACHE_DIR
from src.training import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRAINING_PROFILE,
    TRAINING_PROFILES,
    build_failure_hint,
    build_grpo_config,
    build_lora_config,
    build_run_summary,
    collect_dependency_versions,
    get_training_profile,
    load_stage4_smoke_training_examples,
    load_grpo_training_examples,
    print_summary,
    score_grpo_correctness_rewards,
    score_grpo_format_rewards,
    self_check_reward_adapters,
    training_examples_to_dataset,
    validate_grpo_training_examples,
    write_run_summary_json,
    write_training_metrics_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-machine RLVR training. Use --dry-run before real training.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Check dependencies, data, rewards, and config only.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name or local model path.")
    parser.add_argument(
        "--experiment-profile",
        choices=sorted(TRAINING_PROFILES),
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile that provides recommended defaults.",
    )
    parser.add_argument("--split", default="train", help="GSM8K split to use.")
    parser.add_argument("--train-limit", type=int, help="Maximum GSM8K examples to use.")
    parser.add_argument("--max-steps", type=int, help="Maximum GRPO training steps.")
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
        default=None,
        help="Directory for training output and LoRA adapter.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Optional trainer checkpoint path to resume a previous run.",
    )
    parser.add_argument(
        "--initial-adapter-path",
        help="Optional existing LoRA adapter path to initialize trainable GRPO from, such as the stage 7 SFT adapter.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=None,
        help="Per-device batch size tuned for local single-machine training.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument(
        "--steps-per-generation",
        type=int,
        default=None,
        help="Generation refresh interval; default keeps generation batch divisible by num generations.",
    )
    parser.add_argument("--num-generations", type=int, default=None, help="Completions sampled per prompt.")
    parser.add_argument("--max-prompt-length", type=int, default=None, help="Maximum prompt tokens.")
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=None,
        help="Maximum completion tokens.",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="GRPO learning rate.")
    parser.add_argument(
        "--correctness-reward-weight",
        type=float,
        default=None,
        help="GRPO weight for answer correctness reward.",
    )
    parser.add_argument(
        "--format-reward-weight",
        type=float,
        default=None,
        help="GRPO weight for XML format reward.",
    )
    parser.add_argument("--beta", type=float, default=None, help="KL coefficient for GRPO reference regularization.")
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature used during GRPO rollout.")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top-p used during GRPO rollout.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling used during GRPO rollout.")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for GRPO rollout generation when supported.")
    parser.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default=None,
        help="vLLM integration mode. RunPod L40S single-GPU runs usually start with colocate.",
    )
    parser.add_argument("--vllm-server-base-url", default=None, help="Optional vLLM server base URL.")
    parser.add_argument("--vllm-server-host", default=None, help="vLLM server host for server mode.")
    parser.add_argument("--vllm-server-port", type=int, default=None, help="vLLM server port for server mode.")
    parser.add_argument("--vllm-server-timeout", type=float, default=None, help="vLLM server request timeout.")
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help="Fraction of generation GPU memory reserved by vLLM.",
    )
    parser.add_argument("--vllm-max-model-length", type=int, default=None, help="Maximum vLLM model length.")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=None, help="vLLM tensor parallel size.")
    parser.add_argument("--vllm-enable-sleep-mode", action="store_true", help="Enable vLLM sleep mode if supported.")
    parser.add_argument("--vllm-group-port", type=int, default=None, help="Optional vLLM group port.")
    parser.add_argument("--logging-steps", type=int, default=None, help="Training log interval.")
    parser.add_argument("--save-steps", type=int, default=None, help="Trainer save interval.")
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU mode for GRPOConfig.")
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout.")
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help="Comma-separated LoRA target modules. Defaults work for Qwen-style models.",
    )
    args = parser.parse_args()
    return apply_training_profile_defaults(args)


def apply_training_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    profile = get_training_profile(args.experiment_profile)
    args.training_profile = profile.name
    args.training_profile_description = profile.description

    defaults = {
        "train_limit": profile.train_limit,
        "max_steps": profile.max_steps,
        "per_device_train_batch_size": profile.per_device_train_batch_size,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
        "steps_per_generation": profile.steps_per_generation,
        "num_generations": profile.num_generations,
        "max_prompt_length": profile.max_prompt_length,
        "max_completion_length": profile.max_completion_length,
        "learning_rate": profile.learning_rate,
        "correctness_reward_weight": profile.correctness_reward_weight,
        "format_reward_weight": profile.format_reward_weight,
        "beta": profile.beta,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "top_k": profile.top_k,
        "logging_steps": profile.logging_steps,
        "save_steps": profile.save_steps,
        "lora_r": profile.lora_r,
        "lora_alpha": profile.lora_alpha,
        "lora_dropout": profile.lora_dropout,
        "lora_target_modules": ",".join(profile.lora_target_modules),
    }
    for field_name, default_value in defaults.items():
        if getattr(args, field_name) is None:
            setattr(args, field_name, default_value)

    if args.output_dir is None:
        args.output_dir = str(ROOT / profile.output_dir)
    if args.lora_target_modules is None:
        args.lora_target_modules = ",".join(DEFAULT_LORA_TARGET_MODULES)
    if args.lora_r is None:
        args.lora_r = DEFAULT_LORA_R
    if args.lora_alpha is None:
        args.lora_alpha = DEFAULT_LORA_ALPHA
    if args.lora_dropout is None:
        args.lora_dropout = DEFAULT_LORA_DROPOUT
    if args.vllm_mode is None:
        args.vllm_mode = "colocate"
    if args.vllm_server_host is None:
        args.vllm_server_host = "127.0.0.1"
    if args.vllm_server_port is None:
        args.vllm_server_port = 8000
    if args.vllm_server_timeout is None:
        args.vllm_server_timeout = 240.0
    if args.vllm_gpu_memory_utilization is None:
        args.vllm_gpu_memory_utilization = 0.35
    if args.vllm_max_model_length is None:
        args.vllm_max_model_length = 1024
    if args.vllm_tensor_parallel_size is None:
        args.vllm_tensor_parallel_size = 1
    return args


def load_training_examples(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.experiment_profile == "stage4-dual-reward-smoke-test":
        return load_stage4_smoke_training_examples(
            model_name=args.model_name,
            model_local_files_only=args.dry_run,
        )
    return load_grpo_training_examples(
        split=args.split,
        train_limit=args.train_limit,
        cache_dir=args.cache_dir,
        local_files_only=args.dry_run or not args.allow_dataset_download,
        model_name=args.model_name,
        model_local_files_only=args.dry_run,
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

def run_training(args: argparse.Namespace) -> int:
    print(f"{args.training_profile} training")
    print("=" * (len(args.training_profile) + 9))
    print("mode: real training; model weights may be downloaded if not cached")

    from trl import GRPOTrainer

    versions = collect_dependency_versions()
    training_examples = load_training_examples(args)
    validate_grpo_training_examples(training_examples)
    self_check_reward_adapters()

    training_args = build_grpo_config(args)
    lora_config = build_lora_config(args)
    model = args.model_name
    peft_config = lora_config
    train_dataset = training_examples_to_dataset(training_examples)
    output_dir = Path(args.output_dir)
    summary = build_run_summary(args, training_examples, versions)

    print_summary(summary)
    write_run_summary_json(output_dir, summary)

    if args.initial_adapter_path:
        adapter_path = Path(args.initial_adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"initial adapter path does not exist: {adapter_path}")

        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            is_trainable=True,
        )
        peft_config = None

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[score_grpo_correctness_rewards, score_grpo_format_rewards],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    train_output = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    metrics = dict(train_output.metrics)
    metrics["adapter_dir"] = str(adapter_dir)
    write_training_metrics_json(output_dir, metrics)
    print(f"\nSaved LoRA adapter to: {adapter_dir}")
    print(f"Saved training metrics to: {output_dir / 'train_metrics.json'}")
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
        print("\nTraining command failed.", file=sys.stderr)
        print(f"error_type: {type(error).__name__}", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        print(f"hint: {build_failure_hint(error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
