"""Runtime helpers for local RLVR training entrypoints."""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path
from typing import Any

from src.training.profiles import get_training_profile


def collect_dependency_versions() -> dict[str, str | bool]:
    import datasets
    import peft
    import torch
    import transformers
    import trl

    versions = {
        "trl": trl.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    try:
        versions["vllm"] = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        versions["vllm"] = False
    return versions


def build_grpo_config(args: Any):
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
        "reward_weights": [args.correctness_reward_weight, args.format_reward_weight],
        "beta": args.beta,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.use_vllm:
        vllm_kwargs = {
            "use_vllm": True,
            "vllm_mode": args.vllm_mode,
            "vllm_server_base_url": args.vllm_server_base_url,
            "vllm_server_host": args.vllm_server_host,
            "vllm_server_port": args.vllm_server_port,
            "vllm_server_timeout": args.vllm_server_timeout,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_max_model_length": args.vllm_max_model_length,
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "vllm_enable_sleep_mode": args.vllm_enable_sleep_mode,
            "vllm_group_port": args.vllm_group_port,
        }
        config_kwargs.update({key: value for key, value in vllm_kwargs.items() if value is not None})
    else:
        config_kwargs["use_vllm"] = False
    supported = inspect.signature(GRPOConfig).parameters
    return GRPOConfig(**{key: value for key, value in config_kwargs.items() if key in supported})


def build_lora_config(args: Any):
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
    args: Any,
    training_examples: list[dict[str, str]],
    versions: dict[str, Any],
) -> dict[str, Any]:
    effective_use_cpu = args.use_cpu or (
        versions.get("cuda_available") is False and versions.get("mps_available") is False
    )
    profile = get_training_profile(args.training_profile)
    return {
        "training_profile": args.training_profile,
        "training_profile_description": args.training_profile_description,
        "profile_defaults": profile.to_summary(),
        "cloud_target": profile.cloud_target,
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
        "reward_weights": {
            "correctness": args.correctness_reward_weight,
            "format": args.format_reward_weight,
        },
        "beta": args.beta,
        "generation_sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "vllm": {
            "requested": args.use_vllm,
            "mode": args.vllm_mode,
            "server_base_url": args.vllm_server_base_url,
            "server_host": args.vllm_server_host,
            "server_port": args.vllm_server_port,
            "server_timeout": args.vllm_server_timeout,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "max_model_length": args.vllm_max_model_length,
            "tensor_parallel_size": args.vllm_tensor_parallel_size,
            "enable_sleep_mode": args.vllm_enable_sleep_mode,
            "group_port": args.vllm_group_port,
        },
        "use_cpu": effective_use_cpu,
        "auto_cpu_fallback": effective_use_cpu and not args.use_cpu,
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "initial_adapter_path": args.initial_adapter_path,
        "resume_from_checkpoint": args.resume_from_checkpoint,
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


def write_run_summary_json(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_training_metrics_json(output_dir: Path, metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_failure_hint(error: BaseException) -> str:
    text = str(error)
    lowered = text.lower()
    if "connection" in lowered or "network" in lowered or "not found" in lowered or "401" in lowered:
        return "模型或数据下载/访问可能失败。先运行 --dry-run；真实训练需要模型已缓存或网络可访问。"
    if "mps" in lowered or "cuda" in lowered or "out of memory" in lowered or "memory" in lowered:
        return "设备或内存可能不足。减小 --train-limit、--max-steps、--max-completion-length，或换更小模型。"
    if "vllm" in lowered or "server" in lowered or "port" in lowered:
        return "vLLM 配置可能不匹配。RunPod L40S 单卡优先试 --use-vllm --vllm-mode colocate；双卡 server 模式需先确认 trl vllm-serve 的 host/port 可访问。"
    if "dataset" in lowered or "gsm8k" in lowered or "local_files_only" in lowered:
        return "GSM8K 数据可能未初始化。先运行 scripts/data/cache_gsm8k_dataset.py。"
    if "grpo" in lowered or "trainer" in lowered or "num_generations" in lowered:
        return "TRL GRPO 配置可能不兼容。检查 batch size、steps-per-generation 和 num-generations 是否匹配。"
    return "先运行 --dry-run 区分准备问题和真实训练资源问题。"
