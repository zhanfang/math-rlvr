#!/usr/bin/env python
"""Independent held-out evaluation for the stage 6 GSM8K experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.gsm8k_dataset import DEFAULT_DATA_CACHE_DIR
from src.gsm8k_experiment import (
    DEFAULT_STAGE6_EVAL_LIMIT,
    DEFAULT_STAGE6_EVAL_SPLIT,
    DEFAULT_STAGE6_FAILURE_SAMPLE_LIMIT,
    DEFAULT_STAGE6_MAX_NEW_TOKENS,
    DEFAULT_STAGE6_OUTPUT_DIR,
    build_compare_summary,
    build_experiment_summary,
    build_failure_report,
    ensure_directory,
    evaluate_single_case,
    load_generation_stack,
    load_heldout_examples,
    read_optional_json,
    summarize_evaluation,
    write_json,
    write_jsonl,
)
from src.training import DEFAULT_MODEL_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent held-out GSM8K evaluation for base and/or adapter models.",
    )
    parser.add_argument("--base-model", default=DEFAULT_MODEL_NAME, help="Base model name or local model path.")
    parser.add_argument("--adapter-path", help="Optional LoRA adapter path for adapter evaluation.")
    parser.add_argument(
        "--mode",
        choices=("base", "adapter", "both"),
        default="both",
        help="Which model variant to evaluate.",
    )
    parser.add_argument("--split", default=DEFAULT_STAGE6_EVAL_SPLIT, help="Held-out GSM8K split to evaluate.")
    parser.add_argument("--limit", type=int, default=DEFAULT_STAGE6_EVAL_LIMIT, help="Maximum held-out examples.")
    parser.add_argument("--k", type=int, default=4, help="Number of completions to sample per prompt.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_STAGE6_MAX_NEW_TOKENS,
        help="Maximum new tokens per completion.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / DEFAULT_DATA_CACHE_DIR),
        help="Project-local Hugging Face dataset cache directory.",
    )
    parser.add_argument(
        "--allow-dataset-download",
        action="store_true",
        help="Allow downloading held-out GSM8K data if it is not already cached.",
    )
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU mode for evaluation.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / DEFAULT_STAGE6_OUTPUT_DIR / "eval"),
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--train-run-config",
        help="Optional path to a training run_config.json file for experiment-level summaries.",
    )
    parser.add_argument(
        "--train-metrics",
        help="Optional path to a train_metrics.json file for experiment-level summaries.",
    )
    parser.add_argument(
        "--failure-sample-limit",
        type=int,
        default=DEFAULT_STAGE6_FAILURE_SAMPLE_LIMIT,
        help="Representative samples to save per failure type.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mode in {"adapter", "both"} and not args.adapter_path:
        raise ValueError("--adapter-path is required when --mode is adapter or both")
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.failure_sample_limit <= 0:
        raise ValueError("--failure-sample-limit must be positive")


def print_model_summary(label: str, summary: dict[str, Any], failure_report: dict[str, Any]) -> None:
    print(f"\n{label} summary")
    print("=" * (len(label) + 8))
    print(f"samples={summary['samples']}")
    print(f"pass@1={summary['pass_at_1']:.4f}")
    print(f"pass@{summary['requested_k']}={summary['pass_at_k']:.4f}")
    print(f"structured@1={summary['structured_at_1']:.4f}")
    if failure_report["counts"]:
        print("failure_counts=" + ", ".join(f"{name}:{count}" for name, count in sorted(failure_report["counts"].items())))
    else:
        print("failure_counts=none")


def evaluate_model_variant(
    *,
    label: str,
    base_model: str,
    adapter_path: str | None,
    examples: list[Any],
    k: int,
    max_new_tokens: int,
    force_cpu: bool,
    variant_dir: Path,
    failure_sample_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    model, tokenizer, device = load_generation_stack(
        base_model=base_model,
        adapter_path=adapter_path,
        force_cpu=force_cpu,
    )
    print(f"\nEvaluating {label} on device={device}")

    rows = [
        evaluate_single_case(
            model=model,
            tokenizer=tokenizer,
            example=example,
            k=k,
            max_new_tokens=max_new_tokens,
            model_label=label,
        )
        for example in examples
    ]

    summary = summarize_evaluation(rows, requested_k=k)
    failure_report = build_failure_report(rows, max_samples_per_type=failure_sample_limit)
    write_jsonl(variant_dir / "cases.jsonl", rows)
    write_json(variant_dir / "summary.json", summary)
    write_json(variant_dir / "failure_report.json", failure_report)
    return rows, summary, failure_report


def main() -> int:
    args = parse_args()
    validate_args(args)

    print("Stage 6 complete GSM8K evaluation")
    print("=================================")

    output_dir = ensure_directory(args.output_dir)
    examples = load_heldout_examples(
        split=args.split,
        limit=args.limit,
        cache_dir=args.cache_dir,
        local_files_only=not args.allow_dataset_download,
    )
    print(f"Loaded {len(examples)} held-out examples from split={args.split}")

    base_rows = base_summary = base_failure_report = None
    adapter_rows = adapter_summary = adapter_failure_report = None

    if args.mode in {"base", "both"}:
        base_rows, base_summary, base_failure_report = evaluate_model_variant(
            label="base",
            base_model=args.base_model,
            adapter_path=None,
            examples=examples,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            force_cpu=args.use_cpu,
            variant_dir=ensure_directory(output_dir / "base"),
            failure_sample_limit=args.failure_sample_limit,
        )
        print_model_summary("base", base_summary, base_failure_report)

    if args.mode in {"adapter", "both"}:
        adapter_rows, adapter_summary, adapter_failure_report = evaluate_model_variant(
            label="adapter",
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            examples=examples,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            force_cpu=args.use_cpu,
            variant_dir=ensure_directory(output_dir / "adapter"),
            failure_sample_limit=args.failure_sample_limit,
        )
        print_model_summary("adapter", adapter_summary, adapter_failure_report)

    compare_summary = None
    output_paths: dict[str, Any] = {
        "output_dir": str(output_dir),
        "base_cases": str(output_dir / "base" / "cases.jsonl") if base_rows is not None else None,
        "adapter_cases": str(output_dir / "adapter" / "cases.jsonl") if adapter_rows is not None else None,
    }
    if base_rows is not None and adapter_rows is not None:
        compare_summary = build_compare_summary(base_rows, adapter_rows, requested_k=args.k)
        write_json(output_dir / "compare" / "summary.json", compare_summary)
        output_paths["compare_summary"] = str(output_dir / "compare" / "summary.json")
        print("\ncomparison summary")
        print("==================")
        print(f"delta_pass@1={compare_summary['delta_pass_at_1']:.4f}")
        print(f"delta_pass@{args.k}={compare_summary['delta_pass_at_k']:.4f}")
        print(
            "paired_counts="
            + ", ".join(
                f"{name}:{count}"
                for name, count in sorted(compare_summary["paired_counts"].items())
            )
        )

    experiment_summary = build_experiment_summary(
        train_config=read_optional_json(args.train_run_config),
        train_metrics=read_optional_json(args.train_metrics),
        base_summary=base_summary,
        adapter_summary=adapter_summary,
        compare_summary=compare_summary,
        base_failure_report=base_failure_report,
        adapter_failure_report=adapter_failure_report,
        output_paths=output_paths,
    )
    write_json(output_dir / "experiment_summary.json", experiment_summary)
    print(f"\nSaved experiment summary to: {output_dir / 'experiment_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
