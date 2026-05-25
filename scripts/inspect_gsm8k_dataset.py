#!/usr/bin/env python
"""Inspect a small GSM8K subset without model generation or training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gsm8k_dataset import (
    DEFAULT_DATA_CACHE_DIR,
    GSM8K_CONFIG_NAME,
    GSM8K_DATASET_NAME,
    load_gsm8k_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 read-only GSM8K data inspection. No model weights, training, reward, or evaluation are used.",
    )
    parser.add_argument("--split", default="train", help="GSM8K split to inspect, such as train or test.")
    parser.add_argument("--limit", type=int, default=3, help="Number of examples to print.")
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / DEFAULT_DATA_CACHE_DIR),
        help="Dataset cache directory. Defaults to the project-local data/hf_datasets path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Stage 2 GSM8K data inspection")
    print("=" * 31)
    print(f"dataset: {GSM8K_DATASET_NAME}")
    print(f"config: {GSM8K_CONFIG_NAME}")
    print(f"split: {args.split}")
    print(f"limit: {args.limit}")
    print(f"cache_dir: {args.cache_dir}")
    print("mode: read-only data inspection; no model download, generation, training, reward, or scoring")

    field_names, examples = load_gsm8k_examples(
        split=args.split,
        limit=args.limit,
        cache_dir=args.cache_dir,
    )

    print(f"\nfields: {field_names}")
    print(f"examples_loaded: {len(examples)}")

    for idx, example in enumerate(examples, start=1):
        print(f"\n--- example {idx} ---")
        print("question:")
        print(example.question)
        print("\nraw_answer:")
        print(example.raw_answer)
        print("\nfinal_answer:")
        print(example.final_answer)

    print("\nInspection complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
