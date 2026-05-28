#!/usr/bin/env python
"""Initialize a project-local GSM8K dataset cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.gsm8k_dataset import (
    DEFAULT_DATA_CACHE_DIR,
    GSM8K_CONFIG_NAME,
    GSM8K_DATASET_NAME,
    gsm8k_example_from_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/cache GSM8K under the project data directory. No model weights are downloaded.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / DEFAULT_DATA_CACHE_DIR),
        help="Project-local Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="GSM8K splits to initialize.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the dataset to be already cached locally and avoid any network access.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from datasets import DownloadConfig, load_dataset

    print("Initialize project-local GSM8K cache")
    print("=" * 38)
    print(f"dataset: {GSM8K_DATASET_NAME}")
    print(f"config: {GSM8K_CONFIG_NAME}")
    print(f"cache_dir: {cache_dir}")
    print("mode: dataset cache only; no model weights, generation, training, reward, or scoring")
    print(f"local_files_only: {args.local_files_only}")

    download_config = DownloadConfig(
        local_files_only=args.local_files_only,
        max_retries=0 if args.local_files_only else 1,
    )

    for split in args.splits:
        dataset = load_dataset(
            GSM8K_DATASET_NAME,
            GSM8K_CONFIG_NAME,
            split=split,
            cache_dir=str(cache_dir),
            download_config=download_config,
        )
        print(f"\n[{split}] rows={len(dataset)} fields={list(dataset.column_names)}")

        if len(dataset):
            example = gsm8k_example_from_record(dataset[0])
            print(f"[{split}] first_final_answer={example.final_answer}")

        cache_files = dataset.cache_files
        for cache_file in cache_files:
            print(f"[{split}] cache_file={cache_file.get('filename')}")

    print("\nDataset cache initialized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
