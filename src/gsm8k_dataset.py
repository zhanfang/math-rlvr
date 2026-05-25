"""Utilities for inspecting GSM8K examples in stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


GSM8K_DATASET_NAME = "openai/gsm8k"
GSM8K_CONFIG_NAME = "main"
DEFAULT_DATA_CACHE_DIR = Path("data/hf_datasets")


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    raw_answer: str
    final_answer: str


def normalize_gsm8k_final_answer(answer: str) -> str:
    """Clean the final GSM8K answer string for simple inspection."""
    return answer.strip().replace(",", "")


def extract_gsm8k_final_answer(answer: str) -> str:
    """Extract the final answer from a GSM8K answer field.

    GSM8K answers usually end with a marker such as `#### 42`.
    Stage 2 intentionally keeps this simple: no numeric equivalence,
    symbolic math, or LaTeX parsing yet.
    """
    if "####" in answer:
        return normalize_gsm8k_final_answer(answer.rsplit("####", 1)[-1])
    return normalize_gsm8k_final_answer(answer)


def gsm8k_example_from_record(record: dict[str, Any]) -> GSM8KExample:
    raw_answer = str(record["answer"])
    return GSM8KExample(
        question=str(record["question"]),
        raw_answer=raw_answer,
        final_answer=extract_gsm8k_final_answer(raw_answer),
    )


def load_gsm8k_examples(
    split: str = "train",
    limit: int = 3,
    cache_dir: str | Path | None = None,
) -> tuple[list[str], list[GSM8KExample]]:
    """Load a small GSM8K subset for read-only data inspection."""
    if limit < 0:
        raise ValueError("limit must be non-negative")

    from datasets import load_dataset

    dataset = load_dataset(
        GSM8K_DATASET_NAME,
        GSM8K_CONFIG_NAME,
        split=split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    field_names = list(dataset.column_names)

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    else:
        dataset = dataset.select([])

    return field_names, [gsm8k_example_from_record(record) for record in dataset]
