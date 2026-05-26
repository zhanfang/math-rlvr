"""Training-example loaders for RLVR stages."""

from __future__ import annotations

from pathlib import Path

from src.gsm8k_dataset import (
    DEFAULT_DATA_CACHE_DIR,
    GSM8K_CONFIG_NAME,
    GSM8K_DATASET_NAME,
    gsm8k_example_from_record,
)
from src.training.profiles import DEFAULT_MODEL_NAME, STAGE4_SMOKE_CASES
from src.training.prompts import build_grpo_training_prompt, maybe_load_prompt_tokenizer


def load_grpo_training_examples(
    split: str = "train",
    train_limit: int = 200,
    cache_dir: str | Path | None = DEFAULT_DATA_CACHE_DIR,
    local_files_only: bool = True,
    model_name: str = DEFAULT_MODEL_NAME,
    model_local_files_only: bool = True,
) -> list[dict[str, str]]:
    """Load a tiny GSM8K subset as GRPO training examples."""
    if train_limit < 0:
        raise ValueError("train_limit must be non-negative")

    from datasets import DownloadConfig, load_dataset

    download_config = DownloadConfig(
        local_files_only=local_files_only,
        max_retries=0 if local_files_only else 1,
    )
    dataset = load_dataset(
        GSM8K_DATASET_NAME,
        GSM8K_CONFIG_NAME,
        split=split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        download_config=download_config,
    )

    if train_limit:
        dataset = dataset.select(range(min(train_limit, len(dataset))))
    else:
        dataset = dataset.select([])

    prompt_tokenizer = maybe_load_prompt_tokenizer(
        model_name=model_name,
        local_files_only=model_local_files_only,
    )
    training_examples = []
    for row in dataset:
        example = gsm8k_example_from_record(row)
        training_examples.append(
            {
                "prompt": build_grpo_training_prompt(
                    example.question,
                    tokenizer=prompt_tokenizer,
                ),
                "answer": example.final_answer,
            }
        )
    return training_examples


def load_stage4_smoke_training_examples(
    model_name: str = DEFAULT_MODEL_NAME,
    model_local_files_only: bool = True,
) -> list[dict[str, str]]:
    """Build a tiny deterministic training set for the stage 4 dual-reward smoke test."""
    prompt_tokenizer = maybe_load_prompt_tokenizer(
        model_name=model_name,
        local_files_only=model_local_files_only,
    )
    training_examples = []
    for question, answer in STAGE4_SMOKE_CASES:
        training_examples.append(
            {
                "prompt": build_grpo_training_prompt(question, tokenizer=prompt_tokenizer),
                "answer": answer,
            }
        )
    return training_examples


def training_examples_to_dataset(training_examples: list[dict[str, str]]):
    """Convert training examples to a Hugging Face Dataset for TRL."""
    from datasets import Dataset

    return Dataset.from_list(training_examples)


def validate_grpo_training_examples(training_examples: list[dict[str, str]]) -> None:
    """Validate the minimal fields expected by GRPOTrainer and rewards."""
    if not training_examples:
        raise ValueError("no GRPO training examples were loaded")

    for idx, training_example in enumerate(training_examples):
        prompt = training_example.get("prompt", "")
        answer = training_example.get("answer", "")
        if not prompt.strip():
            raise ValueError(f"training example {idx} has an empty prompt")
        if not answer.strip():
            raise ValueError(f"training example {idx} has an empty answer")
        if "<reasoning>" not in prompt or "<answer>" not in prompt:
            raise ValueError(
                f"training example {idx} prompt is missing the required XML-style output instructions"
            )
