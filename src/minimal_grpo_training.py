"""Helpers for preparing and scoring single-machine RLVR training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.gsm8k_dataset import (
    DEFAULT_DATA_CACHE_DIR,
    GSM8K_CONFIG_NAME,
    GSM8K_DATASET_NAME,
    gsm8k_example_from_record,
)
from src.answer_rewards import score_answer_correctness, score_answer_format


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = Path("outputs/stage-5-real-rlvr")
DEFAULT_TRAIN_LIMIT = 200
DEFAULT_MAX_STEPS = 100
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 8
DEFAULT_MAX_COMPLETION_LENGTH = 48
DEFAULT_LOGGING_STEPS = 10
DEFAULT_SAVE_STEPS = 100


PROMPT_TEMPLATE = """Solve this grade-school math problem.

Return exactly two XML-style sections:
<reasoning>your concise reasoning</reasoning><answer>final answer only</answer>

Problem:
{question}
"""

def build_grpo_training_prompt(question: str) -> str:
    """Build a prompt that teaches the stage 3 answer format."""
    return PROMPT_TEMPLATE.format(question=question.strip())


def load_grpo_training_examples(
    split: str = "train",
    train_limit: int = DEFAULT_TRAIN_LIMIT,
    cache_dir: str | Path | None = DEFAULT_DATA_CACHE_DIR,
    local_files_only: bool = True,
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

    training_examples = []
    for row in dataset:
        example = gsm8k_example_from_record(row)
        training_examples.append(
            {
                "prompt": build_grpo_training_prompt(example.question),
                "answer": example.final_answer,
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


def normalize_completion_text(completion: Any) -> str:
    """Normalize TRL completion objects into plain text for reward helpers."""
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        content = completion.get("content")
        return "" if content is None else str(content)

    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content")
                if content is not None:
                    parts.append(str(content))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(completion)


def broadcast_expected_answers(answer: Any, batch_size: int) -> list[str]:
    if answer is None:
        return [""] * batch_size
    if isinstance(answer, str):
        return [answer] * batch_size
    if isinstance(answer, list):
        if len(answer) == batch_size:
            return ["" if item is None else str(item) for item in answer]
        if len(answer) == 1:
            return ["" if answer[0] is None else str(answer[0])] * batch_size
    return [str(answer)] * batch_size


def score_grpo_correctness_rewards(
    prompts: list[Any],
    completions: list[Any],
    answer: Any = None,
    **_: Any,
) -> list[float]:
    """TRL reward adapter for answer correctness."""
    del prompts
    answers = broadcast_expected_answers(answer, len(completions))
    return [
        score_answer_correctness(normalize_completion_text(completion), expected_answer)
        for completion, expected_answer in zip(completions, answers, strict=True)
    ]


def score_grpo_format_rewards(
    prompts: list[Any],
    completions: list[Any],
    **_: Any,
) -> list[float]:
    """TRL reward adapter for stage 3 XML-style output format."""
    del prompts
    return [score_answer_format(normalize_completion_text(completion)) for completion in completions]


def self_check_reward_adapters() -> None:
    """Run a tiny local check for the TRL reward adapters."""
    prompts = ["prompt"] * 3
    completions = [
        "<reasoning>6 * 7 = 42.</reasoning><answer>42</answer>",
        "<reasoning>Guessing.</reasoning><answer>41</answer>",
        "No tags here",
    ]
    answers = ["42", "42", "42"]

    correctness = score_grpo_correctness_rewards(prompts=prompts, completions=completions, answer=answers)
    fmt = score_grpo_format_rewards(prompts=prompts, completions=completions)

    if correctness != [1.0, 0.0, 0.0]:
        raise ValueError(f"unexpected correctness rewards: {correctness}")
    if fmt != [1.0, 1.0, 0.0]:
        raise ValueError(f"unexpected format rewards: {fmt}")
