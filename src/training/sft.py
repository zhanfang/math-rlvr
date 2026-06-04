"""SFT warm-start data helpers for GSM8K."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_GSM8K_EQUATION_TAG_PATTERN = re.compile(r"<<[^>]*>>")

from src.gsm8k_dataset import (
    DEFAULT_DATA_CACHE_DIR,
    GSM8K_CONFIG_NAME,
    GSM8K_DATASET_NAME,
    extract_gsm8k_final_answer,
    gsm8k_example_from_record,
)
from src.training.profiles import DEFAULT_MODEL_NAME
from src.training.prompts import (
    CHAT_SYSTEM_PROMPT,
    build_grpo_chat_messages,
    build_grpo_training_prompt,
    maybe_load_prompt_tokenizer,
)


def extract_gsm8k_reasoning(answer: str) -> str:
    """Return the supervised reasoning text before GSM8K's final answer marker.

    Strips GSM8K's ``<<expr=value>>`` calculator-annotation tags so SFT does not
    teach the model to copy the short single-equation template that hijacks
    multi-step reasoning (see iter3 analysis).
    """
    if "####" in answer:
        reasoning = answer.rsplit("####", 1)[0]
    else:
        reasoning = answer
    reasoning = _GSM8K_EQUATION_TAG_PATTERN.sub("", reasoning)
    return reasoning.strip()


def build_sft_assistant_completion(reasoning: str, final_answer: str) -> str:
    """Render the supervised assistant response in the project XML format."""
    return (
        f"<reasoning>{reasoning.strip()}</reasoning>\n"
        f"<answer>{final_answer.strip()}</answer>"
    )


def render_sft_training_text(
    question: str,
    completion: str,
    tokenizer: Any | None = None,
) -> str:
    """Render a full chat transcript for SFT."""
    messages = build_grpo_chat_messages(question)
    messages.append({"role": "assistant", "content": completion})
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    return (
        "<|im_start|>system\n"
        f"{CHAT_SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{messages[1]['content']}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{completion}\n"
        "<|im_end|>\n"
    )


def load_sft_training_examples(
    split: str = "train",
    train_limit: int = 200,
    cache_dir: str | Path | None = DEFAULT_DATA_CACHE_DIR,
    local_files_only: bool = True,
    model_name: str = DEFAULT_MODEL_NAME,
    model_local_files_only: bool = True,
) -> list[dict[str, str]]:
    """Load GSM8K examples formatted for supervised warm-start training."""
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
        reasoning = extract_gsm8k_reasoning(example.raw_answer)
        final_answer = extract_gsm8k_final_answer(example.raw_answer)
        completion = build_sft_assistant_completion(reasoning, final_answer)
        training_examples.append(
            {
                "prompt": build_grpo_training_prompt(
                    example.question,
                    tokenizer=prompt_tokenizer,
                ),
                "completion": completion,
                "text": render_sft_training_text(
                    example.question,
                    completion,
                    tokenizer=prompt_tokenizer,
                ),
                "reasoning": reasoning,
                "answer": final_answer,
                "question": example.question,
            }
        )
    return training_examples


def sft_examples_to_dataset(training_examples: list[dict[str, str]]):
    """Convert SFT examples to a Hugging Face Dataset."""
    from datasets import Dataset

    return Dataset.from_list(training_examples)


def validate_sft_training_examples(training_examples: list[dict[str, str]]) -> None:
    """Validate SFT rows before any model weights are loaded."""
    if not training_examples:
        raise ValueError("no SFT training examples were loaded")

    for idx, training_example in enumerate(training_examples):
        prompt = training_example.get("prompt", "")
        completion = training_example.get("completion", "")
        text = training_example.get("text", "")
        reasoning = training_example.get("reasoning", "")
        answer = training_example.get("answer", "")
        if not prompt.strip():
            raise ValueError(f"SFT example {idx} has an empty prompt")
        if not reasoning.strip():
            raise ValueError(f"SFT example {idx} has an empty reasoning field")
        if not answer.strip():
            raise ValueError(f"SFT example {idx} has an empty answer field")
        if not completion.strip() or "<reasoning>" not in completion or "<answer>" not in completion:
            raise ValueError(f"SFT example {idx} completion is missing XML answer tags")
        if not text.strip() or completion not in text:
            raise ValueError(f"SFT example {idx} text does not contain the supervised completion")
