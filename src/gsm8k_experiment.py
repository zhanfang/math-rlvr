"""Helpers for stage 6 complete GSM8K train/eval experiments."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.answer_rewards import (
    answers_match_numerically,
    extract_answer_candidate_from_completion,
    extract_answer_from_completion,
    has_structured_answer_format,
)
from src.gsm8k_dataset import (
    DEFAULT_DATA_CACHE_DIR,
    GSM8K_CONFIG_NAME,
    GSM8K_DATASET_NAME,
    gsm8k_example_from_record,
)
from src.training import build_grpo_training_prompt


DEFAULT_STAGE6_OUTPUT_DIR = Path("outputs/stage-6-gsm8k-experiment")
DEFAULT_STAGE6_EVAL_SPLIT = "test"
DEFAULT_STAGE6_EVAL_LIMIT = 200
DEFAULT_STAGE6_MAX_NEW_TOKENS = 192
DEFAULT_STAGE6_FAILURE_SAMPLE_LIMIT = 3

_NUMERIC_TOKEN_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


@dataclass(frozen=True)
class HeldoutExample:
    case_id: str
    question: str
    gold_answer: str


def load_heldout_examples(
    split: str = DEFAULT_STAGE6_EVAL_SPLIT,
    limit: int = DEFAULT_STAGE6_EVAL_LIMIT,
    cache_dir: str | Path | None = DEFAULT_DATA_CACHE_DIR,
    local_files_only: bool = True,
) -> list[HeldoutExample]:
    """Load held-out GSM8K examples for stage 6 evaluation."""
    if limit < 0:
        raise ValueError("limit must be non-negative")

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
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    else:
        dataset = dataset.select([])

    examples = []
    for index, row in enumerate(dataset):
        example = gsm8k_example_from_record(row)
        examples.append(
            HeldoutExample(
                case_id=f"{split}-{index:05d}",
                question=example.question,
                gold_answer=example.final_answer,
            )
        )
    return examples


def select_torch_device(force_cpu: bool = False) -> tuple[str, Any]:
    import torch

    if force_cpu:
        return "cpu", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def load_generation_stack(
    base_model: str,
    adapter_path: str | None = None,
    force_cpu: bool = False,
):
    """Load model + tokenizer for held-out evaluation."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = select_torch_device(force_cpu=force_cpu)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(torch.device(device))
    model.eval()
    return model, tokenizer, device


def build_eval_prompt(tokenizer: Any, question: str) -> str:
    """Build the shared chat-style prompt for held-out evaluation."""
    return build_grpo_training_prompt(question, tokenizer=tokenizer)


def generate_completions(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    k: int,
) -> list[str]:
    """Generate one or more completions for a GSM8K prompt."""
    import torch

    if k <= 0:
        raise ValueError("k must be positive")

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}

    do_sample = k > 1
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": k,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": 0.7,
                "top_p": 0.95,
            }
        )

    with torch.no_grad():
        outputs = model.generate(**encoded, **generation_kwargs)

    prompt_length = encoded["input_ids"].shape[-1]
    completions = []
    for output in outputs:
        completion_ids = output[prompt_length:]
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
    return completions


def evaluate_single_case(
    model: Any,
    tokenizer: Any,
    example: HeldoutExample,
    k: int,
    max_new_tokens: int,
    model_label: str,
) -> dict[str, Any]:
    """Evaluate one held-out example and return a serializable row."""
    prompt = build_eval_prompt(tokenizer, example.question)
    completions = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        k=k,
    )
    strict_predicted_answers = [extract_answer_from_completion(completion) for completion in completions]
    predicted_answers = [extract_answer_candidate_from_completion(completion) for completion in completions]
    structured = [has_structured_answer_format(completion) for completion in completions]
    correctness = [
        answers_match_numerically(predicted_answer, example.gold_answer) if predicted_answer else False
        for predicted_answer in predicted_answers
    ]
    return {
        "case_id": example.case_id,
        "model_label": model_label,
        "question": example.question,
        "gold_answer": example.gold_answer,
        "prompt": prompt,
        "k": k,
        "completions": completions,
        "strict_predicted_answers": strict_predicted_answers,
        "predicted_answers": predicted_answers,
        "structured": structured,
        "correctness": correctness,
        "pass_at_1": bool(correctness[0]),
        "pass_at_k": any(correctness),
    }


def summarize_evaluation(rows: list[dict[str, Any]], requested_k: int) -> dict[str, Any]:
    """Summarize held-out evaluation rows into pass metrics."""
    total = len(rows)
    if total == 0:
        return {
            "samples": 0,
            "requested_k": requested_k,
            "pass_at_1": 0.0,
            "pass_at_k": 0.0,
            "structured_at_1": 0.0,
        }

    pass_at_1 = sum(1 for row in rows if row["pass_at_1"]) / total
    pass_at_k = sum(1 for row in rows if row["pass_at_k"]) / total
    structured_at_1 = sum(1 for row in rows if row["structured"][0]) / total
    return {
        "samples": total,
        "requested_k": requested_k,
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "structured_at_1": structured_at_1,
    }


def classify_failure_case(row: dict[str, Any]) -> str:
    """Map an evaluation row to a simple failure type for manual inspection."""
    if row["pass_at_1"]:
        return "correct"

    primary_completion = row["completions"][0] if row["completions"] else ""
    primary_answer = row["predicted_answers"][0] if row["predicted_answers"] else ""

    if not has_structured_answer_format(primary_completion):
        return "bad_format"
    if not primary_answer:
        return "extractor_miss"
    if _NUMERIC_TOKEN_RE.search(primary_answer):
        return "arithmetic_error"
    return "question_misread"


def build_failure_report(
    rows: list[dict[str, Any]],
    max_samples_per_type: int = DEFAULT_STAGE6_FAILURE_SAMPLE_LIMIT,
) -> dict[str, Any]:
    """Aggregate failure counts and representative samples."""
    counts: dict[str, int] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        failure_type = classify_failure_case(row)
        if failure_type == "correct":
            continue
        counts[failure_type] = counts.get(failure_type, 0) + 1
        samples.setdefault(failure_type, [])
        if len(samples[failure_type]) < max_samples_per_type:
            samples[failure_type].append(
                {
                    "case_id": row["case_id"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "primary_completion": row["completions"][0] if row["completions"] else "",
                    "predicted_answer": row["predicted_answers"][0] if row["predicted_answers"] else "",
                    "structured": row["structured"][0] if row["structured"] else False,
                    "pass_at_1": row["pass_at_1"],
                    "pass_at_k": row["pass_at_k"],
                }
            )
    return {"counts": counts, "samples": samples}


def build_compare_summary(
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
    requested_k: int,
) -> dict[str, Any]:
    """Compare base and adapter results on aligned held-out rows."""
    if len(base_rows) != len(adapter_rows):
        raise ValueError("base_rows and adapter_rows must have the same length")

    paired_counts = {
        "adapter_only_win": 0,
        "base_only_win": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }
    for base_row, adapter_row in zip(base_rows, adapter_rows, strict=True):
        if base_row["case_id"] != adapter_row["case_id"]:
            raise ValueError("base_rows and adapter_rows must be aligned by case_id")
        if adapter_row["pass_at_k"] and not base_row["pass_at_k"]:
            paired_counts["adapter_only_win"] += 1
        elif base_row["pass_at_k"] and not adapter_row["pass_at_k"]:
            paired_counts["base_only_win"] += 1
        elif base_row["pass_at_k"] and adapter_row["pass_at_k"]:
            paired_counts["both_correct"] += 1
        else:
            paired_counts["both_wrong"] += 1

    base_summary = summarize_evaluation(base_rows, requested_k=requested_k)
    adapter_summary = summarize_evaluation(adapter_rows, requested_k=requested_k)
    return {
        "requested_k": requested_k,
        "base": base_summary,
        "adapter": adapter_summary,
        "delta_pass_at_1": adapter_summary["pass_at_1"] - base_summary["pass_at_1"],
        "delta_pass_at_k": adapter_summary["pass_at_k"] - base_summary["pass_at_k"],
        "paired_counts": paired_counts,
    }


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_experiment_summary(
    *,
    train_config: dict[str, Any] | None,
    train_metrics: dict[str, Any] | None,
    base_summary: dict[str, Any] | None,
    adapter_summary: dict[str, Any] | None,
    compare_summary: dict[str, Any] | None,
    base_failure_report: dict[str, Any] | None,
    adapter_failure_report: dict[str, Any] | None,
    output_paths: dict[str, Any],
) -> dict[str, Any]:
    """Build a single stage-6 experiment summary payload."""
    return {
        "experiment_profile": "stage-6-complete-gsm8k-experiment",
        "train_config": train_config,
        "train_metrics": train_metrics,
        "base_summary": base_summary,
        "adapter_summary": adapter_summary,
        "compare_summary": compare_summary,
        "base_failure_report": base_failure_report,
        "adapter_failure_report": adapter_failure_report,
        "output_paths": output_paths,
    }
