"""TRL reward adapters shared by training entrypoints."""

from __future__ import annotations

from typing import Any

from src.answer_rewards import score_answer_candidate_correctness, score_answer_format


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
        score_answer_candidate_correctness(normalize_completion_text(completion), expected_answer)
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
