"""Stage 3 reward helpers for simple Math RLVR experiments."""

from __future__ import annotations

import re
from fractions import Fraction


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)


def clean_reward_answer(answer: str) -> str:
    """Clean an answer string before reward comparison."""
    return answer.strip().replace(",", "")


def extract_xml_answer(completion: str) -> str:
    """Extract the final answer from an XML-like model completion."""
    match = _ANSWER_RE.search(completion)
    if not match:
        return ""
    return match.group(1).strip()


def has_required_output_format(completion: str) -> bool:
    """Return whether completion has non-empty reasoning and answer blocks."""
    reasoning_match = _REASONING_RE.search(completion)
    answer_match = _ANSWER_RE.search(completion)
    if not reasoning_match or not answer_match:
        return False
    return bool(reasoning_match.group(1).strip()) and bool(answer_match.group(1).strip())


def _parse_basic_number(answer: str) -> Fraction | None:
    cleaned = clean_reward_answer(answer)
    if not cleaned:
        return None

    try:
        return Fraction(cleaned)
    except ValueError:
        return None


def numeric_equal(left: str, right: str) -> bool:
    """Compare answers with lightweight numeric equivalence.

    Integers, decimals, fractions, and comma-grouped numbers are supported.
    If either side cannot be parsed as a basic number, comparison falls back
    to exact equality after the same simple cleaning.
    """
    left_number = _parse_basic_number(left)
    right_number = _parse_basic_number(right)

    if left_number is not None and right_number is not None:
        return left_number == right_number

    return clean_reward_answer(left) == clean_reward_answer(right)


def correctness_reward(completion: str, expected_answer: str) -> float:
    """Return 1.0 when the extracted model answer matches expected_answer."""
    model_answer = extract_xml_answer(completion)
    if not model_answer:
        return 0.0
    return 1.0 if numeric_equal(model_answer, expected_answer) else 0.0


def format_reward(completion: str) -> float:
    """Return 1.0 when completion follows the stage 3 output format."""
    return 1.0 if has_required_output_format(completion) else 0.0
