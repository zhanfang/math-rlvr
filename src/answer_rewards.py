"""Stage 3 reward helpers for simple Math RLVR experiments."""

from __future__ import annotations

import re
from fractions import Fraction


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
_NUMERIC_TOKEN_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_BOXED_ANSWER_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER_PHRASE_RE = re.compile(
    r"(?is)(?:therefore|final answer|answer|total|profit|makes|takes|is|=)[^\n]{0,120}?"
    r"(-?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?)"
)


def normalize_answer_text(answer: str) -> str:
    """Clean an answer string before reward comparison."""
    return answer.strip().replace(",", "").replace("$", "").rstrip(".")


def extract_answer_from_completion(completion: str) -> str:
    """Extract the final answer from an XML-like model completion."""
    match = _ANSWER_RE.search(completion)
    if not match:
        return ""
    return match.group(1).strip()


def extract_answer_candidate_from_completion(completion: str) -> str:
    """Extract an answer candidate from structured or plain GSM8K-style text."""
    structured_answer = extract_answer_from_completion(completion)
    if structured_answer:
        return structured_answer

    boxed_matches = _BOXED_ANSWER_RE.findall(completion)
    if boxed_matches:
        return boxed_matches[-1].strip()

    phrase_matches = [match.group(1) for match in _ANSWER_PHRASE_RE.finditer(completion)]
    if phrase_matches:
        return phrase_matches[-1].strip()

    numeric_tokens = _NUMERIC_TOKEN_RE.findall(completion)
    if numeric_tokens:
        return numeric_tokens[-1].strip()

    return ""


def has_structured_answer_format(completion: str) -> bool:
    """Return whether completion has non-empty reasoning and answer blocks."""
    reasoning_match = _REASONING_RE.search(completion)
    answer_match = _ANSWER_RE.search(completion)
    if not reasoning_match or not answer_match:
        return False
    return bool(reasoning_match.group(1).strip()) and bool(answer_match.group(1).strip())


def _parse_basic_number(answer: str) -> Fraction | None:
    cleaned = normalize_answer_text(answer)
    if not cleaned:
        return None

    try:
        return Fraction(cleaned)
    except ValueError:
        numeric_tokens = _NUMERIC_TOKEN_RE.findall(cleaned)
        if not numeric_tokens:
            return None
        try:
            return Fraction(numeric_tokens[-1].replace(",", "").replace("$", ""))
        except ValueError:
            return None


def answers_match_numerically(left: str, right: str) -> bool:
    """Compare answers with lightweight numeric equivalence.

    Integers, decimals, fractions, and comma-grouped numbers are supported.
    If either side cannot be parsed as a basic number, comparison falls back
    to exact equality after the same simple cleaning.
    """
    left_number = _parse_basic_number(left)
    right_number = _parse_basic_number(right)

    if left_number is not None and right_number is not None:
        return left_number == right_number

    return normalize_answer_text(left) == normalize_answer_text(right)


def score_answer_correctness(completion: str, expected_answer: str) -> float:
    """Return 1.0 when the extracted model answer matches expected_answer."""
    model_answer = extract_answer_from_completion(completion)
    if not model_answer:
        return 0.0
    return 1.0 if answers_match_numerically(model_answer, expected_answer) else 0.0


def score_answer_candidate_correctness(completion: str, expected_answer: str) -> float:
    """Return 1.0 when a structured or plain-text answer candidate matches."""
    model_answer = extract_answer_candidate_from_completion(completion)
    if not model_answer:
        return 0.0
    return 1.0 if answers_match_numerically(model_answer, expected_answer) else 0.0


def score_answer_format(completion: str) -> float:
    """Return 1.0 when completion follows the stage 3 output format."""
    return 1.0 if has_structured_answer_format(completion) else 0.0
