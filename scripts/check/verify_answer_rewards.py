#!/usr/bin/env python
"""Offline checks for stage 3 reward functions."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.gsm8k_dataset import extract_gsm8k_final_answer
from src.answer_rewards import (
    score_answer_correctness,
    extract_answer_from_completion,
    score_answer_format,
    answers_match_numerically,
)


@dataclass(frozen=True)
class RewardCheckCase:
    name: str
    completion: str
    raw_gsm8k_answer: str
    expected_extracted_answer: str
    expected_correctness: float
    expected_format: float


REWARD_CHECK_CASES = [
    RewardCheckCase(
        name="correct integer",
        completion="<reasoning>6 * 7 = 42.</reasoning><answer>42</answer>",
        raw_gsm8k_answer="The answer is 42.\n#### 42",
        expected_extracted_answer="42",
        expected_correctness=1.0,
        expected_format=1.0,
    ),
    RewardCheckCase(
        name="wrong integer",
        completion="<reasoning>Guessing one less.</reasoning><answer>41</answer>",
        raw_gsm8k_answer="The answer is 42.\n#### 42",
        expected_extracted_answer="41",
        expected_correctness=0.0,
        expected_format=1.0,
    ),
    RewardCheckCase(
        name="fraction equals decimal",
        completion="<reasoning>One half is 0.5.</reasoning><answer>1/2</answer>",
        raw_gsm8k_answer="The answer is 0.5.\n#### 0.5",
        expected_extracted_answer="1/2",
        expected_correctness=1.0,
        expected_format=1.0,
    ),
    RewardCheckCase(
        name="comma number equals plain number",
        completion="<reasoning>Adding thousands.</reasoning><answer>1,000</answer>",
        raw_gsm8k_answer="The answer is one thousand.\n#### 1000",
        expected_extracted_answer="1,000",
        expected_correctness=1.0,
        expected_format=1.0,
    ),
    RewardCheckCase(
        name="missing answer tag",
        completion="<reasoning>The arithmetic gives 42.</reasoning>42",
        raw_gsm8k_answer="The answer is 42.\n#### 42",
        expected_extracted_answer="",
        expected_correctness=0.0,
        expected_format=0.0,
    ),
    RewardCheckCase(
        name="missing reasoning tag",
        completion="The arithmetic gives 42. <answer>42</answer>",
        raw_gsm8k_answer="The answer is 42.\n#### 42",
        expected_extracted_answer="42",
        expected_correctness=1.0,
        expected_format=0.0,
    ),
    RewardCheckCase(
        name="empty answer tag",
        completion="<reasoning>The arithmetic gives 42.</reasoning><answer>   </answer>",
        raw_gsm8k_answer="The answer is 42.\n#### 42",
        expected_extracted_answer="",
        expected_correctness=0.0,
        expected_format=0.0,
    ),
]


NUMERIC_EQUIVALENCE_CASES = [
    ("42", "42", True),
    ("1,000", "1000", True),
    ("1/2", "0.5", True),
    ("41", "42", False),
    ("final", "final", True),
    ("final", "answer", False),
]


def main() -> int:
    print("Stage 3 reward function offline check")
    print("=" * 37)
    print("mode: local hard-coded examples; no dataset download, model weights, generation, or training")

    failures = []

    print("\nNumeric equivalence cases")
    for left, right, expected in NUMERIC_EQUIVALENCE_CASES:
        actual = answers_match_numerically(left, right)
        status = "OK" if actual is expected else "FAIL"
        print(f"[{status}] {left!r} vs {right!r}: expected={expected} actual={actual}")
        if actual is not expected:
            failures.append((f"numeric:{left}:{right}", expected, actual))

    print("\nReward cases")
    for case in REWARD_CHECK_CASES:
        expected_answer = extract_gsm8k_final_answer(case.raw_gsm8k_answer)
        extracted_answer = extract_answer_from_completion(case.completion)
        correctness = score_answer_correctness(case.completion, expected_answer)
        fmt = score_answer_format(case.completion)

        ok = (
            extracted_answer == case.expected_extracted_answer
            and correctness == case.expected_correctness
            and fmt == case.expected_format
        )
        status = "OK" if ok else "FAIL"
        print(
            f"[{status}] {case.name}: "
            f"model_answer={extracted_answer!r} "
            f"expected_answer={expected_answer!r} "
            f"correctness_score={correctness:.1f} "
            f"format_score={fmt:.1f}"
        )
        if not ok:
            failures.append(
                (
                    case.name,
                    (case.expected_extracted_answer, case.expected_correctness, case.expected_format),
                    (extracted_answer, correctness, fmt),
                )
            )

    if failures:
        print("\nReward check failed.")
        return 1

    print("\nReward check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
