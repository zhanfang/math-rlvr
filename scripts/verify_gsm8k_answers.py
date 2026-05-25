#!/usr/bin/env python
"""Offline checks for GSM8K final-answer extraction."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gsm8k_dataset import extract_gsm8k_final_answer


ANSWER_EXTRACTION_CASES = [
    ("We compute it step by step.\n#### 42", "42"),
    ("Final arithmetic gives a large number.\n#### 1,234", "1234"),
    ("Reasoning omitted.\n####   17   ", "17"),
    ("No marker here, just 8,000 ", "No marker here just 8000"),
]


def main() -> int:
    print("GSM8K answer extraction offline check")
    print("=" * 40)

    failures = []
    for raw, expected in ANSWER_EXTRACTION_CASES:
        actual = extract_gsm8k_final_answer(raw)
        status = "OK" if actual == expected else "FAIL"
        print(f"[{status}] expected={expected!r} actual={actual!r}")
        if actual != expected:
            failures.append((raw, expected, actual))

    if failures:
        print("\nExtraction check failed.")
        return 1

    print("\nExtraction check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
