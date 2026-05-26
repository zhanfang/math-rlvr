#!/usr/bin/env python
"""Run and validate the stage 4 dual-reward smoke test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.training import DEFAULT_MODEL_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the stage 4 dual-reward smoke test and verify both rewards become positive.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config only; skip real training.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name or local model path.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs/stage-4-dual-reward-smoke-test"),
        help="Directory for smoke-test outputs.",
    )
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU mode.")
    return parser.parse_args()


def run_training_command(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run" / "minimal_grpo_training.py"),
        "--experiment-profile",
        "stage4-dual-reward-smoke-test",
        "--model-name",
        args.model_name,
        "--output-dir",
        args.output_dir,
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.use_cpu:
        command.append("--use-cpu")
    subprocess.run(command, cwd=ROOT, check=True)


def locate_trainer_state(output_dir: Path) -> Path:
    checkpoints = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
    if not checkpoints:
        raise FileNotFoundError(f"no trainer_state.json found under {output_dir}")
    return checkpoints[-1]


def verify_dual_rewards(output_dir: Path) -> dict[str, float]:
    trainer_state_path = locate_trainer_state(output_dir)
    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    best_correctness = 0.0
    best_format = 0.0

    for row in state.get("log_history", []):
        best_correctness = max(
            best_correctness,
            float(row.get("rewards/score_grpo_correctness_rewards/mean", 0.0)),
        )
        best_format = max(
            best_format,
            float(row.get("rewards/score_grpo_format_rewards/mean", 0.0)),
        )

    if best_correctness <= 0.0 or best_format <= 0.0:
        raise RuntimeError(
            "stage4 dual-reward smoke test did not observe positive values for both rewards: "
            f"correctness_max={best_correctness}, format_max={best_format}"
        )

    adapter_dir = output_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"expected adapter directory missing: {adapter_dir}")

    return {
        "best_correctness_reward_mean": best_correctness,
        "best_format_reward_mean": best_format,
        "trainer_state_path": str(trainer_state_path),
        "adapter_dir": str(adapter_dir),
    }


def main() -> int:
    args = parse_args()
    run_training_command(args)

    if args.dry_run:
        print("\nStage 4 dual-reward smoke dry-run passed.")
        return 0

    result = verify_dual_rewards(Path(args.output_dir))
    print("\nStage 4 dual-reward smoke test passed.")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
