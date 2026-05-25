#!/usr/bin/env python
"""Validate the stage-1 local Python environment."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PackageCheck:
    import_name: str
    display_name: str
    version_attr: str = "__version__"


PACKAGES = [
    PackageCheck("torch", "PyTorch"),
    PackageCheck("transformers", "Transformers"),
    PackageCheck("datasets", "Datasets"),
    PackageCheck("accelerate", "Accelerate"),
    PackageCheck("peft", "PEFT"),
    PackageCheck("trl", "TRL"),
]


def check_python() -> bool:
    version = sys.version_info
    ok = version >= (3, 10)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Python: {platform.python_version()} ({sys.executable})")
    return ok


def check_package(package: PackageCheck) -> bool:
    try:
        module = importlib.import_module(package.import_name)
    except Exception as exc:
        print(f"[FAIL] {package.display_name}: import failed ({exc})")
        return False

    version = getattr(module, package.version_attr, "unknown")
    print(f"[OK] {package.display_name}: {version}")
    return True


def report_torch_backends() -> None:
    try:
        import torch
    except Exception:
        print("[SKIP] PyTorch backends: torch is not importable")
        return

    cuda_available = torch.cuda.is_available()
    mps_available = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )

    print("PyTorch backends:")
    print("  - CPU: available")
    print(f"  - CUDA: {'available' if cuda_available else 'not available'}")
    if cuda_available:
        print(f"  - CUDA device count: {torch.cuda.device_count()}")
        print(f"  - Current CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  - MPS: {'available' if mps_available else 'not available'}")


def main() -> int:
    print("Stage 1 environment smoke check")
    print("=" * 33)

    checks = [check_python()]
    checks.extend(check_package(package) for package in PACKAGES)
    report_torch_backends()

    if all(checks):
        print("\nSmoke check passed.")
        return 0

    print("\nSmoke check failed. Install or update the environment first.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
