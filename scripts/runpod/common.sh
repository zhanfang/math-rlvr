#!/usr/bin/env bash

resolve_runpod_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    export PYTHON_BIN
    return
  fi

  if [ -x "${PROJECT_ROOT}/.venv/bin/python" ]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
  export PYTHON_BIN
}

require_runpod_python_env() {
  "$PYTHON_BIN" - <<'PY'
import importlib
import sys

required = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "vllm",
]
missing = []
for module_name in required:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        missing_name = error.name or module_name
        if missing_name not in missing:
            missing.append(missing_name)

if missing:
    print("RunPod Python dependency check failed.", file=sys.stderr)
    print(f"python: {sys.executable}", file=sys.stderr)
    print(f"missing imports: {', '.join(sorted(missing))}", file=sys.stderr)
    print(
        "Use the prebuilt Stage 7 image built from docker/runpod-stage7/Dockerfile, "
        "or set PYTHON_BIN to an environment with requirements-runpod-stage7.txt installed.",
        file=sys.stderr,
    )
    print(
        "Fallback on an already-running pod: python3 -m pip install -r requirements-runpod-stage7.txt",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(f"RunPod Python dependency check passed: {sys.executable}")
PY
}
