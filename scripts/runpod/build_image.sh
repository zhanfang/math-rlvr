#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <registry/image:tag>" >&2
  echo "example: $0 docker.io/yourname/math-rlvr-stage7:cu128-vllm018" >&2
  exit 2
fi

IMAGE="$1"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PLATFORM="${PLATFORM:-linux/amd64}"
HOST_ARCH="$(uname -m)"

if [ "$HOST_ARCH" = "arm64" ] && [ "$PLATFORM" = "linux/amd64" ] && [ "${ALLOW_SLOW_ARM64_BUILD:-0}" != "1" ]; then
  cat >&2 <<'EOF'
Refusing to build linux/amd64 on an Apple Silicon host by default.

This RunPod image is large and the dependency install runs under emulation,
so local builds can take a very long time. Build it on native amd64 instead
(GitHub Actions, Docker Hub Build, or an x86 cloud/dev machine), or override
with:

  ALLOW_SLOW_ARM64_BUILD=1 scripts/runpod/build_image.sh <registry/image:tag>
EOF
  exit 3
fi

cd "$PROJECT_ROOT"

docker buildx build \
  --platform "$PLATFORM" \
  -f docker/runpod-stage7/Dockerfile \
  -t "$IMAGE" \
  --push \
  .

echo "Pushed image: $IMAGE"
