#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
SRC="$ROOT_DIR/src/qwen35_delta.hip"
OUT="${1:-/tmp/qwen35_delta_hip.o}"
ARCH="${HIP_ARCH:-gfx1150}"

hipcc -c -x hip --offload-arch="$ARCH" -O2 "$SRC" -o "$OUT"
echo "$OUT"
