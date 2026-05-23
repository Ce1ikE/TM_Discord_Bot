#!/usr/bin/env bash
set -euo pipefail

# Build/download the torchvision CUDA 13.0 wheel into wheelhouse/dist.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/../dist}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130/}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1+cu130}"

mkdir -p "${OUT_DIR}"

uv run --no-project --python 3.13 --with pip python -m pip wheel \
  --index-url "${PYTORCH_INDEX_URL}" \
  --no-deps \
  -w "${OUT_DIR}" \
  "torchvision==${TORCHVISION_VERSION}"

echo "Built torchvision wheel(s) in: ${OUT_DIR}"
