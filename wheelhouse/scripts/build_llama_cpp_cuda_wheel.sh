#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y libopenblas-dev ccache

# Build llama-cpp-python from source with CUDA enabled into wheelhouse/dist.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/../dist}"
LLAMA_CPP_VERSION="${LLAMA_CPP_VERSION:-0.3.21}"

# Allow overrides while providing sensible CUDA defaults.
export FORCE_CMAKE="${FORCE_CMAKE:-1}"
export CMAKE_ARGS="${CMAKE_ARGS:--DGGML_CUDA=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS}"
# Little explanation of the above:
# -DGGML_CUDA=ON: This enables CUDA support in llama-cpp-python, which is necessary for GPU acceleration.
# -DGGML_BLAS=ON: This enables BLAS (Basic Linear Algebra Subprograms) support, which can improve performance for certain operations.
#                 especially when falling back to CPU execution, namely CUDA provides it's own math librarry (cuBLAS) but does not provide a CPU fallback,
#                 so we need to enable BLAS support to ensure that we can still run on CPU if CUDA is not available. or when VRAM is insufficient for the model size.
# -DGGML_BLAS_VENDOR=OpenBLAS: This specifies that we want to use OpenBLAS as the BLAS implementation (see libopenblas-dev dependency above).
#                              OpenBLAS is a popular open-source BLAS library that provides optimized implementations of BLAS functions, which can further improve performance.

# Use all cores by default unless caller sets a value explicitly.
if [[ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
  fi
fi

mkdir -p "${OUT_DIR}"

# --with pip: Forces uv to include pip in the temporary environment
# --no-project: Ignores your currently 'broken' pyproject.toml
uv run --no-project --python 3.13 --with pip python -m pip wheel \
  --no-deps \
  --no-binary :all: \
  -w "${OUT_DIR}" \
  "llama-cpp-python==${LLAMA_CPP_VERSION}"

echo "Built llama-cpp-python wheel(s) in: ${OUT_DIR}"
