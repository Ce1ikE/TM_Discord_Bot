#!/usr/bin/env bash
set -euo pipefail


# to run this correctly, first create a Python 3.13 virtual environment and activate it:
# uv venv -p python3.13
# source .venv/bin/activate
# cd into this directory (src/df_packages/df_api/wheelhouse/scripts) and make sure the build scripts are executable:
# chmod +x build_torch_cu130_wheel.sh
# chmod +x build_torchvision_cu130_wheel.sh
# chmod +x build_llama_cpp_cuda_wheel.sh
# then run these scripts to build the necessary wheels for torch, torchvision, and llama-cpp-python with CUDA support.

# This script builds all the necessary wheels for our project and places them in the wheelhouse/dist directory.
# This includes:
# - torch with CUDA 13.0 support
# - torchvision with CUDA 13.0 support
# - llama-cpp-python built from source with CUDA support and OpenBLAS for CPU fallback
# The wheels are built/downloaded into the wheelhouse/dist directory, and the pyproject.toml file 
# is configured to use these local wheel files as sources for the torch, torchvision, and llama-cpp-python dependencies. This ensures that when we run uv install, it will use the pre-built wheels.
"$(dirname "${BASH_SOURCE[0]}")/build_torch_cu130_wheel.sh"
"$(dirname "${BASH_SOURCE[0]}")/build_torchvision_cu130_wheel.sh"
"$(dirname "${BASH_SOURCE[0]}")/build_llama_cpp_cuda_wheel.sh"
