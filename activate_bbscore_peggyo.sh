#!/bin/bash
# Activate BBScore environment (server-specific)
# Usage: source activate_bbscore_SUFFIX.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="/Volumes/Lab/Users/xzhou25/bbscore_cs375_project_2026/.conda_env"
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH"
else
    echo "Error: conda not found"
    return 1
fi

export SCIKIT_LEARN_DATA="/Volumes/Lab/Users/xzhou25/bbscore_cs375_project_2026/bbscore_data"

# GPU selection: default to GPU 0 to reserve larger GPUs for heavy tasks
# Override before sourcing: export CUDA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# torch.compile/inductor Triton errors on driver 470 are handled by
# sitecustomize.py in the conda env (suppress_errors = True → eager fallback).
# Do NOT set TORCHDYNAMO_DISABLE=1 — it strips the _orig_mod wrapper that
# project code relies on for layer name resolution.

PYTHON_PATH=$(which python)
echo "BBScore environment activated!"
echo "Python: $PYTHON_PATH"
echo "Data: $SCIKIT_LEARN_DATA"
echo ""
echo "Quick start:"
echo "  $PYTHON_PATH run.py --model resnet18 --layer layer4 --benchmark OnlineTVSDV1 --metric ridge"
