#!/bin/bash
#SBATCH --job-name=sap-rl-cu126-setup
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
VENV_DIR="$PROJECT_DIR/.venv-cu126-v1"
cd "$PROJECT_DIR"
mkdir -p logs .cache/pip
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"

test ! -e "$VENV_DIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install \
  torch==2.14.0 \
  --index-url https://download.pytorch.org/whl/cu126
"$VENV_DIR/bin/python" -m pip install -e ".[rl,dev]"
"$VENV_DIR/bin/python" -c 'import torch; arches = torch.cuda.get_arch_list(); print("torch", torch.__version__); print("built_cuda", torch.version.cuda); print("arches", arches); assert "sm_70" in arches'
"$VENV_DIR/bin/python" -m pytest
