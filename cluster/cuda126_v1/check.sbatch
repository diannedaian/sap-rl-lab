#!/bin/bash
#SBATCH --job-name=sap-rl-cu126-check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
PYTHON="$PROJECT_DIR/.venv-cu126-v1/bin/python"
cd "$PROJECT_DIR"
test -x "$PYTHON"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PYTHON" -c 'import torch; print("torch", torch.__version__); print("built_cuda", torch.version.cuda); print("arches", torch.cuda.get_arch_list()); print("cuda_available", torch.cuda.is_available()); assert torch.cuda.is_available(); x = torch.arange(1024, device="cuda"); print("device", torch.cuda.get_device_name(0)); print("sum", x.sum().item())'
