#!/bin/bash
#SBATCH --job-name=sap-rl-gpu-check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
cd "$PROJECT_DIR"
test -x .venv/bin/python

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/python -c 'import torch; print("torch", torch.__version__); print("built_cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); assert torch.cuda.is_available(); x = torch.arange(1024, device="cuda"); print("device", torch.cuda.get_device_name(0)); print("sum", x.sum().item())'
