#!/bin/bash
#SBATCH --job-name=sap-rl-round2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit from the new versioned checkout}"
PYTHON="${SAP_RL_PYTHON:?Supply an existing compatible personal environment}"
BASE_RUN="${SAP_RL_BASE_RUN:?Supply the prior experiment directory}"
DEVICE="${SAP_RL_DEVICE:-cpu}"
RUN_ROOT="$PROJECT_DIR/runs/round2-job${SLURM_JOB_ID}"
cd "$PROJECT_DIR"
test -x "$PYTHON"
test -f "$BASE_RUN/train/final_model.zip"
test ! -e "$RUN_ROOT"
export PYTHONPATH="$PROJECT_DIR/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PROJECT_DIR/runs/mpl-job${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

# CPU by default: this small policy can spend more time on GPU transfers than
# on arithmetic. Supply --gres=gpu:1 and SAP_RL_DEVICE=cuda for a GPU experiment.
if [ "$DEVICE" = cuda ]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  "$PYTHON" -c 'import torch; x = torch.arange(1024, device="cuda"); assert x.sum().item() == 523776'
fi
"$PYTHON" -m pytest -q
"$PYTHON" -m sap_rl_lab.experiments \
  --base-run "$BASE_RUN" \
  --output "$RUN_ROOT" \
  --timesteps 1000000 \
  --seeds 17 23 41 \
  --device "$DEVICE" \
  --vector-backend subproc \
  --source-commit "$(git rev-parse HEAD)"
