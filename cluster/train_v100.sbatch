#!/bin/bash
#SBATCH --job-name=sap-rl-ppo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
RUN_ROOT="$PROJECT_DIR/runs/v100-seed17-job${SLURM_JOB_ID}"
cd "$PROJECT_DIR"
test -x .venv/bin/python
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT/data" "$RUN_ROOT/cache/matplotlib"

export MPLCONFIGDIR="$RUN_ROOT/cache/matplotlib"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SAP_RL_COMMIT="$(git rev-parse HEAD)"

nvidia-smi
.venv/bin/python -m pytest -q
.venv/bin/sap-rl build-league \
  --episodes 100 \
  --seed 0 \
  --output "$RUN_ROOT/data/train.json"
.venv/bin/sap-rl build-league \
  --episodes 100 \
  --seed 10000 \
  --output "$RUN_ROOT/data/eval.json"

.venv/bin/python -m sap_rl_lab.training train \
  --timesteps 500000 \
  --environments 8 \
  --vector-backend subproc \
  --device cuda \
  --seed 17 \
  --source-commit "$SAP_RL_COMMIT" \
  --opponent-league "$RUN_ROOT/data/train.json" \
  --output-dir "$RUN_ROOT/train"

.venv/bin/python -m sap_rl_lab.training evaluate \
  "$RUN_ROOT/train/final_model.zip" \
  --episodes 500 \
  --seed 20000 \
  --device cuda \
  --opponent-league "$RUN_ROOT/data/eval.json" \
  > "$RUN_ROOT/evaluation.json"

echo "$RUN_ROOT"
