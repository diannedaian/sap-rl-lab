#!/bin/bash
#SBATCH --job-name=sap-rl-setup
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the project directory}"
cd "$PROJECT_DIR"
mkdir -p logs .cache/pip
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"

test ! -e .venv
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[rl,dev]"
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m sap_rl_lab.cli random --episodes 100 --seed 0
.venv/bin/python -m sap_rl_lab.cli greedy --episodes 100 --seed 0
