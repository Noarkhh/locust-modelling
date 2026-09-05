#!/bin/bash
# Sobol screening stage: one array task per batch of design rows.
#
# Before submitting:
#   python -m paramsearch.screening sample --out $SCREEN_DIR --power 5
# then size the array to the printed evaluation count (or use --stride):
#   sbatch --array=0-543 slurm/screening.sbatch
# afterwards:
#   python -m paramsearch.screening analyze --out $SCREEN_DIR
#
# Each task is one single-worker (single-core) simulation chain; the
# scheduler packs many tasks per node, which out-throughputs multi-worker
# simulations (see Scenario in paramsearch/evaluation.py). Size the array
# freely — e.g. one task per design row.
#SBATCH --job-name=locust-screening
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/screening-%a.out

set -euo pipefail

# --- site configuration ------------------------------------------------------
module load java/21 python/3.11 2>/dev/null || true
export LOCUST_SIM_JAR="${LOCUST_SIM_JAR:?set to the assembly jar path}"
SCREEN_DIR="${SCREEN_DIR:?set to the screening output directory (shared FS)}"
cd "$(dirname "$0")/.."
# -----------------------------------------------------------------------------

STRIDE="${SLURM_ARRAY_TASK_COUNT:-1}"
python -m paramsearch.screening run \
    --out "$SCREEN_DIR" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" \
    --stride "$STRIDE"
