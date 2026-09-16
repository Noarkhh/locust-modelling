#!/bin/bash
# Density-variation experiment: one array task per (candidate, density) cell.
#   sbatch --array=0-23 slurm/density_experiment.sbatch
# Each task runs 5 concurrent replicates of a 2000-agent campaign scenario at
# the cell's density; snapshots retained for profile / band-length analysis.
#SBATCH --job-name=locust-density
#SBATCH -A plglocustsim01-cpu
#SBATCH -p plgrid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=/net/people/plgrid/plgjakubpryc/paramsearch/logs/%x-%A_%a.out

set -euo pipefail
module load java/21 2>/dev/null || true
CONDA_ROOT="${LOCUST_CONDA_ROOT:-$(conda info --base 2>/dev/null || true)}"
if [ -z "$CONDA_ROOT" ]; then
    for candidate in miniconda3 anaconda3 miniforge3 mambaforge; do
        [ -d "$HOME/$candidate" ] && CONDA_ROOT="$HOME/$candidate" && break
    done
fi
set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${LOCUST_CONDA_ENV:-base}"
set -u
export LOCUST_SIM_JAR="${LOCUST_SIM_JAR:?set to the assembly jar path}"
PARAMSEARCH_DIR="${PARAMSEARCH_DIR:?set to the paramsearch directory}"
export PYTHONPATH="$PARAMSEARCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LOCUST_KEEP_SNAPSHOTS=1
export LOCUST_REPLICATE_WORKERS="${LOCUST_REPLICATE_WORKERS:-5}"

python "$PARAMSEARCH_DIR/scripts/density_experiment.py" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" \
    --out "${OUT_DIR:?}" \
    --bo-root "${BO_ROOT:?}"
