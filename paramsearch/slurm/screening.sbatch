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
#SBATCH -A plglocustsim01-cpu
#SBATCH -p plgrid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=/net/people/plgrid/plgjakubpryc/paramsearch/logs/%x-%A_%a.out

set -euo pipefail

# --- site configuration ------------------------------------------------------
module load java/21 2>/dev/null || true
# The job must use the same interpreter the study was set up with: activate
# the conda environment explicitly instead of inheriting PATH from the
# submission shell (which module load python would half-shadow).
# Locate the conda installation: explicit override, then the conda binary
# inherited from the submission environment, then common install locations.
CONDA_ROOT="${LOCUST_CONDA_ROOT:-$(conda info --base 2>/dev/null || true)}"
if [ -z "$CONDA_ROOT" ]; then
    for candidate in miniconda3 anaconda3 miniforge3 mambaforge; do
        [ -d "$HOME/$candidate" ] && CONDA_ROOT="$HOME/$candidate" && break
    done
fi
[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ] || {
    echo "cannot locate conda (set LOCUST_CONDA_ROOT)" >&2; exit 1; }
set +u  # conda's activation scripts reference unset variables
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${LOCUST_CONDA_ENV:-base}"
set -u
export LOCUST_SIM_JAR="${LOCUST_SIM_JAR:?set to the assembly jar path}"
SCREEN_DIR="${SCREEN_DIR:?set to the screening output directory (shared FS)}"
PARAMSEARCH_DIR="${PARAMSEARCH_DIR:?set to the absolute path of the paramsearch directory}"
export PYTHONPATH="$PARAMSEARCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
# -----------------------------------------------------------------------------

STRIDE="${SLURM_ARRAY_TASK_COUNT:-1}"
python -m paramsearch.screening run \
    --out "$SCREEN_DIR" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" \
    --stride "$STRIDE"
