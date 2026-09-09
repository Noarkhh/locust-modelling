#!/bin/bash
# A4: replicate-spread measurement — 6 evaluations x 5 seeds at the campaign
# scenarios, all six concurrent (each simulation uses 2 xinuk workers).
#
#   sbatch slurm/replicate_spread.sbatch
#
#SBATCH --job-name=locust-spread
#SBATCH -A plglocustsim01-cpu
#SBATCH -p plgrid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
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
# source /path/to/locust-venv/bin/activate
export LOCUST_SIM_JAR="${LOCUST_SIM_JAR:?set to the versioned jar path}"
PARAMSEARCH_DIR="${PARAMSEARCH_DIR:?set to the absolute path of the paramsearch directory}"
export SPREAD_DIR="${SPREAD_DIR:-$PARAMSEARCH_DIR/runs/replicate-spread}"
# -----------------------------------------------------------------------------

export SPREAD_WORKERS=6
# This script parallelizes across evaluations itself; keep each
# evaluation's replicates sequential so the CPU allocation is not
# oversubscribed by nested JVM pools.
export LOCUST_REPLICATE_WORKERS=1
python3 "$PARAMSEARCH_DIR/scripts/replicate_spread.py"
