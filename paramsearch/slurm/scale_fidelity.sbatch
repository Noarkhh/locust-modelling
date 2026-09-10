#!/bin/bash
# Scale-fidelity check. One array task per (model, parameter set, scale)
# cell; task count = `python scripts/scale_fidelity.py count`.
#
#   FIDELITY_CANDIDATES=... sbatch --array=0-47 slurm/scale_fidelity.sbatch
# afterwards:
#   FIDELITY_CANDIDATES=... python scripts/scale_fidelity.py analyze
#
# Each task is one evaluation whose 5 replicates run concurrently on the
# allocated CPUs (single-worker simulations; see evaluation.py).
#SBATCH --job-name=locust-fidelity
#SBATCH -A plglocustsim01-cpu
#SBATCH -p plgrid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
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
export FIDELITY_DIR="${FIDELITY_DIR:-$PARAMSEARCH_DIR/runs/scale-fidelity}"
export FIDELITY_CANDIDATES="${FIDELITY_CANDIDATES:-}"
# Replicates run concurrently on the task's CPUs (LOCUST_REPLICATE_WORKERS
# follows SLURM_CPUS_PER_TASK); each simulation stays single-worker.
# Full-scale replicates exceed runner.py's 4 h default timeout on one Ares
# core — export LOCUST_RUN_TIMEOUT at submission.
# -----------------------------------------------------------------------------

python3 "$PARAMSEARCH_DIR/scripts/scale_fidelity.py" run \
  --index "${SLURM_ARRAY_TASK_ID:?}" --workers 1
