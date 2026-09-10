#!/bin/bash
# Bayesian optimization stage: every array task is an independent Optuna
# worker sharing one journal-file study on the shared filesystem.
#
#   sbatch --array=0-49 slurm/optimize.sbatch
#
# Total trials = array size x --trials. Resubmit with the same BO_DIR and
# study name to continue the search after wallclock limits.
#
# Each array task runs 4-worker simulations (horizontal slabs) with a
# hybrid replicate schedule: replicate 1 alone (the pruning decision), then
# up to 4 survivors concurrently — peak 4 JVMs x 4 worker threads = 16
# CPUs (see evaluation.evaluate_point and Scenario.workers_x).
#SBATCH --job-name=locust-bo
#SBATCH -A plglocustsim01-cpu
#SBATCH -p plgrid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=24:00:00
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
BO_DIR="${BO_DIR:?set to the study output directory (shared FS)}"
ONLY_PARAMS="${ONLY_PARAMS:-}"   # post-screening restriction, comma-separated
PARAMSEARCH_DIR="${PARAMSEARCH_DIR:?set to the absolute path of the paramsearch directory}"
export PYTHONPATH="$PARAMSEARCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
# 16 allocated CPUs = 4 concurrent replicate JVMs x 4 xinuk workers each;
# without this cap the replicate pool would try 16 concurrent JVMs.
export LOCUST_REPLICATE_WORKERS=4
# -----------------------------------------------------------------------------

python -m paramsearch.optimize \
    --out "$BO_DIR" \
    --study-name "${STUDY_NAME:-locust-calibration}" \
    --model "${MODEL:-NeuralFieldAgentFactory}" \
    --trials "${TRIALS_PER_WORKER:-25}" \
    ${ONLY_PARAMS:+--only "$ONLY_PARAMS"} \
    ${WARM_START:+--warm-start "$WARM_START"} \
    ${REPLICATES:+--replicates "$REPLICATES"}
