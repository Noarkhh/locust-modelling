#!/bin/bash
# Bayesian optimization stage: every array task is an independent Optuna
# worker sharing one journal-file study on the shared filesystem.
#
#   sbatch --array=0-49 slurm/optimize.sbatch
#
# Total trials = array size x --trials. Resubmit with the same BO_DIR and
# study name to continue the search after wallclock limits.
#
# Each worker runs single-worker (single-core) simulations sequentially;
# throughput comes from the array width, not per-simulation parallelism.
#SBATCH --job-name=locust-bo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bo-%a.out

set -euo pipefail

# --- site configuration ------------------------------------------------------
module load java/21 python/3.11 2>/dev/null || true
export LOCUST_SIM_JAR="${LOCUST_SIM_JAR:?set to the assembly jar path}"
BO_DIR="${BO_DIR:?set to the study output directory (shared FS)}"
ONLY_PARAMS="${ONLY_PARAMS:-}"   # post-screening restriction, comma-separated
cd "$(dirname "$0")/.."
# -----------------------------------------------------------------------------

python -m paramsearch.optimize \
    --out "$BO_DIR" \
    --study-name "${STUDY_NAME:-locust-calibration}" \
    --trials "${TRIALS_PER_WORKER:-25}" \
    ${ONLY_PARAMS:+--only "$ONLY_PARAMS"}
