#!/bin/bash
# A5: scale-fidelity check. One array task per (model, parameter set, scale)
# cell; task count = `python scripts/scale_fidelity.py count` (40 by default).
#
#   sbatch --array=0-39 ... slurm/scale_fidelity.sbatch
# afterwards:
#   python scripts/scale_fidelity.py analyze
#
#SBATCH --job-name=locust-fidelity
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=12:00:00

set -euo pipefail

# --- site configuration ------------------------------------------------------
# module load java/21 python/3.11   # or export LOCUST_JAVA=/path/to/jdk-21/bin/java
# source /path/to/locust-venv/bin/activate
export LOCUST_SIM_JAR="${LOCUST_SIM_JAR:?set to the versioned jar path}"
PARAMSEARCH_DIR="${PARAMSEARCH_DIR:?set to the absolute path of the paramsearch directory}"
export FIDELITY_DIR="${FIDELITY_DIR:-$PARAMSEARCH_DIR/runs/scale-fidelity}"
# -----------------------------------------------------------------------------

python3 "$PARAMSEARCH_DIR/scripts/scale_fidelity.py" run \
    --index "${SLURM_ARRAY_TASK_ID:?}" --workers 8
