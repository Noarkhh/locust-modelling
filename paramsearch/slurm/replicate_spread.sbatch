#!/bin/bash
# A4: replicate-spread measurement — 6 evaluations x 5 seeds at the campaign
# scenarios, all six concurrent (each simulation uses 2 xinuk workers).
#
#   sbatch slurm/replicate_spread.sbatch
#
#SBATCH --job-name=locust-spread
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/spread.out

set -euo pipefail

# --- site configuration ------------------------------------------------------
# module load java/21 python/3.11   # or export LOCUST_JAVA=/path/to/jdk-21/bin/java
# source /path/to/locust-venv/bin/activate
export LOCUST_SIM_JAR="${LOCUST_SIM_JAR:?set to the versioned jar path}"
PARAMSEARCH_DIR="${PARAMSEARCH_DIR:?set to the absolute path of the paramsearch directory}"
export SPREAD_DIR="${SPREAD_DIR:-$PARAMSEARCH_DIR/runs/replicate-spread}"
# Per-simulation kill timeout: a hang guard, sized so one stuck run cannot
# eat the task's wall time. Search-scale runs finish in well under an hour.
export LOCUST_RUN_TIMEOUT="${LOCUST_RUN_TIMEOUT:-5400}"
# -----------------------------------------------------------------------------

export SPREAD_WORKERS=6
python3 "$PARAMSEARCH_DIR/scripts/replicate_spread.py"
