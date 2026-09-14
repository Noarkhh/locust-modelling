#!/bin/bash
# Validation-ladder large-scale run: one candidate, one seed, one full node.
#
#   TRIAL_RESULT=<result.json> AGENTS=250000 SEED=1 OUT_DIR=<dir> \
#       sbatch slurm/large_scale.sbatch
#
#SBATCH --job-name=locust-large
#SBATCH -A plglocustsim01-cpu
#SBATCH -p plgrid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=90G
#SBATCH --time=16:00:00
#SBATCH --output=/net/people/plgrid/plgjakubpryc/paramsearch/logs/%x-%A.out

set -euo pipefail

# --- site configuration ------------------------------------------------------
module load java/21 2>/dev/null || true
CONDA_ROOT="${LOCUST_CONDA_ROOT:-$(conda info --base 2>/dev/null || true)}"
if [ -z "$CONDA_ROOT" ]; then
    for candidate in miniconda3 anaconda3 miniforge3 mambaforge; do
        [ -d "$HOME/$candidate" ] && CONDA_ROOT="$HOME/$candidate" && break
    done
fi
[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ] || {
    echo "cannot locate conda (set LOCUST_CONDA_ROOT)" >&2; exit 1; }
set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${LOCUST_CONDA_ENV:-base}"
set -u
export LOCUST_SIM_JAR="${LOCUST_SIM_JAR:?set to the assembly jar path}"
PARAMSEARCH_DIR="${PARAMSEARCH_DIR:?set to the absolute path of the paramsearch directory}"
export PYTHONPATH="$PARAMSEARCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
# One JVM using the whole node: generous heap, 15 h runner timeout inside
# the 16 h wall.
export LOCUST_JAVA_XMX="${LOCUST_JAVA_XMX:-16g}"
export LOCUST_RUN_TIMEOUT="${LOCUST_RUN_TIMEOUT:-54000}"
# -----------------------------------------------------------------------------

python3 "$PARAMSEARCH_DIR/scripts/large_scale_run.py" \
    --trial "${TRIAL_RESULT:?}" \
    --agents "${AGENTS:-250000}" \
    --workers "${WORKERS:-1}" \
    --workers-y "${WORKERS_Y:-48}" \
    --seed "${SEED:-1}" \
    --out "${OUT_DIR:?}" \
    ${SNAPSHOT_START:+--snapshot-start "$SNAPSHOT_START"} \
    ${ITERATIONS:+--iterations "$ITERATIONS"} \
    ${SNAPSHOT_FREQ:+--snapshot-frequency "$SNAPSHOT_FREQ"} \
    ${PATCH_WIDTH:+--patch-width "$PATCH_WIDTH"}
