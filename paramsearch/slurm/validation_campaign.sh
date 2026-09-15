#!/bin/bash
# Validation campaign: top-4 candidates of each model x {25k, 100k, 250k}.
#
# Cell layout (approved 2026-09-15):
#   NF   25k -> 12 workers (1/4 node), wall 16 h
#   NF  100k -> 48 workers (1 node),   wall 16 h
#   NF  250k -> 144 workers (3 nodes), wall 12 h   (distributed)
#   spin 25k -> 24 workers (1/2 node), wall 22 h
#   spin 100k -> 96 workers (2 nodes), wall 24 h   (distributed)
#   spin 250k -> 240 workers (5 nodes), wall 24 h  (distributed)
# Walls are ~1.25x the slowest candidate estimate (trimmed from 1.5x to
# reduce the requested-hours reservation counted against the grant).
#
# Requires the usual submission environment:
#   LOCUST_SIM_JAR, PARAMSEARCH_DIR (and conda via LOCUST_CONDA_ENV/ROOT)
#
#   DRY_RUN=1 ./validation_campaign.sh       # print sbatch commands only
#   ./validation_campaign.sh nf              # only neural field cells
#   ./validation_campaign.sh spin 250k       # only spin 250k cells
#   ./validation_campaign.sh nf 250k 00756   # single cell

set -euo pipefail

SCRATCH_RUNS=/net/afscra/people/plgjakubpryc/paramsearch-runs
SLURM_DIR="${PARAMSEARCH_DIR:?}/slurm"
NF_TRIALS="00756 01030 01273 00886"
SPIN_TRIALS="00514 00305 00457 00682"
ONLY_MODEL="${1:-}"
ONLY_SCALE="${2:-}"
ONLY_TRIAL="${3:-}"

submit() {  # model trial agents workers nodes wall
    local model=$1 trial=$2 agents=$3 workers=$4 nodes=$5 wall=$6
    local scale_label=$((agents / 1000))k
    [ -n "$ONLY_MODEL" ] && [ "$ONLY_MODEL" != "$model" ] && return 0
    [ -n "$ONLY_SCALE" ] && [ "$ONLY_SCALE" != "$scale_label" ] && return 0
    [ -n "$ONLY_TRIAL" ] && [ "$ONLY_TRIAL" != "$trial" ] && return 0
    local trial_result="$SCRATCH_RUNS/bo2-$model/trials/trial-$trial/result.json"
    local out_dir="$SCRATCH_RUNS/valcamp/$model-$trial-$scale_label"
    local name="vc-$model-$trial-$scale_label"
    local cmd
    if [ "$nodes" -eq 1 ]; then
        # single JVM; fractional nodes get proportional memory and a runner
        # timeout 1 h under the wall
        local mem_gb=$((workers * 2 < 20 ? 20 : workers * 2))
        local timeout_s=$(( (${wall%%:*} - 1) * 3600 ))
        cmd=(sbatch --job-name="$name" --cpus-per-task="$workers"
             --mem="${mem_gb}G" --time="$wall"
             --export=ALL,TRIAL_RESULT="$trial_result",AGENTS="$agents",WORKERS=1,WORKERS_Y="$workers",SEED=1,OUT_DIR="$out_dir",SNAPSHOT_START=0,ITERATIONS=36100,LOCUST_KEEP_SNAPSHOTS=1,LOCUST_RUN_TIMEOUT="$timeout_s"
             "$SLURM_DIR/large_scale.sbatch")
    else
        cmd=(sbatch --job-name="$name" --nodes="$nodes" --time="$wall"
             --export=ALL,TRIAL_RESULT="$trial_result",AGENTS="$agents",WORKERS_Y="$workers",SEED=1,OUT_DIR="$out_dir",SNAPSHOT_START=0,ITERATIONS=36100,LOCUST_KEEP_SNAPSHOTS=1
             "$SLURM_DIR/large_scale_distributed.sbatch")
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '%q ' "${cmd[@]}"; echo
    else
        mkdir -p "$out_dir"
        "${cmd[@]}"
    fi
}

for trial in $NF_TRIALS; do
    submit nf "$trial"  25000  12 1 16:00:00
    submit nf "$trial" 100000  48 1 16:00:00
    submit nf "$trial" 250000 144 3 12:00:00
done
for trial in $SPIN_TRIALS; do
    submit spin "$trial"  25000  24 1 22:00:00
    submit spin "$trial" 100000  96 2 24:00:00
    submit spin "$trial" 250000 240 5 24:00:00
done
