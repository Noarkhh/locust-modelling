#!/bin/bash -l
# Distributed ladder run — direct adaptation of xinuk's reference
# multi-node launcher (xinuk/script/validation/run.sh): workers launched
# in the background via `srun -w<host>`, one per node, supervisor JVM
# last, in the foreground, on the batch host; all gated by
# akka.cluster.min-nr-of-members = node count.
#
#   TRIAL_RESULT=<result.json> AGENTS=1000000 WORKERS_Y=384 SEED=1 \
#       OUT_DIR=<dir> sbatch --nodes=8 slurm/large_scale_distributed.sbatch
#
#SBATCH --job-name=locust-dist
#SBATCH -A plglocustsim01-cpu
#SBATCH -p plgrid
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=90G
#SBATCH --time=40:00:00
#SBATCH --output=/net/people/plgrid/plgjakubpryc/paramsearch/logs/%x-%A.out

cd "${SLURM_SUBMIT_DIR}"

module load java/21 2>/dev/null || true
CONDA_ROOT="${LOCUST_CONDA_ROOT:-$(conda info --base 2>/dev/null || true)}"
if [ -z "$CONDA_ROOT" ]; then
    for candidate in miniconda3 anaconda3 miniforge3 mambaforge; do
        [ -d "$HOME/$candidate" ] && CONDA_ROOT="$HOME/$candidate" && break
    done
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${LOCUST_CONDA_ENV:-base}"
export LOCUST_SIM_JAR="${LOCUST_SIM_JAR:?set to the assembly jar path}"
PARAMSEARCH_DIR="${PARAMSEARCH_DIR:?set to the absolute path of the paramsearch directory}"
export PYTHONPATH="$PARAMSEARCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
JAVA_BIN="${LOCUST_JAVA:-java}"

#===============================================================================
# global config (after run.sh)
#===============================================================================
SUPERVISOR_HOSTNAME=$(/bin/hostname)
HOSTSS=$(scontrol show hostnames)
NHOSTS=$(echo ${HOSTSS} | wc -w)
echo NHOSTS ${NHOSTS}
HOSTNAMES=$(echo ${HOSTSS} | sed "s/\b$SUPERVISOR_HOSTNAME\b//g")

SNAP_DIR="${OUT_DIR:?}/snapshots"
mkdir -p "$SNAP_DIR"

PARAMS=$(python3 "$PARAMSEARCH_DIR/scripts/print_overrides.py" \
    --trial "${TRIAL_RESULT:?}" \
    --agents "${AGENTS:-1000000}" \
    --workers-y "${WORKERS_Y:-384}" \
    --seed "${SEED:-1}" \
    ${SNAPSHOT_START:+--snapshot-start "$SNAPSHOT_START"} \
    ${ITERATIONS:+--iterations "$ITERATIONS"} \
    ${PATCH_WIDTH:+--patch-width "$PATCH_WIDTH"})
PARAMS="$PARAMS -Dparticle-agent.config.snapshotPath=$SNAP_DIR"

#===============================================================================
# launch (after run.sh: workers backgrounded, supervisor foreground last)
#===============================================================================
for WORKER_HOST in ${HOSTNAMES}; do
    srun -w${WORKER_HOST} -c${SLURM_CPUS_PER_TASK} -N1 -n1 \
        ${JAVA_BIN} \
        -Xmx${LOCUST_JAVA_XMX:-65536m} -Xms512m -XX:+UseG1GC \
        --add-modules=jdk.incubator.vector \
        -Dclustering.ip=${WORKER_HOST} \
        -Dclustering.supervisor.ip=${SUPERVISOR_HOSTNAME} \
        -Dparticle-agent.config.isSupervisor=false \
        -Dakka.cluster.min-nr-of-members=${NHOSTS} \
        -Dclustering.min-nr-of-members=${NHOSTS} \
        ${PARAMS} \
        -jar "${LOCUST_SIM_JAR}" > "${OUT_DIR}/worker-${WORKER_HOST}.log" 2>&1 &
done

${JAVA_BIN} \
    -Xmx${LOCUST_JAVA_XMX:-65536m} -Xms512m -XX:+UseG1GC \
    --add-modules=jdk.incubator.vector \
    -Dclustering.ip=${SUPERVISOR_HOSTNAME} \
    -Dclustering.supervisor.ip=${SUPERVISOR_HOSTNAME} \
    -Dparticle-agent.config.isSupervisor=true \
    -Dakka.cluster.min-nr-of-members=${NHOSTS} \
    -Dclustering.min-nr-of-members=${NHOSTS} \
    ${PARAMS} \
    -jar "${LOCUST_SIM_JAR}" > "${OUT_DIR}/supervisor.log" 2>&1

sleep 20
echo "$(date -u +%H:%M:%S) all done; snapshots: $(ls "$SNAP_DIR" | wc -l) files"
exit 0
