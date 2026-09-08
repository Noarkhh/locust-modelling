"""Launch a single headless simulation run and collect its snapshots.

The simulation is the assembly jar of the Scala project; every parameter is
passed as a -Dparticle-agent.config.<key>=<value> system property override.
Each run gets its own directory holding snapshots, the resolved parameter set,
and the captured log tail, so any trial can be re-examined later.
"""

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

from .metrics import SNAPSHOT_DTYPE

SIM_JAR = os.environ.get("LOCUST_SIM_JAR", "")
JAVA = os.environ.get("LOCUST_JAVA", "java")
# Explicit heap cap so many JVMs pack onto one node without each claiming a
# fraction of total node memory.
JAVA_FLAGS = [
    "--add-modules=jdk.incubator.vector",
    f"-Xmx{os.environ.get('LOCUST_JAVA_XMX', '3g')}",
]
# Crashed/hung runs must not kill the worker; generous default, override per site.
TIMEOUT_S = int(os.environ.get("LOCUST_RUN_TIMEOUT", 4 * 3600))

# Collapse abort: a run whose global heading order sits below this threshold
# for COLLAPSE_WINDOW_SIM_S consecutive sim-seconds after burn-in is killed
# and reported as a failed replicate. Every sweep run so far showed order
# pinned near 0.02 after a collapse never recovering, so waiting the run out
# only burns core-hours. Snapshots only start after burn-in, which is why the
# check is automatically burn-in-aware. Set LOCUST_COLLAPSE_ORDER=0 to
# disable (e.g. for exploratory runs where recovery is the question).
COLLAPSE_ORDER_THRESHOLD = float(os.environ.get("LOCUST_COLLAPSE_ORDER", 0.1))
COLLAPSE_WINDOW_SIM_S = float(os.environ.get("LOCUST_COLLAPSE_WINDOW", 300.0))
COLLAPSE_POLL_WALL_S = 30.0
# How much of each snapshot file's tail to parse per poll; covers several
# complete snapshots even at 10k+ agents (25 bytes per agent per snapshot).
SNAPSHOT_TAIL_BYTES = 4 * 1024 * 1024

# Harness preconditions applied to every launch, beneath all other overrides:
# batch runs must be headless and must not flood sim.log. Experiment-level
# setup belongs in evaluation.Scenario, not here.
BASE_OVERRIDES = {
    "guiType": "none",
    "iterationFinishedLogFrequency": 1000,
}


class SimulationError(RuntimeError):
    pass


def run_simulation(
    values: dict[str, float],
    run_dir: str | Path,
    seed: int = 0,
    sim_overrides: dict | None = None,
) -> Path:
    """Execute one run; returns the run directory containing snapshots/*.bin.

    values         searched parameter values {hoconKey: value}
    sim_overrides  scenario setup (iterationsNumber, agentAmount, world size,
                   snapshotFrequency, particleAgentFactory, ...)
    """
    if not SIM_JAR:
        raise SimulationError("LOCUST_SIM_JAR is not set (path to assembly jar)")

    run_dir = Path(run_dir)
    snap_dir = run_dir / "snapshots"
    run_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(exist_ok=True)

    overrides = dict(BASE_OVERRIDES)
    overrides.update(sim_overrides or {})
    overrides.update(values)
    overrides["randomSeed"] = seed
    overrides["snapshotPath"] = str(snap_dir)

    # Every simulation binds an Akka clustering port; concurrent runs on one
    # host (SLURM packs many per node) must each get their own free port or
    # all but the first crash on bind.
    port = _free_port()
    cmd = [JAVA, *JAVA_FLAGS, f"-Dclustering.port={port}",
           f"-Dclustering.supervisor.port={port}"]
    cmd += [f"-Dparticle-agent.config.{k}={_hocon(v)}" for k, v in overrides.items()]
    cmd += ["-jar", SIM_JAR]

    (run_dir / "run.json").write_text(
        json.dumps({"overrides": overrides, "seed": seed, "cmd": cmd}, indent=2)
    )

    started = time.time()
    timestep_override = overrides.get("timestepDuration", 0.0)
    timestep_duration = (
        float(timestep_override) if isinstance(timestep_override, (int, float)) else 0.0
    )
    with open(run_dir / "sim.log", "w") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        returncode = _supervise(process, run_dir, snap_dir, timestep_duration)

    snapshots = list(snap_dir.glob("*.bin"))
    if returncode != 0 or not snapshots:
        raise SimulationError(
            f"run failed (rc={returncode}, {len(snapshots)} snapshot files): "
            f"see {run_dir / 'sim.log'}"
        )

    (run_dir / "run.json").write_text(
        json.dumps(
            {"overrides": overrides, "seed": seed, "cmd": cmd,
             "wall_seconds": time.time() - started, "returncode": returncode},
            indent=2,
        )
    )
    return run_dir


def _supervise(
    process: subprocess.Popen,
    run_dir: Path,
    snap_dir: Path,
    timestep_duration: float,
) -> int:
    """Wait for the simulation, enforcing the wall timeout and the collapse
    abort; returns the process exit code.

    Polls the snapshot stream every COLLAPSE_POLL_WALL_S seconds of wall time
    and kills the run once the global heading order has stayed below
    COLLAPSE_ORDER_THRESHOLD for COLLAPSE_WINDOW_SIM_S consecutive
    sim-seconds. The collapse check needs the timestep to convert iterations
    to sim-time, so a run launched without timestepDuration is only subject
    to the timeout.
    """
    deadline = started = time.time()
    deadline += TIMEOUT_S
    collapse_check_enabled = COLLAPSE_ORDER_THRESHOLD > 0.0 and timestep_duration > 0.0
    collapse_start_iteration: int | None = None
    while True:
        wait_seconds = min(COLLAPSE_POLL_WALL_S, max(deadline - time.time(), 1.0))
        try:
            return process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            pass
        if time.time() >= deadline:
            process.kill()
            process.wait()
            raise SimulationError(f"run timed out after {TIMEOUT_S}s: {run_dir}")
        if not collapse_check_enabled:
            continue
        latest = _latest_global_order(snap_dir)
        if latest is None:
            continue
        iteration, global_order = latest
        if global_order >= COLLAPSE_ORDER_THRESHOLD:
            collapse_start_iteration = None
        elif collapse_start_iteration is None:
            collapse_start_iteration = iteration
        elif (
            (iteration - collapse_start_iteration) * timestep_duration
            >= COLLAPSE_WINDOW_SIM_S
        ):
            process.kill()
            process.wait()
            raise SimulationError(
                f"aborted after {time.time() - started:.0f}s: global order "
                f"{global_order:.3f} stayed below {COLLAPSE_ORDER_THRESHOLD} "
                f"for {COLLAPSE_WINDOW_SIM_S:.0f} sim-seconds "
                f"(iterations {collapse_start_iteration}-{iteration}): {run_dir}"
            )


def _latest_global_order(snap_dir: Path) -> tuple[int, float] | None:
    """Global heading order at the newest fully-written snapshot iteration,
    read from the tails of the snapshot files; None if nothing is parseable
    yet.

    Only the last SNAPSHOT_TAIL_BYTES of each file are parsed (the files are
    append-only streams of fixed-size records, so the tail is always
    record-aligned relative to the file start). The newest iteration present
    may still be mid-write, so the order is computed at the second-newest
    when two are available.
    """
    record_size = SNAPSHOT_DTYPE.itemsize
    tails = []
    for snapshot_file in snap_dir.glob("*.bin"):
        try:
            with open(snapshot_file, "rb") as stream:
                file_size = stream.seek(0, os.SEEK_END)
                offset = max(0, file_size - SNAPSHOT_TAIL_BYTES)
                offset += -offset % record_size
                stream.seek(offset)
                raw = stream.read()
        except OSError:
            continue
        raw = raw[: len(raw) - len(raw) % record_size]
        if raw:
            tails.append(np.frombuffer(raw, dtype=SNAPSHOT_DTYPE))
    if not tails:
        return None
    records = np.concatenate(tails)
    iterations = np.unique(records["iter"])
    if len(iterations) == 0:
        return None
    iteration = int(iterations[-2] if len(iterations) > 1 else iterations[-1])
    headings = records["heading"][records["iter"] == iteration]
    order_vector = np.array([np.cos(headings).mean(), np.sin(headings).mean()])
    return iteration, float(np.linalg.norm(order_vector))


def cleanup_snapshots(run_dir: str | Path) -> None:
    """Delete bulky snapshot data once metrics are extracted (run.json stays)."""
    shutil.rmtree(Path(run_dir) / "snapshots", ignore_errors=True)


def _free_port() -> int:
    """Ask the OS for a currently-free TCP port (small race window is
    acceptable: a collision surfaces as a recorded failed replicate)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _hocon(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
