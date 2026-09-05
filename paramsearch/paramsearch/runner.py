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
    with open(run_dir / "sim.log", "w") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise SimulationError(f"run timed out after {TIMEOUT_S}s: {run_dir}")

    snapshots = list(snap_dir.glob("*.bin"))
    if proc.returncode != 0 or not snapshots:
        raise SimulationError(
            f"run failed (rc={proc.returncode}, {len(snapshots)} snapshot files): "
            f"see {run_dir / 'sim.log'}"
        )

    (run_dir / "run.json").write_text(
        json.dumps(
            {"overrides": overrides, "seed": seed, "cmd": cmd,
             "wall_seconds": time.time() - started, "returncode": proc.returncode},
            indent=2,
        )
    )
    return run_dir


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
