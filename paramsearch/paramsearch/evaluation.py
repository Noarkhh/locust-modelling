"""Evaluate one parameter set end-to-end: replicate runs -> metrics -> score.

This is the single entry point shared by the Sobol screening stage and the
Optuna workers, so both stages score parameter sets identically. Every
evaluation leaves a self-contained directory (per-replicate run dirs with
their resolved config and logs, plus result.json) for post-hoc analysis.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import metrics as metrics_module
from .objective import evaluate as score_metrics
from .parameters import NEURAL_FIELD, SPP
from .runner import SimulationError, cleanup_snapshots, run_simulation


@dataclass(frozen=True)
class Scenario:
    """Everything about a run that is NOT searched: the simulation setup.

    ``agent_amount`` scales the starting patch (via ``initial_density``) so
    populations of any size begin at the same density. The WORLD is sized
    per evaluation from the candidate's own speed (see ``world_side_meters``)
    so the marching band can never lap the torus and collide with its own
    tail within the run. Validate a reduced scale against the full one (rank
    correlation over a handful of parameter sets) before trusting it for a
    campaign.
    """

    model: str = NEURAL_FIELD
    agent_amount: int = 100
    # Initial packing of the starting patch, locusts/m^2 (750 = the basking
    # density used for hopper bands at dawn in Bach 2018 / Buhl's field
    # setup). The patch AREA is derived as agent_amount / initial_density, so
    # changing the population size keeps the starting density constant
    # instead of silently diluting or compressing the group.
    initial_density: float = 750.0
    # Height/width ratio of the starting patch (reference strip: 21.6/4.12).
    # The patch is clamped to the world height, widening to preserve area.
    initial_area_aspect: float = 5.2
    # Safety factor on the band's travel budget when sizing the world. The
    # band's centre cannot move faster than its agents, so
    # averageSpeed * run duration bounds the travel; the margin absorbs the
    # hop-speed bonus on top of that bound.
    band_travel_margin: float = 1.2
    # World height = this factor x initial patch height. Escape-driven bands
    # widen well past the patch (anti-goal sweep: width reached ~2.7x patch
    # height); too little room makes the band wrap onto itself laterally and
    # caps the measured band_width at the world height.
    world_height_patch_factor: float = 6.0
    iterations_number: int = 20000
    timestep_duration: float = 0.3
    # Spatial cell size of the agent containers (meters). Must stay >= the
    # largest interaction range: the plan creator only sees agents in the
    # 8 neighbouring containers, so ranges beyond the container size would
    # silently truncate perception.
    agent_container_size: float = 0.3
    snapshot_frequency: int = 200
    replicates: int = 3
    # Absolute burn-in: snapshots (and hence all metrics) start after this
    # many iterations. Set from measured transients, NOT a fraction of the
    # run: heading order equilibrates in seconds, but profile formation and
    # activity-timer desynchronization take minutes (quasi-1D neural field:
    # ~1500 iterations at dt=0.3; Bach-style SPP at dt=1.0: ~1800).
    burn_in_iterations: int = 1500
    # Quasi-1D "infinite front": the initial patch spans the full (wrapped)
    # world height, so the band has no lateral edges — the campaign geometry
    # for the neural-field model. The patch height is aligned down to a
    # whole number of containers (agents initialized outside the grid are
    # silently dropped otherwise) and the width widens to preserve density.
    full_height_patch: bool = False
    # Search runs are single-worker on purpose: evaluations are independent,
    # so packing one simulation per core beats splitting one simulation
    # across cores (xinuk's spatial strips synchronize every iteration and
    # the band concentrates the work in few strips). Use a multi-worker
    # layout only for latency-sensitive full-scale validation runs.
    workers_x: int = 1
    workers_y: int = 1
    sharding_mod: int = 144
    extra_overrides: dict = field(default_factory=dict)

    def initial_area(self) -> tuple[float, float]:
        """Width and height (meters) of the starting patch.

        Sized so the patch holds ``agent_amount`` at ``initial_density`` with
        the requested aspect ratio. With ``full_height_patch`` the height is
        aligned down to a whole number of containers (it becomes the world
        height) and the width compensates to preserve the density.
        """
        area = self.agent_amount / self.initial_density
        height = (area * self.initial_area_aspect) ** 0.5
        if self.full_height_patch:
            height = max(
                math.floor(height / self.agent_container_size), 2
            ) * self.agent_container_size
        width = area / height
        return width, height

    def world_size(self, average_speed: float) -> tuple[float, float]:
        """Width and height (meters) of the world, sized so the band cannot
        lap the torus and collide with its own tail.

        The band marches mostly along the x-axis (the initial patch is a
        tall strip at the left edge), so only the width needs to cover the
        travel: on a torus the front meets its own tail after traveling
        ``world_width - band_length``. The travel over the whole run is
        bounded by ``average_speed * duration`` (the band's centre cannot
        outrun its agents), padded by ``band_travel_margin``; the patch
        allowances cover the band's own extent. The candidate's OWN sampled
        speed sizes its world, so slow candidates stay cheap and fast ones
        stay uncontaminated.
        """
        duration = self.iterations_number * self.timestep_duration
        patch_width, patch_height = self.initial_area()
        width = average_speed * duration * self.band_travel_margin + 3.0 * patch_width
        # Align the width up to whole containers: the grid truncates
        # worldWidthMeters to floor(width / container), and a mismatch between
        # the config value and the effective grid desyncs the metrics' torus
        # arithmetic (and drops agents initialized in the cut-off sliver).
        width = math.ceil(width / self.agent_container_size) * self.agent_container_size
        if self.full_height_patch:
            height = patch_height  # already container-aligned by initial_area
        else:
            height = self.world_height_patch_factor * patch_height
            height = math.ceil(height / self.agent_container_size) * self.agent_container_size
        return width, height

    def simulation_overrides(self, values: dict[str, float]) -> dict:
        """Translate the scenario into HOCON config overrides for the runner.

        ``values`` is the searched parameter set of this evaluation — the
        world size depends on its ``averageSpeed``.
        """
        patch_width, patch_height = self.initial_area()
        world_width, world_height = self.world_size(values["averageSpeed"])
        overrides = {
            "particleAgentFactory": self.model,
            "agentAmount": self.agent_amount,
            "worldWidthMeters": world_width,
            "worldHeightMeters": world_height,
            "iterationsNumber": self.iterations_number,
            "timestepDuration": self.timestep_duration,
            "agentContainerSize": self.agent_container_size,
            "snapshotFrequency": self.snapshot_frequency,
            "snapshotStartIteration": self.burn_in_iterations,
            "workersX": self.workers_x,
            "workersY": self.workers_y,
            "shardingMod": self.sharding_mod,
            # Initial band: a patch near the left edge whose size follows
            # from agent_amount and initial_density (see initial_area).
            "initialAreaCenterX": patch_width,
            "initialAreaCenterY": world_height / 2,
            "initialAreaRadiusX": patch_width / 2,
            "initialAreaRadiusY": patch_height / 2,
        }
        overrides.update(self.extra_overrides)
        return overrides


def neural_field_band_scenario(agent_amount: int = 2000, replicates: int = 3) -> Scenario:
    """Campaign scenario for the neural-field model: quasi-1D infinite front.

    The patch spans the full wrapped world height (no lateral edges), the
    geometry where the escape-driven marching regime was characterized
    (2026-09-05): order sustained ~0.7-0.97 for 40+ sim-minutes, frontal
    profile with pooled decay R^2 ~0.99. Burn-in 1500 iterations (7.5
    sim-min) covers profile formation and timer desynchronization; heading
    order itself equilibrates in seconds.
    """
    return Scenario(
        model=NEURAL_FIELD,
        agent_amount=agent_amount,
        full_height_patch=True,
        iterations_number=8000,
        timestep_duration=0.3,
        snapshot_frequency=100,
        burn_in_iterations=1500,
        replicates=replicates,
    )


def spp_band_scenario(agent_amount: int = 10000, replicates: int = 3) -> Scenario:
    """Campaign scenario for the SPP model: Bach 2018's regime.

    dt = 1 s (Bach's integration step — the SPP update rule is
    timestep-dependent), 4 h of marching (his measurement horizon; the
    validated frontal run scored 12.6 here). Burn-in 1800 iterations
    (30 sim-min) covers band formation at Bach's slow walking speeds.
    """
    return Scenario(
        model=SPP,
        agent_amount=agent_amount,
        iterations_number=14400,
        timestep_duration=1.0,
        snapshot_frequency=60,
        burn_in_iterations=1800,
        replicates=replicates,
    )


def evaluate_point(
    values: dict[str, float],
    evaluation_dir: str | Path,
    scenario: Scenario,
    base_seed: int = 1,
    keep_snapshots: bool = False,
) -> dict:
    """Run all replicates of one parameter set and score it.

    Each replicate gets its own subdirectory and seed. A replicate that
    crashes or times out is recorded and skipped rather than aborting the
    evaluation; if every replicate fails the score falls back to the
    objective's failure score. Returns (and writes to result.json) a dict
    with the score, per-target breakdown, and per-replicate metrics.
    """
    evaluation_dir = Path(evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    simulation_overrides = scenario.simulation_overrides(values)
    world_width, world_height = scenario.world_size(values["averageSpeed"])

    replicate_metrics = []
    failures = []
    for replicate in range(scenario.replicates):
        run_dir = evaluation_dir / f"replicate-{replicate}"
        seed = base_seed + replicate
        try:
            run_simulation(
                values, run_dir, seed=seed, sim_overrides=simulation_overrides
            )
            replicate_metrics.append(
                metrics_module.compute_metrics(
                    run_dir / "snapshots",
                    world_width=world_width,
                    world_height=world_height,
                    timestep_duration=scenario.timestep_duration,
                    snapshot_frequency=scenario.snapshot_frequency,
                    # Snapshots already start after burn-in; keep them all.
                    burn_in_fraction=0.0,
                )
            )
        except (SimulationError, FileNotFoundError, ValueError) as error:
            failures.append({"replicate": replicate, "error": str(error)})
        finally:
            if not keep_snapshots:
                cleanup_snapshots(run_dir)

    # With no successful replicates every target scores its failure penalty.
    score, breakdown = score_metrics(replicate_metrics)
    result = {
        "values": values,
        "scenario": asdict(scenario),
        "base_seed": base_seed,
        "score": score,
        "breakdown": breakdown,
        "replicate_metrics": replicate_metrics,
        "failures": failures,
    }
    (evaluation_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    return result
