"""Evaluate one parameter set end-to-end: replicate runs -> metrics -> score.

This is the single entry point shared by the Sobol screening stage and the
Optuna workers, so both stages score parameter sets identically. Every
evaluation leaves a self-contained directory (per-replicate run dirs with
their resolved config and logs, plus result.json) for post-hoc analysis.
"""

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import metrics as metrics_module
from .objective import evaluate as score_metrics
from .parameters import NEURAL_FIELD, SPIN, SPP
from .runner import SimulationError, cleanup_snapshots, run_simulation

# How many replicates of one evaluation may run concurrently (each is its own
# single-worker JVM; the runner's per-run port allocation keeps them apart).
# Follows the SLURM CPU allocation by default, so `sbatch --cpus-per-task=3`
# is the only knob a submission needs; override with LOCUST_REPLICATE_WORKERS.
# Only used when no interim callback is set — the BO stage's pruning depends
# on replicates completing one at a time.
REPLICATE_WORKERS = int(
    os.environ.get(
        "LOCUST_REPLICATE_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "1")
    )
)

# Pinned marching activity period (seconds) — MUST match the activityPeriod
# default in the simulation's reference.conf (the empirical 45-minute bout,
# Simpson 1981; pinned there 2026-09-11). Needed Python-side to bound each
# candidate's duty cycle for world sizing.
ACTIVITY_PERIOD_SECONDS = 2700.0


def duty_cycle(values: dict[str, float]) -> float:
    """Upper bound on the fraction of time a candidate's agents march.

    With staggered activity timers the population's active fraction equals
    activity / (activity + mean rest) from the first iteration, where the
    mean rest is the minimum inactivity plus the mean geometric wait of the
    per-second resumption draw. The band's centre cannot move faster than
    duty x walking speed, which is what world sizing needs.
    """
    mean_rest = values["minimalInactivityPeriod"] + 1.0 / values[
        "resumeMarchProbabilityPerSecond"
    ]
    return ACTIVITY_PERIOD_SECONDS / (ACTIVITY_PERIOD_SECONDS + mean_rest)


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
    # Horizontal-slab decomposition: workers_x splits the world into
    # full-width rows, so in the quasi-1D geometry every worker owns a
    # cross-section of the band at all times (splitting along the width
    # would idle every worker the band is not in). Default single-worker:
    # the per-iteration synchronization only pays off for models with heavy
    # per-agent compute — measured 2026-09-11: the neural field gains just
    # 1.15x from 4 workers, so it stays single-worker; the spin preset
    # overrides this (Glauber loop amortizes the sync).
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

    def world_size(self, values: dict[str, float]) -> tuple[float, float]:
        """Width and height (meters) of the world, sized so the band's dense
        body cannot lap the torus and collide with itself.

        The band marches mostly along the x-axis (the initial patch is a
        tall strip at the left edge), so only the width needs to cover the
        travel. The band's centre cannot move faster than the candidate's
        duty cycle times its walking speed (only active agents move), so
        the travel budget is ``speed x duty x duration``, padded by
        ``band_travel_margin``; the patch allowances cover the band's own
        extent. Sizing from the candidate's OWN sampled speed and pause
        parameters keeps slow or resty candidates cheap and gives
        short-pause (high-duty) candidates the full room they need. A few
        slow stragglers may still be overtaken by the front — harmless and
        field-realistic recycling; what the budget excludes is the dense
        band meeting itself.
        """
        duration = self.iterations_number * self.timestep_duration
        patch_width, patch_height = self.initial_area()
        travel_budget = (
            values["averageSpeed"] * duty_cycle(values) * duration
        )
        width = travel_budget * self.band_travel_margin + 3.0 * patch_width
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
        world_width, world_height = self.world_size(values)
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
    (2026-05-09): order sustained ~0.7-0.97 for 40+ sim-minutes, frontal
    profile with pooled decay R^2 ~0.99.

    Run length 36000 iterations (3 h) with burn-in 12000 (1 h): with the
    intermittency pinned to the empirical 45/15-min cycle (2026-09-11),
    the burn-in spans one full activity-rest cycle so every agent has
    cycled at least once before measurement, and the measured window
    covers two full cycles of the front-recycling dynamics.
    """
    return Scenario(
        model=NEURAL_FIELD,
        agent_amount=agent_amount,
        full_height_patch=True,
        iterations_number=36000,
        timestep_duration=0.3,
        snapshot_frequency=100,
        burn_in_iterations=12000,
        replicates=replicates,
    )


def spin_band_scenario(agent_amount: int = 2000, replicates: int = 3) -> Scenario:
    """Campaign scenario for the spin-system model: same quasi-1D infinite
    front as the neural-field preset (the two ring models are compared in
    identical geometry). Validated 2026-09-08: with the ported escape
    mechanism and marching intermittency, order stabilizes ~0.55-0.6 over
    20+ sim-minutes at the NF-tuned strength 0.72; the spin model's own
    optimum is the campaign's job to find. Note the Glauber loop makes spin
    evaluations several times more expensive than neural-field ones.

    Run length and burn-in follow the neural-field preset: 36000
    iterations (3 h) with a one-full-cycle burn-in of 12000 under the
    pinned 45/15-min intermittency, leaving a two-cycle measured window.
    """
    return Scenario(
        model=SPIN,
        agent_amount=agent_amount,
        full_height_patch=True,
        workers_x=4,
        iterations_number=36000,
        timestep_duration=0.3,
        snapshot_frequency=100,
        burn_in_iterations=12000,
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


def campaign_scenario(model: str, replicates: int = 3) -> Scenario:
    """The validated campaign preset for a model — the single source both
    the screening and BO entry points draw their scenario from, so a model
    name always maps to the geometry its regime was characterized in."""
    presets = {
        NEURAL_FIELD: neural_field_band_scenario,
        SPIN: spin_band_scenario,
        SPP: spp_band_scenario,
    }
    if model not in presets:
        raise ValueError(f"no campaign scenario for model {model!r}")
    return presets[model](replicates=replicates)


def evaluate_point(
    values: dict[str, float],
    evaluation_dir: str | Path,
    scenario: Scenario,
    base_seed: int = 1,
    keep_snapshots: bool = False,
    interim_callback: Callable[[int, float], bool] | None = None,
) -> dict:
    """Run all replicates of one parameter set and score it.

    Each replicate gets its own subdirectory and seed. A replicate that
    crashes or times out is recorded and skipped rather than aborting the
    evaluation; if every replicate fails the score falls back to the
    objective's failure score. Returns (and writes to result.json) a dict
    with the score, per-target breakdown, and per-replicate metrics.

    ``interim_callback`` (used by the BO stage for pruning) is called after
    every replicate except the last with the number of replicates completed
    so far and the score those replicates alone would produce; returning
    True stops the evaluation early, recorded as ``pruned_after_replicates``
    in the result. The screening stage must leave this None: Sobol indices
    need every design row fully evaluated.

    Without an interim callback the replicates run concurrently, up to
    REPLICATE_WORKERS at a time. With a callback and multiple workers the
    schedule is hybrid: the first replicate runs alone and the callback
    decides pruning on it (deterministic failures — tripped guards,
    collapse aborts — reveal themselves on any single replicate), then the
    surviving replicates run concurrently with no further pruning
    opportunity. Wall time is ~2 replicates instead of the full sequential
    count. With a single worker the fully sequential per-replicate pruning
    behaviour is kept.
    """
    evaluation_dir = Path(evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    simulation_overrides = scenario.simulation_overrides(values)

    def run_replicate(replicate: int) -> tuple[dict[str, float] | None, str | None]:
        """One replicate: simulate, extract metrics, clean up. Returns the
        metrics dict, or None with the error message on failure."""
        run_dir = evaluation_dir / f"replicate-{replicate}"
        try:
            run_simulation(
                values,
                run_dir,
                seed=base_seed + replicate,
                sim_overrides=simulation_overrides,
            )
            world_width, world_height = scenario.world_size(values)
            return (
                metrics_module.compute_metrics(
                    run_dir / "snapshots",
                    world_width=world_width,
                    world_height=world_height,
                    timestep_duration=scenario.timestep_duration,
                    snapshot_frequency=scenario.snapshot_frequency,
                    # Snapshots already start after burn-in; keep them all.
                    burn_in_fraction=0.0,
                ),
                None,
            )
        except (SimulationError, FileNotFoundError, ValueError) as error:
            return None, str(error)
        finally:
            if not keep_snapshots:
                cleanup_snapshots(run_dir)

    replicate_metrics = []
    failures = []
    pruned_after_replicates = None

    def record(replicate: int, outcome: tuple[dict[str, float] | None, str | None]) -> None:
        metrics, error = outcome
        if metrics is not None:
            replicate_metrics.append(metrics)
        else:
            failures.append({"replicate": replicate, "error": error})

    def run_concurrently(replicate_indices: range) -> None:
        workers = min(REPLICATE_WORKERS, len(replicate_indices))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for replicate, outcome in zip(
                replicate_indices, executor.map(run_replicate, replicate_indices)
            ):
                record(replicate, outcome)

    if interim_callback is None and REPLICATE_WORKERS > 1:
        run_concurrently(range(scenario.replicates))
    elif interim_callback is not None and REPLICATE_WORKERS > 1:
        record(0, run_replicate(0))
        interim_score, _ = score_metrics(replicate_metrics)
        if scenario.replicates > 1 and interim_callback(1, interim_score):
            pruned_after_replicates = 1
        else:
            run_concurrently(range(1, scenario.replicates))
    else:
        for replicate in range(scenario.replicates):
            record(replicate, run_replicate(replicate))
            if interim_callback is not None and replicate < scenario.replicates - 1:
                interim_score, _ = score_metrics(replicate_metrics)
                if interim_callback(replicate + 1, interim_score):
                    pruned_after_replicates = replicate + 1
                    break

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
        "pruned_after_replicates": pruned_after_replicates,
    }
    (evaluation_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    return result
