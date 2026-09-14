"""One large-scale validation-ladder simulation of a calibrated candidate.

Runs a single replicate of the candidate stored in a trial's result.json at
a chosen population, with a multi-worker horizontal-slab layout, computes
the full metric set, and writes metrics.json next to the snapshots (which
are always kept — these runs ARE the validation evidence).

  python scripts/large_scale_run.py \
      --trial <path to result.json> --agents 250000 --workers 48 \
      --seed 1 --out <run directory>
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paramsearch import metrics as metrics_module
from paramsearch.evaluation import campaign_scenario
from paramsearch.runner import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True,
                        help="result.json of the calibrated candidate")
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--workers", type=int, default=48,
                        help="workers along the x split (workers_x)")
    parser.add_argument("--workers-y", type=int, default=1,
                        help="workers along the y split (workers_y); the "
                        "2026-09-14 thread-dump diagnosis indicates the "
                        "band-friendly split is along the full-height axis")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="override iterations_number (short diagnostic runs)",
    )
    parser.add_argument(
        "--snapshot-frequency",
        type=int,
        default=None,
        help="override snapshotFrequency (1 = every iteration)",
    )
    parser.add_argument(
        "--snapshot-start",
        type=int,
        default=None,
        help="override snapshotStartIteration (0 = record from launch, "
        "including burn-in — for diagnosing formation and early dynamics); "
        "default keeps the scenario's burn-in start",
    )
    arguments = parser.parse_args()

    trial = json.loads(arguments.trial.read_text())
    scenario = replace(
        campaign_scenario(trial["scenario"]["model"]),
        agent_amount=arguments.agents,
        workers_x=arguments.workers,
        workers_y=arguments.workers_y,
    )
    if arguments.iterations is not None:
        scenario = replace(scenario, iterations_number=arguments.iterations)
    if arguments.snapshot_frequency is not None:
        scenario = replace(scenario, snapshot_frequency=arguments.snapshot_frequency)
    values = trial["values"]
    overrides = scenario.simulation_overrides(values)
    if arguments.snapshot_start is not None:
        overrides["snapshotStartIteration"] = arguments.snapshot_start


    run_simulation(values, arguments.out, seed=arguments.seed, sim_overrides=overrides)
    world_width, world_height = scenario.world_size(values)
    burn_in_fraction = 0.0
    if arguments.snapshot_start is not None and arguments.snapshot_start < scenario.burn_in_iterations:
        burn_in_fraction = (
            (scenario.burn_in_iterations - arguments.snapshot_start)
            / (scenario.iterations_number - arguments.snapshot_start)
        )
    if arguments.iterations is not None and arguments.iterations < scenario.burn_in_iterations:
        print("diagnostic run shorter than burn-in: skipping metric extraction")
        return
    metrics = metrics_module.compute_metrics(
        arguments.out / "snapshots",
        world_width=world_width,
        world_height=world_height,
        timestep_duration=scenario.timestep_duration,
        snapshot_frequency=scenario.snapshot_frequency,
        burn_in_fraction=burn_in_fraction,
    )
    (arguments.out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: metrics[k] for k in (
        "global_order", "heading_travel_alignment", "profile_peak_position",
        "profile_decay_r2", "band_speed_ratio", "band_length", "band_width",
        "stream_count", "transverse_cv")}, indent=1, default=str))


if __name__ == "__main__":
    main()
