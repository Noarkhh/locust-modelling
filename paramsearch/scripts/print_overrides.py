"""Print the resolved -D override arguments for a distributed ladder run.

Used by slurm/large_scale_distributed.sbatch: the scenario/geometry logic
stays in Python (single source: evaluation.Scenario), the multi-node JVM
launch stays in the sbatch. Prints one -D...=... token per line; the world
dimensions are recoverable from the tokens.

  python scripts/print_overrides.py --trial <result.json> --agents 1000000 \
      --workers-y 384 [--snapshot-start 0] [--iterations N]
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paramsearch.evaluation import campaign_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--workers-y", type=int, required=True)
    parser.add_argument("--snapshot-start", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()

    trial = json.loads(arguments.trial.read_text())
    scenario = replace(
        campaign_scenario(trial["scenario"]["model"]),
        agent_amount=arguments.agents,
        workers_x=1,
        workers_y=arguments.workers_y,
    )
    if arguments.iterations is not None:
        scenario = replace(scenario, iterations_number=arguments.iterations)
    overrides = scenario.simulation_overrides(trial["values"])
    overrides.update(trial["values"])
    overrides["randomSeed"] = arguments.seed
    if arguments.snapshot_start is not None:
        overrides["snapshotStartIteration"] = arguments.snapshot_start
    overrides["guiType"] = "none"
    overrides["iterationFinishedLogFrequency"] = 1000
    for key, value in overrides.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        print(f"-Dparticle-agent.config.{key}={value}")


if __name__ == "__main__":
    main()
