"""Local SPP (Bach 2018) density experiment, stretch-the-world design.

Runs Bach's validated frontal three-zone SPP parameters at 2000 agents across
densities 750/500/250/100 with the band thickness held fixed and the front
stretched (the SAME stretch design as the original NF/spin density experiment,
i.e. replace(campaign_scenario, agent_amount=2000, initial_density=d)), so the
SPP baseline can be compared cell-for-cell against the ring-attractor models.

  LOCUST_SIM_JAR=<assembly.jar> LOCUST_REPLICATE_WORKERS=<n> \
      python scripts/spp_density_experiment.py --out <dir> [--replicates 5]
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paramsearch.evaluation import campaign_scenario, evaluate_point
from paramsearch.parameters import SPP

# Bach 2018 validated frontal SPP parameter set (the bach-frontal run), i.e. the
# best three-zone parameters reported for the marching-band regime.
BACH_VALUES = {
    "averageSpeed": 0.0025,
    "previousDirectionWeight": 0.6,
    "randomComponentWeight": 0.05,
    "repulsionRange": 0.035,
    "alignmentRange": 0.135,
    "attractionRange": 0.3,
    "repulsionWeight": 0.1,
    "alignmentWeight": 1.5,
    "attractionWeight": 0.0,
    "hopProbability": 0.01,
    "crowdedHopProbability": 0.1,
    "hopSpeed": 0.1,
    "hopDuration": 1.0,
    "minimalInactivityPeriod": 900.0,
    "resumeMarchProbabilityPerSecond": 0.001,
    "occlusionThreshold": 25,
}
DENSITIES = [750.0, 500.0, 250.0, 100.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--densities", type=float, nargs="+", default=DENSITIES)
    parser.add_argument("--iterations", type=int, default=None,
                        help="override iterations_number (short timing runs)")
    arguments = parser.parse_args()

    for density in arguments.densities:
        # stretch design: fix agent_amount + initial_patch_width, lower the
        # density -> the front length grows as 1/density.
        scenario = replace(
            campaign_scenario(SPP, replicates=arguments.replicates),
            agent_amount=2000,
            initial_density=float(density),
        )
        if arguments.iterations is not None:
            scenario = replace(scenario, iterations_number=arguments.iterations)
        cell_dir = arguments.out / f"spp-bach-d{int(density)}"
        if (cell_dir / "result.json").exists():
            print(f"{cell_dir.name}: skipped (result.json exists)")
            continue
        result = evaluate_point(BACH_VALUES, cell_dir, scenario, base_seed=91000)
        print(f"{cell_dir.name}: score {result['score']:.3g}, "
              f"failures {len(result['failures'])}, "
              f"world {scenario.world_size(BACH_VALUES)}, "
              f"patch {scenario.initial_area()}")


if __name__ == "__main__":
    main()
