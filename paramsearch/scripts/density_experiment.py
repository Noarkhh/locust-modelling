"""Density-variation experiment: top-4 candidates of each model at 2000 agents,
varying initial density (500/250/100 locusts/m^2) with the band thickness held
constant, so only the front length changes.

The scenario fixes ``initial_patch_width`` (along-march thickness) and derives
the patch height (front length) as agent_amount / (density * patch_width); with
agent_amount fixed at 2000, lowering the density stretches the front while the
thickness stays put. Everything else follows the campaign preset (36000
iterations, one-cycle burn-in, 5 replicates).

  python scripts/density_experiment.py --index $SLURM_ARRAY_TASK_ID \
      --out <dir> --bo-root <$SCRATCH/paramsearch-runs>
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paramsearch.evaluation import campaign_scenario, evaluate_point

DENSITIES = [500.0, 250.0, 100.0]
TRIALS = {
    "bo2-nf": ["00756", "01030", "01273", "00886"],
    "bo2-spin": ["00514", "00305", "00457", "00682"],
}
CELLS = [
    (model_dir, trial, density)
    for model_dir, trials in TRIALS.items()
    for trial in trials
    for density in DENSITIES
]  # 24 cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True, help="cell 0..23")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bo-root", type=Path, required=True,
                        help="dir holding bo2-nf/ and bo2-spin/ trial result.json")
    parser.add_argument("--replicates", type=int, default=5)
    arguments = parser.parse_args()

    if arguments.index >= len(CELLS):
        print(f"index {arguments.index} past the {len(CELLS)}-cell design; nothing to do")
        return
    model_dir, trial, density = CELLS[arguments.index]
    trial_file = arguments.bo_root / model_dir / "trials" / f"trial-{trial}" / "result.json"
    trial_data = json.loads(trial_file.read_text())

    # campaign preset, but only the initial density changes; initial_patch_width
    # (thickness) stays at the preset value, so the front length scales with 1/density.
    scenario = replace(
        campaign_scenario(trial_data["scenario"]["model"], replicates=arguments.replicates),
        agent_amount=2000,
        initial_density=float(density),
    )
    cell_dir = arguments.out / f"{model_dir.split('-')[1]}-{trial}-d{int(density)}"
    if (cell_dir / "result.json").exists():
        print(f"{cell_dir.name}: skipped (result.json exists)")
        return
    result = evaluate_point(
        trial_data["values"], cell_dir, scenario, base_seed=1000 * (arguments.index + 1)
    )
    print(f"{cell_dir.name}: score {result['score']:.3g}, "
          f"failures {len(result['failures'])}, "
          f"world {scenario.world_size(trial_data['values'])}")


if __name__ == "__main__":
    main()
