"""Density-dilution experiment: top-4 candidates of each model in a band of
FIXED footprint, varying the density (500/250/100 locusts/m^2) by lowering the
population rather than by stretching the world.

The starting patch area is derived as agent_amount / initial_density, so holding
that area constant keeps both the band thickness (initial_patch_width) and the
front length fixed; the world size (which depends only on the candidate's speed
and the fixed patch width) is likewise identical across densities. To vary the
density within that fixed footprint we scale the population: anchoring on the
densest cell (2000 agents at 500/m^2, area 4 m^2), density D uses round(D * area)
agents -- 2000 / 1000 / 400 agents for 500 / 250 / 100 /m^2. This isolates
density (dilution) from geometry, unlike the earlier stretch-the-front design.

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
# Fixed band footprint: anchor on the densest cell (2000 agents at 500/m^2).
# area = agents / density is held constant, so the patch geometry is identical
# across densities and only the population changes.
REF_AGENTS = 2000
REF_DENSITY = 500.0
PATCH_AREA = REF_AGENTS / REF_DENSITY  # m^2, held constant across all cells
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

    # Fixed footprint: agent_amount / initial_density == PATCH_AREA, so the patch
    # geometry (thickness and front length) and the world are identical across
    # densities; lowering the density lowers the population, not the geometry.
    agent_amount = round(density * PATCH_AREA)
    scenario = replace(
        campaign_scenario(trial_data["scenario"]["model"], replicates=arguments.replicates),
        agent_amount=agent_amount,
        initial_density=float(density),
    )
    cell_dir = arguments.out / f"{model_dir.split('-')[1]}-{trial}-d{int(density)}"
    if (cell_dir / "result.json").exists():
        print(f"{cell_dir.name}: skipped (result.json exists)")
        return
    result = evaluate_point(
        trial_data["values"], cell_dir, scenario, base_seed=1000 * (arguments.index + 1)
    )
    print(f"{cell_dir.name}: {agent_amount} agents, score {result['score']:.3g}, "
          f"failures {len(result['failures'])}, "
          f"world {scenario.world_size(trial_data['values'])}, "
          f"patch {scenario.initial_area()}")


if __name__ == "__main__":
    main()
