"""A5: scale-fidelity check — do reduced-scale scores rank like full-scale?

Draws parameter sets across each model's search box, evaluates every set at
the campaign (search) scale and at a larger reference scale, and reports the
Spearman rank correlation of the scores per model. High correlation is the
justification for running the campaign at reduced scale.

Designed for a SLURM array: `run --index N` evaluates one (model, set, scale)
cell idempotently; `analyze` collects and correlates.

  python scripts/scale_fidelity.py run --index $SLURM_ARRAY_TASK_ID
  python scripts/scale_fidelity.py analyze
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paramsearch.evaluation import (evaluate_point, neural_field_band_scenario,
                                    spp_band_scenario)
from paramsearch.parameters import (NEURAL_FIELD, SPP, active_parameters,
                                    decode_sample, salib_problem)

OUTPUT_DIR = Path(os.environ.get("FIDELITY_DIR", "runs/scale-fidelity"))
SETS_PER_MODEL = 10
SAMPLING_SEED = 7

# (model, preset factory, search-scale agents, reference-scale agents,
#  replicates at search / reference scale). Reference scales chosen to fit a
# 12 h task: NF 10x, SPP 5x.
PLANS = [
    (NEURAL_FIELD, neural_field_band_scenario, 2000, 20000, 3, 2),
    (SPP, spp_band_scenario, 10000, 50000, 3, 2),
]


def parameter_sets(model: str) -> list[dict]:
    parameters = active_parameters(model)
    problem = salib_problem(parameters)
    bounds = np.array(problem["bounds"])
    rng = np.random.default_rng(SAMPLING_SEED)
    samples = bounds[:, 0] + rng.random((SETS_PER_MODEL, len(parameters))) * (
        bounds[:, 1] - bounds[:, 0]
    )
    return [decode_sample(row, parameters) for row in samples]


def cells() -> list[tuple]:
    """Flat task list: one cell per (model, set index, scale)."""
    result = []
    for model, factory, search_n, full_n, search_reps, full_reps in PLANS:
        for set_index, values in enumerate(parameter_sets(model)):
            for scale_name, agents, replicates in (
                ("search", search_n, search_reps), ("full", full_n, full_reps)
            ):
                result.append((model, factory, set_index, values,
                               scale_name, agents, replicates))
    return result


def run(index: int, workers: int) -> None:
    model, factory, set_index, values, scale_name, agents, replicates = cells()[index]
    scenario = factory(agent_amount=agents, replicates=replicates)
    import dataclasses
    scenario = dataclasses.replace(scenario, workers_x=1, workers_y=workers)
    cell_dir = OUTPUT_DIR / model / f"set-{set_index:02d}" / scale_name
    if (cell_dir / "result.json").exists():
        print(f"index {index}: already done, skipping")
        return
    result = evaluate_point(values, cell_dir, scenario,
                            base_seed=10_000 * (set_index + 1))
    print(f"index {index} ({model} set-{set_index} {scale_name}): "
          f"score={result['score']:.2f} failures={len(result['failures'])}")


def analyze() -> None:
    from scipy.stats import spearmanr

    for model, *_ in PLANS:
        search_scores, full_scores, missing = [], [], 0
        for set_index in range(SETS_PER_MODEL):
            pair = []
            for scale_name in ("search", "full"):
                path = OUTPUT_DIR / model / f"set-{set_index:02d}" / scale_name / "result.json"
                if not path.exists():
                    missing += 1
                    break
                pair.append(json.loads(path.read_text())["score"])
            else:
                search_scores.append(pair[0])
                full_scores.append(pair[1])
        if len(search_scores) < 3:
            print(f"{model}: only {len(search_scores)} complete pairs ({missing} cells missing)")
            continue
        correlation, p_value = spearmanr(search_scores, full_scores)
        print(f"{model}: Spearman rho={correlation:.3f} (p={p_value:.3g}, "
              f"n={len(search_scores)} pairs, {missing} cells missing)")
        for i, (a, b) in enumerate(zip(search_scores, full_scores)):
            print(f"  set-{i:02d}: search={a:9.2f} full={b:9.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    run_parser = subcommands.add_parser("run")
    run_parser.add_argument("--index", type=int, required=True)
    run_parser.add_argument("--workers", type=int, default=8)
    subcommands.add_parser("analyze")
    subcommands.add_parser("count")
    arguments = parser.parse_args()
    if arguments.command == "run":
        run(arguments.index, arguments.workers)
    elif arguments.command == "analyze":
        analyze()
    else:
        print(len(cells()))


if __name__ == "__main__":
    main()
