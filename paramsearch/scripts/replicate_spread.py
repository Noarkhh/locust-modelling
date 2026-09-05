"""A4: replicate-spread measurement at the campaign scenarios.

Three parameter sets per model (good / marginal / poor) x 5 seeds, run at the
exact campaign presets (2 xinuk workers per run, three evaluations of a model
concurrently). Output: per-metric replicate std vs the objective's tolerance
scale, and the per-seed score spread — the evidence for (or against)
replicates=3 and the current tolerance choices.
"""

import dataclasses
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paramsearch.evaluation import (evaluate_point, neural_field_band_scenario,
                                    spp_band_scenario)
from paramsearch.parameters import NEURAL_FIELD, SPP, active_parameters

OUTPUT_DIR = Path(os.environ.get("SPREAD_DIR", "runs/replicate-spread"))


def neural_field_cases() -> list[tuple[str, dict]]:
    base = {p.name: p.default for p in active_parameters(NEURAL_FIELD)}
    good = dict(base, averageSpeed=0.1, hopSpeed=0.5, totalSocialAttraction=0.72,
                antiGoalStimulusStrength=0.72)
    marginal = dict(base, averageSpeed=0.1, hopSpeed=0.5, totalSocialAttraction=0.72,
                    antiGoalStimulusStrength=0.05)
    poor = dict(base, averageSpeed=0.1, hopSpeed=0.5, totalSocialAttraction=0.24,
                antiGoalStimulusStrength=0.01)
    return [("nf-good", good), ("nf-marginal", marginal), ("nf-poor", poor)]


def spp_cases() -> list[tuple[str, dict]]:
    base = {p.name: p.default for p in active_parameters(SPP)}
    bach = dict(base, averageSpeed=0.0025, previousDirectionWeight=0.6,
                randomComponentWeight=0.05, repulsionRange=0.035, alignmentRange=0.135,
                attractionRange=0.3, repulsionWeight=0.1, alignmentWeight=1.5,
                attractionWeight=1e-4, hopProbability=0.01, crowdedHopProbability=0.1,
                hopSpeed=0.1, hopDuration=1.0, activityPeriod=2700.0,
                minimalInactivityPeriod=900.0, resumeMarchProbabilityPerSecond=0.001,
                occlusionThreshold=25)
    columnar = dict(bach, crowdedHopProbability=0.3, minimalInactivityPeriod=90.0,
                    resumeMarchProbabilityPerSecond=0.5)
    poor = dict(base, averageSpeed=0.0025, alignmentWeight=0.1, attractionWeight=0.05)
    return [("spp-bach", bach), ("spp-columnar", columnar), ("spp-poor", poor)]


def run_case(arguments: tuple) -> tuple[str, dict]:
    name, values, scenario = arguments
    result = evaluate_point(values, OUTPUT_DIR / name, scenario, base_seed=87000)
    return name, result


def main() -> None:
    jobs = []
    nf_scenario = dataclasses.replace(neural_field_band_scenario(replicates=5),
                                      workers_y=2)
    for name, values in neural_field_cases():
        jobs.append((name, values, nf_scenario))
    spp_scenario = dataclasses.replace(spp_band_scenario(replicates=5), workers_y=2)
    for name, values in spp_cases():
        jobs.append((name, values, spp_scenario))

    summaries = {}
    # Concurrent evaluations, each using 2 xinuk workers: cores needed =
    # 2 x SPREAD_WORKERS. Default 3 fits a laptop; on the cluster set
    # SPREAD_WORKERS=6 (12 cores) to run all six cases at once.
    max_workers = int(os.environ.get("SPREAD_WORKERS", 3))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for name, result in pool.map(run_case, jobs):
            summaries[name] = {
                "score": result["score"],
                "failures": len(result["failures"]),
                "replicate_metrics": result["replicate_metrics"],
            }
            print(f"{name}: score={result['score']:.2f} "
                  f"failures={len(result['failures'])}", flush=True)

    (OUTPUT_DIR / "spread-summary.json").write_text(json.dumps(summaries, indent=2))
    print("done -> spread-summary.json", flush=True)


if __name__ == "__main__":
    main()
