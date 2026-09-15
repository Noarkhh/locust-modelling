"""Run the held-out validation post-processors over BO trial replicate streams.

  python scripts/run_validation.py <trial_dir>... --out <results.json>

Each trial directory holds replicate-*/{run.json,snapshots/}; both validators
run per replicate (post burn-in) and are aggregated as mean +/- sd per trial.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paramsearch.validation import marking_experiment, neighbour_anisotropy

BURN_IN_ITERATION = 12000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    results = {}
    for trial_dir in arguments.trial_dirs:
        replicates = []
        for replicate_dir in sorted(trial_dir.glob("replicate-*")):
            snapshot_dir = replicate_dir / "snapshots"
            run_file = replicate_dir / "run.json"
            if not run_file.exists() or not any(snapshot_dir.glob("*.bin")):
                continue
            overrides = json.loads(run_file.read_text())["overrides"]
            width = overrides["worldWidthMeters"]
            height = overrides["worldHeightMeters"]
            marking = marking_experiment(
                snapshot_dir, width, height, burn_in_iteration=BURN_IN_ITERATION
            )
            anisotropy = neighbour_anisotropy(
                snapshot_dir, width, height, burn_in_iteration=BURN_IN_ITERATION
            )
            replicates.append({"marking": marking, "anisotropy": anisotropy})
            print(
                f"{trial_dir.name}/{replicate_dir.name}: "
                f"mixing {marking['mixing_index']:.3f}, "
                f"front_back_excess {anisotropy['front_back_excess']:.3f}",
                flush=True,
            )
        mixing = [r["marking"]["mixing_index"] for r in replicates]
        excess = [r["anisotropy"]["front_back_excess"] for r in replicates]
        histograms = np.array(
            [r["anisotropy"]["bearing_histogram"] for r in replicates]
        )
        results[trial_dir.name] = {
            "replicate_count": len(replicates),
            "mixing_index_mean": float(np.mean(mixing)),
            "mixing_index_sd": float(np.std(mixing)),
            "front_back_excess_mean": float(np.mean(excess)),
            "front_back_excess_sd": float(np.std(excess)),
            "bearing_histogram_mean": histograms.mean(axis=0).tolist(),
            "bearing_bin_centres": replicates[0]["anisotropy"][
                "bearing_bin_centres"
            ],
            "cohort_mean_rank_by_replicate": [
                r["marking"]["cohort_mean_rank"] for r in replicates
            ],
            "final_quintile_matrix_by_replicate": [
                r["marking"]["final_quintile_matrix"] for r in replicates
            ],
            "iterations_from_reference": replicates[0]["marking"][
                "iterations_from_reference"
            ],
        }
    arguments.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
