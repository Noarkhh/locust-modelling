"""Select stratified candidate sets for the scale-fidelity check.

Draws from completed screening campaigns: the top healthy rows (the region
the optimizer actually explores, where ranking precision matters most),
plus mid-scoring and poor-scoring healthy rows so the rank correlation is
measured over a realistic score spread rather than a compressed one.

  python scripts/select_fidelity_candidates.py \
      --neural-field <screening-nf dir> --spin <screening-spin dir> \
      --out fidelity-candidates.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paramsearch.optimize import warm_start_values
from paramsearch.parameters import NEURAL_FIELD, SPIN, active_parameters

TOP_COUNT = 5
MID_COUNT = 4
POOR_COUNT = 3
POOR_SCORE_RANGE = (150.0, 500.0)


def healthy_rows(campaign_dir: Path) -> list[tuple[float, dict]]:
    """(score, values) for every row whose replicates all succeeded."""
    rows = []
    for result_file in campaign_dir.glob("evaluations/eval-*/result.json"):
        result = json.loads(result_file.read_text())
        if not result["failures"] and result["replicate_metrics"]:
            rows.append((result["score"], result["values"]))
    rows.sort(key=lambda scored: scored[0])
    return rows


def spread_picks(
    rows: list[tuple[float, dict]], low: float, high: float, count: int
) -> list[dict]:
    """Up to ``count`` value dicts with scores in [low, high), spread evenly."""
    pool = [values for score, values in rows if low <= score < high]
    step = max(len(pool) // count, 1)
    return pool[::step][:count]


def select(campaign_dir: Path, model: str) -> list[dict]:
    rows = healthy_rows(campaign_dir)
    scores = [score for score, _ in rows]
    top = warm_start_values(campaign_dir, active_parameters(model), TOP_COUNT)
    mid = spread_picks(
        rows, np.percentile(scores, 35), np.percentile(scores, 65), MID_COUNT
    )
    poor = spread_picks(rows, *POOR_SCORE_RANGE, POOR_COUNT)
    print(
        f"{model}: {len(top)} top + {len(mid)} mid + {len(poor)} poor "
        f"from {len(rows)} healthy rows"
    )
    return top + mid + poor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-field", type=Path, required=True,
                        help="neural-field screening campaign directory")
    parser.add_argument("--spin", type=Path, required=True,
                        help="spin screening campaign directory")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    candidates = {
        NEURAL_FIELD: select(arguments.neural_field, NEURAL_FIELD),
        SPIN: select(arguments.spin, SPIN),
    }
    arguments.out.write_text(json.dumps(candidates, indent=1))
    print(f"written {arguments.out}")


if __name__ == "__main__":
    main()
