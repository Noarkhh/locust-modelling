"""Sobol sensitivity screening: which parameters actually move the metrics.

Run before Bayesian optimization to shrink the search space. Three
subcommands, designed around SLURM job arrays:

  sample   generate the Saltelli/Sobol design once, writing samples.csv —
           one row per simulation evaluation — plus the problem definition
  run      evaluate row(s) of samples.csv; each SLURM array task calls this
           with its own index, so the design executes embarrassingly parallel
  analyze  collect all result.json files, compute Sobol indices (SALib) for
           every metric and for the objective score, and write a sensitivity
           ranking to guide which parameters stay active in the BO stage

The design must be evaluated completely (in row order) for the indices to be
valid; failed evaluations are median-imputed and reported, but rerun them if
there are more than a few.

Example:
  python -m paramsearch.screening sample --out screening/ --power 5
  python -m paramsearch.screening run --out screening/ --index $SLURM_ARRAY_TASK_ID
  python -m paramsearch.screening analyze --out screening/
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluation import Scenario, evaluate_point
from .parameters import NEURAL_FIELD, active_parameters, decode_sample, salib_problem


def generate_samples(
    output_dir: Path, model: str, power: int, scenario: Scenario
) -> None:
    """Write the Saltelli design: 2^power * (n_parameters + 2) evaluations."""
    from SALib.sample.sobol import sample

    parameters = active_parameters(model)
    problem = salib_problem(parameters)
    sample_matrix = sample(problem, N=2**power, calc_second_order=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "problem.json").write_text(
        json.dumps({"model": model, "power": power, "problem": problem}, indent=2)
    )
    (output_dir / "scenario.json").write_text(
        json.dumps(scenario.__dict__, indent=2, default=str)
    )
    with open(output_dir / "samples.csv", "w", newline="") as samples_file:
        writer = csv.writer(samples_file)
        writer.writerow(problem["names"])
        writer.writerows(sample_matrix)
    print(
        f"{len(sample_matrix)} evaluations x {scenario.replicates} replicates "
        f"over {len(parameters)} parameters -> {output_dir / 'samples.csv'}"
    )


def run_rows(output_dir: Path, index: int, stride: int) -> None:
    """Evaluate design rows index, index+stride, ... (one call per array task)."""
    rows, parameters, scenario = _load_design(output_dir)
    for row_index in range(index, len(rows), stride):
        evaluation_dir = output_dir / "evaluations" / f"eval-{row_index:05d}"
        if (evaluation_dir / "result.json").exists():
            continue  # already done; makes reruns after node failures cheap
        values = decode_sample(rows[row_index], parameters)
        result = evaluate_point(
            values, evaluation_dir, scenario, base_seed=1000 * (row_index + 1)
        )
        print(
            f"eval-{row_index:05d}: score={result['score']:.3g} "
            f"failures={len(result['failures'])}"
        )


def analyze(output_dir: Path) -> None:
    """Compute first/total-order Sobol indices per metric and rank parameters."""
    from SALib.analyze import sobol as sobol_analyze

    rows, parameters, _ = _load_design(output_dir)
    problem = json.loads((output_dir / "problem.json").read_text())["problem"]

    scores, metric_table = _collect_results(output_dir, len(rows))
    outputs = {"objective_score": scores, **metric_table}

    ranking: dict[str, dict] = {}
    for output_name, values in outputs.items():
        finite = np.isfinite(values)
        if finite.sum() < len(values):
            median = np.median(values[finite]) if finite.any() else 0.0
            values = np.where(finite, values, median)
            print(f"{output_name}: imputed {int((~finite).sum())} missing values")
        indices = sobol_analyze.analyze(
            problem, values, calc_second_order=False, print_to_console=False
        )
        ranking[output_name] = {
            name: {"S1": float(s1), "ST": float(st)}
            for name, s1, st in zip(problem["names"], indices["S1"], indices["ST"])
        }

    (output_dir / "sensitivity.json").write_text(json.dumps(ranking, indent=2))

    # Rank by the maximum total-order index across all outputs: a parameter
    # matters if it moves ANY metric, not just the aggregate score.
    max_total_order = {
        name: max(ranking[output][name]["ST"] for output in ranking)
        for name in problem["names"]
    }
    print(f"\n{'parameter':40s} {'max ST':>8s}   (ST on objective)")
    for name, total_order in sorted(max_total_order.items(), key=lambda item: -item[1]):
        objective_st = ranking["objective_score"][name]["ST"]
        print(f"{name:40s} {total_order:8.3f}   ({objective_st:.3f})")
    print(
        "\nSuggestion: keep parameters with max ST >~ 0.05 for the BO stage; "
        "fix the rest at defaults. Full table: sensitivity.json"
    )


def _load_design(output_dir: Path) -> tuple[np.ndarray, list, Scenario]:
    problem_info = json.loads((output_dir / "problem.json").read_text())
    parameters = active_parameters(problem_info["model"])
    scenario = Scenario(**json.loads((output_dir / "scenario.json").read_text()))
    with open(output_dir / "samples.csv", newline="") as samples_file:
        reader = csv.reader(samples_file)
        next(reader)  # header
        rows = np.array([[float(cell) for cell in row] for row in reader])
    return rows, parameters, scenario


def _collect_results(
    output_dir: Path, row_count: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Gather scores and replicate-mean metrics for every design row, in order."""
    scores = np.full(row_count, np.nan)
    metric_table: dict[str, np.ndarray] = {}
    missing = 0
    for row_index in range(row_count):
        result_file = (
            output_dir / "evaluations" / f"eval-{row_index:05d}" / "result.json"
        )
        if not result_file.exists():
            missing += 1
            continue
        result = json.loads(result_file.read_text())
        scores[row_index] = result["score"]
        replicate_metrics = result["replicate_metrics"]
        if replicate_metrics:
            for key in replicate_metrics[0]:
                mean_value = np.nanmean([metrics[key] for metrics in replicate_metrics])
                metric_table.setdefault(key, np.full(row_count, np.nan))[
                    row_index
                ] = mean_value
    if missing:
        print(
            f"WARNING: {missing}/{row_count} evaluations missing — "
            "indices will be imputed; prefer rerunning the missing rows"
        )
    return scores, metric_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    sample_parser = subcommands.add_parser("sample")
    sample_parser.add_argument("--out", type=Path, required=True)
    sample_parser.add_argument("--model", default=NEURAL_FIELD)
    sample_parser.add_argument(
        "--power", type=int, default=5, help="Saltelli base sample = 2^power"
    )
    sample_parser.add_argument("--replicates", type=int, default=3)

    run_parser = subcommands.add_parser("run")
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--index", type=int, required=True)
    run_parser.add_argument(
        "--stride", type=int, default=1, help="array size when tasks < design rows"
    )

    analyze_parser = subcommands.add_parser("analyze")
    analyze_parser.add_argument("--out", type=Path, required=True)

    arguments = parser.parse_args()
    if arguments.command == "sample":
        scenario = Scenario(model=arguments.model, replicates=arguments.replicates)
        generate_samples(arguments.out, arguments.model, arguments.power, scenario)
    elif arguments.command == "run":
        run_rows(arguments.out, arguments.index, arguments.stride)
    else:
        analyze(arguments.out)


if __name__ == "__main__":
    main()
