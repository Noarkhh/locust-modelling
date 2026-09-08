"""Optuna Bayesian-optimization stage, designed for SLURM job arrays.

Every array task runs this same script as an independent worker. Workers
coordinate through a shared JournalFileStorage on the cluster's shared
filesystem — no database server needed — and each asks the sampler for the
next trial, evaluates it with paramsearch.evaluation (replicate runs of the
Scala simulation), and reports the score. The study is resumable: rerunning
with the same --storage and --study-name continues where it stopped.

After the Sobol screening stage, restrict the space with --only, e.g.
  --only totalSocialAttraction,activityPeriod,crowdedHopProbability
Unlisted parameters stay at their reference.conf defaults.

Example (one worker; SLURM array launches many):
  python -m paramsearch.optimize --out bo/ --study-name locust-nf \
      --trials 50 --only totalSocialAttraction,activityPeriod
"""

import argparse
import json
from pathlib import Path

import optuna
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from .evaluation import Scenario, campaign_scenario, evaluate_point
from .parameters import NEURAL_FIELD, Parameter, active_parameters

# Prune a trial once its interim score (the replicates completed so far)
# reaches this value. One failure-scored target contributes 100 (see
# objective.FAILURE_SCORE), so the threshold means "at least one target in
# outright failure territory" — well clear of the 18-60 single-replicate
# scores the replicate-spread analysis measured for genuinely good
# candidates, whose replicate noise must never trigger pruning.
PRUNE_SCORE_DEFAULT = 100.0


def suggest_values(
    trial: optuna.Trial, parameters: list[Parameter]
) -> dict[str, float]:
    """Ask the sampler for one value per searched parameter.

    Bounds, log-scaling and integrality come from the parameter definitions,
    so the BO stage searches exactly the space the screening stage measured.
    """
    values: dict[str, float] = {}
    for parameter in parameters:
        low, high = parameter.bounds
        if parameter.integer:
            values[parameter.name] = trial.suggest_int(
                parameter.name, int(low), int(high), log=parameter.log
            )
        else:
            values[parameter.name] = trial.suggest_float(
                parameter.name, low, high, log=parameter.log
            )
    return values


def make_objective(parameters: list[Parameter], scenario: Scenario, output_dir: Path):
    """Build the Optuna objective closure around the shared evaluation path."""

    def objective(trial: optuna.Trial) -> float:
        values = suggest_values(trial, parameters)
        evaluation_dir = output_dir / "trials" / f"trial-{trial.number:05d}"

        def report_interim_score(replicates_completed: int, interim_score: float) -> bool:
            trial.report(interim_score, step=replicates_completed)
            return trial.should_prune()

        result = evaluate_point(
            values,
            evaluation_dir,
            scenario,
            # Fresh seeds per trial so no two trials share a replicate draw.
            base_seed=1000 * (trial.number + 1),
            interim_callback=report_interim_score,
        )
        # Persist the diagnosis with the trial so `optuna.load_study` alone
        # can answer "which metric killed this candidate".
        trial.set_user_attr("breakdown", result["breakdown"])
        trial.set_user_attr("failures", len(result["failures"]))
        if result["pruned_after_replicates"] is not None:
            raise optuna.TrialPruned(
                f"score {result['score']:.1f} after "
                f"{result['pruned_after_replicates']} replicate(s)"
            )
        return result["score"]

    return objective


def load_study(
    storage_path: Path, study_name: str, prune_above: float = PRUNE_SCORE_DEFAULT
) -> optuna.Study:
    """Open (or create) the shared study on journal-file storage.

    JournalFileStorage with an OpenLock is safe for concurrent workers on a
    POSIX shared filesystem, which is exactly the SLURM array setup.

    Pruning is a fixed threshold on the interim score, not a cross-trial
    statistic: single-replicate scores of good candidates are too noisy
    (replicate-spread analysis, 2026-09-07) for median-style pruning, while
    a score past ``prune_above`` means a deterministic failure (tripped
    guard or collapsed run) that further replicates cannot redeem.
    ``prune_above <= 0`` disables pruning.
    """
    pruner = (
        optuna.pruners.ThresholdPruner(upper=prune_above)
        if prune_above > 0
        else optuna.pruners.NopPruner()
    )
    storage = optuna.storages.JournalStorage(
        JournalFileBackend(
            str(storage_path),
            lock_obj=JournalFileOpenLock(str(storage_path)),
        )
    )
    sampler = optuna.samplers.TPESampler(
        multivariate=True,  # model parameter interactions jointly
        constant_liar=True,  # avoid duplicate suggestions across parallel workers
        seed=None,  # workers must not propose identical points
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="study directory (journal, trials, summary)",
    )
    parser.add_argument("--study-name", default="locust-calibration")
    parser.add_argument("--model", default=NEURAL_FIELD)
    parser.add_argument(
        "--trials", type=int, default=25, help="trials THIS worker contributes"
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument(
        "--prune-above",
        type=float,
        default=PRUNE_SCORE_DEFAULT,
        help="prune a trial once its interim score reaches this value "
        "(<= 0 disables pruning)",
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated parameter names to search "
        "(post-screening restriction); empty = all active",
    )
    arguments = parser.parse_args()

    parameters = active_parameters(arguments.model)
    if arguments.only:
        requested = set(arguments.only.split(","))
        unknown = requested - {parameter.name for parameter in parameters}
        if unknown:
            parser.error(
                f"unknown parameters for model {arguments.model}: {sorted(unknown)}"
            )
        parameters = [
            parameter for parameter in parameters if parameter.name in requested
        ]

    scenario = campaign_scenario(arguments.model, replicates=arguments.replicates)
    arguments.out.mkdir(parents=True, exist_ok=True)
    study = load_study(
        arguments.out / "journal.log", arguments.study_name, arguments.prune_above
    )
    study.optimize(
        make_objective(parameters, scenario, arguments.out),
        n_trials=arguments.trials,
        gc_after_trial=True,
    )

    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        print("no completed trials (all pruned or failed); no summary written")
        return
    best = study.best_trial
    summary = {
        "best_score": best.value,
        "best_values": best.params,
        "best_breakdown": best.user_attrs.get("breakdown", {}),
        "completed_trials": len(completed),
        "pruned_trials": sum(
            trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials
        ),
    }
    (arguments.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
