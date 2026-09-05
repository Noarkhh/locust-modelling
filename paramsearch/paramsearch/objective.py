"""Combine simulation metrics into the scalar objective minimized by Optuna.

The objective is a synthetic-likelihood chi-square: for each target, the
squared discrepancy between the replicate-mean metric and its empirical
target, normalized by a scale that blends the target's intrinsic tolerance
with the between-replicate standard deviation. Guard targets (``upper`` /
``lower``) are one-sided penalties: zero anywhere inside the allowed region,
quadratic outside — they exclude pathological regimes (aggregation collapse,
band evaporation) without distorting the optimum inside the valid region.

TARGETS ships with PROVISIONAL values assembled from Buhl et al. (2011)-style
field figures; replace them with the numbers you extract from the papers
before a production search. Per-target breakdowns are always returned so a
failed candidate can be attributed to specific metrics.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

TargetKind = Literal["match", "upper", "lower"]


@dataclass(frozen=True)
class Target:
    """One term of the objective.

    metric  key into the dict produced by metrics.compute_metrics
    value   empirical target ("match") or bound ("upper"/"lower")
    scale   discrepancy that counts as one standard unit; sets the floor of
            the normalization so a metric with tiny replicate variance cannot
            dominate the sum
    kind    "match" = two-sided fit, "upper"/"lower" = one-sided guard
    weight  relative importance multiplier
    """

    metric: str
    value: float
    scale: float
    kind: TargetKind = "match"
    weight: float = 1.0


# PROVISIONAL targets — replace `value`/`scale` with figures extracted from
# the field papers before a production search.
TARGETS = [
    # Morphology. The profile target tests only the FACT of an exponential
    # rearward decay (Buhl et al. 2011 frontal-band signature): a one-sided
    # threshold on the log-linear fit quality, zero penalty above it — R^2 of
    # 0.97 is not "better" than 0.93, and rewarding higher R^2 would favour
    # dense low-noise profiles.
    # NOTE: elongation is deliberately untargeted for now — a target > 1
    # encodes columnar shapes while APL frontal bands sit < 1, so it needs a
    # formation-specific value from the field papers before targeting.
    Target("profile_decay_r2", value=0.9, scale=0.05, kind="lower"),
    # Kinematics: bands travel 3-4x slower than the individuals marching in
    # them (Uvarov 1977, quoted in Ariel & Ayali's review: individual hopper
    # speed relates to the band's marching rate "by a factor of 3-4"), i.e.
    # a ratio of 0.25-0.33. VALIDATED against that source, not provisional.
    Target("band_speed_ratio", value=0.3, scale=0.05),
    # marching_fraction is deliberately untargeted: its field value is the
    # least certain (Uvarov: "as low as 10%" at any moment), it is a
    # near-deterministic function of the intermittency parameters rather
    # than an emergent observable, and band_speed_ratio already penalizes
    # its consequences. It stays in the metrics as a diagnostic.
    # Collective order: marching bands are highly aligned.
    Target("global_order", value=0.9, scale=0.1),
    # Guards: exclude aggregation collapse without rewarding any particular
    # density inside the valid region.
    Target("local_density_p99", value=1000.0, scale=200.0, kind="upper"),
    Target("nn_distance_p5", value=0.005, scale=0.002, kind="lower"),
    Target("area_per_agent_trend", value=-1e-4, scale=5e-5, kind="lower"),
    # Guard against the opposite failure: the band evaporating into vapor.
    Target("area_per_agent", value=0.05, scale=0.02, kind="upper"),
]

# Per-target score when a metric is non-finite (e.g. no fittable density
# profile) or every replicate crashed. Equivalent to a 10-sigma miss: clearly
# worse than any acceptable candidate, but not a cliff that erases the
# ordering among failing candidates and starves the optimizer of signal.
FAILURE_SCORE = 100.0


def evaluate(
    replicate_metrics: list[dict[str, float]],
    targets: list[Target] | None = None,
) -> tuple[float, dict[str, float]]:
    """Score a parameter set from its replicate runs' metric dicts.

    Averages each metric over replicates, computes every target's normalized
    squared discrepancy (replicate scatter widens the normalization, so noisy
    metrics are automatically down-weighted), and sums them. Returns the
    scalar score and a per-target breakdown for logging; a non-finite metric
    yields FAILURE_SCORE for that term.
    """
    targets = targets if targets is not None else TARGETS
    breakdown: dict[str, float] = {}
    for target in targets:
        values = np.array([metrics.get(target.metric, np.nan) for metrics in replicate_metrics])
        breakdown[target.metric] = _score_target(target, values)
    return sum(breakdown.values()), breakdown


def _score_target(target: Target, replicate_values: np.ndarray) -> float:
    """Normalized squared discrepancy of one target given replicate values."""
    finite = replicate_values[np.isfinite(replicate_values)]
    if len(finite) == 0:
        return FAILURE_SCORE * target.weight
    mean = float(finite.mean())
    replicate_std = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
    normalization = math.hypot(target.scale, replicate_std)

    if target.kind == "match":
        discrepancy = mean - target.value
    elif target.kind == "upper":
        discrepancy = max(0.0, mean - target.value)
    else:  # "lower"
        discrepancy = max(0.0, target.value - mean)
    return target.weight * (discrepancy / normalization) ** 2
