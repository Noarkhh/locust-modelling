"""Combine simulation metrics into the scalar objective minimized by Optuna.

The objective is a history-matching-style implausibility sum: for each
target, the squared discrepancy between the replicate-mean metric and its
empirical target, normalized by the target's fixed tolerance scale. Guard
targets (``upper`` / ``lower``) are one-sided penalties: zero anywhere
inside the allowed region, quadratic outside — they exclude pathological
regimes (aggregation collapse, band evaporation) without distorting the
optimum inside the valid region.

TARGETS ships with PROVISIONAL values assembled from Buhl et al. (2011)-style
field figures; replace them with the numbers you extract from the papers
before a production search. Per-target breakdowns are always returned so a
failed candidate can be attributed to specific metrics.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

TargetKind = Literal["match", "upper", "lower"]


@dataclass(frozen=True)
class Target:
    """One term of the objective.

    metric  key into the dict produced by metrics.compute_metrics
    value   empirical target ("match") or bound ("upper"/"lower")
    scale   discrepancy that counts as one standard unit. The sole
            normalization: candidate replicate scatter is deliberately kept
            OUT of the denominator — a candidate-dependent width lets
            erratic dynamics widen their own tolerance (variance directly
            improving the score), an exploit observed in both screening and
            BO leaderboards. Campaign-wide replicate noise (measured: well
            below every scale) is treated as absorbed into the scale.
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
    # Per-snapshot mean fit quality (redefined 2026-09-12: the pooled fit
    # manufactured smooth exponentials from time-averaging — 0.98 on runs
    # with visibly non-exponential instantaneous profiles). ASPIRATIONAL
    # target 1.0 (2026-09-12, user decision): the literature states the
    # decay is exponential, and a log-linear R^2 cannot sharply separate
    # exponential from other monotone decays at a reachable threshold —
    # so the target is perfection itself. No candidate satisfies it; every
    # candidate pays proportionally to its distance, so the optimizer is
    # always pulled toward more exponential profiles (exp ~0.99 pays ~0,
    # linear ~0.92 pays ~0.6, bumpy ~0.5 pays ~25).
    Target("profile_decay_r2", value=1.0, scale=0.03, kind="lower"),
    # Peak location: the frontal-band signature is a dense front with the
    # mass trailing BEHIND it, i.e. the profile peak near the band's leading
    # edge (Buhl et al. 2011). The metric is the time-averaged band-proper
    # rank of the densest slice (peak within the p5-p95 extent): ~0.5 =
    # symmetric blob or peak wandering between relay stations, ~1 = peak
    # pinned at the front. Reinstated 2026-09-10 after symmetric blobs
    # topped the BO leaderboard; redefined the same day (the pooled
    # extent-based version rewarded long straggler trails instead of
    # frontal structure). PROVISIONAL value/scale: re-anchor against the
    # redefined metric on validated frontal runs before the next campaign.
    Target("profile_peak_position", value=1.0, scale=0.1, kind="lower"),
    # Kinematics: bands travel 3-4x slower than their marching individuals.
    # MEASURED, not just quoted: Telenga 1930 (via Uvarov 1977, p. 173) —
    # Schistocerca instar I bands 25 vs 100 cm/min individual (ratio 0.25),
    # instar V 333 vs 1000 cm/min (ratio 0.33). Mechanism per the same
    # page: marching fraction, "as low as 10%" (Ellis & Ashall 1957). The metric's denominator
    # (individual_marching_rate) mirrors Telenga's method: distance covered
    # by a marching hopper per minute-scale window. Caveats: Schistocerca,
    # instar-dependent, and Uvarov notes band speed also varies with band
    # size, temperature, and terrain.
    Target("band_speed_ratio", value=0.3, scale=0.05),
    # marching_fraction is deliberately untargeted: its field value is the
    # least certain (Uvarov: "as low as 10%" at any moment), it is a
    # near-deterministic function of the intermittency parameters rather
    # than an emergent observable, and band_speed_ratio already penalizes
    # its consequences. It stays in the metrics as a diagnostic.
    # Alignment: marching bands are highly aligned WITH their direction of
    # travel. heading_travel_alignment (mean projection of moving agents'
    # headings onto the realized COM travel direction) replaced the plain
    # global_order target on 2026-09-11: it penalizes the same heading
    # dispersion (alignment = order x cos(mean-heading-vs-travel angle))
    # while additionally failing non-translating clumps (NaN -> failure;
    # order alone scored coherently-pointing frozen marchers as excellent)
    # and coherent sideways/milling motion, which the band_speed_ratio
    # target otherwise rewards. global_order stays as a diagnostic; the
    # difference between the two isolates the misalignment angle.
    # ASPIRATIONAL target 1.0 (2026-09-12, user decision, same rationale
    # as profile_decay_r2): perfect travel-alignment is unreachable, so
    # every candidate is pulled toward tighter alignment in proportion to
    # its dispersion instead of coasting once past a threshold.
    Target("heading_travel_alignment", value=1.0, scale=0.05),
    # Guards: exclude aggregation collapse without rewarding any particular
    # density inside the valid region.
    Target("local_density_p99", value=1500.0, scale=200.0, kind="upper"),
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
    squared discrepancy against its fixed tolerance scale, and sums them.
    Returns the scalar score and a per-target breakdown for logging; a
    non-finite metric yields FAILURE_SCORE for that term.
    """
    targets = targets if targets is not None else TARGETS
    breakdown: dict[str, float] = {}
    for target in targets:
        values = np.array(
            [metrics.get(target.metric, np.nan) for metrics in replicate_metrics]
        )
        breakdown[target.metric] = _score_target(target, values)
    return sum(breakdown.values()), breakdown


def _score_target(target: Target, replicate_values: np.ndarray) -> float:
    """Normalized squared discrepancy of one target given replicate values."""
    finite = replicate_values[np.isfinite(replicate_values)]
    if len(finite) == 0:
        return FAILURE_SCORE * target.weight
    mean = float(finite.mean())
    normalization = target.scale

    if target.kind == "match":
        discrepancy = mean - target.value
    elif target.kind == "upper":
        discrepancy = max(0.0, mean - target.value)
    else:  # "lower"
        discrepancy = max(0.0, target.value - mean)
    # Cap at the failure penalty: the chi-square is unbounded above, and a
    # single blown metric (screening saw band_speed_ratio ~15 -> a score of
    # ~80000) would otherwise dominate any variance-based analysis and stretch
    # the score axis meaninglessly. Beyond a 10-sigma miss, "how much worse"
    # carries no calibration information.
    return min(
        target.weight * (discrepancy / normalization) ** 2,
        target.weight * FAILURE_SCORE,
    )
