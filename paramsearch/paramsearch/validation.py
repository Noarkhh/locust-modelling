"""Held-out validation post-processors, computed from stored snapshot streams.

Two literature-anchored observables, both untouched by calibration:

- Marking experiment (Ashall & Ellis 1962; Bach 2018): cohorts marked by
  their initial position along the march axis are tracked over time. Field
  bands show full redistribution — front individuals are later found at
  the back and vice versa (the "rolling structure"): actively marching
  hoppers overtake to the front, rest, and are left behind.
- Neighbour-bearing anisotropy (Buhl et al., field group structure):
  locusts keep nearest neighbours preferentially in front of and behind
  themselves rather than to the sides.

Both operate on the binary snapshot streams (per-agent ids are stable), so
they apply retroactively to any retained run.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from .metrics import SNAPSHOT_DTYPE, _circular_center_of_mass, load_snapshots


def _band_frame_along(
    snapshot: np.ndarray, world_size: tuple[float, float]
) -> np.ndarray:
    """Along-march coordinates of a snapshot's agents (band frame)."""
    positions = (
        np.column_stack([snapshot["x"], snapshot["y"]]).astype(np.float64) % world_size
    )
    center = _circular_center_of_mass(positions, world_size)
    world = np.array(world_size)
    relative = (positions - center + world / 2) % world - world / 2
    headings = snapshot["heading"].astype(np.float64)
    moving = ((snapshot["flags"] & 1) != 0) & (snapshot["speed"] > 0)
    reference = headings[moving] if moving.any() else headings
    forward = np.array([np.cos(reference).mean(), np.sin(reference).mean()])
    forward /= max(np.linalg.norm(forward), 1e-12)
    return relative @ forward


def marking_experiment(
    snapshot_dir: str | Path,
    world_width: float,
    world_height: float,
    cohort_quantiles: tuple[tuple[float, float], ...] = (
        (0.0, 0.2),
        (0.4, 0.6),
        (0.8, 1.0),
    ),
    burn_in_iteration: int = 0,
) -> dict:
    """Track position-rank distributions of initially-marked cohorts.

    At the reference frame, agents are 'marked' into cohorts by the
    quantile band of their along-march position (0 = rearmost). For every
    subsequent frame each cohort's mean position rank and its final
    quintile occupancy are computed. Full redistribution (all cohorts'
    mean ranks converging toward 0.5, final quintile rows near-uniform)
    is the field signature; rank preservation falsifies the rolling
    structure for that candidate.

    Returns a dict with per-cohort mean-rank time series, the final
    redistribution matrix (cohort x final quintile), and a scalar
    ``mixing_index``: 1 - mean |final mean rank - 0.5| / initial
    |mean rank - 0.5| over cohorts (1 = fully mixed, 0 = ranks preserved).
    """
    records = load_snapshots(snapshot_dir)
    iterations = np.unique(records["iter"])
    iterations = iterations[iterations >= burn_in_iteration]
    world_size = (world_width, world_height)

    reference = records[records["iter"] == iterations[0]]
    reference = reference[np.argsort(reference["id"])]
    along = _band_frame_along(reference, world_size)
    ranks = np.argsort(np.argsort(along)) / max(len(along) - 1, 1)
    cohorts = {
        f"cohort_{int(low*100):02d}_{int(high*100):02d}": reference["id"][
            (ranks >= low) & (ranks < high) | ((high == 1.0) & (ranks == 1.0))
        ]
        for low, high in cohort_quantiles
    }

    time_series: dict[str, list[float]] = {name: [] for name in cohorts}
    times_seconds: list[float] = []
    final_matrix: dict[str, list[float]] = {}
    for index, iteration in enumerate(iterations):
        snapshot = records[records["iter"] == iteration]
        snapshot = snapshot[np.argsort(snapshot["id"])]
        along = _band_frame_along(snapshot, world_size)
        frame_ranks = np.argsort(np.argsort(along)) / max(len(along) - 1, 1)
        snapshot_ids = snapshot["id"]
        times_seconds.append(float(iteration - iterations[0]))
        for name, ids in cohorts.items():
            positions = np.searchsorted(snapshot_ids, ids)
            positions = positions[positions < len(snapshot_ids)]
            present = snapshot_ids[positions] == ids[: len(positions)]
            cohort_ranks = frame_ranks[positions[present]]
            time_series[name].append(float(cohort_ranks.mean()))
            if index == len(iterations) - 1:
                final_matrix[name] = [
                    float(((cohort_ranks >= q / 5) & (cohort_ranks < (q + 1) / 5)).mean())
                    for q in range(5)
                ]

    initial_offsets = [abs(series[0] - 0.5) for series in time_series.values()]
    final_offsets = [abs(series[-1] - 0.5) for series in time_series.values()]
    mixing_index = 1.0 - float(np.mean(final_offsets)) / max(
        float(np.mean(initial_offsets)), 1e-12
    )
    return {
        "iterations_from_reference": times_seconds,
        "cohort_mean_rank": time_series,
        "final_quintile_matrix": final_matrix,
        "mixing_index": mixing_index,
    }


def neighbour_anisotropy(
    snapshot_dir: str | Path,
    world_width: float,
    world_height: float,
    neighbour_count: int = 4,
    bearing_bins: int = 12,
    frame_stride: int = 5,
    burn_in_iteration: int = 0,
) -> dict:
    """Angular distribution of nearest-neighbour bearings relative to heading.

    For every ``frame_stride``-th snapshot and every agent, the bearings of
    its ``neighbour_count`` nearest neighbours (periodic KD-tree) are taken
    relative to the agent's own heading and histogrammed (bin 0 centred on
    dead ahead). Field bands show excess density ahead and behind the focal
    individual relative to the sides (Buhl et al. group structure).

    Returns the normalized bearing histogram, its bin centres [rad], and a
    scalar ``front_back_excess``: (mean density in the two bins around 0
    and pi) / (mean density in the two bins around +/- pi/2) — 1 =
    isotropic, > 1 = field-like fore-aft alignment of neighbours.
    """
    records = load_snapshots(snapshot_dir)
    iterations = np.unique(records["iter"])
    iterations = iterations[iterations >= burn_in_iteration][::frame_stride]
    world_size = (world_width, world_height)
    histogram = np.zeros(bearing_bins)
    edges = np.linspace(-np.pi, np.pi, bearing_bins + 1)

    for iteration in iterations:
        snapshot = records[records["iter"] == iteration]
        positions = (
            np.column_stack([snapshot["x"], snapshot["y"]]).astype(np.float64)
            % world_size
        )
        tree = KDTree(positions, boxsize=world_size)
        _, neighbour_indices = tree.query(positions, k=neighbour_count + 1)
        headings = snapshot["heading"].astype(np.float64)
        world = np.array(world_size)
        for k in range(1, neighbour_count + 1):
            offsets = (
                positions[neighbour_indices[:, k]] - positions + world / 2
            ) % world - world / 2
            bearings = np.arctan2(offsets[:, 1], offsets[:, 0]) - headings
            bearings = (bearings + np.pi) % (2 * np.pi) - np.pi
            histogram += np.histogram(bearings, bins=edges)[0]

    histogram /= max(histogram.sum(), 1e-12)
    centres = (edges[:-1] + edges[1:]) / 2
    def bins_near(angle: float) -> np.ndarray:
        distance = np.abs((centres - angle + np.pi) % (2 * np.pi) - np.pi)
        return histogram[np.argsort(distance)[:2]]
    fore_aft = float(np.mean(np.concatenate([bins_near(0.0), bins_near(np.pi)])))
    lateral = float(np.mean(np.concatenate([bins_near(np.pi / 2), bins_near(-np.pi / 2)])))
    return {
        "bearing_bin_centres": centres.tolist(),
        "bearing_histogram": histogram.tolist(),
        "front_back_excess": fore_aft / max(lateral, 1e-12),
    }


def marking_experiment_v2(
    snapshot_dir: str | Path,
    world_width: float,
    world_height: float,
    front_band: tuple[float, float] = (0.7, 0.9),
    rear_band: tuple[float, float] = (0.1, 0.3),
    sample_fraction: float = 0.10,
    sample_cap: int = 1000,
    burn_in_iteration: int = 0,
    seed: int = 0,
) -> dict:
    """Cohort-redistribution mixing measured from sub-sampled band bands.

    A variant of :func:`marking_experiment` that (a) marks cohorts from
    interior position bands rather than the extreme tails — the frontmost
    ``front_band`` and rearmost ``rear_band`` quantile ranges of the
    along-march position rank (0 = rear, 1 = front), excluding the outer
    10% at each end — and (b) tracks a fixed random sub-sample of
    ``min(sample_fraction * N, sample_cap)`` marked individuals per cohort,
    so the cost is bounded at large populations.

    At the first post-burn-in frame each cohort's members are fixed by id;
    for every subsequent frame the mean and standard deviation of their
    position rank are recorded. Mixing shows up as the cohort means drifting
    toward 0.5 and the standard deviations growing.

    Returns the per-cohort mean/std rank time series and a signed scalar
    ``cohort_separation``: the final ``front_mean - rear_mean`` (~0.6 at the
    reference, → 0 under full redistribution, < 0 if the cohorts cross over).
    """
    records = load_snapshots(snapshot_dir)
    iterations = np.unique(records["iter"])
    iterations = iterations[iterations >= burn_in_iteration]
    world_size = (world_width, world_height)

    reference = records[records["iter"] == iterations[0]]
    reference = reference[np.argsort(reference["id"])]
    along = _band_frame_along(reference, world_size)
    ranks = np.argsort(np.argsort(along)) / max(len(along) - 1, 1)
    population = len(reference)
    sample_size = min(int(sample_fraction * population), sample_cap)
    generator = np.random.default_rng(seed)

    def sample(band: tuple[float, float]) -> np.ndarray:
        low, high = band
        pool = reference["id"][(ranks >= low) & (ranks < high)]
        drawn = generator.choice(pool, size=min(sample_size, len(pool)), replace=False)
        return np.sort(drawn)

    cohorts = {"front": sample(front_band), "rear": sample(rear_band)}
    series = {name: {"mean": [], "std": []} for name in cohorts}
    times_seconds: list[float] = []
    for iteration in iterations:
        snapshot = records[records["iter"] == iteration]
        snapshot = snapshot[np.argsort(snapshot["id"])]
        frame_ranks = np.argsort(np.argsort(_band_frame_along(snapshot, world_size)))
        frame_ranks = frame_ranks / max(len(snapshot) - 1, 1)
        snapshot_ids = snapshot["id"]
        times_seconds.append(float(iteration - iterations[0]))
        for name, ids in cohorts.items():
            positions = np.searchsorted(snapshot_ids, ids)
            positions = positions[positions < len(snapshot_ids)]
            present = snapshot_ids[positions] == ids[: len(positions)]
            cohort_ranks = frame_ranks[positions[present]]
            series[name]["mean"].append(float(cohort_ranks.mean()))
            series[name]["std"].append(float(cohort_ranks.std()))

    cohort_separation = series["front"]["mean"][-1] - series["rear"]["mean"][-1]
    return {
        "iterations_from_reference": times_seconds,
        "sample_size_per_cohort": sample_size,
        "population": population,
        "cohort_mean_rank": {n: series[n]["mean"] for n in cohorts},
        "cohort_std_rank": {n: series[n]["std"] for n in cohorts},
        "cohort_separation": cohort_separation,
    }


def neighbour_anisotropy_v2(
    snapshot_dir: str | Path,
    world_width: float,
    world_height: float,
    inner_radius: float = 0.01,
    outer_radius: float = 0.07,
    frame_stride: int = 5,
    burn_in_iteration: int = 0,
) -> dict:
    """State-conditioned angular neighbour-density anisotropy (Weinburd 2024).

    For every focal locust the bearings of neighbours in the annulus
    ``[inner_radius, outer_radius]`` (metres) are taken relative to its
    heading (0 = ahead) and pooled per motion state — ``stationary`` (rest),
    ``walking`` (active, not hopping), ``hopping`` (active + hopping) — read
    from the snapshot flags. The field signature (Weinburd et al. 2024) is a
    depleted frontal/axial sector with the highest density to the sides
    around MOVING locusts, and near-isotropy around stationary ones.

    Each state is summarized by the low-order angular Fourier moments
    ``a1 = <cos theta>`` (front-back: < 0 = front-depleted) and
    ``a2 = <cos 2theta>`` (axis vs lateral: < 0 = lateral-dense, the
    field-like packing; > 0 = fore-aft/columnar files), plus 45-degree
    sector occupancies. Two held-out scalars condition the metric on state,
    matching the field result rather than any calibration target:
    ``lateral_packing_walking`` = -a2 for walking (> 0 = field-like) and
    ``state_contrast`` = |a2_walking| - |a2_stationary| (> 0 = movers more
    anisotropic than stationary, as observed).
    """
    records = load_snapshots(snapshot_dir)
    iterations = np.unique(records["iter"])
    iterations = iterations[iterations >= burn_in_iteration][::frame_stride]
    world = np.array([world_width, world_height])
    bearings: dict[str, list[float]] = {"stationary": [], "walking": [], "hopping": []}

    for iteration in iterations:
        snapshot = records[records["iter"] == iteration]
        positions = (
            np.column_stack([snapshot["x"], snapshot["y"]]).astype(np.float64) % world
        )
        tree = KDTree(positions, boxsize=(world_width, world_height))
        headings = snapshot["heading"].astype(np.float64)
        active = (snapshot["flags"] & 1) != 0
        hopping = (snapshot["flags"] & 2) != 0
        neighbour_lists = tree.query_ball_point(positions, r=outer_radius)
        for focal, neighbours in enumerate(neighbour_lists):
            state = "stationary" if not active[focal] else ("hopping" if hopping[focal] else "walking")
            store = bearings[state]
            for other in neighbours:
                if other == focal:
                    continue
                offset = (positions[other] - positions[focal] + world / 2) % world - world / 2
                distance = float(np.hypot(offset[0], offset[1]))
                if distance < inner_radius or distance > outer_radius:
                    continue
                bearing = np.arctan2(offset[1], offset[0]) - headings[focal]
                store.append(float((bearing + np.pi) % (2 * np.pi) - np.pi))

    def summarize(values: list[float]) -> dict | None:
        if len(values) < 50:
            return None
        angles = np.array(values)
        wedge = np.pi / 8  # 45-degree sectors
        front = float(np.mean(np.abs(angles) < wedge))
        rear = float(np.mean(np.abs(np.abs(angles) - np.pi) < wedge))
        lateral = float(np.mean(np.abs(np.abs(angles) - np.pi / 2) < wedge)) / 2
        a1c, a1s = float(np.mean(np.cos(angles))), float(np.mean(np.sin(angles)))
        a2c, a2s = float(np.mean(np.cos(2 * angles))), float(np.mean(np.sin(2 * angles)))
        return {
            "count": len(angles),
            "a1": a1c,
            "a2": a2c,
            # moduli of the complex trigonometric moments <e^{i n theta}>, the
            # non-negative, rotation-invariant strengths Weinburd (2024) reports
            # (|M1|, |M2|); the sign/direction lives in a1, a2 above.
            "m1_modulus": float(np.hypot(a1c, a1s)),
            "m2_modulus": float(np.hypot(a2c, a2s)),
            "front_fraction": front,
            "rear_fraction": rear,
            "lateral_fraction": lateral,
            "front_over_lateral": front / (lateral + 1e-9),
        }

    states = {name: summarize(vals) for name, vals in bearings.items()}
    walking = states.get("walking")
    stationary = states.get("stationary")
    lateral_packing_walking = -walking["a2"] if walking else float("nan")
    state_contrast = (
        abs(walking["a2"]) - abs(stationary["a2"])
        if walking and stationary
        else float("nan")
    )
    return {
        "states": states,
        "lateral_packing_walking": lateral_packing_walking,
        "state_contrast": state_contrast,
    }
