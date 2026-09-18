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

from .metrics import SNAPSHOT_DTYPE, _circular_center_of_mass, frame_slices, load_snapshots


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
    frames = frame_slices(records)
    iterations = np.unique(records["iter"])
    iterations = iterations[iterations >= burn_in_iteration]
    world_size = (world_width, world_height)

    reference = records[frames[int(iterations[0])]]
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
        snapshot = records[frames[int(iteration)]]
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
    frames = frame_slices(records)
    iterations = np.unique(records["iter"])
    iterations = iterations[iterations >= burn_in_iteration][::frame_stride]
    world_size = (world_width, world_height)
    histogram = np.zeros(bearing_bins)
    edges = np.linspace(-np.pi, np.pi, bearing_bins + 1)

    for iteration in iterations:
        snapshot = records[frames[int(iteration)]]
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
    frames = frame_slices(records)
    iterations = np.unique(records["iter"])
    iterations = iterations[iterations >= burn_in_iteration]
    world_size = (world_width, world_height)

    reference = records[frames[int(iterations[0])]]
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
        snapshot = records[frames[int(iteration)]]
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


WEINBURD_TABLE4 = {
    # Weinburd et al. 2024, Appendix Table 4: trigonometric moments of relative
    # neighbour angles within 7 cm, all bands. Keys as in the paper.
    "stationary": {"count": 1505672, "M1": 0.0040, "psi1": -2.5370, "Ms1": -0.0023, "Mc1": -0.0033,
                   "M2": 0.0146, "psi2": -0.2191, "Ms2": -0.0032, "Mc2": 0.0142},
    "walking": {"count": 1527233, "M1": 0.0173, "psi1": -1.5242, "Ms1": -0.0173, "Mc1": 0.0008,
                "M2": 0.0248, "psi2": 0.0670, "Ms2": 0.0017, "Mc2": 0.0247},
    "hopping": {"count": 3536882, "M1": 0.0203, "psi1": 0.1393, "Ms1": 0.0028, "Mc1": 0.0201,
                "M2": 0.0358, "psi2": -0.0492, "Ms2": -0.0018, "Mc2": 0.0357},
}


def neighbour_anisotropy_v2(
    snapshot_dir: str | Path,
    world_width: float,
    world_height: float,
    outer_radius: float = 0.07,
    inner_radius: float = 0.0,
    frame_stride: int = 5,
    burn_in_iteration: int = 0,
) -> dict:
    """Trigonometric moments of relative neighbour angles, exactly as defined in
    Weinburd et al. 2024 (Appendix F, "Trigonometric moments"; Table 4).

    For every focal locust and every neighbour within ``outer_radius`` (7 cm in
    the paper, no inner exclusion) the neighbour's relative position is expressed
    in the focal's frame with the focal facing UP: ``phi = atan2(y, x)`` where
    x is to the focal's right and y is ahead, so phi = 0 is right, pi/2 ahead,
    pi left, -pi/2 behind. Angles are pooled per motion state of the focal —
    ``stationary`` (rest), ``walking`` (active, not hopping), ``hopping``
    (active + hopping) — from the snapshot flags. For each state and p = 1, 2:

        Ms_p = mean(sin(p phi)),  Mc_p = mean(cos(p phi)),
        |M_p| = hypot(Mc_p, Ms_p),  psi_p = atan2(Ms_p, Mc_p) / p.

    The paper's headline quantities are ``-Ms1`` (front-back asymmetry: > 0
    means lower density in front and higher behind the focal) and ``Mc2``
    (four-fold anisotropy: > 0 means high density to the left and right, low
    in front and behind). Field values (all bands): walking -Ms1 = 0.0173,
    Mc2 = 0.0247, |M2| = 0.0248; stationary |M2| = 0.0146; hopping
    Mc2 = 0.0357 — see ``WEINBURD_TABLE4``.
    """
    records = load_snapshots(snapshot_dir)
    frames = frame_slices(records)
    iterations = np.unique(records["iter"])
    iterations = iterations[iterations >= burn_in_iteration][::frame_stride]
    world = np.array([world_width, world_height])
    angles: dict[str, list[np.ndarray]] = {"stationary": [], "walking": [], "hopping": []}

    for iteration in iterations:
        snapshot = records[frames[int(iteration)]]
        positions = (
            np.column_stack([snapshot["x"], snapshot["y"]]).astype(np.float64) % world
        )
        headings = snapshot["heading"].astype(np.float64)
        active = (snapshot["flags"] & 1) != 0
        hopping = (snapshot["flags"] & 2) != 0
        tree = KDTree(positions, boxsize=(world_width, world_height))
        pairs = tree.query_pairs(outer_radius, output_type="ndarray")
        if len(pairs) == 0:
            continue
        # every member of a pair is a focal once
        focal = np.concatenate([pairs[:, 0], pairs[:, 1]])
        other = np.concatenate([pairs[:, 1], pairs[:, 0]])
        offset = (positions[other] - positions[focal] + world / 2) % world - world / 2
        if inner_radius > 0:
            keep = np.hypot(offset[:, 0], offset[:, 1]) >= inner_radius
            focal, offset = focal[keep], offset[keep]
        heading = headings[focal]
        # focal frame, focal facing up: x = right (clockwise from heading), y = ahead
        ahead = offset[:, 0] * np.cos(heading) + offset[:, 1] * np.sin(heading)
        right = offset[:, 0] * np.sin(heading) - offset[:, 1] * np.cos(heading)
        phi = np.arctan2(ahead, right)
        state_of = np.where(~active[focal], 0, np.where(hopping[focal], 2, 1))
        for code, name in enumerate(("stationary", "walking", "hopping")):
            angles[name].append(phi[state_of == code])

    def moments(values: list[np.ndarray]) -> dict | None:
        phi = np.concatenate(values) if values else np.zeros(0)
        if phi.size < 50:
            return None
        out: dict = {"count": int(phi.size)}
        for p in (1, 2):
            ms, mc = float(np.mean(np.sin(p * phi))), float(np.mean(np.cos(p * phi)))
            out[f"Ms{p}"], out[f"Mc{p}"] = ms, mc
            out[f"M{p}"] = float(np.hypot(mc, ms))
            out[f"psi{p}"] = float(np.arctan2(ms, mc) / p)
        return out

    states = {name: moments(vals) for name, vals in angles.items()}
    walking, stationary = states["walking"], states["stationary"]
    return {
        "states": states,
        # the paper's two headline scalars, for the walking focal
        "front_back_asymmetry": -walking["Ms1"] if walking else float("nan"),
        "four_fold_anisotropy": walking["Mc2"] if walking else float("nan"),
        # movers more anisotropic than resters, as in the field (> 0)
        "state_contrast": (walking["M2"] - stationary["M2"]) if walking and stationary else float("nan"),
    }
