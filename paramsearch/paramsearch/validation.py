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
