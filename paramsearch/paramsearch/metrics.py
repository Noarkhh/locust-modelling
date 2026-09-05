"""Compute band metrics from binary agent snapshots.

Snapshot record layout must match AgentSnapshotWriter.scala. The world is
toroidal, so centering uses circular means and neighbour queries use periodic
KD-trees; all band-frame quantities are computed on minimum-image coordinates
relative to the circular centre of mass.

All profile metrics are computed in the band frame: the x'-axis is the mean
heading of moving agents, positive x' pointing where the band is going, so
"front" = large x'.
"""

import glob
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

SNAPSHOT_DTYPE = np.dtype(
    [
        ("iter", "<u4"),
        ("id", "<u4"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("heading", "<f4"),
        ("speed", "<f4"),
        ("flags", "u1"),
    ]
)

PROFILE_BIN_METERS = 0.5  # bin size for density profiles and occupancy grids
MIN_DECAY_FIT_BINS = 4  # minimum profile bins behind the peak to fit a decay
CROWDED_RANGE_METERS = 0.035  # neighbour distance defining "crowded" (repulsion range)
DECAY_FIT_FLOOR_FRACTION = 0.1  # decay fit spans the band proper, not the straggler tail

WorldSize = tuple[float, float]


def load_snapshots(snapshot_dir: str | Path) -> np.ndarray:
    """Read every per-JVM snapshot file in a run's snapshot directory into one
    structured array.

    A run produces one file per JVM (one per node in distributed mode); records
    from different files interleave arbitrarily within an iteration, so the
    concatenation is stably sorted by iteration and callers group by the
    ``iter`` field.
    """
    files = sorted(glob.glob(str(Path(snapshot_dir) / "*.bin")))
    if not files:
        raise FileNotFoundError(f"no snapshot files in {snapshot_dir}")
    records = np.concatenate(
        [np.fromfile(file, dtype=SNAPSHOT_DTYPE) for file in files]
    )
    return records[np.argsort(records["iter"], kind="stable")]


def compute_metrics(
    snapshot_dir: str | Path,
    world_width: float,
    world_height: float,
    timestep_duration: float,
    snapshot_frequency: int,
    burn_in_fraction: float = 0.5,
) -> dict[str, float]:
    """Reduce one simulation run to a flat dict of scalar band metrics.

    Discards the first ``burn_in_fraction`` of the run as initialization
    transient, computes per-snapshot metrics for the remainder (see
    ``_snapshot_metrics``), and averages them over time. Two quantities need
    consecutive snapshots and are added on top of the averages:

    - ``band_speed`` / ``band_speed_ratio``: displacement rate of the band's
      centre of mass, absolute and relative to the mean speed of moving
      agents. Field bands travel far slower than the locusts in them, so the
      ratio constrains the intermittency parameters.
    - ``area_per_agent_trend``: slope of the (normalized) occupied area per
      agent over time. A negative value at the end of the run means the swarm
      is still contracting — the progressive-clumping failure mode.
    """
    records = load_snapshots(snapshot_dir)
    iterations = np.unique(records["iter"])
    burn_in_end = iterations[0] + burn_in_fraction * (iterations[-1] - iterations[0])
    kept_iterations = iterations[iterations >= burn_in_end]
    if len(kept_iterations) < 2:
        kept_iterations = iterations[-2:]
    snapshot_interval_seconds = snapshot_frequency * timestep_duration
    world_size: WorldSize = (world_width, world_height)

    snapshot_metrics = []
    centers_of_mass = []
    along_band_per_snapshot = []
    for iteration in kept_iterations:
        snapshot = records[records["iter"] == iteration]
        metrics, center_of_mass, along_band = _snapshot_metrics(snapshot, world_size)
        snapshot_metrics.append(metrics)
        centers_of_mass.append(center_of_mass)
        along_band_per_snapshot.append(along_band)

    aggregated = {
        key: _nanmean([metrics[key] for metrics in snapshot_metrics])
        for key in snapshot_metrics[0]
    }

    # Profile-shape metrics are re-derived from the time-pooled, peak-aligned
    # profile rather than averaged per snapshot: single-snapshot bins carry
    # counting noise that systematically depresses the log-linear fit R^2
    # (Bach 2018 likewise fits replicate-averaged profiles).
    mean_band_width = aggregated["band_width"]
    pooled = np.concatenate(
        [along - _profile_peak_position(along) for along in along_band_per_snapshot]
    )
    aggregated.update(_density_profile(pooled, mean_band_width))
    # Pooling sums counts over snapshots; renormalize the one absolute density.
    aggregated["front_peak_density"] /= len(along_band_per_snapshot)

    # Band speed: minimum-image COM displacement per snapshot interval.
    center_track = np.array(centers_of_mass)
    world = np.array(world_size)[None, :]
    displacements = (np.diff(center_track, axis=0) + world / 2) % world - world / 2
    band_speed = float(
        np.mean(np.linalg.norm(displacements, axis=1)) / snapshot_interval_seconds
    )
    aggregated["band_speed"] = band_speed
    mean_moving_speed = aggregated.pop("_mean_moving_speed")
    aggregated["band_speed_ratio"] = (
        band_speed / mean_moving_speed if mean_moving_speed > 0 else 0.0
    )

    area_per_agent = np.array(
        [metrics["area_per_agent"] for metrics in snapshot_metrics]
    )
    times = np.arange(len(area_per_agent)) * snapshot_interval_seconds
    if len(area_per_agent) > 2:
        normalized_area = area_per_agent / max(area_per_agent.mean(), 1e-12)
        aggregated["area_per_agent_trend"] = float(
            np.polyfit(times, normalized_area, 1)[0]
        )
    else:
        aggregated["area_per_agent_trend"] = 0.0
    return aggregated


def _nanmean(values: list[float]) -> float:
    """Mean over finite values; NaN (without a warning) when there are none."""
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _snapshot_metrics(
    snapshot: np.ndarray, world_size: WorldSize
) -> tuple[dict[str, float], np.ndarray]:
    """Compute all single-snapshot metrics plus the band's centre of mass.

    Establishes the band frame — minimum-image coordinates around the circular
    centre of mass, rotated so x' points along the mean heading of moving
    agents — then delegates to the per-family helpers (density profile,
    transverse structure, neighbour statistics, occupancy). Directly computed
    here:

    - ``global_order``: norm of the mean heading unit vector (0..1), the
      standard polarization order parameter.
    - ``marching_fraction`` / ``hopping_fraction``: shares of agents whose
      active/hopping flag bits are set; identify the intermittency and
      hopping parameters.
    - ``band_length`` / ``band_width`` / ``elongation``: 5th-95th percentile
      extents along and across the direction of motion and their ratio;
      round blobs give elongation near 1, fronts and columns well above it.
    """
    agent_count = len(snapshot)
    positions = (
        np.column_stack([snapshot["x"], snapshot["y"]]).astype(np.float64) % world_size
    )
    headings = snapshot["heading"].astype(np.float64)
    is_active = (snapshot["flags"] & 1).astype(bool)
    is_hopping = (snapshot["flags"] & 2).astype(bool)
    is_moving = is_active & (snapshot["speed"] > 0)

    center_of_mass = _circular_center_of_mass(positions, world_size)
    world = np.array(world_size)
    relative_positions = (positions - center_of_mass + world / 2) % world - world / 2

    # Band frame from moving agents' mean heading (all agents if none move).
    reference_headings = headings[is_moving] if is_moving.any() else headings
    mean_direction = np.array(
        [np.cos(reference_headings).mean(), np.sin(reference_headings).mean()]
    )
    mean_direction_norm = np.linalg.norm(mean_direction)
    forward_axis = (
        mean_direction / mean_direction_norm
        if mean_direction_norm > 1e-12
        else np.array([1.0, 0.0])
    )
    transverse_axis = np.array([-forward_axis[1], forward_axis[0]])
    along_band = relative_positions @ forward_axis
    across_band = relative_positions @ transverse_axis

    heading_unit_vectors = np.column_stack([np.cos(headings), np.sin(headings)])
    moving_speeds = snapshot["speed"][is_moving]
    metrics = {
        "global_order": float(np.linalg.norm(heading_unit_vectors.mean(axis=0))),
        "marching_fraction": float(is_active.mean()),
        "hopping_fraction": float(is_hopping.mean()),
        "_mean_moving_speed": float(moving_speeds.mean()) if is_moving.any() else 0.0,
        "band_length": float(
            np.percentile(along_band, 95) - np.percentile(along_band, 5)
        ),
        "band_width": float(
            np.percentile(across_band, 95) - np.percentile(across_band, 5)
        ),
    }
    metrics["elongation"] = metrics["band_length"] / max(metrics["band_width"], 1e-9)

    metrics.update(_density_profile(along_band, metrics["band_width"]))
    metrics.update(_transverse_structure(across_band))
    metrics.update(_neighbour_metrics(positions, world_size, is_hopping))
    metrics.update(_occupancy(positions, world_size, agent_count))
    return metrics, center_of_mass, along_band


def _circular_center_of_mass(
    positions: np.ndarray, world_size: WorldSize
) -> np.ndarray:
    """Centre of mass on a torus, computed independently per axis.

    Each coordinate is mapped to an angle on a circle, the angles are averaged
    as unit vectors, and the mean angle is mapped back — the standard circular
    mean, which is well-defined even when the band straddles the wrap-around
    boundary (an arithmetic mean would land it on the wrong side of the world).
    """
    center = np.empty(2)
    for axis in range(2):
        angles = positions[:, axis] / world_size[axis] * 2 * np.pi
        mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        center[axis] = (mean_angle % (2 * np.pi)) / (2 * np.pi) * world_size[axis]
    return center


def _profile_peak_position(along_band: np.ndarray) -> float:
    """Position (in band-frame meters) of the densest profile slice, used to
    align per-snapshot profiles before pooling them across time."""
    low, high = along_band.min(), along_band.max()
    bin_count = max(int(np.ceil((high - low) / PROFILE_BIN_METERS)), 1)
    counts, edges = np.histogram(along_band, bins=bin_count, range=(low, high))
    peak_index = int(np.argmax(counts))
    return float((edges[peak_index] + edges[peak_index + 1]) / 2)


def _density_profile(along_band: np.ndarray, band_width: float) -> dict[str, float]:
    """Characterize the SHAPE of the front-to-rear density profile.

    Histograms agent positions along the direction of motion and extracts the
    empirical signature of Australian plague locust frontal bands (Buhl et
    al. 2011): a sharp density peak at the front followed by an exponential
    decay toward the rear. All targeted quantities are dimensionless, so the
    shape comparison is independent of population size, world size, and
    absolute density. Returns:

    - ``profile_peak_position``: fractional position of the densest slice
      along the band, 0 = rearmost, 1 = frontmost. A frontal band peaks near
      the front (~1); a symmetric blob peaks near 0.5.
    - ``profile_peak_contrast``: peak slice density divided by the mean slice
      density — how front-loaded the mass is (1 = flat profile).
    - ``profile_decay_fraction``: e-folding distance of the rearward decay
      divided by the profile's full extent — what fraction of the band one
      decay constant spans (small = sharp front with a long thin tail).
    - ``profile_decay_r2``: goodness of the log-linear fit over the band
      proper — the region from the peak back to where density falls below
      ``DECAY_FIT_FLOOR_FRACTION`` of it. The straggler plateau behind the
      band is deliberately excluded (it is measured by area/band-length
      metrics instead).
    - ``front_peak_density`` and ``profile_decay_length`` (locusts/m^2,
      meters) are kept as scale-carrying DIAGNOSTICS, not objective targets.
    """
    low, high = along_band.min(), along_band.max()
    bin_count = max(
        int(np.ceil((high - low) / PROFILE_BIN_METERS)), MIN_DECAY_FIT_BINS + 1
    )
    counts, _ = np.histogram(along_band, bins=bin_count, range=(low, high))
    densities = counts / (PROFILE_BIN_METERS * max(band_width, 1e-9))
    peak_index = int(np.argmax(counts))
    profile_extent = max(high - low, 1e-9)
    profile = {
        "front_peak_density": float(densities[peak_index]),
        "profile_peak_position": peak_index / max(bin_count - 1, 1),
        "profile_peak_contrast": float(counts[peak_index] / max(counts.mean(), 1e-9)),
    }

    rearward = densities[: peak_index + 1][::-1]  # rearward[k] = k bins behind the peak
    # Fit only the band proper: stop where density first drops below
    # DECAY_FIT_FLOOR_FRACTION of the peak (or at the first empty bin).
    # Marching bands trail a roughly flat straggler plateau; including it
    # would conflate "is the band's decay exponential" (this metric) with
    # "how large is the straggler tail" (area/band-length metrics).
    below_floor = rearward < DECAY_FIT_FLOOR_FRACTION * rearward[0]
    fit_length = int(np.argmax(below_floor)) if below_floor.any() else len(rearward)
    rearward = rearward[:fit_length]
    if len(rearward) >= MIN_DECAY_FIT_BINS:
        distances_behind_peak = np.arange(len(rearward)) * PROFILE_BIN_METERS
        log_density = np.log(rearward)
        slope, intercept = np.polyfit(distances_behind_peak, log_density, 1)
        predicted = slope * distances_behind_peak + intercept
        residual_sum = float(np.sum((log_density - predicted) ** 2))
        total_sum = float(np.sum((log_density - log_density.mean()) ** 2))
        decay_length = float(-1.0 / slope) if slope < 0 else np.inf
        profile["profile_decay_length"] = decay_length
        profile["profile_decay_fraction"] = decay_length / profile_extent
        profile["profile_decay_r2"] = (
            1 - residual_sum / total_sum if total_sum > 0 else 0.0
        )
    else:
        profile["profile_decay_length"] = np.nan
        profile["profile_decay_fraction"] = np.nan
        profile["profile_decay_r2"] = np.nan
    return profile


def _transverse_structure(across_band: np.ndarray) -> dict[str, float]:
    """Distinguish frontal from columnar formations by their cross-section.

    Histograms agent positions across the direction of motion (2nd-98th
    percentile span, to ignore stragglers). A frontal band is transversely
    homogeneous; a columnar formation concentrates into distinct streams.
    Returns:

    - ``transverse_cv``: coefficient of variation of the cross-band histogram
      — near 0 for a uniform front, large when density bunches into columns.
    - ``stream_count``: number of contiguous runs of bins above half the
      median occupancy, a direct count of density streams (1 for a single
      coherent front).
    """
    low, high = np.percentile(across_band, [2, 98])
    bin_count = max(int(np.ceil((high - low) / PROFILE_BIN_METERS)), 4)
    counts, _ = np.histogram(across_band, bins=bin_count, range=(low, high))
    mean_count = counts.mean()
    coefficient_of_variation = (
        float(counts.std() / mean_count) if mean_count > 0 else 0.0
    )
    threshold = np.median(counts) / 2
    is_above = counts > threshold
    stream_count = int(np.sum(is_above[1:] & ~is_above[:-1]) + int(is_above[0]))
    return {"transverse_cv": coefficient_of_variation, "stream_count": stream_count}


def _neighbour_metrics(
    positions: np.ndarray, world_size: WorldSize, is_hopping: np.ndarray
) -> dict[str, float]:
    """Nearest-neighbour spacing and the density dependence of hopping.

    Builds a periodic KD-tree and queries each agent's nearest neighbour.
    Returns:

    - ``nn_distance_median``: typical spacing, anchoring the interaction
      ranges against field densities.
    - ``nn_distance_p5``: lower tail of spacing; collapsing toward zero means
      repulsion is being overwhelmed (agents stacking — the clumping
      diagnostic at individual scale).
    - ``crowded_fraction``: share of agents with a neighbour inside the
      repulsion range.
    - ``hop_rate_crowded`` / ``hop_rate_uncrowded``: fraction of hopping
      agents among crowded vs uncrowded ones. Their contrast is the
      observable that makes ``crowdedHopProbability`` identifiable; NaN when
      a group is empty.
    """
    tree = KDTree(positions, boxsize=world_size)
    distances, _ = tree.query(positions, k=2)
    nearest_neighbour = distances[:, 1]
    is_crowded = nearest_neighbour < CROWDED_RANGE_METERS
    metrics = {
        "nn_distance_median": float(np.median(nearest_neighbour)),
        "nn_distance_p5": float(np.percentile(nearest_neighbour, 5)),
        "crowded_fraction": float(is_crowded.mean()),
    }
    metrics["hop_rate_crowded"] = (
        float(is_hopping[is_crowded].mean()) if is_crowded.any() else np.nan
    )
    metrics["hop_rate_uncrowded"] = (
        float(is_hopping[~is_crowded].mean()) if (~is_crowded).any() else np.nan
    )
    return metrics


def _occupancy(
    positions: np.ndarray, world_size: WorldSize, agent_count: int
) -> dict[str, float]:
    """Coarse-grained density statistics for the clumping penalties.

    Bins the world into a fixed grid and looks at occupied cells only.
    Returns:

    - ``local_density_p99``: 99th percentile of per-cell density
      (locusts/m^2). Compared against the empirical ceiling on front
      densities; a quantile is used instead of the maximum because the max of
      tens of thousands of cells is a noisy extreme-value statistic.
    - ``area_per_agent``: occupied area divided by agent count — the inverse
      of mean packing. Shrinking values over time indicate progressive
      aggregation (tracked by ``area_per_agent_trend`` upstream).
    """
    bins_x, bins_y = (
        max(int(np.ceil(side / PROFILE_BIN_METERS)), 1) for side in world_size
    )
    counts, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=(bins_x, bins_y),
        range=((0, world_size[0]), (0, world_size[1])),
    )
    cell_area = (world_size[0] / bins_x) * (world_size[1] / bins_y)
    occupied = counts > 0
    occupied_densities = counts[occupied] / cell_area
    return {
        "local_density_p99": float(np.percentile(occupied_densities, 99)),
        "area_per_agent": float(occupied.sum() * cell_area / agent_count),
    }
