"""Regenerate every static results figure into figures/output/static/.

Single source of truth for the five generated figures:
  sensitivity_indices.svg, bo_history.svg, marking_ranks.svg,
  anisotropy_hist.svg, morphology_frames.png

Run from anywhere:  python figures/make_figures.py

Data dependencies (all repo-relative):
  - Sobol indices: hardcoded below, from docs/sensitivity_screening2.md.
  - BO history:    figures/data/bo_histories.jsonl
                   (per-model [trial_number, score] of completed trials).
  - Marking / anisotropy: figures/data/validation_results.json.
  - Morphology frames:    snapshot streams under runs/ (paths in MORPHOLOGY).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # paramsearch/
DATA = Path(__file__).resolve().parent / "data"
IMG = Path(__file__).resolve().parent / "output" / "static"
IMG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from paramsearch.metrics import load_snapshots

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "figure.dpi": 150,
    }
)
NF_COLOR, SPIN_COLOR = "#2b6a99", "#b0503c"


def _display_trial(trial_dir):
    """Raw Optuna trial number offset by the neural-field 255-479 gap (225
    missing trials); spin numbering is left unchanged."""
    raw = int(trial_dir.split("-")[-1])
    nf_trials = {756, 1030, 1273, 886}
    return raw - 225 if raw in nf_trials and raw > 479 else raw



# Sobol total- and first-order indices (rank-transformed score), power-5
# screening; negative S1 estimates clamped to 0. Source: screening2.
NF_SENS = [
    ("antiGoalStimulusStrength", 0.568, 0.0),
    ("totalSocialAttraction", 0.410, 0.0),
    ("synapticConnectivityCoefficient", 0.395, 0.19),
    ("neuralInhibitionCoefficient", 0.338, 0.05),
    ("inverseTemperatureCoefficient", 0.315, 0.0),
    ("receptiveFieldStd", 0.267, 0.0),
    ("averageSpeed", 0.264, 0.0),
    ("pursuerHeadingAngleEnd", 0.253, 0.09),
    ("antiGoalAngleRangeStart", 0.213, 0.04),
    ("antiGoalOverrideRange", 0.196, 0.06),
    ("resumeMarchProbability", 0.151, 0.0),
    ("hopSpeed", 0.136, 0.0),
    ("hopDuration", 0.074, 0.06),
    ("occlusionThreshold", 0.034, 0.0),
    ("minimalInactivityPeriod", 0.021, 0.0),
    ("crowdedHopProbability", 0.013, 0.0),
    ("hopProbability", 0.008, 0.03),
]
# Spin recomputed after the timeout-rerun recovered 87 of 217 censored evals;
# the earlier S1 spread was a timeout-censoring artifact and collapses to ~0.
SPIN_SENS = [
    ("neuralInhibitionCoefficient", 0.600, 0.0),
    ("antiGoalOverrideRange", 0.564, 0.0),
    ("antiGoalStimulusStrength", 0.480, 0.0),
    ("totalSocialAttraction", 0.410, 0.0),
    ("glauberDynamicsIterations", 0.399, 0.066),
    ("receptiveFieldStd", 0.377, 0.0),
    ("crowdedHopProbability", 0.303, 0.0),
    ("minimalInactivityPeriod", 0.281, 0.0),
    ("synapticConnectivityCoefficient", 0.227, 0.0),
    ("hopSpeed", 0.207, 0.0),
    ("hopDuration", 0.190, 0.0),
    ("inverseTemperatureCoefficient", 0.181, 0.0),
    ("antiGoalAngleRangeStart", 0.178, 0.0),
    ("pursuerHeadingAngleEnd", 0.113, 0.0),
    ("resumeMarchProbability", 0.095, 0.0),
    ("occlusionThreshold", 0.093, 0.0),
    ("averageSpeed", 0.009, 0.0),
    ("hopProbability", 0.000, 0.0),
]

# Morphology panels: (snapshot dir relative to runs/, world W, H, title).
MORPHOLOGY = [
    (
        "validation-data/bo2-nf/trial-01030/replicate-0",
        23.4,
        3.6,
        "Neural field 805 (intermittent) — frontal band",
    ),
    (
        "bo2-analysis/bo2-nf-trial-00916/replicate-0",
        None,
        None,
        "Neural field 691 (continuous) — dense exponential front",
    ),
    (
        "bo2-nf-top/trial-00000/replicate-0",
        None,
        None,
        "Neural field 0 (continuous) — very dense front",
    ),
    (
        "validation-data/bo2-spin/trial-00305/replicate-0",
        None,
        None,
        "Spin system 305 (intermittent) — columnar streams",
    ),
    (
        "validation-data/bo2-spin/trial-00514/replicate-0",
        None,
        None,
        "Spin system 514 (continuous) — dense band",
    ),
]

MARKING_PANELS = [
    ("trial-00756", "Neural field 531 (intermittent)"),
    ("trial-00886", "Neural field 661 (continuous)"),
    ("trial-00305", "Spin system 305 (columnar)"),
    ("trial-00514", "Spin system 514 (continuous)"),
]
COHORT_STYLE = {
    "cohort_80_100": ("front cohort (80–100%)", NF_COLOR),
    "cohort_40_60": ("middle cohort (40–60%)", "#777777"),
    "cohort_00_20": ("rear cohort (0–20%)", SPIN_COLOR),
}
ANISO_GROUPS = [
    ("Neural field", ["trial-00756", "trial-01030", "trial-01273", "trial-00886"]),
    ("Spin system", ["trial-00514", "trial-00305", "trial-00457", "trial-00682"]),
]


def sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    for ax, data, color, title in [
        (axes[0], NF_SENS, NF_COLOR, "Neural Field"),
        (axes[1], SPIN_SENS, SPIN_COLOR, "Spin System"),
    ]:
        names = [d[0] for d in data][::-1]
        st = [d[1] for d in data][::-1]
        s1 = [max(d[2], 0) for d in data][::-1]
        y = np.arange(len(names))
        ax.barh(
            y, st, height=0.72, color=color, alpha=0.45, label="$S_T$ (total-order)"
        )
        ax.barh(y, s1, height=0.72, color=color, label="$S_1$ (first-order)")
        ax.set_yticks(y, names, fontsize=7.5)
        ax.set_xlabel("Sobol index (rank-transformed score)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 0.65)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "sensitivity_indices.svg", bbox_inches="tight")
    plt.close(fig)


def bo_history():
    histories = {}
    for line in (DATA / "bo_histories.jsonl").read_text().splitlines():
        if line.strip().startswith("{"):
            histories.update(json.loads(line))
    window = 100  # trials in the rolling-mean window
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, key, color, title in [
        (axes[0], "bo2-nf", NF_COLOR, "Neural field"),
        (axes[1], "bo2-spin", SPIN_COLOR, "Spin system"),
    ]:
        rows = np.array(sorted(histories[key], key=lambda r: r[0]), dtype=float)
        scores = rows[:, 1]
        index = np.arange(len(scores))  # compact completed-trial index
        ax.scatter(index, scores, s=4, color=color, alpha=0.3, linewidths=0)
        rolling = np.convolve(scores, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(len(rolling)) + window // 2,
            rolling,
            color=color,
            linewidth=1.8,
            label=f"rolling mean of {window} trials",
        )
        ax.plot(
            index,
            np.minimum.accumulate(scores),
            color="black",
            linewidth=1.4,
            label="best so far",
        )
        ax.set_xlabel("Completed trial")
        ax.set_title(
            f"{title} ({len(scores)} trials, best {scores.min():.1f})", fontsize=10
        )
        ax.legend(frameon=False, fontsize=8, loc="center right")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 810)
    fig.tight_layout()
    fig.savefig(IMG / "bo_history.svg", bbox_inches="tight")
    plt.close(fig)


def marking():
    validation = json.loads(
        (DATA / "validation_results.json").read_text()
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.4), sharex=True, sharey=True)
    for ax, (trial, title) in zip(axes.flat, MARKING_PANELS):
        entry = validation[trial]
        hours = np.array(entry["iterations_from_reference"]) * 0.3 / 3600
        for cohort, (label, color) in COHORT_STYLE.items():
            series = np.array(
                [r[cohort] for r in entry["cohort_mean_rank_by_replicate"]]
            )
            ax.plot(hours, series.mean(0), color=color, linewidth=1.5, label=label)
            ax.fill_between(
                hours,
                series.min(0),
                series.max(0),
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        ax.axhline(0.5, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("Time since marking [h]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean position rank")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(IMG / "marking_ranks.svg", bbox_inches="tight")
    plt.close(fig)


def anisotropy():
    validation = json.loads(
        (DATA / "validation_results.json").read_text()
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0), sharey=True)
    for (title, trials), ax in zip(ANISO_GROUPS, axes):
        for trial in trials:
            entry = validation[trial]
            centres = np.degrees(entry["bearing_bin_centres"])
            hist = np.array(entry["bearing_histogram_mean"]) * 12
            order = np.argsort(centres)
            ax.plot(
                centres[order],
                hist[order],
                marker="o",
                markersize=2.5,
                linewidth=1.1,
                label=f"trial {_display_trial(trial)}",
            )
        ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel("Neighbour bearing relative to heading [deg]")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=7.5, ncol=2)
    axes[0].set_ylabel("Relative density\n(1 = isotropic)")
    axes[0].set_ylim(0.85, 1.15)
    fig.tight_layout()
    fig.savefig(IMG / "anisotropy_hist.svg", bbox_inches="tight")
    plt.close(fig)


def morphology():
    fig, axes = plt.subplots(5, 1, figsize=(8.4, 7.8))
    for ax, (rel, w, h, title) in zip(axes, MORPHOLOGY):
        rep = ROOT / "runs" / rel
        if w is None:
            overrides = json.loads((rep / "run.json").read_text())["overrides"]
            w, h = overrides["worldWidthMeters"], overrides["worldHeightMeters"]
        records = load_snapshots(rep / "snapshots")
        last = records[records["iter"] == records["iter"].max()]
        x = np.asarray(last["x"], float) % w
        y = np.asarray(last["y"], float) % h
        angle = x / w * 2 * np.pi
        com = (
            np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
            % (2 * np.pi)
            / (2 * np.pi)
            * w
        )
        xs = (x - com + w / 2) % w  # band-centred
        lo, hi = np.percentile(xs, 2) - 1.0, np.percentile(xs, 98) + 1.0
        moving = (last["flags"] & 1) != 0
        ax.scatter(xs[~moving], y[~moving], s=1.1, color="#b8b8b8", linewidths=0)
        ax.scatter(xs[moving], y[moving], s=1.1, color="#1a1a1a", linewidths=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.set_title(
            f"{title}   [{hi - lo:.0f} m shown, {int(moving.sum())}/{len(last)} marching]",
            fontsize=8.5,
            loc="left",
        )
        ax.set_yticks([0, 3])
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel(
        "Along-march position, band-centred [m]  (march direction →)", fontsize=8.5
    )
    fig.tight_layout(h_pad=1.2)
    fig.savefig(IMG / "morphology_frames.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sensitivity()
    bo_history()
    marking()
    anisotropy()
    morphology()
    print(f"wrote figures to {IMG}")
