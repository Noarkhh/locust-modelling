"""Neighbour-bearing statistics around a focal locust (Weinburd et al. 2024 style)
for the validation-campaign top-4 of each model: the data module shared by the
polar figure below and by neighbour_heatmap.py.

For every focal in every `stride`-th post-burn-in frame, the offsets of all
neighbours within 7 cm (no inner cut, as in Weinburd et al. 2024) are taken in
the focal's frame (ahead = along its heading, right = clockwise from it),
separately for active (walking or hopping) and inactive (stationary) focals,
and accumulated as (a) a bearing histogram and (b) a 2-D map of
`cell`-sized bins. Counts are pooled over all replicates of a trial.

Data: the 36k-iteration, 2k-agent replicate runs under runs/validation-data/bo2-{nf,spin}/.
Cache: figures/data/neighbour_density_c<cell mm>_s<stride>.json (re-styling the
figures does not need the snapshot streams).

  python figures/neighbour_density.py [--cell MM] [--stride N] [--recompute]
      -> figures/output/static/neighbour_density.{svg,png}   (polar bearing curves)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paramsearch.metrics import load_snapshots

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "output" / "static"
NF_COLOR, SPIN_COLOR = "#2b6a99", "#b0503c"  # thesis model colours (make_figures.py)
COL_COLOR, SPP_COLOR = "#c98a2b", "#6f6f6f"  # columnar spin candidate; SPP reference
MODEL_COLORS = {"nf": NF_COLOR, "spin": SPIN_COLOR, "col": COL_COLOR, "spp": SPP_COLOR}
INNER, OUTER = (
    0.0,
    0.07,
)  # Weinburd window [m]: all neighbours within 7 cm, no inner cut
CELL = 0.0025  # default heatmap cell size [m]
STRIDE = 1  # default frame stride (1 = every post-burn-in frame)
BINS, BURN_IN = 24, 12000
# focal motion states from the snapshot flags (bit 0 = active, bit 1 = hopping):
# Weinburd's three (walking = active & not hopping, hopping, stationary = inactive) plus the
# pooled "active" (walking + hopping) and "inactive" (= stationary). All are accumulated;
# DRAW_STATES is what the figures show by default.
STATES = ("active", "inactive", "walking", "hopping", "stationary", "all")
DRAW_STATES = ("active", "inactive")
# Weinburd 2024 walking-focal moments (|M1|, |M2|); orientation from the paper:
# lateral maxima (a2 < 0), slight rear excess (a1 < 0).
FIELD_M1, FIELD_M2 = 0.0173, 0.0248

# Run sets: (model, id, display label, directory holding replicate-*/ runs).
# NF thesis numbering = raw trial - 225.
SETS = {
    "nf": [
        ("nf", "00756", "Neural Field model 531", "runs/validation-data/bo2-nf/trial-00756"),
        ("nf", "01030", "Neural Field model 805", "runs/validation-data/bo2-nf/trial-01030"),
        ("nf", "01273", "Neural Field model 1048", "runs/validation-data/bo2-nf/trial-01273"),
        ("nf", "00886", "Neural Field model 661", "runs/validation-data/bo2-nf/trial-00886"),
    ],
    "spin": [
        ("spin", "00514", "Spin System model 514", "runs/validation-data/bo2-spin/trial-00514"),
        ("spin", "01048", "Spin System model 1048", "runs/new_top10_36k/spin-01048"),
        ("spin", "00783", "Spin System model 783", "runs/new_top10_36k/spin-00783"),
        ("spin", "00937", "Spin System model 937", "runs/new_top10_36k/spin-00937"),
    ],
    # spin 305 is the columnar spin candidate: its own model type "col" so it gets its own
    # row/colour instead of being grouped with the frontal spin candidates
    "col": [
        ("col", "00305", "Spin System model 305", "runs/validation-data/bo2-spin/trial-00305"),
    ],
    # the fixed validation candidate set (decided 2026-09-17): 4 NF + 5 spin.
    # NF and spin 514/305: 5-replicate calibration-scale runs; spin 1048/783/937: single-seed 36k regens.
    "validation": [
        ("nf", "00756", "Neural Field model 531", "runs/validation-data/bo2-nf/trial-00756"),
        ("nf", "01030", "Neural Field model 805", "runs/validation-data/bo2-nf/trial-01030"),
        ("nf", "01273", "Neural Field model 1048", "runs/validation-data/bo2-nf/trial-01273"),
        ("nf", "00886", "Neural Field model 661", "runs/validation-data/bo2-nf/trial-00886"),
        ("spin", "00514", "Spin System model 514", "runs/validation-data/bo2-spin/trial-00514"),
        ("spin", "01048", "Spin System model 1048", "runs/new_top10_36k/spin-01048"),
        ("spin", "00783", "Spin System model 783", "runs/new_top10_36k/spin-00783"),
        ("spin", "00937", "Spin System model 937", "runs/new_top10_36k/spin-00937"),
        ("col", "00305", "Spin System model 305", "runs/validation-data/bo2-spin/trial-00305"),
    ],
    # Bach 2018 frontal SPP reference at the campaign's initial density (750 /m^2)
    "spp": [
        ("spp", "d750", "SPP (Bach 2018), 750 /m²", "runs/spp-density/spp-bach-d750"),
    ],
}
# large-scale ladder cells (runs/final_scale_validation/<model>-<thesis>-<scale>, one run each,
# snapshots from iteration 0 -> the campaign burn-in of 12000 applies): sets "scale-25k" etc.
SCALE_CELLS = [
    ("nf", "531"), ("nf", "805"), ("nf", "1048"), ("nf", "661"),
    ("spin", "514"), ("spin", "1048"), ("spin", "783"), ("spin", "937"), ("spin", "305"),
]
for _scale in ("25k", "100k", "250k"):
    SETS[f"scale-{_scale}"] = [
        (m, t, f"{'Neural Field' if m == 'nf' else 'Spin System'} model {t}, {_scale}",
         f"runs/final_scale_validation/{m}-{t}-{_scale}")
        for m, t in SCALE_CELLS
    ]
RUNS = SETS["validation"]


def register_runs(paths, name="custom"):
    """Ad-hoc set from explicit run directories (`DIR` or `DIR:label`); each DIR is a
    run holding snapshots/ + run.json, or a trial dir holding replicate-*/ runs."""
    runs = []
    for spec in paths:
        path, _, label = spec.partition(":")
        p = Path(path)
        model = "nf" if "nf" in p.name.lower() else ("spin" if "spin" in p.name.lower() else "spp")
        runs.append((model, p.name, label or p.name, str(p if p.is_absolute() else p)))
    SETS[name] = runs
    return name


def cache_path(set_name="validation", cell=CELL, stride=STRIDE):
    return DATA_DIR / f"neighbour_density_{set_name}_c{cell * 1000:g}_s{stride}.json"


def bearing_histograms(run_dir, cell=CELL, stride=STRIDE):
    """Per state: bearing histogram and 2-D offset map (raw counts) over all neighbours within 7 cm."""
    ov = json.loads((run_dir / "run.json").read_text())["overrides"]
    world = np.array([ov["worldWidthMeters"], ov["worldHeightMeters"]])
    records = load_snapshots(run_dir / "snapshots")  # sorted by iteration
    iterations, starts, counts = np.unique(records["iter"], return_index=True, return_counts=True)
    frames = {int(it): slice(int(s), int(s + c)) for it, s, c in zip(iterations, starts, counts)}
    # keep complete frames only: large-scale streams lose their last frame(s) to the
    # writer flush / wall-time cut, leaving partial agent sets
    agents = int(ov.get("agentAmount", counts.max()))
    iterations = iterations[counts == agents]
    # the run's own burn-in (snapshots start there); BURN_IN is the campaign default,
    # which also applies to campaign cells that snapshot from iteration 0
    burn_in = int(ov.get("snapshotStartIteration", BURN_IN)) or BURN_IN
    iterations = iterations[iterations >= burn_in][::stride]
    edges = np.linspace(-np.pi, np.pi, BINS + 1)
    hist = {state: np.zeros(BINS) for state in STATES}
    # 2-D map of neighbour offsets in the focal frame (x = right, y = ahead)
    map_edges = np.arange(-OUTER, OUTER + cell / 2, cell)
    n_cells = len(map_edges) - 1
    maps = {state: np.zeros((n_cells, n_cells)) for state in STATES}
    for it in iterations:
        snap = records[frames[int(it)]]
        pos = np.column_stack([snap["x"], snap["y"]]).astype(np.float64) % world
        heading = snap["heading"].astype(np.float64)
        active = (snap["flags"] & 1) != 0
        hopping = (snap["flags"] & 2) != 0
        tree = KDTree(pos, boxsize=world)
        pairs = tree.query_pairs(OUTER, output_type="ndarray")
        # both directions: each member of a pair is a focal once
        focal = np.concatenate([pairs[:, 0], pairs[:, 1]])
        other = np.concatenate([pairs[:, 1], pairs[:, 0]])
        off = (pos[other] - pos[focal] + world / 2) % world - world / 2
        dist = np.hypot(off[:, 0], off[:, 1])
        bearing = (np.arctan2(off[:, 1], off[:, 0]) - heading[focal] + np.pi) % (
            2 * np.pi
        ) - np.pi
        hf = heading[focal]
        ahead = off[:, 0] * np.cos(hf) + off[:, 1] * np.sin(hf)
        right = off[:, 0] * np.sin(hf) - off[:, 1] * np.cos(hf)
        # both the bearing histogram and the 2-D map use every neighbour within 7 cm,
        # as in Weinburd et al. 2024 (no inner exclusion)
        act, hop = active[focal], hopping[focal]
        for state, sel in (
            ("active", act),
            ("inactive", ~act),
            ("walking", act & ~hop),
            ("hopping", act & hop),
            ("stationary", ~act),
            ("all", np.ones(len(focal), dtype=bool)),
        ):
            hist[state] += np.histogram(bearing[sel], bins=edges)[0]
            maps[state] += np.histogram2d(
                right[sel], ahead[sel], bins=[map_edges, map_edges]
            )[0]
    out = {}
    for state, h in hist.items():
        out[state] = {
            "count": int(h.sum()),
            "relative": (h / max(h.mean(), 1e-12)).tolist(),
            "map": maps[state].tolist(),
        }  # raw counts, [right, ahead] indexing
    out["centres_deg"] = np.degrees((edges[:-1] + edges[1:]) / 2).tolist()
    out["map_edges_cm"] = (map_edges * 100).tolist()
    return out


def compute(set_name="validation", cell=CELL, stride=STRIDE):
    """Pool the counts over every replicate of each run in the set and cache the result."""
    data = {}
    for model, trial, label, run_root in SETS[set_name]:
        # either a trial dir holding replicate-*/ runs, or a single run dir holding snapshots/
        reps = sorted((ROOT / run_root).glob("replicate-*")) or [ROOT / run_root]
        pooled = bearing_histograms(reps[0], cell, stride)
        for rep in reps[1:]:
            h = bearing_histograms(rep, cell, stride)
            for state in STATES:
                n0, n1 = pooled[state]["count"], h[state]["count"]
                pooled[state]["relative"] = (
                    (
                        np.array(pooled[state]["relative"]) * n0
                        + np.array(h[state]["relative"]) * n1
                    )
                    / max(n0 + n1, 1)
                ).tolist()
                pooled[state]["count"] = n0 + n1
                pooled[state]["map"] = (
                    np.array(pooled[state]["map"]) + np.array(h[state]["map"])
                ).tolist()
        pooled["replicates"] = len(reps)
        data[label] = pooled
        print(
            label,
            f"{len(reps)} reps",
            {s: pooled[s]["count"] for s in STATES},
            flush=True,
        )
    path = cache_path(set_name, cell, stride)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def load(set_name="validation", cell=CELL, stride=STRIDE, recompute=False):
    path = cache_path(set_name, cell, stride)
    if recompute or not path.exists():
        return compute(set_name, cell, stride)
    data = json.loads(path.read_text())
    # a cache written under an older state split (e.g. walking/stationary) or run list is stale
    labels = [run[2] for run in SETS[set_name]]
    if set(labels) != set(data) or any(s not in data[label] for label in labels for s in STATES):
        print(f"{path.name}: stale (states/runs changed), recomputing", flush=True)
        return compute(set_name, cell, stride)
    return data


def model_subset(set_name, model):
    """Register and return `<set>-<model>`: the runs of one model (nf / spin / spp) only."""
    if model is None:
        return set_name
    name = f"{set_name}-{model}"
    SETS[name] = [run for run in SETS[set_name] if run[0] == model]
    if not SETS[name]:
        raise SystemExit(f"set {set_name!r} has no {model} runs")
    return name


def chosen_set(a):
    """Set name selected on the command line: `--runs` registers an ad-hoc set of run
    dirs, `--model` narrows any set to one model (cached and named separately)."""
    name = register_runs(a.runs) if a.runs else a.set
    return model_subset(name, getattr(a, "model", None))


def data_arguments(ap):
    ap.add_argument(
        "--set",
        choices=sorted(SETS),
        default="validation",
        help="which run set to analyse (scale-25k / scale-100k / scale-250k = the ladder cells)",
    )
    ap.add_argument(
        "--model",
        choices=["nf", "spin", "col", "spp"],
        default=None,
        help="keep only this model's runs from the chosen set (e.g. --set scale-250k --model nf)",
    )
    ap.add_argument(
        "--runs",
        nargs="*",
        metavar="DIR[:LABEL]",
        help="analyse these run directories instead of a named set (a run dir with snapshots/ "
             "+ run.json, or a trial dir of replicate-*/ runs), e.g. "
             "runs/final_scale_validation/nf-531-250k:'NF 531, 250k'",
    )
    ap.add_argument(
        "--cell", type=float, default=CELL * 1000, help="heatmap cell size [mm]"
    )
    ap.add_argument(
        "--stride", type=int, default=STRIDE, help="use every n-th post-burn-in frame"
    )
    ap.add_argument(
        "--recompute", action="store_true", help="re-read the snapshot streams"
    )
    ap.add_argument(
        "--states",
        default=",".join(DRAW_STATES),
        help="comma-separated focal states to draw: " + ", ".join(STATES),
    )


def plot(data, runs=RUNS, states=DRAW_STATES):
    """Polar panels of relative density vs bearing (walking and stationary focals):
    one row per model, one column per run."""
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "figure.dpi": 150})
    models = list(dict.fromkeys(run[0] for run in runs))
    n_cols = max(sum(1 for run in runs if run[0] == m) for m in models)
    fig, axes = plt.subplots(
        len(models),
        n_cols,
        figsize=(2.5 * n_cols, 3.3 * len(models)),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )
    theta_fine = np.linspace(-np.pi, np.pi, 361)
    field = (
        1 - 2 * FIELD_M1 * np.cos(theta_fine) - 2 * FIELD_M2 * np.cos(2 * theta_fine)
    )
    slots = []
    for r_i, m in enumerate(models):
        model_runs = [run for run in runs if run[0] == m]
        slots += [(axes[r_i, c_i], run) for c_i, run in enumerate(model_runs)]
        for ax in axes[r_i, len(model_runs) :]:
            ax.set_visible(False)
    for ax, (model, _, label, _) in slots:
        d = data[label]
        colour = MODEL_COLORS.get(model, SPIN_COLOR)
        theta = np.radians(d["centres_deg"] + [d["centres_deg"][0]])  # close the loop
        line_style = {"active": ("-", 1.0), "inactive": ("--", 0.55)}
        for state in states:
            ls, alpha = line_style.get(state, ("-", 1.0))
            r = d[state]["relative"] + [d[state]["relative"][0]]
            ax.plot(
                theta,
                r,
                ls,
                color=colour,
                alpha=alpha,
                lw=1.6,
                label=f"{state} (n = {d[state]['count']:,})",
            )
        ax.plot(theta_fine, np.ones_like(theta_fine), color="#9a9a9a", lw=0.8)
        ax.plot(theta_fine, field, color="#9a9a9a", lw=0.8, ls=":")
        ax.set_theta_zero_location("N")  # ahead = up
        ax.set_theta_direction(
            -1
        )  # drawn clockwise, so a positive (counter-clockwise) bearing = left
        ax.set_thetagrids([0, 90, 180, 270], ["ahead", "left", "behind", "right"])
        ax.set_ylim(0, 1.5)
        ax.set_yticks([0.5, 1.0, 1.5])
        ax.set_yticklabels(["0.5", "1", "1.5"], color="#6f6f6f")
        ax.tick_params(axis="x", pad=-2)
        ax.grid(color="#d8d8d8", lw=0.6)
        ax.set_title(label, pad=10)
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=7
        )
    fig.suptitle(
        "Relative neighbour density by bearing, 1–7 cm annulus (1 = isotropic; "
        "dotted grey = Weinburd 2024 walking-focal moments)",
        y=0.995,
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.04, right=0.98, top=0.9, bottom=0.08, wspace=0.35, hspace=0.75
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "neighbour_density.svg", bbox_inches="tight")
    fig.savefig(OUT / "neighbour_density.png", dpi=200, bbox_inches="tight")
    print("wrote", OUT / "neighbour_density.svg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    data_arguments(ap)
    a = ap.parse_args()
    name = chosen_set(a)
    plot(load(name, a.cell / 1000, a.stride, a.recompute), SETS[name], tuple(a.states.split(",")))
