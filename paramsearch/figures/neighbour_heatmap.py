"""Heatmap of the relative neighbour density around a focal locust, for the
validation-campaign top-4 of each model (data from neighbour_density.py).

4 x 4 grid: rows = NF walking / NF stationary / spin walking / spin stationary,
columns = the four trials. Each panel is the 2-D map of neighbour offsets in the
focal's frame (ahead = up, right = right), 0-7 cm, divided by the mean count of
the cells in the same radial shell, so the radial density profile drops out and
colour shows angular structure only: 1 = isotropic at that radius, red = excess,
blue = deficit (log scale, 1/2x .. 2x). Cells not wholly inside the 7 cm disc are
masked.

  python figures/neighbour_heatmap.py [--cell MM] [--stride N] [--recompute] [--range R]
      -> figures/output/static/neighbour_density_heatmap.{svg,png}
"""

import argparse
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from neighbour_density import (
    CELL,
    INNER,
    OUT,
    OUTER,
    SETS,
    STRIDE,
    chosen_set,
    data_arguments,
    load,
)


def relative_map(counts, edges_cm, norm="shell"):
    """Counts -> relative density.

    norm="shell":  each cell / mean of the cells in its radial shell (isotropic
                   expectation at that radius; radial profile removed, angular
                   structure only).
    norm="window": Weinburd et al. 2024 / Buhl et al. 2012 convention (fig4.m):
                   density per cm^2 divided by total neighbours / area of the
                   (2 x 7 cm)^2 window, i.e. relative to the window-average
                   density; the radial profile is kept."""
    counts = np.asarray(counts, dtype=float)
    edges_cm = np.asarray(edges_cm)
    cell_cm = edges_cm[1] - edges_cm[0]
    centres = (edges_cm[:-1] + edges_cm[1:]) / 2
    xc, yc = np.meshgrid(centres, centres, indexing="ij")  # [right, ahead]
    r = np.hypot(xc, yc)
    # cells lying wholly inside the annulus (partial cells are under-counted by geometry)
    half_diag = cell_cm * np.sqrt(2) / 2
    inside = r + half_diag <= OUTER * 100  # full disc down to the focal itself
    if norm == "window":
        # Weinburd's window is filled to its corners by unrestricted neighbours, so their
        # window average equals the average over the plotted area; our counts stop at 7 cm,
        # so the equivalent is the mean over the displayed (annulus) cells -> uniform = 1.
        expected = np.full_like(counts, counts[inside].mean())
    else:
        shell = np.floor(r / cell_cm).astype(int)
        expected = np.zeros_like(counts)
        for s in np.unique(shell):
            cells = shell == s
            expected[cells] = counts[cells].mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = counts / expected
    rel[~inside] = np.nan
    return rel


DEFAULT_STATES = ("active", "inactive")


def panel_layout(runs, states, ncols=None):
    """Panels in drawing order and their (row, col) slots.

    ncols=None: one row per (model, state), one column per run of that model
    (rows of different models may leave trailing slots empty).
    ncols=N:    all panels in a plain grid, state-major (all runs for the first
    state, then the next state), wrapped every N columns."""
    if ncols is None:
        models = list(dict.fromkeys(run[0] for run in runs))
        n_cols = max(sum(1 for run in runs if run[0] == m) for m in models)
        slots, r = [], 0
        for m in models:
            for s in states:
                for c, run in enumerate(run for run in runs if run[0] == m):
                    slots.append((r, c, run, s))
                r += 1
        return slots, r, n_cols
    panels = [(run, s) for s in states for run in runs]
    n_rows = -(-len(panels) // ncols)
    return (
        [(i // ncols, i % ncols, run, s) for i, (run, s) in enumerate(panels)],
        n_rows,
        ncols,
    )


def plot_heatmap(
    data,
    runs,
    set_name,
    cell_mm,
    log_range=1.0,
    norm="shell",
    vmax_window=1.6,
    states=DEFAULT_STATES,
    ncols=None,
):
    """One panel per (run, focal state); arrangement per `panel_layout`.
    norm="shell": log2 ratio, diverging map centred on 1.
    norm="window": Weinburd-style linear 0..vmax_window, turbo map (their fig4 used 0..1.6).
    """
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "figure.dpi": 150})
    slots, n_rows, n_cols = panel_layout(runs, states, ncols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols + 1.2, 2.7 * n_rows + 0.5),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    used = np.zeros((n_rows, n_cols), dtype=bool)
    lim = OUTER * 100
    im = None
    for r_i, c_i, (_, _, label, _), state in slots:
        ax = cast(Axes, axes[r_i, c_i])
        used[r_i, c_i] = True
        d = data[label]
        rel = relative_map(d[state]["map"], d["map_edges_cm"], norm)
        # imshow wants [row = ahead (Δy), col = right (Δx)] with Δy increasing upward
        if norm == "window":
            im = ax.imshow(
                rel.T,
                origin="lower",
                extent=(-lim, lim, -lim, lim),
                cmap="turbo",
                vmin=0,
                vmax=vmax_window,
                interpolation="nearest",
            )
        else:
            with np.errstate(divide="ignore"):
                im = ax.imshow(
                    np.log2(rel).T,
                    origin="lower",
                    extent=(-lim, lim, -lim, lim),
                    cmap="RdBu_r",
                    vmin=-log_range,
                    vmax=log_range,
                    interpolation="nearest",
                )
        # focal marker: white dot + white heading arrow, both black-outlined, as in Weinburd's fig. 4
        ax.plot(0, 0, "o", ms=7, mfc="white", mec="black", mew=1.0, zorder=3)
        ax.annotate(
            "",
            xy=(0, 2.2),
            xytext=(0, 0),
            zorder=4,
            arrowprops=dict(
                arrowstyle="simple,head_width=0.9,head_length=1.0,tail_width=0.32",
                fc="white",
                ec="black",
                lw=1.0,
                shrinkA=0,
                shrinkB=0,
            ),
        )
        ax.set_title(f"{label} — {state}\n(n = {d[state]['count']:,})", fontsize=8.5)
        ax.set_aspect("equal")
        # same ticks on both axes (the panel is square): every 2 cm, −6 … 6
        tick_values = np.arange(-6, 6.1, 2)
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        # shared axes hide inner tick labels by default; every panel gets its numbers
        ax.tick_params(length=2, labelsize=7.5, colors="#6f6f6f", labelbottom=True, labelleft=True)
        for spine in ax.spines.values():
            spine.set_color("#c8c8c8")
    # hide empty slots; axis titles on the lowest used panel of each column / leftmost of each row
    for r_i in range(n_rows):
        for c_i in range(n_cols):
            if not used[r_i, c_i]:
                axes[r_i, c_i].set_visible(False)
    for c_i in range(n_cols):
        rows_used = np.flatnonzero(used[:, c_i])
        if rows_used.size:
            axes[rows_used[-1], c_i].set_xlabel("Δx [cm]", fontsize=8)
            axes[rows_used[-1], c_i].tick_params(labelbottom=True)
    for r_i in range(n_rows):
        cols_used = np.flatnonzero(used[r_i])
        if cols_used.size:
            axes[r_i, cols_used[0]].set_ylabel("Δy [cm]", fontsize=8)
    if norm == "window":
        what = "1 = window-average density (Weinburd 2024 / Buhl 2012 convention)"
    else:
        what = "1 = isotropic at that radius"
    # fig.suptitle(f"Relative neighbour density around a focal locust, 0–7 cm, {cell_mm:g} mm cells", fontsize=9)
    assert im is not None, "no panels drawn"
    cb = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.6 if n_rows > 2 else 0.9,
        pad=0.02,
        aspect=30,
    )
    if norm == "window":
        cb.set_ticks(np.arange(0, vmax_window + 1e-9, 0.4).tolist())
    else:
        cb.set_ticks([-log_range, 0, log_range])
        cb.set_ticklabels([f"1/{2 ** log_range:g}×", "1", f"{2 ** log_range:g}×"])
    cb.set_label("relative density", fontsize=8)
    cb.ax.tick_params(labelsize=7.5, colors="#6f6f6f")
    OUT.mkdir(parents=True, exist_ok=True)
    stem = (
        "neighbour_density_heatmap"
        + ("" if set_name == "validation" else f"_{set_name}")
        + ("" if tuple(states) == DEFAULT_STATES else "_" + "-".join(states))
        + ("" if ncols is None else f"_grid{ncols}")
        + ("_window" if norm == "window" else "")
    )
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    print("wrote", OUT / f"{stem}.svg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    data_arguments(ap)
    ap.add_argument(
        "--range",
        type=float,
        default=1.0,
        help="shell norm: colour range as log2 (1 = 1/2x .. 2x)",
    )
    ap.add_argument(
        "--norm",
        choices=["shell", "window"],
        default="shell",
        help="shell = per-radius isotropic expectation; window = Weinburd/Buhl window average",
    )
    ap.add_argument(
        "--vmax",
        type=float,
        default=1.6,
        help="window norm: top of the linear colour scale",
    )
    ap.add_argument(
        "--ncols",
        type=int,
        default=None,
        help="arrange all panels in a plain grid with this many columns "
        "(default: one row per model and state, one column per run)",
    )
    a = ap.parse_args()
    name = chosen_set(a)
    plot_heatmap(
        load(name, a.cell / 1000, a.stride, a.recompute),
        SETS[name],
        name,
        a.cell,
        a.range,
        a.norm,
        a.vmax,
        tuple(a.states.split(",")),
        a.ncols,
    )
