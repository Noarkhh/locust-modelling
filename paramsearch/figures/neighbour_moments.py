"""Trigonometric moments of relative neighbour angles (Weinburd et al. 2024,
Appendix F / Table 4) for the run sets defined in neighbour_density.py, laid out
column-for-column against the paper's Table 4.

Per run and motion state: N, |M1|, psi1, Ms1, Mc1, |M2|, psi2, Ms2, Mc2, computed
over all neighbours within 7 cm of each focal (no inner cut), focal facing up,
phi = 0 to the right. Replicates are pooled by count-weighting the moments
(equivalent to pooling the angles).

  python figures/neighbour_moments.py [--set top4|spp|spin-new ...] [--stride N]
      -> figures/output/static/neighbour_moments.md  (+ .json)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paramsearch.validation import WEINBURD_TABLE4, neighbour_anisotropy_v2
from neighbour_density import OUT, SETS, model_subset, register_runs

BASE_STATES = ("stationary", "walking", "hopping")  # Weinburd's three motion states
# pooled states, as unions of the base ones (moments are means, so they pool by count)
DERIVED = {"active": ("walking", "hopping"), "inactive": ("stationary",), "all": BASE_STATES}
STATES = BASE_STATES  # default table rows
COLS = ("M1", "psi1", "Ms1", "Mc1", "M2", "psi2", "Ms2", "Mc2")


def _blank():
    return {"count": 0, **{f"{k}{p}": 0.0 for k in ("Ms", "Mc") for p in (1, 2)}}


def _pool(acc, m, n):
    for key in ("Ms1", "Mc1", "Ms2", "Mc2"):
        acc[key] = (acc[key] * acc["count"] + m[key] * n) / (acc["count"] + n)
    acc["count"] += n


def pooled_moments(run_dirs, stride, states=STATES):
    """Count-weighted pooling of per-replicate moments -> same as pooling the angles.
    Base states come from the snapshot flags; derived ones (active, inactive, all)
    are count-weighted unions of them."""
    acc = {s: _blank() for s in BASE_STATES}
    for run_dir in run_dirs:
        ov = json.loads((run_dir / "run.json").read_text())["overrides"]
        res = neighbour_anisotropy_v2(run_dir / "snapshots", ov["worldWidthMeters"], ov["worldHeightMeters"],
                                      frame_stride=stride, burn_in_iteration=int(ov.get("snapshotStartIteration", 0)))
        for s in BASE_STATES:
            m = res["states"][s]
            if m is not None:
                _pool(acc[s], m, m["count"])
    for name, parts in DERIVED.items():
        acc[name] = _blank()
        for s in parts:
            if acc[s]["count"]:
                _pool(acc[name], acc[s], acc[s]["count"])
    out = {}
    for s in states:
        a = acc[s]
        if a["count"] < 50:
            out[s] = None
            continue
        row = {"count": a["count"]}
        for p in (1, 2):
            ms, mc = a[f"Ms{p}"], a[f"Mc{p}"]
            row.update({f"Ms{p}": ms, f"Mc{p}": mc, f"M{p}": float(np.hypot(mc, ms)),
                        f"psi{p}": float(np.arctan2(ms, mc) / p)})
        out[s] = row
    return out


def field_moments(states):
    """Weinburd's Table 4 rows, plus the same derived unions (active / inactive / all)
    pooled by their reported counts."""
    acc = {s: dict(WEINBURD_TABLE4[s]) for s in BASE_STATES}
    for name, parts in DERIVED.items():
        acc[name] = _blank()
        for s in parts:
            _pool(acc[name], acc[s], acc[s]["count"])
        for p in (1, 2):
            ms, mc = acc[name][f"Ms{p}"], acc[name][f"Mc{p}"]
            acc[name][f"M{p}"] = float(np.hypot(mc, ms))
            acc[name][f"psi{p}"] = float(np.arctan2(ms, mc) / p)
    return {s: acc[s] for s in states}


def fmt_row(label, state, m):
    if m is None:
        return f"| {label} | {state} | — |" + " — |" * len(COLS)
    return (f"| {label} | {state} | {m['count']:,} | " +
            " | ".join(f"{m[c]:+.4f}" if c.startswith(("Ms", "Mc")) else f"{m[c]:.4f}" for c in COLS) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="+", default=["validation", "spp"], choices=sorted(SETS))
    ap.add_argument("--runs", nargs="*", metavar="DIR[:LABEL]",
                    help="analyse these run directories instead of the named sets")
    ap.add_argument("--model", choices=["nf", "spin", "col", "spp"], default=None,
                    help="keep only this model's runs from each set")
    ap.add_argument("--stride", type=int, default=5, help="use every n-th post-burn-in frame")
    ap.add_argument("--states", default=",".join(STATES),
                    help="comma-separated focal states to tabulate: stationary, walking, hopping "
                         "(Weinburd's), active (= walking + hopping), inactive (= stationary), all")
    a = ap.parse_args()
    states = tuple(a.states.split(","))
    unknown = set(states) - set(BASE_STATES) - set(DERIVED)
    if unknown:
        raise SystemExit(f"unknown states: {sorted(unknown)}")

    results = {"field (Weinburd 2024, Table 4)": field_moments(states)}
    for set_name in ([register_runs(a.runs)] if a.runs else a.set):
        set_name = model_subset(set_name, a.model)
        for _, _, label, run_root in SETS[set_name]:
            run_dirs = sorted((ROOT / run_root).glob("replicate-*")) or [ROOT / run_root]
            results[label] = pooled_moments(run_dirs, a.stride, states)
            lead = states[0]
            w = results[label][lead]
            print(label, f"({len(run_dirs)} runs)",
                  f"{lead}: -Ms1=%.4f Mc2=%.4f |M2|=%.4f" % (-w["Ms1"], w["Mc2"], w["M2"]) if w else f"{lead}: n/a",
                  flush=True)

    lines = ["# Trigonometric moments of relative neighbour angles (Weinburd 2024 definition)", "",
             "All neighbours within 7 cm of the focal, focal facing up, phi = 0 to the right; "
             "Ms_p = <sin p phi>, Mc_p = <cos p phi>, |M_p| = hypot(Mc_p, Ms_p), psi_p = atan2(Ms_p, Mc_p)/p. "
             "Headline: -Ms1 > 0 = lower density in front / higher behind; Mc2 > 0 = high density left & right, "
             f"low front & back. Every {a.stride}-th post-burn-in frame; replicates pooled.", "",
             "| run | state | N | \\|M1\\| | psi1 | Ms1 | Mc1 | \\|M2\\| | psi2 | Ms2 | Mc2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, per_state in results.items():
        for s in states:
            lines.append(fmt_row(label, s, per_state.get(s)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "neighbour_moments.md").write_text("\n".join(lines) + "\n")
    (OUT / "neighbour_moments.json").write_text(json.dumps(results, indent=1))
    print("wrote", OUT / "neighbour_moments.md")


if __name__ == "__main__":
    main()
