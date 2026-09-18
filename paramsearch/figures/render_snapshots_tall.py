"""Render a TALL world (long front, short march axis) rotated 90 degrees:
screen-horizontal = world y (the front), screen-vertical = world x (march,
forward = up). Agent stride for large populations.

  python render_snapshots_tall.py <snap_dir> <world_width_x> <world_height_y> <dt> <stride> <title> <out.html>
"""

import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/noarkhh/Masters/locust-modelling/paramsearch")
from paramsearch.metrics import SNAPSHOT_DTYPE

def load_snapshots_single(path):
    data = np.fromfile(path, dtype=SNAPSHOT_DTYPE)
    return data[np.argsort(data["iter"], kind="stable")]

snap_dir, world_x, world_y, dt, stride, title, out = (
    sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
    int(sys.argv[5]), sys.argv[6], Path(sys.argv[7]))

# a directory may hold streams from several launches (reused OUT_DIR);
# use only the largest file = the current run
import glob as _glob, os as _os
_files = sorted(_glob.glob(str(Path(snap_dir) / "*.bin")), key=_os.path.getsize)
records = load_snapshots_single(_files[-1])
iterations = np.unique(records["iter"])
frames = []
for iteration in iterations:
    snap = records[records["iter"] == iteration]
    snap = snap[np.argsort(snap["id"])][::stride]
    march = snap["x"].astype(np.float64) % world_x     # short axis, forward
    front = snap["y"].astype(np.float64) % world_y     # long axis
    headings = snap["heading"].astype(np.float64)
    order = float(np.linalg.norm([np.cos(headings).mean(), np.sin(headings).mean()]))
    payload = struct.pack("<Lf", int(iteration), order)
    per_agent = np.zeros(len(snap), dtype=[("x", "<u2"), ("y", "<u2"), ("h", "u1"), ("f", "u1")])
    per_agent["x"] = np.clip(front / world_y * 65535, 0, 65535)   # screen horizontal
    per_agent["y"] = np.clip(march / world_x * 65535, 0, 65535)   # screen vertical (template flips)
    per_agent["h"] = ((headings % (2 * np.pi)) / (2 * np.pi) * 255).astype("u1")
    per_agent["f"] = snap["flags"]
    frames.append(payload + per_agent.tobytes())

blob = b"".join(struct.pack("<L", len(f)) + f for f in frames)
meta = {"worldWidth": world_y, "worldHeight": world_x, "timestep": dt,
        "frameCount": len(frames), "agentCount": int((len(frames[0]) - 8) / 6)}
template = (Path(__file__).parent / "viz_template_world.html").read_text().replace(
    "NF trial 305 — fixed world view", title)
# rotated view: the density profile must run along the MARCH axis
# (screen-vertical, world x) — bin wy instead of wx, right = front
template = template.replace(
    "counts[Math.min(Math.floor(wx / META.worldWidth * 160), 159)]++;",
    "counts[Math.min(Math.floor(wy / META.worldHeight * 160), 159)]++;")
template = template.replace(
    'pctx.fillText("along-world density (full torus)", 8, 12);',
    'pctx.fillText("along-march density (left = rear, right = front)", 8, 12);')
out.write_text(template.replace("__META__", json.dumps(meta)).replace("__DATA__", base64.b64encode(blob).decode()))
print(f"{len(frames)} frames, {meta['agentCount']} agents/frame -> {out} ({out.stat().st_size/1e6:.1f} MB)")
