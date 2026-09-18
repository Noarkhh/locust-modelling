"""Render a snapshot stream in fixed world coordinates (no camera tracking).

The full torus is drawn; positions are stored as uint16 fractions of the
world extent (~2 mm resolution). The vertical axis is exaggerated so the
3.6 m-tall strip is visible against the 113.7 m width.
"""

import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/noarkhh/Masters/locust-modelling/paramsearch")
from paramsearch.metrics import load_snapshots

snapshot_dir = Path(sys.argv[1])
world_width = float(sys.argv[2])
world_height = float(sys.argv[3])
timestep = float(sys.argv[4])
output_path = Path(sys.argv[5])

records = load_snapshots(snapshot_dir)
iterations = np.unique(records["iter"])
frames = []
for iteration in iterations:
    snap = records[records["iter"] == iteration]
    snap = snap[np.argsort(snap["id"])]
    x = snap["x"].astype(np.float64) % world_width
    y = snap["y"].astype(np.float64) % world_height
    headings = snap["heading"].astype(np.float64)
    order = float(np.linalg.norm([np.cos(headings).mean(), np.sin(headings).mean()]))
    payload = struct.pack("<Lf", int(iteration), order)
    per_agent = np.zeros(len(snap), dtype=[("x", "<u2"), ("y", "<u2"), ("h", "u1"), ("f", "u1")])
    per_agent["x"] = np.clip(x / world_width * 65535, 0, 65535)
    per_agent["y"] = np.clip(y / world_height * 65535, 0, 65535)
    per_agent["h"] = ((headings % (2 * np.pi)) / (2 * np.pi) * 255).astype("u1")
    per_agent["f"] = snap["flags"]
    frames.append(payload + per_agent.tobytes())

blob = b"".join(struct.pack("<L", len(f)) + f for f in frames)
meta = {
    "worldWidth": world_width, "worldHeight": world_height,
    "timestep": timestep, "frameCount": len(frames),
    "agentCount": int((len(frames[0]) - 8) / 6),
}
template = (Path(__file__).parent / "viz_template_world.html").read_text()
output_path.write_text(
    template.replace("__META__", json.dumps(meta)).replace("__DATA__", base64.b64encode(blob).decode())
)
print(f"{len(frames)} frames -> {output_path} ({output_path.stat().st_size/1e6:.1f} MB)")
