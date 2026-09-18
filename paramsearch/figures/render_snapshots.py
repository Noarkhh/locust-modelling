"""Render a snapshot stream into a self-contained HTML canvas animation.

The camera follows the band's circular center of mass; agents are colored by
heading (cyclic HSV), dimmed when paused, brightened when hopping. A side
panel shows the live along-band density histogram and global order.
"""

import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/noarkhh/Masters/locust-modelling/paramsearch")
from paramsearch.metrics import SNAPSHOT_DTYPE, _circular_center_of_mass, load_snapshots

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
    positions = np.column_stack([snap["x"], snap["y"]]).astype(np.float64) % (world_width, world_height)
    center = _circular_center_of_mass(positions, (world_width, world_height))
    headings = snap["heading"].astype(np.float64)
    order = float(np.linalg.norm([np.cos(headings).mean(), np.sin(headings).mean()]))
    # Pack per-frame binary: center (2f), order (f), then per agent
    # relative min-image position (2 x int16, mm precision) + heading
    # (uint8 turn) + flags (uint8).
    relative = (positions - center + [world_width / 2, world_height / 2]) % [world_width, world_height] - [world_width / 2, world_height / 2]
    payload = struct.pack("<ffLf", *center, int(iteration), order)
    quantized_xy = np.clip(relative * 1000, -32000, 32000).astype("<i2")
    heading_byte = ((headings % (2 * np.pi)) / (2 * np.pi) * 255).astype("u1")
    per_agent = np.zeros(len(snap), dtype=[("x", "<i2"), ("y", "<i2"), ("h", "u1"), ("f", "u1")])
    per_agent["x"], per_agent["y"] = quantized_xy[:, 0], quantized_xy[:, 1]
    per_agent["h"] = heading_byte
    per_agent["f"] = snap["flags"]
    frames.append(payload + per_agent.tobytes())

blob = b"".join(struct.pack("<L", len(f)) + f for f in frames)
encoded = base64.b64encode(blob).decode()
meta = {
    "worldWidth": world_width, "worldHeight": world_height,
    "timestep": timestep, "frameCount": len(frames), "agentCount": int((len(frames[0]) - 16) / 6),
}
template = (Path(__file__).parent / "viz_template.html").read_text()
output_path.write_text(
    template.replace("__META__", json.dumps(meta)).replace("__DATA__", encoded)
)
print(f"{len(frames)} frames, {len(blob)/1e6:.1f} MB raw -> {output_path} ({output_path.stat().st_size/1e6:.1f} MB)")
