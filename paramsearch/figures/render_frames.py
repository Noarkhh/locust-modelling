"""Render every frame of a snapshot stream with the render_frame pipeline (all
agents, no subsampling) into figures/output/dynamic/<run_name>/frames/, then
encode a viewing mp4 (CRF 18) and optionally a lossless mkv next to them.

  python render_frames.py <snap_dir> <run_name> [--lo 12000 --hi 36000 --step 100]
                          [--fps 10] [--lossless] [style options of render_frame]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from render_frame import DYNAMIC_OUT, build_index, profile_max, render_one, style_arguments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap_dir"); ap.add_argument("run_name")
    ap.add_argument("--lo", type=int, default=None); ap.add_argument("--hi", type=int, default=None)
    ap.add_argument("--step", type=int, default=1, help="keep every n-th available iteration")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--lossless", action="store_true", help="also encode a bit-exact libx264rgb mkv")
    ap.add_argument("--no-video", action="store_true")
    style_arguments(ap)
    a = ap.parse_args()

    iterations = sorted(build_index(a.snap_dir))
    iterations = [i for i in iterations if (a.lo is None or i >= a.lo) and (a.hi is None or i <= a.hi)][::a.step]
    run_dir = DYNAMIC_OUT / a.run_name
    fdir = run_dir / "frames"
    shutil.rmtree(fdir, ignore_errors=True); fdir.mkdir(parents=True)
    print(f"{a.run_name}: {len(iterations)} frames {iterations[0]}..{iterations[-1]} -> {fdir}", flush=True)
    # one histogram scale for the whole video: the simulation-wide profile maximum
    prof_max = profile_max(a.snap_dir, iterations, a)
    print(f"  fixed histogram scale: {prof_max:.1f} agents/m²", flush=True)
    for k, it in enumerate(iterations):
        render_one(a.snap_dir, it, fdir / f"frame-{k:04d}.png", a, prof_max=prof_max)
        if k % 20 == 0:
            print(f"  frame {k}/{len(iterations)} (iter {it})", flush=True)
    if a.no_video:
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found; frames written, no video", file=sys.stderr); return
    src = ["-framerate", str(a.fps), "-i", str(fdir / "frame-%04d.png")]
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", *src, "-c:v", "libx264", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt", "yuv420p", "-crf", "18", str(run_dir / f"{a.run_name}.mp4")], check=True)
    print("encoded", run_dir / f"{a.run_name}.mp4")
    if a.lossless:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", *src, "-c:v", "libx264rgb", "-qp", "0",
                        str(run_dir / f"{a.run_name}-lossless.mkv")], check=True)
        print("encoded", run_dir / f"{a.run_name}-lossless.mkv")


if __name__ == "__main__":
    main()
