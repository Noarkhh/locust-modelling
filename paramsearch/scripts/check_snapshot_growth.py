"""Guard against the silent snapshot-loss failure: verify every run under a
parent directory is actively writing snapshots.

Run it a few minutes after launching a batch of runs; it fails loudly (exit 1)
listing any cell whose snapshots/ is below a size threshold, so a
non-writing run is caught within minutes instead of after it completes.

  python scripts/check_snapshot_growth.py <parent_dir> [--min-mb 5]
"""

import argparse
import sys
from pathlib import Path


def dir_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*.bin")) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path, help="dir holding per-run subdirs")
    parser.add_argument("--min-mb", type=float, default=5.0,
                        help="minimum snapshots/ size [MB] to count as writing")
    arguments = parser.parse_args()

    runs = sorted(p for p in arguments.parent.iterdir()
                  if p.is_dir() and (p / "snapshots").exists())
    if not runs:
        print(f"no run dirs with snapshots/ under {arguments.parent}")
        sys.exit(1)

    stalled = []
    for run in runs:
        mb = dir_mb(run / "snapshots")
        state = "ok" if mb >= arguments.min_mb else "STALLED"
        if mb < arguments.min_mb:
            stalled.append(run.name)
        print(f"  {run.name:32s} {mb:9.1f} MB  {state}")

    if stalled:
        print(f"\nFAIL: {len(stalled)} run(s) not writing snapshots "
              f"(< {arguments.min_mb} MB): {', '.join(stalled)}")
        sys.exit(1)
    print(f"\nOK: all {len(runs)} runs are writing snapshots.")


if __name__ == "__main__":
    main()
