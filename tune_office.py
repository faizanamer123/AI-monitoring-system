"""
Measure one rule setting at a time on the MM-Office tuning split.

    python tune_office.py TRACKS_DIR OUT_DIR NAME [--confirm-s 3] [--stitch-gap 5]
                                                  [--hold 3] [--edge-only]

Replays the cached tracks with those settings and prints the fused scores, once per
camera-fusion method. TRACKS_DIR should hold only tuning-split caches.
"""

import argparse
import subprocess
import sys
from pathlib import Path

LABELS = "../mm-office/MM_Office_Dataset/label/testlabel"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tracks")
    parser.add_argument("out")
    parser.add_argument("name")
    parser.add_argument("--confirm-s", default="0")
    parser.add_argument("--stitch-gap", default="0")
    parser.add_argument("--hold", default="0")
    parser.add_argument("--edge-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.out) / args.name
    subprocess.run([sys.executable, "office_monitor.py", "--tracks", args.tracks, "--out", str(out),
                    "--frame-size", "320x240", "--confirm-s", args.confirm_s,
                    "--stitch-gap", args.stitch_gap],
                   check=True, stdout=subprocess.DEVNULL)
    score = [sys.executable, "evaluate_office.py", "--labels", LABELS, "score",
             str(out / "events.csv"), "--split", "tuning", "--hold", args.hold]
    if args.edge_only:
        score.append("--edge-only")
    report = subprocess.run(score, check=True, capture_output=True, text=True).stdout
    # the two fused tables; the per-camera ones remain available from evaluate_office
    print(f"=== {args.name}")
    print(report.split("camera id00")[0].strip())


if __name__ == "__main__":
    main()
