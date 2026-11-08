"""
Measure one rule setting on the MM-Office tuning split.

    python tune_office.py ../track_caches/v1/tuning OUT_DIR NAME [--confirm-s 3] [--stitch-gap 5]

Replays the cached tracks with those settings, then prints the scores: both camera
fusion methods, then each camera alone. Point it at tuning caches only.
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
    parser.add_argument("--confirm-s", default="3")
    parser.add_argument("--stitch-gap", default="5")
    args = parser.parse_args()

    out = Path(args.out) / args.name
    subprocess.run([sys.executable, "office_monitor.py", "--tracks", args.tracks, "--out", str(out),
                    "--frame-size", "320x240", "--confirm-s", args.confirm_s,
                    "--stitch-gap", args.stitch_gap],
                   check=True, stdout=subprocess.DEVNULL)
    report = subprocess.run([sys.executable, "evaluate_office.py", "--labels", LABELS, "score",
                             str(out / "events.csv"), "--split", "tuning",
                             "--report", str(out / "mistakes.csv")],
                            check=True, capture_output=True, text=True).stdout
    (out / "scores.txt").write_text(report)
    print(f"=== {args.name}\n{report.strip()}")


if __name__ == "__main__":
    main()
