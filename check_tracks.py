"""
Check a folder of track caches against a split: which are missing, which are corrupt.

    python check_tracks.py ../mm-office/kaggle_tracks_v1/tracks --split tuning

A download that drops mid-file leaves a truncated .npz that only fails when the rules
are replayed, so every cache is opened and read in full here first.
"""

import argparse
import zipfile
from pathlib import Path

import numpy as np

import evaluate_office


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tracks")
    parser.add_argument("--split", default="tuning", choices=["tuning", "final", "all"])
    parser.add_argument("--delete-corrupt", action="store_true",
                        help="remove unreadable caches so a re-download fetches them again")
    args = parser.parse_args()

    wanted = set(evaluate_office.split(args.split))
    expected = {(r, cam) for r in wanted for cam in evaluate_office.CAMERAS}
    found, corrupt = set(), []
    for path in sorted(Path(args.tracks).glob("*.npz")):
        m = evaluate_office.SOURCE.match(path.stem)
        if not m or int(m["recid"]) not in wanted:
            continue
        try:
            with np.load(path) as data:
                for key in data.files:
                    data[key]
            found.add((int(m["recid"]), m["camera"]))
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
            corrupt.append((path, error))

    missing = sorted(expected - found)
    print(f"{args.split}: {len(found)}/{len(expected)} caches readable, "
          f"{len(corrupt)} corrupt, {len(missing) - len(corrupt)} not downloaded")
    for path, error in corrupt:
        print(f"  corrupt: {path.name} ({error})")
        if args.delete_corrupt:
            path.unlink()


if __name__ == "__main__":
    main()
