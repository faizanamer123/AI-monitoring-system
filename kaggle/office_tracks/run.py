"""
Kaggle job: run the pose model + tracker over every MM-Office test video, once, on a GPU,
and save the per-video track caches (office/cache.py) as the notebook's output.

Pushed and started from a laptop with:
    kaggle kernels push -p kaggle/office_tracks --accelerator NvidiaTeslaT4
Results come back with:
    kaggle kernels output waqasshoaib07/office-monitor-tracks -p <dir>

Everything the rules need afterwards is in tracks/; tuning and scoring happen locally
with `office_monitor.py --tracks` and `evaluate_office.py`, in seconds, with no GPU.

Needs internet (GitHub for the code, Zenodo for the videos), which Kaggle only allows
on phone-verified accounts.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/faizanamer123/AI-monitoring-system.git"
BRANCH = "office-monitor"
CODE = Path("/kaggle/temp/repo")
DATA = Path("/kaggle/temp/mm-office")
OUT = Path("/kaggle/working")


# child processes inherit this: their progress lines reach the Kaggle log as they are
# printed instead of in 8 KB bursts
os.environ["PYTHONUNBUFFERED"] = "1"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


started = time.time()
sh(f"git clone --depth 1 --branch {BRANCH} {REPO} {CODE}")
sh("pip install -q ultralytics lap")
os.chdir(CODE)
sys.path.insert(0, str(CODE))

import fetch_mm_office as fetch  # noqa: E402

fetch.set_root(DATA)
files = [f for f in fetch.central_directory() if not f[0].endswith("/")]
labels = [f for f in files if "/label/" in f[0]]
videos = sorted(f for f in files if f[0].endswith(".mp4") and "/video/test/" in f[0])
print(f"{len(labels)} label files, {len(videos)} test videos "
      f"({sum(f[3] for f in videos) / 1e6:.0f} MB)", flush=True)

for entry in labels:
    fetch.extract(entry)
paths = []
for i, entry in enumerate(videos, 1):
    paths.append(fetch.extract(entry))
    if i % 25 == 0 or i == len(videos):
        print(f"  downloaded {i}/{len(videos)}  ({time.time() - started:.0f}s)", flush=True)

listing = DATA / "videos.txt"
listing.write_text("\n".join(str(p) for p in paths) + "\n")
sh(f"python office_monitor.py --video {listing} --weights yolo11n-pose.pt "
   f"--save-tracks {OUT / 'tracks'} --out {OUT / 'events_at_push'}")

# the labels travel with the results, so scoring needs nothing else from Zenodo
sh(f"cp -r {DATA / 'MM_Office_Dataset' / 'label'} {OUT / 'label'}")
print(f"done in {(time.time() - started) / 60:.0f} min", flush=True)
