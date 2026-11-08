"""
Build a MEVA Kaggle job from the committed code, without pushing anything to GitHub.

    python kaggle/meva_test/build.py OUT_DIR --clips 30 --highlights 10
    python kaggle/meva_test/build.py OUT_DIR --remainder --objects off --name office-monitor-skeletons
    kaggle kernels push -p OUT_DIR --accelerator NvidiaTeslaT4

Packs git HEAD's code into a private script. Run from the repository root.
"""

import argparse
import base64
import json
import subprocess
from pathlib import Path

FILES = ["office", "office_monitor.py", "detect_hands.py", "evaluate_office.py",
         "evaluate_meva.py", "render_results.py"]
HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--clips", type=int, default=30, help="clips per camera in the test set")
    parser.add_argument("--highlights", type=int, default=10)
    parser.add_argument("--remainder", action="store_true",
                        help="process the labelled clips outside the test set, for training")
    parser.add_argument("--objects", default="person", choices=["person", "frame", "off"])
    parser.add_argument("--name", default="office-monitor-meva", help="Kaggle kernel name")
    args = parser.parse_args()

    archive = subprocess.run(["git", "archive", "--format=tar.gz", "HEAD", *FILES],
                             check=True, capture_output=True).stdout
    script = ((HERE / "run_template.py").read_text()
              .replace("@@CODE@@", base64.b64encode(archive).decode())
              .replace("@@CLIPS@@", str(args.clips))
              .replace("@@HIGHLIGHTS@@", str(args.highlights))
              .replace("@@REMAINDER@@", str(args.remainder))
              .replace("@@OBJECTS@@", args.objects))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(script)
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"waqasshoaib07/{args.name}", "title": args.name,
        "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "dataset_sources": [],
        "competition_sources": [], "kernel_sources": []}, indent=2))
    print(f"wrote {out}/run.py ({len(script) // 1024} KB) for {args.name}")


if __name__ == "__main__":
    main()
