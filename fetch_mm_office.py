"""
Pull individual files out of the MM-Office dataset without downloading all 12 GB.

Zenodo serves the dataset as MM_Office_Dataset.zip.001-.004. The zip's central
directory (the list of every file and its byte offset) sits at the end, in the 40 MB
.004 part. Read it once, then fetch only the byte range of each file we want with an
HTTP Range request.

    python fetch_mm_office.py list                    # print every file in the zip
    python fetch_mm_office.py get video/test/x.mp4 ...  # extract named files
    python fetch_mm_office.py get --from-file names.txt
    python fetch_mm_office.py sample --n 8            # a spread of test videos

Files land under --root (default ../mm-office) as MM_Office_Dataset/..., which is where
evaluate_office.py looks for them.

The dataset's licence is non-commercial research use; see the Zenodo record.
"""

import argparse
import struct
import time
import zlib
from pathlib import Path

import requests

BASE = "https://zenodo.org/api/records/6088572/files/MM_Office_Dataset.zip.{:03d}/content"
PART_SIZES = [4290772992, 4290772992, 4290772992, 44558133]
ROOT = Path("../mm-office")   # entries already start with MM_Office_Dataset/


def set_root(path):
    global ROOT
    ROOT = Path(path)


def get(url, headers=None, stream=False, attempts=6):
    """GET with backoff. Zenodo rate-limits (429) and drops connections when hundreds
    of range requests arrive in a row; one failure must not sink a multi-hour job."""
    for attempt in range(attempts):
        try:
            r = requests.get(url, headers=headers, stream=stream, timeout=120)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as error:
            if attempt == attempts - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  {error}; retrying in {wait}s")
            time.sleep(wait)


def last_part():
    cache = ROOT / "_download"
    path = cache / "MM_Office_Dataset.zip.004"
    if not path.is_file() or path.stat().st_size != PART_SIZES[3]:
        cache.mkdir(parents=True, exist_ok=True)
        print("downloading the 40 MB index part ...")
        with get(BASE.format(4), stream=True) as r:
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    return path.read_bytes()


def read_range(start, length):
    """Bytes [start, start+length) of the concatenated zip, across part boundaries."""
    out = b""
    offset = 0
    for index, size in enumerate(PART_SIZES, 1):
        lo, hi = max(start, offset), min(start + length, offset + size)
        if lo < hi:
            headers = {"Range": f"bytes={lo - offset}-{hi - offset - 1}"}
            r = get(BASE.format(index), headers=headers)
            if r.status_code != 206:
                raise SystemExit("server ignored the Range header; cannot fetch partially")
            out += r.content
        offset += size
    return out


def central_directory():
    """[(name, method, compressed_size, size, global_offset), ...]"""
    tail = last_part()
    total = sum(PART_SIZES)
    tail_start = total - len(tail)

    eocd = tail.rfind(b"PK\x05\x06")
    (_, disk, cd_disk, _, entries, cd_size, cd_offset, _) = struct.unpack(
        "<IHHHHIIH", tail[eocd:eocd + 22])
    if entries == 0xFFFF or cd_offset == 0xFFFFFFFF:
        locator = tail.rfind(b"PK\x06\x07", 0, eocd)
        (_, _, eocd64, _) = struct.unpack("<IIQI", tail[locator:locator + 20])
        record = tail[eocd64 - tail_start:]
        (_, _, _, _, disk, cd_disk, _, entries, cd_size, cd_offset) = struct.unpack(
            "<IQHHIIQQQQ", record[:56])
    if disk != 0 or cd_disk != 0:
        raise SystemExit("this is a multi-disk spanned zip; offsets are per part -- unsupported")

    cd = tail[cd_offset - tail_start:cd_offset - tail_start + cd_size]
    files, p = [], 0
    for _ in range(entries):
        (sig, _, _, _, method, _, _, _, csize, size, nlen, xlen, clen, _, _, _,
         offset) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        assert sig == 0x02014B50, "corrupt central directory"
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        # ZIP64: the real values live in the extra field, in this fixed order
        q = 0
        while q + 4 <= len(extra):
            tag, length = struct.unpack("<HH", extra[q:q + 4])
            if tag == 1:
                fields = iter(struct.unpack(f"<{length // 8}Q", extra[q + 4:q + 4 + length // 8 * 8]))
                if size == 0xFFFFFFFF:
                    size = next(fields)
                if csize == 0xFFFFFFFF:
                    csize = next(fields)
                if offset == 0xFFFFFFFF:
                    offset = next(fields)
            q += 4 + length
        files.append((name, method, csize, size, offset))
        p += 46 + nlen + xlen + clen
    return files


def extract(entry):
    name, method, csize, size, offset = entry
    target = ROOT / name
    if target.is_file() and target.stat().st_size == size:
        return target
    header = read_range(offset, 30)
    nlen, xlen = struct.unpack("<HH", header[26:30])
    data = read_range(offset + 30 + nlen + xlen, csize)
    if method == 8:
        data = zlib.decompress(data, -15)
    elif method != 0:
        raise SystemExit(f"{name}: unsupported compression method {method}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(ROOT), help="where MM_Office_Dataset/ goes")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    get_cmd = sub.add_parser("get")
    get_cmd.add_argument("names", nargs="*")
    get_cmd.add_argument("--from-file", help="a text file of names, one per line")
    sample = sub.add_parser("sample")
    sample.add_argument("--n", type=int, default=8)
    sample.add_argument("--split", default="test")
    args = parser.parse_args()
    set_root(args.root)
    if args.cmd == "get" and args.from_file:
        args.names += Path(args.from_file).read_text().split()

    files = [f for f in central_directory() if not f[0].endswith("/")]
    if args.cmd == "list":
        for name, _, _, size, _ in files:
            print(f"{size / 1e6:9.1f} MB  {name}")
        print(f"{len(files)} files, {sum(f[3] for f in files) / 1e9:.1f} GB")
        return

    if args.cmd == "get":
        wanted = [f for f in files if f[0] in args.names or Path(f[0]).name in args.names]
    else:
        # one clip per camera first, then spread across scenes, so the sample shows
        # every viewpoint rather than n clips from the same corner
        videos = sorted(f for f in files
                        if f[0].endswith(".mp4") and f"/{args.split}/" in f"/{f[0]}")
        by_camera = {}
        for f in videos:
            camera = next((part for part in Path(f[0]).stem.split("_") if part.startswith("id")), "")
            by_camera.setdefault(camera, []).append(f)
        wanted, round_ = [], 0
        while len(wanted) < args.n and any(round_ < len(v) for v in by_camera.values()):
            for clips in by_camera.values():
                if round_ < len(clips) and len(wanted) < args.n:
                    wanted.append(clips[(round_ * 37) % len(clips)])
            round_ += 1
    for entry in wanted:
        print(f"fetching {entry[0]}  ({entry[3] / 1e6:.1f} MB)")
        extract(entry)
    print(f"done -> {ROOT}")


if __name__ == "__main__":
    main()
