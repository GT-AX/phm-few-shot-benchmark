"""Download only the ten original CWRU files needed by this benchmark."""

import argparse
import json
import time
import urllib.request
from pathlib import Path
from integrity import REPO, sha256, verify_data


def download(data_dir):
    for entry in json.loads((REPO / "configs/data_files.json").read_text()):
        dest = Path(data_dir) / entry["path"]
        if dest.exists():
            if sha256(dest) != entry["sha256"]:
                raise ValueError(
                    f"Existing file has the wrong checksum: {dest}. Move it aside before retrying."
                )
            print("Verified", entry["path"], flush=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".mat.part")
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    entry["url"], headers={"User-Agent": "PHM-reproducibility/1.0"}
                )
                with urllib.request.urlopen(req, timeout=60) as response, part.open(
                    "wb"
                ) as f:
                    while chunk := response.read(1024 * 1024):
                        f.write(chunk)
                if sha256(part) != entry["sha256"]:
                    raise ValueError(
                        f'Downloaded file differs from the benchmark: {entry["url"]}'
                    )
                part.replace(dest)
                print("Downloaded and verified", entry["path"], flush=True)
                break
            except Exception:
                part.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    verify_data(data_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/raw/cwru")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    (verify_data if args.verify_only else download)(args.data_dir)
