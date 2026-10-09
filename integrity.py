"""Portable data, code, and experiment fingerprints (no model dependencies)."""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_data(data_dir):
    for entry in json.loads((REPO / "configs/data_files.json").read_text()):
        p = Path(data_dir) / entry["path"]
        if not p.exists():
            raise FileNotFoundError(
                f'{p}: run python download_data.py --data-dir "{data_dir}"'
            )
        if sha256(p) != entry["sha256"]:
            raise ValueError(f"CWRU file checksum mismatch: {p}")


def write_frozen(output):
    output = Path(output)
    code = {
        p: sha256(REPO / p)
        for p in [
            "benchmark.py",
            "integrity.py",
            "configs/protocol.json",
            "configs/data_files.json",
        ]
    }
    data = {
        p: sha256(output / p)
        for p in [
            "protocol.json",
            "windows.npz",
            "data_manifest.json",
            "test_episodes.json",
        ]
    }
    (output / "frozen.json").write_text(
        json.dumps({"code": code, "data": data}, indent=2) + "\n"
    )


def verify_frozen(output):
    output = Path(output)
    lock = json.loads((output / "frozen.json").read_text())
    for base, section in [(REPO, "code"), (output, "data")]:
        for name, expected in lock[section].items():
            if sha256(base / name) != expected:
                raise ValueError(
                    f"{section} changed since prepare: {name}. Use a new output directory."
                )


def verify_reference(reference):
    reference = Path(reference)
    manifest = json.loads((reference / "checksums.json").read_text())
    for name, expected in manifest.items():
        if sha256(reference / name) != expected:
            raise ValueError(f"Reference checksum mismatch: {name}")
    return len(manifest)
