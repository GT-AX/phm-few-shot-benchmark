"""One command for a new benchmark run or reference-checkpoint evaluation."""

import argparse
import subprocess
import sys
from pathlib import Path
from integrity import REPO, verify_frozen, verify_reference


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["train", "checkpoints"], default="train")
    ap.add_argument("--data-dir", type=Path, default=REPO / "data/raw/cwru")
    ap.add_argument("--output", type=Path, default=REPO / "outputs/reproduction")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--download", action="store_true")
    ap.add_argument(
        "--plots", action="store_true", help="Requires requirements-plots.txt"
    )
    args = ap.parse_args()
    args.output = args.output.resolve()
    args.data_dir = args.data_dir.resolve()

    def run(script, *argv):
        subprocess.run(
            [sys.executable, str(REPO / script), *map(str, argv)], check=True, cwd=REPO
        )

    if args.download:
        run("download_data.py", "--data-dir", args.data_dir)
    common = [
        "--output",
        args.output,
        "--data-dir",
        args.data_dir,
        "--device",
        args.device,
    ]
    if (args.output / "frozen.json").exists():
        verify_frozen(args.output)
    else:
        run("benchmark.py", "prepare", *common)
    run("benchmark.py", "check", *common)
    if args.mode == "checkpoints":
        verify_reference(REPO / "reference")
    run("benchmark.py", "evaluate" if args.mode == "checkpoints" else "run", *common)
    run("benchmark.py", "aggregate", *common)
    run(
        "audit.py",
        "--output",
        args.output,
        "--data-dir",
        args.data_dir,
        "--compare",
        REPO / "reference",
    )
    run(
        "report.py",
        "--results",
        args.output / "results.json",
        "--output",
        args.output / "report",
        *(["--plots"] if args.plots else [])
    )


if __name__ == "__main__":
    main()
