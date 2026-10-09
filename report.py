"""Regenerate the slide result tables and labeled scientific plots."""

import argparse
import csv
import json
from pathlib import Path


def report(results, output, plots=False):
    data = json.loads(Path(results).read_text())["results"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "backbone",
        "shots",
        "mean_accuracy_pct",
        "sd_across_training_seeds_pp",
        "params",
    ]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in data)
    lookup = {(r["method"], r["backbone"], r["shots"]): r for r in data}
    methods = ["maml", "reptile", "protonet", "relationnet"]
    names = ["MAML", "Reptile", "ProtoNet", "RelationNet"]
    lines = [
        "| Method | CNN | 1 shot [%] | 5 shots [%] | 10 shots [%] | Parameters |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m, name in zip(methods, names):
        for rep in ["1d", "2d"]:
            values = [
                f"{lookup[m, rep, k]['mean_accuracy_pct']:.2f} ± {lookup[m, rep, k]['sd_across_training_seeds_pp']:.2f}"
                for k in [1, 5, 10]
            ]
            lines.append(
                "| "
                + " | ".join(
                    [name, rep.upper(), *values, f"{lookup[m, rep, 1]['params']:,}"]
                )
                + " |"
            )
    lines += [
        "",
        "Mean accuracy ± sample SD across three training seeds. Each seed averages 200 paired target episodes.",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 11,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "svg.fonttype": "none",
            }
        )
        for m, name in zip(methods, names):
            fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
            for rep, color, marker in [("1d", "#1E293B", "o"), ("2d", "#4F46E5", "s")]:
                rows = [lookup[m, rep, k] for k in [1, 5, 10]]
                ax.errorbar(
                    [1, 5, 10],
                    [r["mean_accuracy_pct"] for r in rows],
                    yerr=[r["sd_across_training_seeds_pp"] for r in rows],
                    label=f"{rep.upper()} CNN",
                    color=color,
                    marker=marker,
                    linewidth=0.875,
                    capsize=3,
                )
            ax.set(
                xlabel="Labeled examples per class [shots]",
                ylabel="Accuracy [%]",
                title=name,
                xticks=[1, 5, 10],
                ylim=(0, 100),
            )
            ax.grid(axis="y", linewidth=0.375, alpha=0.4)
            ax.legend(frameon=False)
            fig.savefig(output / f"{m}.png", dpi=300)
            fig.savefig(output / f"{m}.svg")
            plt.close(fig)
    print(f"Report saved to {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results", type=Path, default=Path(__file__).parent / "reference/results.json"
    )
    ap.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "outputs/report"
    )
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()
    report(args.results, args.output, args.plots)
