"""Recompute results from predictions and validate splits and checkpoint selection."""

import argparse
import json
from pathlib import Path
import numpy as np
from integrity import REPO, verify_data, verify_frozen, verify_reference


def audit(root, reference=False, data_dir=None, compare=None):
    root = Path(root)
    if reference:
        verified = verify_reference(root)
        cfg = json.loads((REPO / "configs/protocol.json").read_text())
    else:
        verify_frozen(root)
        cfg = json.loads((root / "protocol.json").read_text())
        verified = None
    if data_dir is not None:
        verify_data(data_dir)
    manifest = json.loads((root / "data_manifest.json").read_text())
    assert not set(cfg["source_classes"]) & set(cfg["target_classes"])
    for name in cfg["source_classes"] + cfg["target_classes"]:
        a, b = [x for x in manifest if x["class"] == name]
        assert a["sample_stop"] < b["sample_start"], name
        for row in [a, b]:
            assert row["windows"] * 1024 == row["sample_stop"] - row["sample_start"]
    episodes = json.loads((root / "test_episodes.json").read_text())
    assert len(episodes) == cfg["test_episodes"]
    for e in episodes:
        assert sorted(e["classes"]) == [0, 1, 2, 3]
        for j, c in enumerate(e["classes"]):
            for split, count in [("support", 10), ("query", 15)]:
                indices = e[split][j]
                assert len(indices) == len(set(indices)) == count
                pool = next(
                    x["windows"]
                    for x in manifest
                    if x["role"] == "target"
                    and x["split"] == split
                    and x["class_index"] == c
                )
                assert min(indices) >= 0 and max(indices) < pool
    truth = np.repeat(np.arange(4), 15)
    all_results = {}
    matches = []
    if compare:
        compare = Path(compare)
        verify_reference(compare)
        assert episodes == json.loads((compare / "test_episodes.json").read_text())
    for method in cfg["algorithms"]:
        for rep in cfg["representations"]:
            for seed in cfg["train_seeds"]:
                name = f"{method}_{rep}_s{seed}"
                d = root / "runs" / name
                r = json.loads((d / "result.json").read_text())
                pred = json.loads((d / "predictions.json").read_text())
                history = json.loads((d / "training.json").read_text())
                assert history[-1]["step"] == cfg["train_episodes"]
                best = max(history, key=lambda x: x["source_val_acc"])
                assert r["checkpoint_step"] == best["step"]
                assert r["source_validation_accuracy"] == best["source_val_acc"]
                selection = json.loads((d / "adaptation_selection.json").read_text())
                assert selection["source_only"]
                if selection["choices"]:
                    assert (
                        selection["lr"]
                        == max(selection["choices"], key=lambda x: x["source_val_acc"])[
                            "lr"
                        ]
                    )
                assert r["adaptation_lr"] == selection["lr"]
                for k in cfg["test_shots"]:
                    p = np.array(pred[str(k)])
                    assert p.shape == (200, 60) and np.all((p >= 0) & (p < 4))
                    vals = (p == truth[None]).mean(1)
                    assert np.allclose(vals, r["episode_accuracy"][str(k)], atol=1e-7)
                    assert abs(vals.mean() * 100 - r["accuracy"][str(k)]) < 1e-5
                assert (d / "best.pt").exists()
                all_results[name] = r
                if compare:
                    refpred = json.loads(
                        (compare / "runs" / name / "predictions.json").read_text()
                    )
                    matches.append(
                        {"run": name, "identical_predictions": pred == refpred}
                    )
    results = json.loads((root / "results.json").read_text())["results"]
    expected_keys = {
        (m, r, k)
        for m in cfg["algorithms"]
        for r in cfg["representations"]
        for k in cfg["test_shots"]
    }
    assert (
        len(results) == 24
        and {(r["method"], r["backbone"], r["shots"]) for r in results} == expected_keys
    )
    for row in results:
        vals = [
            all_results[f"{row['method']}_{row['backbone']}_s{s}"]["accuracy"][
                str(row["shots"])
            ]
            for s in cfg["train_seeds"]
        ]
        assert np.isclose(row["mean_accuracy_pct"], np.mean(vals))
        assert np.isclose(row["sd_across_training_seeds_pp"], np.std(vals, ddof=1))
    report = {
        "status": "pass",
        "models": len(all_results),
        "result_rows": len(results),
        "data_hashes_checked": data_dir is not None,
        "reference_files_verified": verified,
        "prediction_accuracy_recomputed": True,
        "paired_episodes_checked": True,
        "source_only_selection_checked": True,
        "comparison": matches,
        "all_reference_predictions_identical": (
            all(x["identical_predictions"] for x in matches) if matches else None
        ),
    }
    if not reference:
        (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=REPO / "outputs/reproduction")
    ap.add_argument(
        "--reference",
        action="store_true",
        help="Audit the bundled reference without raw data or training",
    )
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument(
        "--compare",
        type=Path,
        help="Compare all predictions against a reference directory",
    )
    ap.add_argument("--require-exact", action="store_true")
    args = ap.parse_args()
    report = audit(
        REPO / "reference" if args.reference else args.output,
        args.reference,
        args.data_dir,
        args.compare,
    )
    print(json.dumps(report, indent=2))
    if args.require_exact and report["all_reference_predictions_identical"] is not True:
        raise SystemExit("Exact reference-prediction reproduction did not pass.")
