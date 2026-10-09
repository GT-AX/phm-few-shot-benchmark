"""Controlled CWRU T2-class partition, 1/5/10 shots, native PyTorch algorithms."""

import argparse, copy, hashlib, json, random, time, sys, platform, shutil
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from scipy.io import loadmat
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parent
ROOT = REPO / "outputs/reproduction"
CFG = json.loads((REPO / "configs/protocol.json").read_text())
DATA_ROOT = REPO / CFG["data_root"]
from integrity import sha256, verify_data, verify_frozen, write_frozen

torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare():
    data = {}
    manifest = []
    base = DATA_ROOT
    verify_data(base)
    if (ROOT / "runs").exists():
        raise FileExistsError(
            "Use a new --output directory to prepare a new experiment; existing runs are preserved."
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    for role, names in [
        ("source", CFG["source_classes"]),
        ("target", CFG["target_classes"]),
    ]:
        for ci, name in enumerate(names):
            p = (
                base
                / (
                    "Normal Data"
                    if name == "Normal"
                    else "12k Drive End Bearing Fault Data"
                )
                / (name + "_0.mat")
            )
            d = loadmat(p)
            keys = [k for k in d if k.endswith("_DE_time")]
            assert len(keys) == 1, (p, keys)
            x = d[keys[0]].ravel().astype(np.float64)
            raw_count = len(x)
            if name == "Normal":
                x = resample_poly(x, 1, 4)
            roles = (
                (("train", 0, 0.6), ("val", 0.65, 1))
                if role == "source"
                else (("support", 0, 0.3), ("query", 0.4, 1))
            )
            bounds = []
            for split, lo, hi in roles:
                start = int(lo * len(x))
                stop = int(hi * len(x))
                count = (stop - start) // 1024
                a = x[start : start + count * 1024].reshape(count, 1024).copy()
                a = (a - a.mean(1, keepdims=True)) / np.maximum(
                    a.std(1, keepdims=True), 1e-6
                )
                a = a[:, None, :].astype(np.float32)
                assert np.isfinite(a).all()
                minimum = (
                    20
                    if split == "train"
                    else (15 if split in ["val", "query"] else 10)
                )
                assert count >= minimum, (name, split, count)
                data[f"{split}_{ci}"] = a
                bounds.append((start, start + count * 1024))
                manifest.append(
                    {
                        "role": role,
                        "split": split,
                        "class": name,
                        "class_index": ci,
                        "file": p.relative_to(base).as_posix(),
                        "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "mat_key": keys[0],
                        "raw_samples": raw_count,
                        "resampled_samples": len(x),
                        "sample_start": start,
                        "sample_stop": start + count * 1024,
                        "windows": count,
                    }
                )
            assert bounds[0][1] < bounds[1][0]
    np.savez(ROOT / "windows.npz", **data)
    save(ROOT / "data_manifest.json", manifest)
    # Persist common target episodes before any target inference.
    rng = np.random.default_rng(CFG["test_episode_seed"])
    episodes = []
    for i in range(CFG["test_episodes"]):
        classes = rng.permutation(4).tolist()
        episodes.append(
            {
                "classes": classes,
                "support": [
                    rng.choice(len(data[f"support_{c}"]), 10, replace=False).tolist()
                    for c in classes
                ],
                "query": [
                    rng.choice(len(data[f"query_{c}"]), 15, replace=False).tolist()
                    for c in classes
                ],
            }
        )
    save(ROOT / "test_episodes.json", episodes)
    save(ROOT / "protocol.json", CFG)
    write_frozen(ROOT)
    print(
        "Prepared windows:",
        [(m["class"], m["split"], m["windows"]) for m in manifest],
        flush=True,
    )


class Encoder(nn.Module):
    def __init__(self, rep):
        super().__init__()
        self.rep = rep
        conv = nn.Conv1d if rep == "1d" else nn.Conv2d
        pool = nn.MaxPool1d if rep == "1d" else nn.MaxPool2d
        blocks = []
        for i in range(4):
            blocks.extend(
                [
                    conv(1 if i == 0 else 64, 64, 3, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.ReLU(),
                    pool(2),
                ]
            )
        self.blocks = nn.Sequential(*blocks)
        self.pool = (
            nn.AdaptiveAvgPool1d(4) if rep == "1d" else nn.AdaptiveAvgPool2d((2, 2))
        )

    def forward_map(self, x):
        if self.rep == "2d":
            x = x.reshape(-1, 1, 32, 32)
        return self.blocks(x)

    def forward(self, x):
        return self.pool(self.forward_map(x)).flatten(1)


class Model(nn.Module):
    def __init__(self, rep, method):
        super().__init__()
        self.encoder = Encoder(rep)
        self.method = method
        if method in ["maml", "reptile"]:
            self.head = nn.Linear(256, 4)
        if method == "relationnet":
            conv = nn.Conv1d if rep == "1d" else nn.Conv2d
            pool = nn.MaxPool1d if rep == "1d" else nn.MaxPool2d
            avg = nn.AdaptiveAvgPool1d(1) if rep == "1d" else nn.AdaptiveAvgPool2d(1)
            self.relation = nn.Sequential(
                conv(128, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                pool(2, ceil_mode=True),
                conv(64, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                pool(2, ceil_mode=True),
                avg,
                nn.Flatten(),
                nn.Linear(64, 8),
                nn.LeakyReLU(0.1),
                nn.Linear(8, 1),
            )

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z) if self.method in ["maml", "reptile"] else z


def episode(data, rng, split, k=5, q=5, classes=None):
    cs = rng.choice(6, 4, replace=False) if classes is None else classes
    sx = []
    qx = []
    for c in cs:
        a = data[f"{split}_{c}"]
        idx = rng.choice(len(a), k + q, replace=False)
        sx.append(a[idx[:k]])
        qx.append(a[idx[k:]])
    return (
        torch.cat(sx),
        torch.arange(4, device=sx[0].device).repeat_interleave(k),
        torch.cat(qx),
        torch.arange(4, device=sx[0].device).repeat_interleave(q),
    )


def target_episode(data, e, k):
    sx = [data[f"support_{c}"][e["support"][j][:k]] for j, c in enumerate(e["classes"])]
    qx = [data[f"query_{c}"][e["query"][j]] for j, c in enumerate(e["classes"])]
    dev = sx[0].device
    return (
        torch.cat(sx),
        torch.arange(4, device=dev).repeat_interleave(k),
        torch.cat(qx),
        torch.arange(4, device=dev).repeat_interleave(15),
    )


def adapt(model, sx, sy, steps, lr, meta=False):
    if model.method == "reptile":
        clone = copy.deepcopy(model)
        optimizer = torch.optim.Adam(clone.parameters(), lr=lr, betas=(0, 0.999))
        for _ in range(steps):
            x, y = sx, sy
            if model.training and len(sy) > 10:
                idx = torch.randperm(len(sy), device=sy.device)[:10]
                x, y = sx[idx], sy[idx]
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(clone(x), y).backward()
            optimizer.step()
        return OrderedDict(clone.named_parameters())
    p = (
        OrderedDict(model.named_parameters())
        if meta
        else OrderedDict(
            (n, v.detach().clone().requires_grad_(True))
            for n, v in model.named_parameters()
        )
    )
    for step in range(steps):
        loss = F.cross_entropy(functional_call(model, p, (sx,)), sy)
        grad = torch.autograd.grad(loss, tuple(p.values()), create_graph=meta)
        p = OrderedDict((n, v - lr * g) for (n, v), g in zip(p.items(), grad))
        if not meta:
            p = OrderedDict((n, v.detach().requires_grad_(True)) for n, v in p.items())
    return p


def metric_scores(model, sx, qx, k):
    if model.method == "relationnet":
        sz = model.encoder.forward_map(sx)
        sz = sz.reshape(4, k, *sz.shape[1:]).mean(1)
        qz = model.encoder.forward_map(qx)
        pairs = torch.cat(
            [
                sz[None].expand(len(qz), *sz.shape),
                qz[:, None].expand(len(qz), 4, *qz.shape[1:]),
            ],
            2,
        )
        return model.relation(pairs.reshape(-1, *pairs.shape[2:])).reshape(len(qz), 4)
    sz = model(sx).reshape(4, k, -1).mean(1)
    qz = model(qx)
    if model.method == "protonet":
        return -(qz[:, None] - sz[None]).square().mean(-1)
    pair = torch.cat(
        [sz[None].expand(len(qz), -1, -1), qz[:, None].expand(-1, 4, -1)], -1
    )
    return model.relation(pair).squeeze(-1)


def score(model, e, lr=0.05, steps=5):
    sx, sy, qx, qy = e
    k = len(sy) // 4
    if model.method in ["maml", "reptile"]:
        p = adapt(model, sx, sy, steps, lr)
        with torch.no_grad():
            out = functional_call(model, p, (qx,))
    else:
        with torch.no_grad():
            out = metric_scores(model, sx, qx, k)
    return (out.argmax(1) == qy).float().mean().item(), out.argmax(1).cpu().tolist()


def source_validation(model, data, n=20, shots=(5,), lr=0.05, steps=5):
    rng = np.random.default_rng(74019)
    acc = []
    model.eval()
    for k in shots:
        for _ in range(n):
            acc.append(score(model, episode(data, rng, "val", k, 5), lr, steps)[0])
    model.train()
    return float(np.mean(acc))


def load_data(device):
    a = np.load(ROOT / "windows.npz")
    return {k: torch.from_numpy(a[k]).to(device) for k in a.files}


def train_one(rep, method, seed, data, smoke=False):
    out = ROOT / "runs" / f"{method}_{rep}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() and not smoke:
        verify_frozen(ROOT)
        return json.loads((out / "result.json").read_text())
    save(out / "environment.json", environment(data))
    seed_all(seed)
    model = Model(rep, method).to(next(iter(data.values())).device)
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=0.001) if method != "reptile" else None
    )
    rng = np.random.default_rng(seed)
    start = time.time()
    best = -1
    history = []
    total = 600 if smoke else CFG["train_episodes"]
    for step in range(1, total + 1):
        sx, sy, qx, qy = episode(data, rng, "train")
        if method == "maml":
            p = adapt(model, sx, sy, 3, 0.05, meta=True)
            scores = functional_call(model, p, (qx,))
            loss = F.cross_entropy(scores, qy)
        elif method == "reptile":
            p = adapt(model, sx, sy, 5, 0.001)
            eps = 1.0 - 0.9 * (step - 1) / max(1, total - 1)
            with torch.no_grad():
                for n, v in model.named_parameters():
                    v.lerp_(p[n], eps)
                scores = model(qx)
                loss = F.cross_entropy(scores, qy)
        else:
            scores = metric_scores(model, sx, qx, 5)
            loss = F.cross_entropy(scores, qy)
        if not torch.isfinite(loss):
            raise FloatingPointError((rep, method, seed, step))
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if step % CFG["validation_interval"] == 0 or step == total:
            val = source_validation(
                model, data, n=20, steps=5, lr=0.001 if method == "reptile" else 0.05
            )
            row = {
                "step": step,
                "source_val_acc": val,
                "loss": loss.item(),
                "elapsed_s": round(time.time() - start, 2),
            }
            history.append(row)
            if val > best:
                best = val
                torch.save(
                    {
                        "state": model.state_dict(),
                        "step": step,
                        "source_val_accuracy": val,
                        "rep": rep,
                        "method": method,
                        "seed": seed,
                    },
                    out / "best.pt",
                )
            save(out / "training.json", history)
            print(rep, method, seed, json.dumps(row), flush=True)
    if smoke:
        return {
            "seconds": time.time() - start,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    ck = torch.load(
        out / "best.pt",
        map_location=next(iter(data.values())).device,
        weights_only=True,
    )
    model.load_state_dict(ck["state"])
    return evaluate_one(model, data, out, ck, rep, method, seed, start)


def evaluate_one(model, data, out, ck, rep, method, seed, start):
    choices = []
    lr = 0.05
    if method in ["maml", "reptile"]:
        for candidate in (
            CFG["reptile_adaptation_lr_candidates"]
            if method == "reptile"
            else CFG["adaptation_lr_candidates"]
        ):
            a = source_validation(
                model, data, n=10, shots=(1, 5, 10), lr=candidate, steps=5
            )
            choices.append({"lr": candidate, "source_val_acc": a})
        lr = max(choices, key=lambda x: x["source_val_acc"])["lr"]
    save(
        out / "adaptation_selection.json",
        {"source_only": True, "choices": choices, "lr": lr, "steps": 5},
    )
    model.eval()
    episodes = json.loads((ROOT / "test_episodes.json").read_text())
    trials = {str(k): [] for k in CFG["test_shots"]}
    predictions = {str(k): [] for k in CFG["test_shots"]}
    for k in CFG["test_shots"]:
        for e in episodes:
            a, p = score(model, target_episode(data, e, k), lr, 5)
            trials[str(k)].append(a)
            predictions[str(k)].append(p)
        print(
            "TEST",
            rep,
            method,
            seed,
            k,
            round(np.mean(trials[str(k)]) * 100, 3),
            flush=True,
        )
    save(out / "predictions.json", predictions)
    result = {
        "method": method,
        "representation": rep,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_step": ck["step"],
        "source_validation_accuracy": ck["source_val_accuracy"],
        "adaptation_lr": lr,
        "adaptation_steps": 5 if method in ["maml", "reptile"] else 0,
        "accuracy": {k: float(np.mean(v) * 100) for k, v in trials.items()},
        "episode_accuracy": trials,
        "elapsed_s": time.time() - start,
    }
    save(out / "result.json", result)
    return result


def checks(device):
    seed_all(19)
    data = load_data(device)
    rng = np.random.default_rng(5)
    e = episode(data, rng, "train")
    records = []
    for rep in ["1d", "2d"]:
        model = Model(rep, "maml").to(device)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        p = adapt(model, e[0], e[1], 2, 0.05, True)
        assert any(not torch.equal(v, before[n]) for n, v in p.items())
        loss = F.cross_entropy(functional_call(model, p, (e[2],)), e[3])
        g = torch.autograd.grad(loss, tuple(model.parameters()))
        assert all(torch.isfinite(a).all() for a in g)
        assert all(torch.equal(v, before[n]) for n, v in model.named_parameters())
        # Direct SGD and functional adaptation must agree on a cloned model.
        clone = copy.deepcopy(model)
        opt = torch.optim.SGD(clone.parameters(), lr=0.05)
        for _ in range(2):
            opt.zero_grad()
            F.cross_entropy(clone(e[0]), e[1]).backward()
            opt.step()
        delta = max(
            (a - b).abs().max().item() for a, b in zip(p.values(), clone.parameters())
        )
        print("Functional/SGD max difference", rep, delta, flush=True)
        assert delta < 1e-5
        proto = Model(rep, "protonet").to(device)
        sx, sy, qx, qy = e
        with torch.no_grad():
            base = metric_scores(proto, sx, qx, 5)
            perm = torch.tensor([2, 0, 3, 1], device=device)
            swapped = metric_scores(
                proto, sx.reshape(4, 5, 1, 1024)[perm].reshape(-1, 1, 1024), qx, 5
            )
            assert torch.allclose(base[:, perm], swapped, atol=1e-5)
        records.append(
            {
                "rep": rep,
                "maml_adapts": True,
                "full_meta_gradient_finite": True,
                "base_weights_unchanged": True,
                "functional_matches_sgd": True,
                "protonet_class_permutation": True,
            }
        )
        rept = Model(rep, "reptile").to(device).eval()
        fast = adapt(rept, e[0], e[1], 5, 0.001)
        reference = copy.deepcopy(rept)
        opt = torch.optim.Adam(reference.parameters(), lr=0.001, betas=(0, 0.999))
        for _ in range(5):
            opt.zero_grad()
            F.cross_entropy(reference(e[0]), e[1]).backward()
            opt.step()
        adam_delta = max(
            (a - b).abs().max().item()
            for a, b in zip(fast.values(), reference.parameters())
        )
        assert adam_delta < 1e-5, adam_delta
        rel = Model(rep, "relationnet").to(device)
        scores = metric_scores(rel, e[0], e[2], 5)
        F.cross_entropy(scores, e[3]).backward()
        assert sum(p.grad.abs().sum().item() for p in rel.encoder.parameters()) > 0
        assert sum(p.grad.abs().sum().item() for p in rel.relation.parameters()) > 0
        with torch.no_grad():
            perm = torch.tensor([2, 0, 3, 1], device=device)
            permuted = metric_scores(
                rel, e[0].reshape(4, 5, 1, 1024)[perm].reshape(-1, 1, 1024), e[2], 5
            )
            assert torch.allclose(scores[:, perm], permuted, atol=1e-5)
        records[-1].update(
            {
                "functional_adam_matches_torch": True,
                "relation_encoder_and_head_gradients": True,
                "relation_class_permutation": True,
            }
        )
    episodes = json.loads((ROOT / "test_episodes.json").read_text())
    assert all(len(set(ix)) == 10 for e in episodes for ix in e["support"])
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    assert not set(CFG["source_classes"]) & set(CFG["target_classes"])
    save(
        ROOT / "implementation_checks.json",
        {
            "checks": records,
            "nested_support_sizes": [1, 5, 10],
            "disjoint_source_target_classes": True,
            "temporal_separation_verified": True,
        },
    )
    print("Implementation and data checks passed", flush=True)


def aggregate():
    rows = []
    for method in CFG["algorithms"]:
        for rep in CFG["representations"]:
            records = [
                json.loads(
                    (ROOT / "runs" / f"{method}_{rep}_s{s}" / "result.json").read_text()
                )
                for s in CFG["train_seeds"]
            ]
            for k in CFG["test_shots"]:
                values = [r["accuracy"][str(k)] for r in records]
                rows.append(
                    {
                        "method": method,
                        "backbone": rep,
                        "shots": k,
                        "mean_accuracy_pct": float(np.mean(values)),
                        "sd_across_training_seeds_pp": float(np.std(values, ddof=1)),
                        "seed_values": values,
                        "params": records[0]["parameters"],
                    }
                )
    save(ROOT / "results.json", {"protocol": CFG, "results": rows})
    print(json.dumps(rows, indent=2))


def environment(data):
    dev = next(iter(data.values())).device
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(dev),
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "code_sha256": sha256(Path(__file__)),
        "cudnn_deterministic": True,
        "tf32": False,
    }


def evaluate_checkpoints(data, methods, reps, seeds, reference):
    for method in methods:
        for rep in reps:
            for seed in seeds:
                name = f"{method}_{rep}_s{seed}"
                out = ROOT / "runs" / name
                if (out / "result.json").exists():
                    print("SKIP completed", name, flush=True)
                    continue
                out.mkdir(parents=True, exist_ok=True)
                original = reference / "runs" / name
                ck = torch.load(
                    original / "best.pt",
                    map_location=next(iter(data.values())).device,
                    weights_only=True,
                )
                model = Model(rep, method).to(next(iter(data.values())).device)
                model.load_state_dict(ck["state"])
                shutil.copy2(original / "best.pt", out / "best.pt")
                shutil.copy2(original / "training.json", out / "training.json")
                save(
                    out / "environment.json",
                    {**environment(data), "mode": "reference-checkpoint-evaluation"},
                )
                evaluate_one(model, data, out, ck, rep, method, seed, time.time())


def main():
    global ROOT, DATA_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "mode", choices=["prepare", "check", "smoke", "run", "evaluate", "aggregate"]
    )
    ap.add_argument("--output", type=Path, default=REPO / "outputs/reproduction")
    ap.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    ap.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--rep", choices=["1d", "2d"])
    ap.add_argument("--method", choices=CFG["algorithms"])
    ap.add_argument("--seed", type=int, choices=CFG["train_seeds"])
    ap.add_argument("--reference", type=Path, default=REPO / "reference")
    args = ap.parse_args()
    ROOT = args.output.resolve()
    DATA_ROOT = args.data_dir.resolve()
    reference = args.reference.resolve()
    if ROOT == reference or reference in ROOT.parents:
        ap.error(
            "The reference directory is immutable. Choose --output outside reference/."
        )
    if args.mode == "prepare":
        prepare()
        return
    verify_frozen(ROOT)
    if args.mode == "aggregate":
        aggregate()
        return
    if args.mode == "check":
        checks(args.device)
        return
    methods = [args.method] if args.method else CFG["algorithms"]
    reps = [args.rep] if args.rep else ["1d", "2d"]
    seeds = [args.seed] if args.seed else CFG["train_seeds"]
    data = load_data(args.device)
    if args.mode == "evaluate":
        evaluate_checkpoints(data, methods, reps, seeds, reference)
        return
    if args.mode == "smoke" and (ROOT / "runs").exists():
        ap.error(
            "Smoke tests require a separate prepared output directory without existing runs."
        )
    for method in methods:
        for rep in reps:
            for seed in [999] if args.mode == "smoke" else seeds:
                print("START", method, rep, seed, flush=True)
                result = train_one(rep, method, seed, data, args.mode == "smoke")
                if args.mode == "smoke":
                    print("SMOKE", method, rep, result, flush=True)


if __name__ == "__main__":
    main()
