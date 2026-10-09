import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.func import functional_call
import benchmark as b
from integrity import write_frozen, verify_frozen, verify_reference, sha256
from audit import audit


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        b.seed_all(19)
        self.x = torch.randn(20, 1, 1024)
        self.y = torch.arange(4).repeat_interleave(5)

    def test_full_maml_gradients_and_isolation(self):
        for rep in ["1d", "2d"]:
            model = b.Model(rep, "maml")
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            fast = b.adapt(model, self.x, self.y, 2, 0.05, meta=True)
            loss = F.cross_entropy(functional_call(model, fast, (self.x,)), self.y)
            grads = torch.autograd.grad(loss, tuple(model.parameters()))
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertTrue(any(not torch.equal(p, before[n]) for n, p in fast.items()))
            self.assertTrue(
                all(torch.equal(p, before[n]) for n, p in model.named_parameters())
            )

    def test_reptile_matches_independent_adam_and_keeps_base(self):
        model = b.Model("1d", "reptile").eval()
        original = copy.deepcopy(model.state_dict())
        clone = copy.deepcopy(model)
        opt = torch.optim.Adam(clone.parameters(), lr=0.001, betas=(0, 0.999))
        for _ in range(2):
            opt.zero_grad()
            F.cross_entropy(clone(self.x), self.y).backward()
            opt.step()
        fast = b.adapt(model, self.x, self.y, 2, 0.001)
        self.assertTrue(
            all(torch.allclose(a, c) for a, c in zip(fast.values(), clone.parameters()))
        )
        self.assertTrue(
            all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
        )

    def test_metric_class_permutation(self):
        perm = torch.tensor([2, 0, 3, 1])
        for rep in ["1d", "2d"]:
            for method in ["protonet", "relationnet"]:
                model = b.Model(rep, method)
                with torch.no_grad():
                    a = b.metric_scores(model, self.x, self.x[:4], 5)
                    sx = self.x.reshape(4, 5, 1, 1024)[perm].reshape(-1, 1, 1024)
                    c = b.metric_scores(model, sx, self.x[:4], 5)
                self.assertTrue(torch.allclose(a[:, perm], c, atol=1e-5))

    def test_nested_support_and_fixed_query(self):
        data = {
            f"{split}_{c}": torch.arange(20 * 1024, dtype=torch.float32).reshape(
                20, 1, 1024
            )
            + c * 1e5
            for split in ["support", "query"]
            for c in range(4)
        }
        e = {
            "classes": [2, 0, 3, 1],
            "support": [list(range(10))] * 4,
            "query": [list(range(5, 20))] * 4,
        }
        one, five, ten = [b.target_episode(data, e, k) for k in [1, 5, 10]]
        self.assertTrue(
            torch.equal(
                one[0], ten[0].reshape(4, 10, 1, 1024)[:, :1].reshape(4, 1, 1024)
            )
        )
        self.assertTrue(
            torch.equal(
                five[0], ten[0].reshape(4, 10, 1, 1024)[:, :5].reshape(20, 1, 1024)
            )
        )
        self.assertTrue(torch.equal(one[2], ten[2]) and torch.equal(five[2], ten[2]))

    def test_frozen_data_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for n in [
                "protocol.json",
                "windows.npz",
                "data_manifest.json",
                "test_episodes.json",
            ]:
                (root / n).write_text("original")
            write_frozen(root)
            verify_frozen(root)
            (root / "test_episodes.json").write_text("changed")
            with self.assertRaises(ValueError):
                verify_frozen(root)

    def test_reference_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "result.json").write_text("{}")
            (root / "checksums.json").write_text(
                json.dumps({"result.json": sha256(root / "result.json")})
            )
            verify_reference(root)
            (root / "result.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                verify_reference(root)

    def test_all_reference_predictions_reproduce_tables(self):
        result = audit(b.REPO / "reference", reference=True)
        self.assertEqual(result["models"], 24)
        self.assertEqual(result["result_rows"], 24)


if __name__ == "__main__":
    unittest.main()
