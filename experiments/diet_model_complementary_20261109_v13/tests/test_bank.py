"""Actual nested historical bank: closed receipts, provenance and inference order."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from bridge import V6, V8, V10, V11, Memo, imported, sha, write_json
from fusion import linear_fit
from legacy import Reference
from support import ROOT


def cached_bank(package):
    """Create synthetic V10 and V11 cache trees, never inspect native runs."""
    cfg = json.loads((ROOT / "config.json").read_text())
    configs, roots, specs, members = {}, {}, {}, {}
    table = package / "synthetic_input"
    table.write_text("SYNTHETIC_ONLY")
    y, rows = np.array([90.0, 93.0]), np.array([1, 2])
    for version in (V10, V11):
        source = package / "experiments" / version.name
        source.mkdir(parents=True)
        files = [
            "config.json",
            "bridge.py",
            "context.py",
            "features.py",
            "models.py",
            "estimators.py",
            "engine.py",
            "registry.py",
            "fusion.py",
        ]
        if version == V11:
            files += ["food_features.py", "legacy.py"]
        for name in files:
            shutil.copy2(version / name, source / name)
        c = json.loads((source / "config.json").read_text())
        for family in c["families"].values():
            family["grid"] = family["grid"][:1]
        for key in ("interaction_grid", "residual_weights"):
            if key in c:
                c[key] = c[key][:1]
        (source / "config.json").write_text(json.dumps(c))
        registry = imported("v13_test_" + version.name, source / "registry.py")
        spec = {s["name"]: s for s in registry.specifications(c)}
        member = registry.pools(c)["diverse" if version == V10 else "combined"]
        root = (
            source
            / "runs"
            / (
                c["v10_reference_run"]
                if version == V10 and "v10_reference_run" in c
                else ("context_v10_20261108" if version == V10 else cfg["v11_reference_run"])
            )
        )
        if version == V10:
            # The pointer belongs to the historical V11 configuration.
            v11cfg = json.loads((V11 / "config.json").read_text())
            root = source / "runs" / v11cfg["v10_reference_run"]
        root.mkdir(parents=True)
        needed = (
            [source / n for n in files]
            + [table]
            + [V6 / n for n in ("data.py", "numerics.py", "storage.py")]
            + [V8 / "features.py"]
        )
        if version == V11:
            needed += [V10 / "context.py", V10 / "features.py"]
        write_json(root / "registration.json", dict(config=c, hashes={str(p): sha(p) for p in needed}))
        write_json(root / "summary.json", dict(status="COMPLETE"))
        configs[version], roots[version], specs[version], members[version] = c, root, spec, member
    self_index = pd.DataFrame(dict(split_unit_id=["g%d" % j for j in range(6)]))
    roles = [
        SimpleNamespace(index=self_index, split=SimpleNamespace(yv=y, rows=np.arange(2 * k, 2 * k + 2)))
        for k in range(3)
    ]
    target = np.tile(y, 3)
    group = self_index.split_unit_id.to_numpy()
    matrix = np.tile((target + 0.2)[:, None], (1, len(members[V11])))
    # Reconstruct the historical positive fusion with its original formula,
    # independently of the new bank class and its selected_linear wrapper.
    c = configs[V11]
    loss = []
    for penalty in c["meta_penalties"]:
        p = np.full(len(target), np.nan)
        for t, v in GroupKFold(3).split(matrix, target, group):
            p[v] = linear_fit(matrix[t], target[t], "positive", penalty).predict(matrix[v])
        loss.append(np.mean((target - p) ** 2))
    penalty = c["meta_penalties"][int(np.argmin(loss))]
    fit = linear_fit(matrix, target, "positive", penalty)
    replay = fit.predict(np.tile((y + 0.2)[:, None], (1, len(members[V11]))))
    for version in (V10, V11):
        memo = Memo(roots[version] / "TRE_ONLY")
        selected = members[version] if version == V10 else list(specs[version])
        for r in range(2):
            for f in range(5):
                tag = "r%d_f%d" % (r, f)
                pred = {n: y + 0.2 for n in selected}
                if version == V10:
                    pred.update(v8_ridge_wide=y + 0.4, ridge_diverse=y + 0.3)
                else:
                    pred.update(v10_ridge_diverse=y + 0.3, positive_combined=replay, sparse_combined=y + 0.1)
                memo.get(
                    tag + "_result",
                    lambda pred=pred, selected=selected: dict(
                        rows=rows,
                        y=y,
                        predictions=pred,
                        selected={n: 0 for n in selected},
                        outer_diagnostics={n: {} for n in selected},
                        meta=dict(positive_combined=dict(penalty=penalty)),
                    ),
                )
                for name in selected:
                    for k in range(3):
                        for j in range(len(specs[version][name]["grid"])):
                            memo.get(tag + "_i%d_%s_c%d" % (k, name, j), lambda: dict(prediction=y + 0.2))
    return cfg, SimpleNamespace(paths=[table]), roles, rows, y


class BankChecks(unittest.TestCase):
    def test_actual_composite_bank_is_readonly_and_inner_never_deserializes_outer(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            cfg, data, roles, rows, y = cached_bank(package)
            before = {str(p): sha(p) for p in package.rglob("*") if p.is_file()}
            ref = Reference(package, cfg, data)
            with (
                patch.object(ref, "read", wraps=ref.read) as new,
                patch.object(ref.old, "read", wraps=ref.old.read) as old,
            ):
                prior = ref.inner(0, 0, roles)
                self.assertFalse(
                    any(c.args[0].endswith("_result") for reader in (new, old) for c in reader.call_args_list)
                )
            self.assertEqual(len(prior["predictions"]), 90)
            outer, _ = ref.get(0, 0, rows, y, prior["choices"])
            np.testing.assert_allclose(
                prior["replay"].predict(outer), outer["v11_positive_combined"], atol=1e-12
            )
            self.assertEqual(before, {str(p): sha(p) for p in package.rglob("*") if p.is_file()})
            for wrong_rows, wrong_y in ((rows[::-1], y), (rows, y + 1)):
                with self.assertRaisesRegex(ValueError, "rows/processed"):
                    ref.get(0, 0, wrong_rows, wrong_y, prior["choices"])
            bad = copy.deepcopy(prior["choices"])
            bad["v11"][next(iter(bad["v11"]))] = 1
            with self.assertRaisesRegex(ValueError, "choices"):
                ref.get(0, 0, rows, y, bad)
            path, _ = ref.units["r0_f0_i0_hgb_oral_full_c0"]
            payload = path.read_bytes()
            path.write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                ref.inner(0, 0, roles)
            with self.assertRaises(ValueError):
                Reference(package, cfg, data)
            path.write_bytes(payload)
            data.paths[0].write_text("DRIFT")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                Reference(package, cfg, data)
            data.paths[0].write_text("SYNTHETIC_ONLY")
            source = ref.source / "features.py"
            original = source.read_bytes()
            source.write_bytes(original + b"\n# drift\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                Reference(package, cfg, data)
            source.write_bytes(original)
            shutil.rmtree(ref.root / "TRE_ONLY/r0_f0_i0_hgb_oral_full_c0")
            with self.assertRaisesRegex(ValueError, "missing"):
                Reference(package, cfg, data)
