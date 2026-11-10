"""Synthetic saved-bank provenance, read-only access and outer-payload separation."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bridge import V6, V8, V10, Memo, Reference, imported, sha, write_json
from support import fixture


def cache_fixture(package, config):
    source = package / "experiments" / V10.name
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
    for name in files:
        shutil.copy2(V10 / name, source / name)
    old = json.loads((source / "config.json").read_text())
    for family in old["families"].values():
        family["grid"] = family["grid"][:1]
    old["interaction_grid"] = old["interaction_grid"][:1]
    (source / "config.json").write_text(json.dumps(old))
    table = package / "synthetic_input"
    table.write_text("SYNTHETIC_ONLY")
    data = SimpleNamespace(paths=[table])
    root = source / "runs" / config["v10_reference_run"]
    root.mkdir(parents=True)
    write_json(
        root / "registration.json",
        dict(
            config=old,
            hashes={
                str(p): sha(p)
                for p in [source / n for n in files]
                + [table]
                + [V6 / name for name in ("data.py", "numerics.py", "storage.py")]
                + [V8 / "features.py"]
            },
        ),
    )
    write_json(root / "summary.json", dict(status="COMPLETE"))
    registry = imported("v11_test_old_registry", source / "registry.py")
    specs = {s["name"]: s for s in registry.specifications(old)}
    members = registry.pools(old)["diverse"]
    rows, y = np.array([1, 2]), np.array([90.0, 93.0])
    memo = Memo(root / "TRE_ONLY")
    for r in range(2):
        for f in range(5):
            tag = "r%d_f%d" % (r, f)
            predictions = {n: y + 0.2 for n in members + ["v8_ridge_wide", "ridge_diverse"]}
            memo.get(
                tag + "_result",
                lambda: dict(
                    rows=rows,
                    y=y,
                    predictions=predictions,
                    selected={n: 0 for n in members},
                    outer_diagnostics={n: {} for n in members},
                ),
            )
            for name in members:
                for k in range(3):
                    for j in range(len(specs[name]["grid"])):
                        memo.get(tag + "_i%d_%s_c%d" % (k, name, j), lambda: dict(prediction=y + 0.2))
    roles = [SimpleNamespace(split=SimpleNamespace(yv=y)) for _ in range(3)]
    return data, roles, rows, y


class LegacyChecks(unittest.TestCase):
    def test_bank_is_readonly_and_inner_access_never_opens_outer_payload(self):
        _, config = fixture()
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            data, roles, rows, y = cache_fixture(package, config)
            before = {str(p): sha(p) for p in package.rglob("*") if p.is_file()}
            ref = Reference(package, config, data)
            with patch.object(ref, "read", wraps=ref.read) as reader:
                predictions, choices = ref.inner(0, 0, roles)
                self.assertFalse(any(c.args[0].endswith("_result") for c in reader.call_args_list))
            self.assertEqual(len(predictions), 28)
            outer, _ = ref.get(0, 0, rows, y, choices)
            self.assertEqual(len(outer), 30)
            self.assertEqual(before, {str(p): sha(p) for p in package.rglob("*") if p.is_file()})
            with self.assertRaisesRegex(ValueError, "rows/processed"):
                ref.get(0, 0, rows[::-1], y, choices)
            with self.assertRaisesRegex(ValueError, "rows/processed"):
                ref.get(0, 0, rows, y + 1, choices)
            bad = dict(choices)
            bad[next(iter(bad))] = 1
            with self.assertRaisesRegex(ValueError, "selections"):
                ref.get(0, 0, rows, y, bad)
            payload, _ = ref.units["r0_f0_i0_hgb_oral_full_c0"]
            payload.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                ref.inner(0, 0, roles)
            with self.assertRaises(ValueError):
                Reference(package, config, data)

    def test_changed_input_or_source_and_missing_cache_fail_without_refitting(self):
        _, config = fixture()
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            data, _, _, _ = cache_fixture(package, config)
            original = data.paths[0].read_bytes()
            data.paths[0].write_bytes(b"drift")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                Reference(package, config, data)
            data.paths[0].write_bytes(original)
            ref = Reference(package, config, data)
            key = "r0_f0_i0_hgb_oral_full_c0"
            shutil.rmtree(ref.root / "TRE_ONLY" / key)
            with self.assertRaisesRegex(ValueError, "missing"):
                Reference(package, config, data)
