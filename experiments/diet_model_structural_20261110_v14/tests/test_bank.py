"""Actual V10/V11/V13 cache adapter, saved replay, corruption and provenance."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bridge import V13, V13Reference, Memo, imported, sha, write_json
from fusion import selected_linear
from legacy import Reference
from registry import bank_names


def cached_bank(package):
    from cache_fixture import cached_bank as previous_bank

    cfg, data, roles, rows, y = previous_bank(package)
    old = V13Reference(package, cfg, data)
    source = package / "experiments" / V13.name
    source.mkdir()
    for p in list(V13.glob("*.py")) + [V13 / "requirements.txt"]:
        shutil.copy2(p, source / p.name)
    previous = json.loads((V13 / "config.json").read_text())
    for family in previous["families"].values():
        family["grid"] = family["grid"][:1]
    (source / "config.json").write_text(json.dumps(previous))
    registry = imported("v14_synthetic_v13_registry", source / "registry.py")
    specs = registry.specifications(previous)
    root = source / "runs" / cfg["v13_reference_run"]
    root.mkdir(parents=True)
    memo = Memo(root / "TRE_ONLY")
    targets = np.concatenate([r.split.yv for r in roles])
    groups = np.concatenate([r.index.iloc[r.split.rows].split_unit_id.to_numpy() for r in roles])
    for repeat in range(2):
        for fold in range(5):
            tag = "r%d_f%d" % (repeat, fold)
            prior = old.inner(repeat, fold, roles)
            memo.get(tag + "_prior_inner", lambda prior=prior: prior)
            values = dict(prior["predictions"])
            for spec in specs:
                name = spec["name"]
                values[name] = targets + 0.2
                for k, role in enumerate(roles):
                    memo.get(
                        tag + "_i%d_%s_c0" % (k, name), lambda role=role: dict(prediction=role.split.yv + 0.2)
                    )
            members = list(values)
            fit = selected_linear(
                members,
                np.column_stack(list(values.values())),
                targets,
                groups,
                "positive",
                previous["meta_penalties"],
            )
            memo.get(tag + "_meta_fit", lambda fit=fit: dict(positive_combined=fit))
            outer, _ = old.get(repeat, fold, rows, y, prior["choices"])
            outer.update({s["name"]: y + 0.2 for s in specs})
            baseline = fit.predict(outer)
            memo.get(
                tag + "_result",
                lambda outer=outer, baseline=baseline: dict(
                    rows=rows,
                    y=y,
                    selected={s["name"]: 0 for s in specs},
                    predictions=dict(outer, positive_combined=baseline),
                    outer_diagnostics={s["name"]: {} for s in specs},
                ),
            )
    required = (
        list(source.glob("*.py"))
        + [source / "config.json", source / "requirements.txt"]
        + data.paths
        + old.paths
    )
    write_json(root / "registration.json", dict(config=previous, hashes={str(p): sha(p) for p in required}))
    write_json(root / "summary.json", dict(status="COMPLETE"))
    return cfg, data, roles, rows, y


class BankChecks(unittest.TestCase):
    def test_saved_replays_need_no_old_solver_or_outer_payload_during_inner_training(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            cfg, data, roles, rows, y = cached_bank(package)
            before = {str(p): sha(p) for p in package.rglob("*") if p.is_file()}
            ref = Reference(package, cfg, data)
            with (
                patch.object(ref, "read", wraps=ref.read) as reader,
                patch.object(ref.old, "inner", side_effect=AssertionError("Old baseline unexpectedly refit")),
                patch(
                    "fusion.selected_linear", side_effect=AssertionError("Historical NNLS unexpectedly refit")
                ),
            ):
                prior = ref.inner(0, 0, roles)
            self.assertFalse(any(c.args[0].endswith("_result") for c in reader.call_args_list))
            self.assertEqual(list(prior["predictions"]), bank_names(cfg))
            outer, _ = ref.get(0, 0, rows, y, prior["choices"])
            for version, fit in prior["replay"].items():
                np.testing.assert_allclose(
                    fit.predict(outer), outer[version.replace("_replay", "_positive_combined")], atol=1e-6
                )
            self.assertEqual(before, {str(p): sha(p) for p in package.rglob("*") if p.is_file()})
            with self.assertRaisesRegex(ValueError, "rows/processed"):
                ref.get(0, 0, rows[::-1], y, prior["choices"])
            bad = copy.deepcopy(prior["choices"])
            bad["v13"][next(iter(bad["v13"]))] = 1
            with self.assertRaisesRegex(ValueError, "choices"):
                ref.get(0, 0, rows, y, bad)
            path, _ = ref.units["r0_f0_i0_hgb_pc_oral_c0"]
            original = path.read_bytes()
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "changed|Corrupt|corrupt"):
                ref.inner(0, 0, roles)
            with self.assertRaises(ValueError):
                Reference(package, cfg, data)
            path.write_bytes(original)
            data.paths[0].write_text("DRIFT")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                Reference(package, cfg, data)

    def test_missing_cache_or_changed_v13_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)
            cfg, data, roles, rows, y = cached_bank(package)
            source = package / "experiments" / V13.name
            p = source / "features.py"
            original = p.read_bytes()
            p.write_bytes(original + b"\n# drift\n")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                Reference(package, cfg, data)
            p.write_bytes(original)
            shutil.rmtree(source / "runs" / cfg["v13_reference_run"] / "TRE_ONLY/r0_f0_prior_inner")
            with self.assertRaisesRegex(ValueError, "missing"):
                Reference(package, cfg, data)
