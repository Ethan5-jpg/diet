"""Contract tests for the generated integration source (no native HPP reads)."""

import os
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from integration_fixture import Generated, PANEL_TAXA


class GeneratedSourceTests(unittest.TestCase):
    def test_all_roles_are_household_separated_and_complete(self):
        source = Generated()
        self.assertEqual(len(source.index), 180)
        for repeat in range(2):
            seen = []
            for fold in range(5):
                inner, outer = source.roles(repeat, fold)
                seen.extend(outer.rows.tolist())
                outer_groups = source.groups[outer.rows]
                self.assertEqual(len(inner), 3)
                inner_seen = []
                for split in inner:
                    self.assertFalse(np.intersect1d(source.groups[split.train_rows], source.groups[split.rows]).size)
                    self.assertFalse(np.intersect1d(outer_groups, source.groups[np.r_[split.train_rows, split.rows]]).size)
                    inner_seen.extend(split.rows.tolist())
                np.testing.assert_array_equal(np.sort(inner_seen), outer.train_rows)
            np.testing.assert_array_equal(np.sort(seen), np.arange(len(source.y)))
        self.assertFalse(np.array_equal(*source.reference["outer"]))

    def test_train_transform_ignores_held_out_rows_and_targets(self):
        source = Generated()
        outer = source.outer_split(0, 0)
        changed = source.with_y(source.y + (source.reference["outer"][0] == 0) * 1000)
        replay = changed.outer_split(0, 0)
        for name in ("ht", "hv", "dt", "dv", "delta", "ot", "ov", "gt", "gv", "yt"):
            np.testing.assert_array_equal(getattr(outer, name), getattr(replay, name))
        np.testing.assert_allclose(replay.yv - outer.yv, 1000, atol=1e-12, rtol=0)
        np.testing.assert_allclose(outer.ot.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(outer.gt.std(axis=0), 1, atol=1e-14)
        self.assertEqual(source.taxa["gut"][:3], list(PANEL_TAXA))

    def test_invalid_role_and_invalid_replacement_fail(self):
        source = Generated()
        valid = np.zeros(len(source.y), dtype=bool)
        valid[0] = True
        with self.assertRaisesRegex(ValueError, "HOUSEHOLD_ROLE"):
            source.split(~valid, valid)
        with self.assertRaisesRegex(ValueError, "OUTCOME"):
            source.with_y(np.full(len(source.y), np.nan))
        with self.assertRaisesRegex(ValueError, "CONFIG"):
            Generated(n=179)


class RunnerIntegrationTests(unittest.TestCase):
    def test_fixed_baseline_has_no_diet_by_microbe_cross_difference(self):
        from adapters import moderation_models as mm
        from experiment import prepare

        source = Generated()
        prepared = prepare(source, source.outer_split(0, 0))
        role = prepared.role
        self.assertEqual(prepared.feature_audit["rff_blocks"], 6)
        self.assertEqual(role.shared_columns, (2, 3))
        np.testing.assert_array_equal(role.design[0][:, -288:],
                                      np.column_stack([value[0] for value in prepared.modifiers.values()]))
        with threadpool_limits(limits=2):
            fit = mm.fit_base(role.design[0], role.y[0], role.width,
                              ("tiny", 12, 5, 8), role.shared_columns)
            a, b = role.design[0][:12].copy(), role.design[0][12:24].copy()
            # Shared host age and sex must agree while varying diet and microbes.
            b[:, role.shared_columns] = a[:, role.shared_columns]
            c, d = a.copy(), b.copy()
            c[:, role.width:] = b[:, role.width:]
            d[:, role.width:] = a[:, role.width:]
            np.testing.assert_allclose(fit.predict(a) + fit.predict(b)
                                       - fit.predict(c) - fit.predict(d), 0.0, atol=1e-10)

    def test_all_routes_do_not_use_outer_targets(self):
        from experiment import run_fold, route_names, settings

        source = Generated()
        changed = source.with_y(source.y + (source.reference["outer"][0] == 0) * 1000)
        with threadpool_limits(limits=2):
            original = run_fold(source, 0, 0, bases=(("tiny", 12, 5, 8),))
            replay = run_fold(changed, 0, 0, bases=(("tiny", 12, 5, 8),))
        self.assertEqual(len(settings()), 204)
        self.assertEqual(len(route_names()), 78)
        self.assertEqual(original[1]["choices"], replay[1]["choices"])
        self.assertEqual(original[1]["baseline"], replay[1]["baseline"])
        for name in ("rows", "baseline", "full", "contrasts", "groups"):
            np.testing.assert_array_equal(original[0][name], replay[0][name])
        np.testing.assert_allclose(replay[0]["y"] - original[0]["y"], 1000,
                                   atol=1e-12, rtol=0)
        self.assertTrue(np.isfinite(original[0]["full"]).all())


class StorageLifecycleTests(unittest.TestCase):
    @staticmethod
    def fake_fold(source, repeat, fold, **kwargs):
        from experiment import route_names

        split = source.outer_split(repeat, fold)
        baseline = 0.65 * split.yv + 0.35 * source.y.mean()
        names = route_names()
        arrays = dict(rows=split.rows, y=split.yv, baseline=baseline,
                      full=np.tile(baseline[:, None], (1, len(names))),
                      contrasts=np.zeros((len(split.rows), len(names))),
                      groups=source.groups[split.rows])
        metadata = dict(repeat=repeat, fold=fold, names=names, baseline="tiny",
                        choices={name: {"name": "zero"} for name in names},
                        people=len(split.rows), seconds=0.0)
        return arrays, metadata

    def test_fold_checkpoint_roundtrip_and_tamper_rejected(self):
        from run_v6 import save_fold, read_fold

        source = Generated()
        arrays, metadata = self.fake_fold(source, 0, 0)
        with tempfile.TemporaryDirectory(prefix="hpp_v6_test_checkpoint_") as temp:
            run = Path(temp)
            save_fold(run, 0, 0, arrays, metadata)
            actual, detail = read_fold(run, 0, 0)
            self.assertEqual(detail, metadata)
            for name in arrays:
                np.testing.assert_array_equal(actual[name], arrays[name])
            with self.assertRaisesRegex(RuntimeError, "OVERWRITE"):
                save_fold(run, 0, 0, arrays, metadata)
            path = run / "r0_f0/arrays.npz"
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(RuntimeError, "HASH_MISMATCH"):
                read_fold(run, 0, 0)

    def test_interruption_resume_completed_replay_and_registration_guard(self):
        from run_v6 import execute, completed

        source = Generated()
        bases = (("tiny", 12, 5, 8),)
        with tempfile.TemporaryDirectory(prefix="hpp_v6_test_resume_") as temp:
            run = Path(temp) / "run"

            def interrupt(source, repeat, fold, **kwargs):
                if (repeat, fold) == (0, 2):
                    raise RuntimeError("generated interruption")
                return self.fake_fold(source, repeat, fold, **kwargs)

            with contextlib.redirect_stdout(io.StringIO()):
                with patch("experiment.run_fold", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "generated interruption"):
                        execute(source, run, {"synthetic": "test"}, draws=99,
                                bases=bases, emit_progress=False)
                self.assertFalse((run / "RUNNING.lock").exists())
                self.assertFalse((run / "COMPLETE.json").exists())
                with patch("experiment.run_fold", side_effect=self.fake_fold) as fitted:
                    summary = execute(source, run, {"synthetic": "test"}, draws=99,
                                      bases=bases, emit_progress=False)
                    self.assertEqual(fitted.call_count, 8)
                self.assertEqual(completed(run), summary)
                for result in summary["repeats"]:
                    for route in result["routes"].values():
                        self.assertAlmostEqual(route["delta_r2"], 0.0, places=14)
                        self.assertAlmostEqual(route["ratio_over_full_pct"], 0.0, places=12)
                with patch("experiment.run_fold", side_effect=AssertionError("must reuse")):
                    replay = execute(source, run, {"synthetic": "test"}, draws=99,
                                     bases=bases, emit_progress=False)
                    self.assertEqual(replay, summary)
                with self.assertRaisesRegex(RuntimeError, "REGISTRATION_CHANGED"):
                    execute(source, run, {"synthetic": "changed"}, draws=99,
                            bases=bases, emit_progress=False)
                self.assertFalse((run / "RUNNING.lock").exists())


@unittest.skipUnless(os.environ.get("HPP_V6_FULL_FIXTURE") == "1",
                     "optional 10-fold generated model lifecycle check")
class FullGeneratedLifecycleTests(unittest.TestCase):
    def test_full_generated_run_and_completed_resume(self):
        from run_v6 import execute, completed

        source = Generated()
        with tempfile.TemporaryDirectory(prefix="hpp_v6_test_full_") as temp:
            run = Path(temp) / "run"
            with threadpool_limits(limits=2), contextlib.redirect_stdout(io.StringIO()):
                summary = execute(source, run, {"synthetic": "full"}, draws=127,
                                  bases=(("tiny", 12, 5, 8),), emit_progress=False)
                self.assertEqual(completed(run), summary)
                with patch("experiment.run_fold", side_effect=AssertionError("must reuse")):
                    replay = execute(source, run, {"synthetic": "full"}, draws=127,
                                     bases=(("tiny", 12, 5, 8),), emit_progress=False)
            self.assertEqual(replay, summary)
            self.assertEqual(len(summary["repeats"]), 2)
            self.assertEqual(summary["route_count"], 78)
            self.assertFalse((run / "RUNNING.lock").exists())
            for result in summary["repeats"]:
                selected = result["routes"]["nested_selector"]
                self.assertGreater(selected["delta_r2"], 0.05)
                self.assertGreater(selected["ratio_over_full_pct"], 0.0)


@unittest.skipUnless(os.environ.get("HPP_V6_NATIVE_FIXTURE") == "1",
                     "optional local full-archive fixture; not a native HPP run")
class NativeLayoutTests(unittest.TestCase):
    def test_real_adapter_and_one_outer_fold_with_generated_files(self):
        from adapters.person_inputs import load_discovery
        from experiment import run_fold, route_names
        from integration_fixture import make_native_layout

        with tempfile.TemporaryDirectory(prefix="hpp_v6_generated_layout_") as temp:
            with threadpool_limits(limits=2):
                fake_v6 = make_native_layout(Path(temp) / "collaborator_handoff_20261102_v1")
                source = load_discovery(fake_v6)
                self.assertEqual(len(source.index), 4427)
                self.assertEqual(np.isfinite(source.reference["y"][0]).sum(), 3343)
                self.assertEqual(len(np.unique(source.groups[np.isfinite(source.reference["y"][0])])), 3059)
                arrays, metadata = run_fold(source, 0, 0, bases=(("tiny", 12, 5, 8),))
                source.verify_sources()
                expected = np.flatnonzero((source.reference["outer"][0] == 0)
                                         & np.isfinite(source.reference["y"][0]))
                np.testing.assert_array_equal(arrays["rows"], expected)
                self.assertEqual(arrays["full"].shape, (len(expected), len(route_names())))
                self.assertTrue(np.isfinite(arrays["full"]).all())
                self.assertEqual(metadata["feature_audit"]["rff_columns"], 288)


if __name__ == "__main__":
    unittest.main()
