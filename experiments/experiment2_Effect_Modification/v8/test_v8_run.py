"""End-to-end replay, paired ensembles, interrupted runs and immutable output."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

import run_v8
import run_v7
import run_v6 as store
import v8_models
import v8_selection as sel
from integration_fixture import Generated


class RunTests(unittest.TestCase):
    def test_full_nested_run_replays_v7_and_resumes(self):
        source = Generated(n=180, seed=67211, effect=5.)
        release = {"scope": "generated"}
        release7 = store.read_json(run_v8.V7 / "upload_manifest.json")
        with tempfile.TemporaryDirectory() as tmp, threadpool_limits(1), contextlib.redirect_stdout(io.StringIO()):
            run, previous = Path(tmp) / "v8", Path(tmp) / "v7"
            run_v7.execute(source, previous, release7, 1, draws=99)
            fit = sel.fold_predictions
            def interrupt(src, r, f):
                if (r, f) == (0, 1):
                    raise InterruptedError("simulated interruption")
                return fit(src, r, f)
            with patch("v8_selection.fold_predictions", side_effect=interrupt), self.assertRaises(InterruptedError):
                run_v8.execute(source, run, release, 1, draws=99, prior_run=previous)
            self.assertTrue((run/"r0_f0/receipt.json").is_file())
            self.assertFalse((run/"COMPLETE.json").exists())
            with patch("v8_selection.fold_predictions", wraps=fit) as resumed:
                report = run_v8.execute(source, run, release, 1, draws=99, prior_run=previous)
            self.assertEqual(resumed.call_count, 9)
            self.assertEqual(report["v7_replay"]["status"], "v7_predictions_match")
            self.assertEqual(report["v7_replay"]["folds_checked"], 10)
            self.assertFalse(report["reserved_outcomes_read"])
            self.assertEqual(set(report["routes"]), set(sel.NAMES))
            self.assertGreater(report["routes"][sel.PRIMARY]["minimum_ratio_pct"], 0.)
            with (run/"all_routes.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2 * len(sel.NAMES))
            for row in rows:
                self.assertAlmostEqual(float(row["delta_r2"]), float(row["on_r2"]) - float(row["off_r2"]))
            run_v8.compact(report)
            with patch("v8_selection.fold_predictions", side_effect=AssertionError("refit")):
                self.assertEqual(run_v8.execute(source, run, release, 1, draws=99), report)
            with self.assertRaisesRegex(ValueError, "registration differs"):
                run_v8.execute(source, run, release, 1, draws=199)
            # Checkpoint corruption must not pass completed-report verification.
            (run/"r1_f3/arrays.npz").write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                run_v8.completed(run, release)

    def test_selected_ensemble_uses_identical_weights_for_both_arms(self):
        src = Generated(n=180, seed=57771)
        with threadpool_limits(1):
            arrays, meta = sel.fold_predictions(src, 1, 4)
            expected = v8_models.evaluate_split(src, src.outer_split(1, 4))
        choice = meta["choices"][sel.ENSEMBLE]
        self.assertTrue(choice["enabled"])
        self.assertEqual(len(choice["settings"]), 3)
        for index, arm in enumerate(("off", "on")):
            together = np.mean([expected[key][index] for key in choice["settings"]], axis=0)
            np.testing.assert_allclose(arrays[arm][:, sel.NAMES.index(sel.ENSEMBLE)], together, atol=1e-12)
