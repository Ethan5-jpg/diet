"""Runtime guards and shell handoff checks; generated temporary files only."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_v6


class RuntimeTests(unittest.TestCase):
    def test_fresh_cli_limits_numerical_libraries_loaded_after_startup(self):
        # Existing in-process tests already import numpy/sklearn and therefore
        # cannot detect a limiter created before the numerical stack loads.
        script = '''
import json
import sys
import run_v6
from threadpoolctl import threadpool_info
before = dict(numpy_loaded="numpy" in sys.modules, pools=threadpool_info())
run_v6.release = lambda: {}
run_v6.environment = lambda: None
run_v6.preflight = lambda: print(json.dumps(dict(before=before, pools=threadpool_info())))
sys.argv = ["run_v6.py", "--preflight", "--threads", "2"]
run_v6.main()
'''
        env = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "11"
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                cwd=Path(__file__).parent, env=env,
                                capture_output=True, text=True, check=True, timeout=60)
        observed = json.loads(result.stdout)
        self.assertFalse(observed["before"]["numpy_loaded"])
        self.assertEqual(observed["before"]["pools"], [])
        self.assertTrue(observed["pools"])
        self.assertTrue(all(0 < pool["num_threads"] <= 2 for pool in observed["pools"]),
                        observed["pools"])

    @staticmethod
    def _fake_fold(source, repeat, fold, **kwargs):
        import numpy as np
        from experiment import route_names
        split = source.outer_split(repeat, fold)
        names = route_names()
        baseline = .65 * split.yv + .35 * source.y.mean()
        return (dict(rows=split.rows, y=split.yv, baseline=baseline,
                     full=np.tile(baseline[:, None], (1, len(names))),
                     contrasts=np.zeros((len(split.rows), len(names))),
                     groups=source.groups[split.rows]),
                dict(repeat=repeat, fold=fold, names=names, baseline="generated",
                     choices={name: {"name": "zero"} for name in names},
                     people=len(split.rows), seconds=0.0))

    def _complete_generated(self, folder):
        from integration_fixture import Generated
        from threadpoolctl import threadpool_limits
        source = Generated()
        with threadpool_limits(limits=2), contextlib.redirect_stdout(io.StringIO()):
            with patch("experiment.run_fold", side_effect=self._fake_fold):
                summary = run_v6.execute(source, folder, {"generated": "code"}, draws=99,
                                         bases=(("generated", 1, 2, 2),), emit_progress=False,
                                         threads=2)
        return source, summary

    def test_existing_lock_is_never_stolen_even_with_stale_pid(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with run_v6.lock(folder):
                before = (folder / "RUNNING.lock").read_bytes()
                with self.assertRaisesRegex(RuntimeError, "RUN_LOCK_EXISTS"):
                    with run_v6.lock(folder):
                        self.fail("concurrent writer entered")
                self.assertEqual((folder / "RUNNING.lock").read_bytes(), before)
            self.assertFalse((folder / "RUNNING.lock").exists())
            (folder / "RUNNING.lock").write_text(json.dumps(dict(pid=99999999, host="old", token="stale")))
            with self.assertRaisesRegex(RuntimeError, "RUN_LOCK_EXISTS"):
                with run_v6.lock(folder):
                    self.fail("stale path was stolen")
            self.assertTrue((folder / "RUNNING.lock").exists())

    def test_upload_manifest_detects_changed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            code = folder / "code.py"
            code.write_text("original")
            (folder / "upload_manifest.json").write_text(json.dumps(dict(
                version=run_v6.VERSION, files={"code.py": run_v6.digest(code)})))
            with patch.object(run_v6, "ROOT", folder), contextlib.redirect_stdout(io.StringIO()):
                run_v6.release()
                code.write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "UPLOAD_FILE_MISMATCH"):
                    run_v6.release()

    def test_run_all_stops_before_audit_and_models_when_preflight_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            shutil.copy2(Path(__file__).parent / "run_all.sh", folder / "run_all.sh")
            (folder / "server_run.sh").write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$1" >> calls.txt\nexit 7\n')
            result = subprocess.run(["bash", "run_all.sh"], cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7)
            self.assertEqual((folder / "calls.txt").read_text().splitlines(), ["--preflight"])

    def test_shipped_history_audit_keeps_scope_and_missingness(self):
        original = Path(__file__).parent
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "package/experiment2_Effect_Modification/v6"
            folder.mkdir(parents=True)
            for name in ("historical_pairs.json", "historical_missing.json"):
                shutil.copy2(original / name, folder / name)
            with patch.object(run_v6, "ROOT", folder), contextlib.redirect_stdout(io.StringIO()):
                run_v6.audit()
            result = run_v6.read_json(folder / "runs/TRE_ONLY/history_audit/history_ratios.json")
            self.assertEqual(result["shipped_rows"], 182)
            self.assertEqual(result["server_rows"], 0)
            self.assertTrue(result["historical_missing_groups"])
            best = result["fiber_person_D_Glu_exploratory_max"]
            self.assertEqual(best["label"], "legacy_V1 岭PC10 肠道")
            self.assertAlmostEqual(best["ratio_over_full_pct"], 6.456892195933457)
            self.assertTrue(any(not v["positive_r2_denominators"] for v in result["rows"]))

    def test_completed_receipt_requires_exact_artifacts_and_current_version(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "run"
            _, summary = self._complete_generated(folder)
            self.assertEqual(run_v6.completed(folder, expected_code_hashes={"generated": "code"}), summary)
            receipt = run_v6.read_json(folder / "COMPLETE.json")
            del receipt["artifacts"]["r1_f4/arrays.npz"]
            run_v6.write_json(folder / "COMPLETE.json", receipt)
            with self.assertRaisesRegex(RuntimeError, "ARTIFACT_SET_INCOMPLETE"):
                run_v6.completed(folder)
            receipt["artifacts"]["r1_f4/arrays.npz"] = run_v6.digest(folder / "r1_f4/arrays.npz")
            receipt["version"] = "old_version"
            run_v6.write_json(folder / "COMPLETE.json", receipt)
            with self.assertRaisesRegex(RuntimeError, "RUN_VERSION_MISMATCH"):
                run_v6.completed(folder)

    def test_completed_summary_cannot_change_cohort_routes_or_code(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "run"
            _, summary = self._complete_generated(folder)
            with self.assertRaisesRegex(RuntimeError, "RUN_CODE_MISMATCH"):
                run_v6.completed(folder, expected_code_hashes={"generated": "different"})
            summary["repeats"][0]["people"] += 1
            run_v6.write_json(folder / "summary.json", summary)
            receipt = run_v6.read_json(folder / "COMPLETE.json")
            receipt["artifacts"]["summary.json"] = run_v6.digest(folder / "summary.json")
            run_v6.write_json(folder / "COMPLETE.json", receipt)
            with self.assertRaisesRegex(RuntimeError, "COHORT_OR_ROUTE_MISMATCH"):
                run_v6.completed(folder)
            summary["repeats"][0]["people"] -= 1
            del summary["repeats"][0]["routes"]["nested_selector"]
            run_v6.write_json(folder / "summary.json", summary)
            receipt["artifacts"]["summary.json"] = run_v6.digest(folder / "summary.json")
            run_v6.write_json(folder / "COMPLETE.json", receipt)
            with self.assertRaisesRegex(RuntimeError, "COHORT_OR_ROUTE_MISMATCH"):
                run_v6.completed(folder)

    def test_requested_threads_or_runtime_change_blocks_checkpoint_reuse(self):
        from threadpoolctl import threadpool_limits
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "run"
            source, _ = self._complete_generated(folder)
            registration = run_v6.read_json(folder / "registration.json")
            self.assertEqual(registration["runtime"]["requested_threads"], 2)
            self.assertEqual(registration["cohort"][0], dict(repeat=0, people=180, households=90))
            self.assertTrue(registration["runtime"]["threadpools"])
            self.assertEqual(set(registration["runtime"]["package_versions"]), set(run_v6.PINS))
            with threadpool_limits(limits=2), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "REGISTRATION_CHANGED"):
                    run_v6.execute(source, folder, {"generated": "code"}, draws=99,
                                   bases=(("generated", 1, 2, 2),), emit_progress=False, threads=3)
                changed = dict(registration["runtime"], python_patch="3.11.999")
                with patch("run_v6.runtime_identity", return_value=changed):
                    with self.assertRaisesRegex(RuntimeError, "REGISTRATION_CHANGED"):
                        run_v6.execute(source, folder, {"generated": "code"}, draws=99,
                                       bases=(("generated", 1, 2, 2),), emit_progress=False, threads=2)


if __name__ == "__main__":
    unittest.main()
