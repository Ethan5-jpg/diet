"""Read-only historical audit contracts using generated aggregate files only."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from history_reader import scan


class HistoricalAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hpp_v6_history_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.exp2 = self.root / "experiment2_Effect_Modification"

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
        return path

    def v1(self, name="complete", *, complete=True, valid_hash=True):
        path = self.exp2 / "v1/runs_TRE_ONLY" / name / "summary.json"
        self.write(path, dict(exposure="fiber_density", outcome="D_Glu", scores=[
            dict(common_main=dict(r2=.10), candidate=dict(r2=.12, delta=.02, ci=[.01, .03])),
            dict(common_main=dict(r2=.11), candidate=dict(r2=.10, delta=-.01)),
        ]))
        if complete:
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if valid_hash else "bad"
            self.write(path.parent / "COMPLETE.json", dict(summary_sha=digest))
        return path

    def reserved(self, *, valid_hash=True):
        path = self.exp2 / "v5/runs/TRE_ONLY/v5_reserved/reserved_summary.json"
        self.write(path, dict(base_r2=.10, full_r2=.09, delta_r2=-.01,
                             delta_r2_ci95=[-.03, .01]))
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if valid_hash else "bad"
        self.write(path.parent / "COMPLETE.json", dict(summary_sha256=digest))
        return path

    def test_v1_requires_completion_hash_and_keeps_both_repeats(self):
        path = self.v1()
        rows, issues = scan(self.root)
        self.assertEqual(issues, [])
        self.assertEqual([r["repeat"] for r in rows], [0, 1])
        self.assertEqual({r["label"] for r in rows}, {"exp2_v1:candidate"})
        self.assertEqual(rows[0]["source"], str(path.relative_to(self.root)))
        self.assertEqual(rows[0]["stage"], "discovery_family_oof")
        self.assertAlmostEqual(rows[0]["ratio_over_full_pct"], 100 * .02 / .12)
        self.assertEqual(rows[0]["original_delta_ci95"], [.01, .03])
        self.assertLess(rows[1]["delta_r2"], 0)

    def test_v3_v4_parse_completed_logs_and_deduplicate_copies(self):
        for version in (3, 4):
            result = dict(repeat=0, baseline_r2=.10, routes={"candidate": {
                ("r2" if version == 3 else "full_r2"): .12,
                "delta_r2": .02, ("delta_ci" if version == 3 else "ci95"): [.01, .03],
            }})
            if version == 4:
                result["background"] = "panel"
            line = f"EXP2_V{version}_PREDICT " + json.dumps(result) + "\n"
            for name, suffix in (("first", f"EXP2_V{version}_COMPLETE {{}}\n"),
                                 ("copy", f"EXP2_V{version}_COMPLETE {{}}\n"),
                                 ("incomplete", "")):
                path = self.exp2 / f"v{version}" / f"{name}.log"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(line + suffix, encoding="utf-8")
        rows, issues = scan(self.root)
        self.assertEqual(issues, [])
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["label"] for r in rows},
                         {"exp2_v3:shared:candidate", "exp2_v4:panel:candidate"})
        self.assertTrue(all(r["original_delta_ci95"] == [.01, .03] for r in rows))

    def test_legacy_v5_recovers_matched_r2_from_mse(self):
        folder = self.root / "experiments/diet_gut_fiber_modules_20261104_v5"
        for kind, repeat, mse, gain, delta in (("pc", 0, 88., 2., .02),
                                             ("module", 1, 91., -1., -.01)):
            self.write(folder / f"{kind}_TRE_ONLY/round{repeat}.json",
                       dict(MSE_null=90., MSE_model=mse, mse_gain=gain, delta_R2=delta))
        rows, issues = scan(self.root)
        self.assertEqual(issues, [])
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(by_label["legacy_v5:pc"]["base_r2"], .10)
        self.assertAlmostEqual(by_label["legacy_v5:pc"]["full_r2"], .12)
        self.assertAlmostEqual(by_label["legacy_v5:module"]["full_r2"], .09)
        self.assertEqual(by_label["legacy_v5:module"]["repeat"], 1)
        self.assertTrue(all(r["stage"] == "discovery_nested_family_oof" for r in rows))

    def test_reserved_requires_summary_hash_and_retains_negative_gain(self):
        self.reserved()
        rows, issues = scan(self.root)
        self.assertEqual(issues, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stage"], "frozen_HPP_reserved")
        self.assertIsNone(rows[0]["repeat"])
        self.assertEqual(rows[0]["original_delta_ci95"], [-.03, .01])
        self.assertAlmostEqual(rows[0]["ratio_over_full_pct"], -100 * .01 / .09)

    def test_bad_hash_missing_completion_and_incomplete_log_are_not_adopted(self):
        bad = self.v1("bad_hash", valid_hash=False)
        unfinished = self.v1("unfinished", complete=False)
        reserved = self.reserved(valid_hash=False)
        path = self.exp2 / "v3/incomplete.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('EXP2_V3_PREDICT {"repeat":0,"baseline_r2":0.1,'
                        '"routes":{"candidate":{"r2":0.12,"delta_r2":0.02}}}\n',
                        encoding="utf-8")
        rows, issues = scan(self.root)
        self.assertEqual(rows, [])
        self.assertEqual({r["source"] for r in issues},
                         {str(p.relative_to(self.root)) for p in (bad, unfinished, reserved)})


if __name__ == "__main__":
    unittest.main()
