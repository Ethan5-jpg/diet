#!/usr/bin/env python3
"""Tests for diet–CGM alignment and unadjusted association models."""

from __future__ import print_function

import importlib.util
import math
import os
import sys
import unittest

import numpy as np
import pandas as pd


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.abspath(os.path.join(TEST_DIR, "..", "scripts"))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def load_script(module_name, filename):
    path = os.path.join(SCRIPT_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


align = load_script("align_diet_cgm", "00_align_diet_cgm.py")
models = load_script("unadjusted_models", "01_unadjusted_diet_cgm_models.py")


def identity(participant):
    return {
        "participant_id": participant,
        "cohort": "10k",
        "research_stage": "00_00_visit",
    }


def make_alignment_tables():
    amed_rows = []
    for participant, score in [("p1", -1.0), ("p2", 0.0), ("p3", 1.0)]:
        row = identity(participant)
        row["amed_energy_adjusted_score_z"] = score
        amed_rows.append(row)

    hpdi_rows = []
    for participant, score in [("p2", -1.0), ("p3", 0.0), ("p4", 1.0)]:
        row = identity(participant)
        row["hpdi_score_energy_adjusted_z"] = score
        hpdi_rows.append(row)

    cgm_rows = []
    for index, participant in enumerate(["p1", "p2", "p3", "p4", "p5"]):
        row = identity(participant)
        row.update(
            {
                "array_index": 0,
                "connection_id": "c" + participant[1:],
                "cgm_mean_raw": 90.0 + index,
                "cgm_mean_clean": 90.0 + index,
                "cgm_mean_z": -1.0 + index * 0.5,
                "cgm_cv_raw": 15.0 + index,
                "cgm_cv_clean": 15.0 + index,
                "cgm_cv_z": -0.8 + index * 0.4,
                "cgm_above_140_raw": 1.0 + index,
                "cgm_above_140_clean": 1.0 + index,
                "cgm_above_140_z": -0.6 + index * 0.3,
            }
        )
        cgm_rows.append(row)
    return pd.DataFrame(amed_rows), pd.DataFrame(hpdi_rows), pd.DataFrame(cgm_rows)


class AlignmentTests(unittest.TestCase):
    def test_diet_union_is_inner_joined_to_cgm(self):
        amed, hpdi, cgm = make_alignment_tables()

        aligned, membership, summary = align.align_diet_cgm(amed, hpdi, cgm)

        self.assertEqual(set(aligned["participant_id"]), {"p1", "p2", "p3", "p4"})
        self.assertEqual(len(aligned), 4)
        self.assertTrue(
            math.isnan(
                float(
                    aligned.set_index("participant_id").loc[
                        "p1", "hpdi_score_energy_adjusted_z"
                    ]
                )
            )
        )
        self.assertEqual(summary["amed_participants"], 3)
        self.assertEqual(summary["hpdi_participants"], 3)
        self.assertEqual(summary["diet_union_participants"], 4)
        self.assertEqual(summary["cgm_participants"], 5)
        self.assertEqual(summary["diet_union_and_cgm_participants"], 4)
        self.assertEqual(summary["amed_hpdi_and_cgm_participants"], 2)
        self.assertEqual(len(membership), 5)

    def test_old_hpdi_csv_without_z_score_stops_with_rerun_message(self):
        amed, hpdi, cgm = make_alignment_tables()
        hpdi = hpdi.rename(
            columns={
                "hpdi_score_energy_adjusted_z": "hpdi_score_energy_adjusted"
            }
        )

        with self.assertRaisesRegex(ValueError, "05_calculate_hpdi_scores.py"):
            align.align_diet_cgm(amed, hpdi, cgm)

    def test_gut_presence_is_reported_but_does_not_restrict_alignment(self):
        amed, hpdi, cgm = make_alignment_tables()
        gut = pd.DataFrame(
            [
                dict(identity("p2"), sample_name="g1"),
                dict(identity("p2"), sample_name="g2"),
                dict(identity("p5"), sample_name="g3"),
            ]
        )

        aligned, membership, summary = align.align_diet_cgm(
            amed, hpdi, cgm, gut=gut
        )

        self.assertEqual(len(aligned), 4)
        member = membership.set_index("participant_id")
        self.assertTrue(bool(member.loc["p2", "has_gut_sample"]))
        self.assertEqual(int(member.loc["p2", "gut_sample_count"]), 2)
        self.assertEqual(summary["gut_participants"], 2)
        self.assertEqual(summary["diet_cgm_and_gut_participants"], 1)


class ModelTests(unittest.TestCase):
    def make_model_frame(self):
        frame = pd.DataFrame(
            {
                "participant_id": ["a", "b", "c", "d", "e", "f"],
                "amed_energy_adjusted_score_z": [-2, -1, 0, 1, 2, 3],
                "hpdi_score_energy_adjusted_z": [3, 2, 1, 0, -1, -2],
                "cgm_mean_z": [-2.8, -0.9, 1.0, 3.2, 4.9, 7.1],
                "cgm_cv_z": [3.0, 2.0, 1.2, -0.1, -1.0, -2.2],
                "cgm_above_140_z": [-1.0, -0.4, np.nan, 0.7, 1.1, 1.8],
            }
        )
        return frame

    def test_simple_ols_matches_numpy_coefficient(self):
        x = pd.Series([-2, -1, 0, 1, 2, 3], dtype="float64")
        y = pd.Series([-2.8, -0.9, 1.0, 3.2, 4.9, 7.1], dtype="float64")

        result = models.fit_simple_ols(x, y)
        expected_beta, expected_intercept = np.polyfit(x.to_numpy(), y.to_numpy(), 1)

        self.assertAlmostEqual(result["beta"], float(expected_beta), places=12)
        self.assertAlmostEqual(result["intercept"], float(expected_intercept), places=12)
        self.assertEqual(result["n"], 6)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_models_use_outcome_specific_complete_cases(self):
        results, summary = models.run_unadjusted_models(self.make_model_frame())

        self.assertEqual(len(results), 6)
        above = results[results["outcome"] == "cgm_above_140_z"]
        other = results[results["outcome"] != "cgm_above_140_z"]
        self.assertEqual(set(above["n"]), {5})
        self.assertEqual(set(other["n"]), {6})
        self.assertTrue(results["fdr_bh"].between(0.0, 1.0).all())
        self.assertEqual(summary["models_fitted"], 6)
        self.assertEqual(summary["covariate_count"], 0)

    def test_bh_fdr_matches_known_values(self):
        adjusted = models.benjamini_hochberg([0.01, 0.04, 0.03, 0.002])

        expected = [0.02, 0.04, 0.04, 0.008]
        for observed, target in zip(adjusted, expected):
            self.assertAlmostEqual(observed, target, places=12)

    def test_student_t_inference_matches_known_df4_cutoff(self):
        cutoff = 2.7764451051977987

        self.assertAlmostEqual(
            models.student_t_two_sided_p(cutoff, 4), 0.05, places=10
        )
        self.assertAlmostEqual(
            models.student_t_quantile(0.975, 4), cutoff, places=10
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
