#!/usr/bin/env python3
"""Tests for the fixed CGM preprocessing workflow.

The tests use small in-memory tables that mirror the exported CSV structure.  They
can run without access to the server data.
"""

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

import cgm_utils


def load_script(module_name, filename):
    """Load a numbered pipeline script as an importable module."""

    path = os.path.join(SCRIPT_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_baseline", "01_prepare_baseline_connections.py")
connection_qc = load_script("connection_qc", "02_cgm_connection_qc.py")
extract = load_script("extract_phenotypes", "03_extract_iglu_phenotypes.py")
clean = load_script("clean_phenotypes", "04_clean_cgm_phenotypes.py")
standardize = load_script("standardize_phenotypes", "05_standardize_cgm_phenotypes.py")


FULL_KEY = [
    "participant_id",
    "cohort",
    "research_stage",
    "array_index",
    "connection_id",
]

CORE_OUTCOMES = [
    "cgm_mean",
    "cgm_cv",
    "cgm_above_140",
    "cgm_gmi",
    "cgm_in_range_63_140",
    "cgm_mage",
    "cgm_modd",
]


def make_source_tables():
    """Create CGM, daily, and iglu fixtures covering every decision branch."""

    connection_specs = [
        ("p1", "p1_short", 5, 0.10),
        ("p1", "p1_long", 12, 0.20),
        ("p2", "p2_high_loss", 12, 0.20),
        ("p2", "p2_low_loss", 12, 0.10),
        ("p3", "p3_nine_days", 9, 0.10),
        ("p4", "p4_bad_loss", 12, 0.40),
        ("p5", "p5_missing_primary", 12, 0.10),
        ("p6", "p6_high_but_valid", 12, 0.10),
    ]

    cgm_rows = []
    daily_rows = []
    iglu_rows = []
    for offset, (participant, connection, days, loss) in enumerate(connection_specs):
        identity = {
            "participant_id": participant,
            "cohort": "10k",
            "research_stage": "00_00_visit",
            "array_index": 0,
            "connection_id": connection,
        }
        cgm_row = dict(identity)
        cgm_row.update(
            {
                "connection_timestamp": "2022-01-%02d 00:00:00+00:00" % (offset + 1),
                "cgm_device_type": "abbott_freestyle_libre_pro_iq",
                "percentage_of_cgm_datapoints_lost_in_qc": loss,
            }
        )
        cgm_rows.append(cgm_row)

        for day in range(1, days + 1):
            daily_row = dict(identity)
            daily_row.update(
                {
                    # In iglu_daily, array_index identifies the repeated daily row;
                    # it is not part of the connection-level identity.
                    "array_index": day - 1,
                    "connection_day": day,
                    "collection_date": "2022-02-%02d" % day,
                }
            )
            daily_rows.append(daily_row)

        phenotype_values = {
            "cgm_mean": 90.0 + offset,
            "cgm_cv": 15.0 + offset,
            "cgm_above_140": 2.0 + offset,
            "cgm_gmi": 5.5 + offset / 10.0,
            "cgm_in_range_63_140": 90.0 - offset,
            "cgm_mage": 20.0 + offset,
            "cgm_modd": 10.0 + offset,
        }
        if participant == "p5":
            phenotype_values["cgm_mean"] = np.nan
        if participant == "p6":
            phenotype_values.update(
                {
                    "cgm_mean": 300.0,
                    "cgm_cv": 90.0,
                    "cgm_above_140": 100.0,
                }
            )
        iglu_row = dict(identity)
        iglu_row.update(phenotype_values)
        iglu_rows.append(iglu_row)

    return pd.DataFrame(cgm_rows), pd.DataFrame(daily_rows), pd.DataFrame(iglu_rows)


class BaselineSelectionTests(unittest.TestCase):
    def test_selects_one_best_connection_per_participant(self):
        cgm, daily, _ = make_source_tables()

        selected, duplicate_report, summary = prepare.prepare_baseline_connections(
            cgm, daily
        )

        self.assertEqual(len(selected), 6)
        selected_by_participant = selected.set_index("participant_id")
        self.assertEqual(selected_by_participant.loc["p1", "connection_id"], "p1_long")
        self.assertEqual(
            selected_by_participant.loc["p2", "connection_id"], "p2_low_loss"
        )
        self.assertEqual(int(duplicate_report["selected_flag"].sum()), 2)
        self.assertEqual(summary["participants_with_multiple_connections"], 2)

    def test_stops_if_missing_datapoint_count_leaves_an_unresolved_tie(self):
        cgm, daily, _ = make_source_tables()
        tied_cgm = cgm.iloc[[0, 1]].copy()
        tied_cgm["participant_id"] = "tie_person"
        tied_cgm["connection_id"] = ["tie_a", "tie_b"]
        tied_cgm["percentage_of_cgm_datapoints_lost_in_qc"] = 0.10

        tied_daily_rows = []
        for connection in ["tie_a", "tie_b"]:
            for day in range(1, 13):
                tied_daily_rows.append(
                    {
                        "participant_id": "tie_person",
                        "cohort": "10k",
                        "research_stage": "00_00_visit",
                        "array_index": day - 1,
                        "connection_id": connection,
                        "connection_day": day,
                        "collection_date": "2022-03-%02d" % day,
                    }
                )

        with self.assertRaisesRegex(ValueError, "datapoint"):
            prepare.prepare_baseline_connections(
                tied_cgm, pd.DataFrame(tied_daily_rows)
            )


class ConnectionQCTests(unittest.TestCase):
    def test_applies_only_the_prespecified_connection_filters(self):
        cgm, daily, iglu = make_source_tables()
        selected, _, _ = prepare.prepare_baseline_connections(cgm, daily)

        all_qc, passed, excluded, summary = connection_qc.apply_connection_qc(
            selected, iglu
        )

        self.assertEqual(set(passed["participant_id"]), {"p1", "p2", "p6"})
        self.assertEqual(set(excluded["participant_id"]), {"p3", "p4", "p5"})
        self.assertTrue(
            bool(all_qc.set_index("participant_id").loc["p6", "cgm_qc_pass"])
        )
        self.assertEqual(summary["participants_passing_all_connection_qc"], 3)


class PhenotypeExtractionTests(unittest.TestCase):
    def test_illegal_values_become_missing_without_dropping_participants(self):
        cgm, daily, iglu = make_source_tables()
        selected, _, _ = prepare.prepare_baseline_connections(cgm, daily)
        _, passed, _, _ = connection_qc.apply_connection_qc(selected, iglu)
        passed.loc[passed["participant_id"] == "p6", "cgm_above_140"] = 120.0
        passed.loc[passed["participant_id"] == "p6", "cgm_mage"] = -1.0
        passed.loc[passed["participant_id"] == "p6", "cgm_gmi"] = np.inf

        raw, legality_report, summary = extract.extract_core_phenotypes(passed)

        self.assertEqual(len(raw), 3)
        p6 = raw.set_index("participant_id").loc["p6"]
        self.assertTrue(math.isnan(float(p6["cgm_above_140_raw"])))
        self.assertTrue(math.isnan(float(p6["cgm_mage_raw"])))
        self.assertTrue(math.isnan(float(p6["cgm_gmi_raw"])))
        self.assertEqual(len(legality_report), 3)
        self.assertEqual(summary["participants_retained"], 3)


class PhenotypeCleaningTests(unittest.TestCase):
    def test_95_percent_interval_drives_8sd_removal_and_5sd_winsorization(self):
        values = pd.Series(list(np.linspace(0.0, 1.0, 100)) + [2.0, 100.0])

        cleaned, parameters, extreme_mask, winsor_mask = clean.clean_outcome_series(
            values, "example"
        )

        self.assertTrue(bool(extreme_mask.iloc[-1]))
        self.assertTrue(math.isnan(float(cleaned.iloc[-1])))
        self.assertTrue(bool(winsor_mask.iloc[-2]))
        self.assertLess(float(cleaned.iloc[-2]), 2.0)
        self.assertEqual(parameters["coverage_target"], 0.95)


class StandardizationTests(unittest.TestCase):
    def test_clean_values_are_standardized_with_sample_sd(self):
        frame = pd.DataFrame({"participant_id": ["a", "b", "c"]})
        for outcome_index, outcome in enumerate(CORE_OUTCOMES):
            frame[outcome + "_raw"] = [1.0, 2.0, 3.0]
            frame[outcome + "_clean"] = [
                1.0 + outcome_index,
                2.0 + outcome_index,
                3.0 + outcome_index,
            ]

        final, parameters = standardize.standardize_clean_phenotypes(frame)

        for outcome in CORE_OUTCOMES:
            self.assertAlmostEqual(float(final[outcome + "_z"].mean()), 0.0)
            self.assertAlmostEqual(float(final[outcome + "_z"].std(ddof=1)), 1.0)
        self.assertTrue(final["A10_excluded_flag"].isna().all())
        self.assertEqual(set(final["A10_exclusion_status"]), {"not_evaluated"})
        self.assertEqual(len(parameters), len(CORE_OUTCOMES))


class OutputLayoutTests(unittest.TestCase):
    def test_default_outputs_are_grouped_under_outputs_directory(self):
        expected_root = os.path.abspath(os.path.join(TEST_DIR, "..", "outputs"))

        self.assertEqual(cgm_utils.DEFAULT_OUTPUT_DIR, expected_root)
        self.assertEqual(
            cgm_utils.DEFAULT_DATA_DIR, os.path.join(expected_root, "data")
        )
        self.assertEqual(
            cgm_utils.DEFAULT_REPORT_DIR, os.path.join(expected_root, "reports")
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
