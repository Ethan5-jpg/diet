import unittest

import numpy as np
import pandas as pd

from clinical import SPECS, select_antecedent
from evaluate import summarize, terms


class HostTests(unittest.TestCase):
    def test_antecedent_measurements_match_cohort_and_exclude_latest_conflict(self):
        spec = SPECS["hba1c"]
        anchors = pd.DataFrame(dict(participant_id=["a", "b", "c"], cohort="discovery",
                                    input_row=[1, 2, 3], first_day=pd.Timestamp("2026-01-15")))
        values = pd.DataFrame([
            ("a", "discovery", "2026-01-10", 5.6),
            ("a", "discovery", "2026-01-10", 5.6),
            ("a", "discovery", "2026-01-15", 10.1),
            ("a", "discovery", "2026-01-17", 10.9),
            ("a", "other", "2026-01-13", 12.),
            ("b", "discovery", "2026-01-10", 6.),
            ("b", "discovery", "2026-01-10", 6.1),
            ("b", "discovery", "2026-01-01", 5.4),
            ("c", "discovery", "2024-12-01", 5.1),
        ], columns=["participant_id", "cohort", "collection_date", spec["field"]])
        selected, audit = select_antecedent(anchors, values, spec)
        self.assertEqual(selected.index.tolist(), [1])
        self.assertEqual(selected.loc[1].tolist(), [5.6, 5.])
        self.assertEqual(audit["conflicting"], 1)

    def test_three_way_term_uses_registered_first_host_factor(self):
        class Model:
            @staticmethod
            def center(x, groups):
                return x, None

        dose = np.array([1., -2.])
        person = np.array([0, 1])
        host = np.array([[2., 20.], [3., 30.]])
        microbial = np.array([[5., 1.], [7., 1.]])
        h, m, triple = terms(dose, person, person, host, microbial, Model())
        np.testing.assert_equal(h[:, 0], [2., -6.])
        np.testing.assert_equal(m[:, 0], [5., -14.])
        np.testing.assert_equal(triple[:, 0], [10., -42.])

    def test_increment_uses_host_baseline_and_three_way_uses_both_lower_orders(self):
        values = {k: np.array(v) for k, v in dict(original=[20., 30.], host=[18., 28.],
                 microbe_alone=[19., 29.], host_microbe=[17., 27.],
                 combined=[15., 25.], tss=[100., 150.]).items()}
        rows = summarize(values, np.array([[0, 1], [1, 1], [0, 0]]))
        self.assertEqual(rows["host"]["delta_r2"], 4 / 250)
        self.assertEqual(rows["microbe_beyond_host"]["delta_r2"], 2 / 250)
        self.assertEqual(rows["host_microbe_interaction"]["delta_r2"], 4 / 250)


if __name__ == "__main__":
    unittest.main()
