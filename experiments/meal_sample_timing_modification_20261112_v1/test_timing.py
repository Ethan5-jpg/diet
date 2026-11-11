"""Synthetic checks for local dates and within-person support."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from timing import align_dates, design, supported


class TimingTest(unittest.TestCase):
    def test_align_uses_local_date_and_handoff_cutoff(self):
        visits = pd.DataFrame({
            "input_row": [1, 2],
            "first_observed": pd.to_datetime(["2026-11-01T22:30:00Z", "2026-11-02T06:00:00Z"]),
            "timezone": "Asia/Jerusalem",
        })
        meals = pd.DataFrame({"input_row": [1, 1, 2], "date": ["2026-11-02", "2026-11-03", "2026-11-02"]})
        registry = pd.DataFrame({
            "input_row": [1, 2], "descriptive_sample_calendar_date": ["2026-11-10"] * 2,
            "linked_index_calendar_date": ["2026-11-10", "2026-11-11"],
            "date_conflict_between_linked_sources": False,
        })
        result = align_dates(meals, visits, registry)
        self.assertEqual(result.monitor_day.tolist(), [0, 1, 0])
        self.assertEqual(result.handoff_lag.iloc[:2].tolist(), [8, 7])
        self.assertTrue(pd.isna(result.handoff_lag.iloc[2]))
        self.assertEqual(result.near.tolist(), [False, True, False])

    def test_require_varied_dose_and_repeated_dates_in_both_contexts(self):
        rows = []
        for person in range(3):
            for near in (False, True):
                dates = [1, 1, 2, 2] if person != 1 or not near else [1] * 4
                carb = [20, 40] * 2 if person != 2 or not near else [20, 21] * 2
                rows.extend(dict(input_row=person, near=near, date=day, carbs=g, handoff_lag=7 if near else 10)
                            for day, g in zip(dates, carb))
        self.assertEqual(supported(pd.DataFrame(rows), "common_dose").input_row.unique().tolist(), [0])

    def test_timing_term_is_carb_by_score_by_near(self):
        def center(values, unit):
            frame = pd.DataFrame(values)
            return (frame - frame.groupby(unit).transform("mean")).to_numpy(), None

        meals = pd.DataFrame({
            "carbs": [30., 50., 30., 50.], "monitor_day": [1, 2, 5, 6], "handoff_lag": [11, 10, 7, 6],
            "near": [False, False, True, True], "energy": 300., "protein": 10.,
        })
        _, term = design(meals, np.array([[2.]]), np.zeros(4, dtype=int), SimpleNamespace(center=center))
        np.testing.assert_allclose(term[:, 0], [0., 0., -2., 2.])


if __name__ == "__main__":
    unittest.main()
