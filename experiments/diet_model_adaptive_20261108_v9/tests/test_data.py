"""Food-only projections, composite identities and date-aware aggregation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from daily import COLUMNS, NAMES, aggregate_food, read_daily


class DataChecks(unittest.TestCase):
    def frame(self):
        index = pd.DataFrame(
            dict(participant_id=[3, 1, 2], cohort="10k", research_stage="visit", fiber_density=[10, 20, 15])
        )
        data = []
        for person, center in ((1, 20), (2, 15), (3, 10)):
            for day, delta in enumerate((-2, 0, 2, 4)):
                data.append(
                    dict(
                        participant_id=person,
                        cohort="10k",
                        research_stage="visit",
                        date=pd.Timestamp("2026-01-02") + pd.Timedelta(days=day),
                        energy_kcal=2000,
                        fiber_g=2 * (center + delta),
                        energy_eligible=True,
                        fiber_eligible=True,
                        Y_value=900 + person,
                    )
                )
        return pd.DataFrame(data), index

    def test_food_features_are_key_aligned_and_outcome_independent(self):
        frame, index = self.frame()
        a = aggregate_food(frame, index)
        frame = frame.sample(frac=1, random_state=4)
        frame.Y_value *= -100
        b = aggregate_food(frame, index)
        np.testing.assert_allclose(a["values"], b["values"], atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(a["values"][:, NAMES.index("fiber_mean_square")], [126, 446, 261])
        self.assertEqual(a["audit"]["daily_outcome_columns_read"], [])

    def test_reader_projects_only_permitted_food_columns(self):
        frame, index = self.frame()
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "food.parquet"
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), p)
            original = pq.ParquetFile
            seen = []

            class Reader:
                def __init__(self, *args, **kwargs):
                    self.reader = original(*args, **kwargs)
                    self.schema_arrow = self.reader.schema_arrow

                def read(self, **kwargs):
                    seen.extend(kwargs["columns"])
                    return self.reader.read(**kwargs)

            with patch.object(pq, "ParquetFile", Reader):
                result = read_daily(p, index)
            self.assertEqual(seen, list(COLUMNS))
            self.assertNotIn("Y_value", seen)
            self.assertEqual(result["values"].shape, (3, len(NAMES)))

    def test_duplicate_dates_unknown_keys_and_bad_eligible_values_are_rejected(self):
        frame, index = self.frame()
        for changed in (
            pd.concat([frame, frame.iloc[:1]], ignore_index=True),
            frame.assign(cohort="outside"),
            frame.assign(energy_kcal=-1),
        ):
            with self.assertRaises(ValueError):
                aggregate_food(changed, index)

    def test_short_records_remain_missing_without_removing_people(self):
        frame, index = self.frame()
        frame = frame.drop(frame[(frame.participant_id == 2)].index[2:])
        result = aggregate_food(frame, index)
        self.assertEqual(result["values"].shape[0], 3)
        self.assertTrue(np.isnan(result["values"][2]).all())
