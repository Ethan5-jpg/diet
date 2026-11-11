import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from inputs import (DAY, ENERGY_ANCHOR, FIBER, KEY, MACROS, assemble, integer_ids,
                    projected_day)
from models import (background, center, design, fit_background, fit_interaction,
                    holm, household_test, train_standardize)
from run import combine, compare, direct_tests


def generated():
    rng = np.random.default_rng(14)
    n, days = 180, 6
    person = np.repeat(np.arange(n), days)
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(np.tile([0, 1, 3, 6, 10, 12], n), unit="D")
    marker = rng.normal(size=(n, 1))
    diet = 50 + rng.normal(0, 8, len(person))
    frame = pd.DataFrame(dict(
        input_row=person, person_index=person, split_unit_id=person // 2, date=dates,
        carb=diet, fiber=rng.uniform(7, 20, len(person)), protein=rng.uniform(12, 20, len(person)),
        energy_kcal=rng.uniform(1300, 2400, len(person)),
        age_calendar_years=np.repeat(rng.normal(45, 8, n), days),
        sex_code=np.repeat(rng.integers(0, 2, n), days),
        Y_value=100 + np.repeat(rng.normal(0, 8, n), days) + 0.1 * diet
        + 2 * ((diet - 50) / 10) * marker[person, 0] + rng.normal(0, 0.5, len(person)),
    ))
    return frame, marker


class DailyTests(unittest.TestCase):
    def test_weight_and_center(self):
        people = np.array([0, 0, 0, 1, 1])
        x, w = center(np.array([[4], [5], [6], [20], [26]]), people)
        np.testing.assert_allclose(x[:, 0], [-1, 0, 1, -3, 3])
        np.testing.assert_allclose(np.bincount(people, w), [1, 1])

    def test_signed_signal_and_matched_null(self):
        data, marker = generated()
        y, x, w = background(data)
        train = data.split_unit_id.to_numpy() % 5 != 0
        z, _ = design(data, ("carb",), marker[data.person_index], curve=False)
        fitted = fit_background(x[train], y[train], w[train])
        beta, null, alternative = fit_interaction(x, z, y, w, train, ~train, fitted, 0.03)
        self.assertGreater(beta[0], 1.7)
        self.assertLess(alternative, null / 4)
        all_beta, cov, p = household_test(z, fit_background(x, y, w), data.split_unit_id, len(marker))
        self.assertAlmostEqual(all_beta[0], 2, delta=0.1)
        self.assertGreater(cov[0, 0], 0)
        self.assertLess(p, 0.001)
        data["Y_value"] -= 2 * ((data.carb - 50) / 10) * marker[data.person_index, 0]
        y, x, w = background(data)
        null_beta, _, _ = household_test(z, fit_background(x, y, w), data.split_unit_id, len(marker))
        self.assertAlmostEqual(null_beta[0], 0, delta=0.1)

    def test_direct_and_oof_report(self):
        data, marker = generated()
        cfg = dict(routes=[["carb", "gut_shannon", False], ["joint", "gut_shannon", True]],
                   penalties=[0.03, 0.3])
        designs = background(data)
        score = dict(gut_shannon=marker)
        rows = direct_tests(data, designs, score, cfg)
        self.assertGreater(rows[0]["contrasts"]["carb"]["q75_minus_q25"], 2)
        self.assertGreaterEqual(rows[0]["holm_p"], rows[0]["p"])
        cuts = (np.arange(len(marker)) // 2) % 2 == 0
        parts = [dict(repeat=0, **compare(data, designs, score, ~valid, valid, cfg))
                 for valid in (cuts, ~cuts)]
        out = combine(parts)["carb_gut_shannon_linear:0.03"]["r0"]
        self.assertGreater(out["delta_r2"], 0.7)
        self.assertGreater(out["contrast_sd"]["carb"], 1.5)

    def test_training_standardization(self):
        x = np.array([[1.0], [3.0], [100.0]])
        np.testing.assert_allclose(train_standardize(x, np.array([True, True, False]))[:, 0], [-1, 1, 98])

    def test_holm(self):
        np.testing.assert_allclose(holm([0.02, 0.01, 0.9]), [0.04, 0.03, 0.9])

    def test_arrow_filter_scope_and_lossless_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.parquet"
            table = pa.table(dict(participant_id=[3, 3, 4, 2 ** 62], cohort=["a", "a", "a", "a"],
                                  research_stage=["v1", "v2", "v1", "v1"],
                                  date=["2026-01-02"] * 4, Y_value=[1, 2, 3, 4]))
            pq.write_table(table, path)
            people = pd.DataFrame([(3, "a", "v1"), (2 ** 62, "a", "v1")], columns=KEY)
            got = projected_day(path, ["Y_value"], people)
            self.assertEqual(set(got.Y_value), {1, 4})
            self.assertIn(2 ** 62, got.participant_id.tolist())
            with self.assertRaisesRegex(ValueError, "losslessly"):
                integer_ids(pa.table(dict(participant_id=[float(2 ** 53)])))

    def test_aligned_macro_shares_and_qc(self):
        days = pd.date_range("2026-01-01", periods=4)
        keys = pd.DataFrame([(1, "c", "v", date) for date in days], columns=DAY)
        people = keys[KEY].iloc[:1].assign(input_row=0, final_role="discovery", split_unit_id=1,
                                          age_calendar_years=50, sex_code=0)
        food = keys.assign(energy_kcal=2000, energy_eligible=True, fiber_eligible=True, **{FIBER: 10})
        nutrients = keys.assign(**{ENERGY_ANCHOR: 2000, MACROS[0]: 250, MACROS[1]: 75, MACROS[2]: 75})
        cgm = keys.assign(Y_value=100, base_eligible=[True, True, True, False])
        data = assemble(people, food, cgm, nutrients, 3)
        self.assertEqual(len(data), 3)
        np.testing.assert_allclose(data.carb, 100 * 1000 / 1975)
        np.testing.assert_allclose(data.protein, 100 * 300 / 1975)
        with self.assertRaisesRegex(ValueError, "anchors"):
            assemble(people, food, cgm, nutrients.assign(**{ENERGY_ANCHOR: 2200}), 3)


if __name__ == "__main__":
    unittest.main()
