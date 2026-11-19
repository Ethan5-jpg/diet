"""Check matched predictions and scoring on generated inputs."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import run_diet_audit

import numpy as np

from breadth import evaluate_split
from diet_audit import RECIPES, fit_fold, household_moments, simultaneous
import protocol as p
from source import Snapshot, Study, required_files
from test_full_pheno import fixture


class DietAuditTest(unittest.TestCase):
    def test_archive_prediction_and_household_scores(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            study = Study(Snapshot(root, required_files(p.MODES["all"])), with_upf=True, expected=None)
            for exposure, endpoint in (("fiber_density", "D_Glu"), ("UPF", "D_TotalFat")):
                outcome = study.outcome(endpoint)
                fits = [fit_fold(study, exposure, endpoint, outcome, fold) for fold in p.FOLDS]
                for fold, new in enumerate(fits):
                    old = evaluate_split(study, exposure, {endpoint: outcome}, fold)[endpoint]
                    np.testing.assert_array_equal(new.rows, old.rows)
                    ref = old.preds["v1_off"]
                    np.testing.assert_allclose(new.off[:, 0], ref, atol=1e-10)
                    np.testing.assert_allclose(new.on[:, 0],
                                               (ref + old.preds["gut_pc5/linear/ridge_1"]) / 2, atol=1e-10)
                _, groups = np.unique(study.family, return_inverse=True)
                moments = household_moments(fits, groups, len(np.unique(groups)))
                r2, pp, widths = simultaneous(moments[None, ...], 99)
                actual = np.concatenate([fit.outcome for fit in fits])
                base = np.concatenate([fit.no_diet for fit in fits])
                off = np.concatenate([fit.off for fit in fits])
                on = np.concatenate([fit.on for fit in fits])
                sst = np.sum((actual - actual.mean())**2)
                hand = 1. - np.sum((actual[:, None] - np.column_stack((base, off, on)))**2, axis=0) / sst
                np.testing.assert_allclose(r2[0, :, 0], hand[0])
                np.testing.assert_allclose(r2[0, :, 1], hand[1:1+len(RECIPES)])
                np.testing.assert_allclose(r2[0, :, 2], hand[1+len(RECIPES):])
                np.testing.assert_allclose(pp[0, :, 2], pp[0, :, 0] + pp[0, :, 1])
                self.assertTrue(np.isfinite(widths).all())


if __name__ == "__main__":
    unittest.main()
