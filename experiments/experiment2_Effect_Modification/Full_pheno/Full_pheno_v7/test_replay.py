"""Check the reference and held-out isolation using the registered synthetic fixture."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import run_interaction

import numpy as np

from diet_audit import fit_fold
from interaction_fit import fit_split
import protocol as p
from source import Outcome, Snapshot, Study, required_files
from test_full_pheno import fixture


class TestReplayedSplit(unittest.TestCase):
    def test_replay_and_training_roles(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            study = Study(Snapshot(root, required_files(p.MODES["all"])), with_upf=True, expected=None)
            outcomes = {e:study.outcome(e) for e in p.ENDPOINTS}
            for exposure in p.EXPOSURES:
                tested = "D_TotalFat"
                first = fit_split(study, 0, exposure, outcomes)[tested]
                reference = fit_fold(study, exposure, tested, outcomes[tested], 0)
                for label in ("rows", "outcome", "no_diet"):
                    np.testing.assert_allclose(getattr(first, label), getattr(reference, label))
                np.testing.assert_allclose(first.off[:, 0], reference.off[:, 0], atol=1e-10)
                np.testing.assert_allclose(first.on[:, 0], reference.on[:, 0], atol=1e-10)
                perturbation = {e:Outcome(y.y + (study.fold==0)*1e-7, y.unit) for e,y in outcomes.items()}
                second = fit_split(study, 0, exposure, perturbation)[tested]
                np.testing.assert_array_equal(first.rows, second.rows)
                np.testing.assert_array_equal(first.off, second.off)
                np.testing.assert_array_equal(first.on, second.on)
