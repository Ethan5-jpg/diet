"""Check that a PCA reference keeps unscreened training rows but excludes held households."""
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from adapters.person_inputs import PersonSource
from v7_source import wrap_source


class SourceRoleTests(unittest.TestCase):
    def test_outer_and_inner_pca_reference_precedes_endpoint_eligibility(self):
        # Half the individuals lack an endpoint. Every household has two rows.
        n = 60
        groups = np.arange(n) // 2
        outcome = np.where(np.arange(n) % 2, np.nan, np.arange(n).astype(float))
        outer = np.stack([groups % 5, (groups + 1) % 5])
        index = pd.DataFrame({'split_unit_id': groups.astype(str)})

        def folds(labels, count, seed):
            return np.unique(labels, return_inverse=True)[1] % count

        def build(train, valid, contrast):
            eligible = np.isfinite(outcome)
            rows = np.flatnonzero(valid & eligible)
            return SimpleNamespace(train_rows=np.flatnonzero(train & eligible), rows=rows,
                                   yv=outcome[rows], preprocessing={'CLR': {
                                       s: {'training_participants': int(train.sum())}
                                       for s in ('oral', 'gut')}})

        data = SimpleNamespace(index=index, api=SimpleNamespace(folds=folds), raw={}, taxa={},
                               y=outcome, reference={'y': np.stack([outcome, outcome]), 'outer': outer},
                               split=build)
        source = wrap_source(PersonSource(data, None, {}, (0., 0.)))
        for repeat in range(2):
            for fold in range(5):
                inners, outer_split = source.roles(repeat, fold)
                outer_train = outer[repeat] != fold
                self.assertFalse(outer_split.pca_training_rows.flags.writeable)
                np.testing.assert_array_equal(outer_split.pca_training_rows, np.flatnonzero(outer_train))
                for role in (*inners, outer_split):
                    expected = np.flatnonzero(np.isin(groups, groups[role.train_rows]))
                    np.testing.assert_array_equal(role.pca_training_rows, expected)
                    self.assertEqual(len(expected), 2 * len(role.train_rows))
                    self.assertFalse(np.isin(groups[expected], groups[role.rows]).any())
                    self.assertTrue(outer_train[expected].all())
