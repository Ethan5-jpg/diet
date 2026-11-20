"""Ensure feature options preserve the matched v8 comparator."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import run_expanded

import numpy as np
from expanded_fit import fit_split
from feature_policies import BLOCKS
import protocol as p
from source import Snapshot, Study, required_files
from stability_fit import fit_split as old_fit
from stability_rules import FIXED as V8_METHODS
from test_full_pheno import fixture


class TestV8Control(unittest.TestCase):
    def test_prediction_matches_reference(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            study = Study(Snapshot(root,required_files(p.MODES["all"])),with_upf=True,expected=None)
            n = study.n
            q = np.tile([.01,.03,.06],n)
            days = ((np.linspace(.07,.7,n),np.full(n,3)),
                    (np.repeat(np.arange(n),3),q,np.full(3*n,.8),np.full(n,3),np.full(n,q.mean()),np.full(n,.8)))
            values = {k:np.random.default_rng(j).normal(size=(n,3)) for j,k in enumerate(BLOCKS)}
            outcome = {e:study.outcome(e) for e in p.ENDPOINTS}
            j = V8_METHODS.index("blended_main")
            for exposure in p.EXPOSURES:
                new,_ = fit_split(study,0,exposure,outcome,days,values)
                prev = old_fit(study,exposure,outcome,0,days)
                for e,a in new.items():
                    b = prev[e]
                    np.testing.assert_array_equal(a.rows,b.rows)
                    np.testing.assert_allclose(a.no_diet,b.no_diet,atol=1e-10)
                    np.testing.assert_allclose(a.off[:,0],b.off[:,j],atol=1e-10)
                    np.testing.assert_allclose(a.on[:,0],b.on[:,j],atol=1e-10)
