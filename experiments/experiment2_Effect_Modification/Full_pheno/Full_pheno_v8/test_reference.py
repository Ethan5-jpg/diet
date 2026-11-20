"""Replay the fixed precision control on synthetic registered input."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import run_stability

import numpy as np
from run_precision import fit_all, METHODS
import protocol as p
from source import Snapshot, Study, required_files
from stability_fit import fit_split
from test_full_pheno import fixture


class ReferenceTest(unittest.TestCase):
    def test_reference_scores_same_fold_predictions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            study = Study(Snapshot(root,required_files(p.MODES["all"])),with_upf=True,expected=None)
            n = study.n
            people = np.repeat(np.arange(n),3)
            known_upf = np.tile(np.array([.01,.03,.06]),n)
            days = ((np.linspace(.07,.7,n),np.full(n,3)),
                    (people,known_upf,np.full(3*n,.80),np.full(n,3),np.full(n,known_upf.mean()),np.full(n,.8)))
            outcomes = {e:study.outcome(e) for e in p.ENDPOINTS}
            old,_ = fit_all(study,outcomes,days)
            index = METHODS.index("host_clip_dayvariance_pc2")
            for exposure in p.EXPOSURES:
                now = fit_split(study,exposure,outcomes,0,days)
                for e in outcomes:
                    a,b = now[e],old[exposure,e][0]
                    np.testing.assert_array_equal(a.rows,b.rows)
                    np.testing.assert_allclose(a.off[:,0],b.off[:,index],atol=1e-10)
                    anchor = (exposure,e)==("fiber_density","D_Glu")
                    ref = b.off[:,index]+(1. if anchor else 1.37)*(b.on[:,index]-b.off[:,index])
                    np.testing.assert_allclose(a.on[:,0],ref,atol=1e-10)
