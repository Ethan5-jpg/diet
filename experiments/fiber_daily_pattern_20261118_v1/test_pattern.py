import sys
import unittest
from pathlib import Path

import numpy as np

import pattern

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"fiber_surface_dates_20261118_v1"))
import surface
sys.path.pop(0)


class PatternTests(unittest.TestCase):
    def test_equal_mean_distinct_patterns(self):
        days = np.array([[12.5,12.5], [7.5,17.5], [17.5,17.5], [5.,10.]])
        moments = np.array([pattern.daily_moments(d).mean(axis=0) for d in days])
        ctx = pattern.pattern(days.mean(axis=1),moments,np.full(len(days),2))
        np.testing.assert_allclose(ctx[:,0], [0,.5,0,0])
        np.testing.assert_allclose(ctx[:,1], days.std(axis=1,ddof=1)/5)

    def test_prediction_and_leakage(self):
        rng = np.random.default_rng(14)
        n = 900
        train = np.arange(n)<600
        held = ~train
        mean, marker, shannon = rng.normal(size=(3,n))
        gap, spread = rng.uniform(0,1,(2,n))
        role = (1.8*mean*marker + 4.6*(spread-.5)*marker + rng.normal(0,.15,n),
                np.column_stack((np.ones(n),mean)), np.column_stack((mean,np.maximum(mean,0))))
        eco,pcs,scores,ctx = rng.normal(size=(n,4)),rng.normal(size=(n,8)),np.column_stack((marker,shannon)),np.column_stack((gap,spread))
        pred = pattern.predictions(surface,role,eco,pcs,scores,ctx,train,held)
        ref = surface.candidates(*surface.design(role,eco,pcs,scores,train),train,held)
        np.testing.assert_allclose(pred[:,0],ref["baseline",None],atol=1e-10)
        np.testing.assert_allclose(pred[:,1],ref["panel",surface.FIXED],atol=1e-10)
        error = ((role[0][held,None]-pred)**2).mean(axis=0)
        self.assertLess(error[5], error[3]/2)
        changed = (role[0]+900*held, *role[1:])
        np.testing.assert_array_equal(pred,pattern.predictions(surface,changed,eco,pcs,scores,ctx,train,held))

    def test_bootstrap_repeats_share_families(self):
        rng = np.random.default_rng(836)
        n = 100
        totals = rng.uniform(1,2,(n,16))
        totals[:,0], totals[:,1] = 1.,4.
        # Opposite deviations in two repeats cancel for every paired household draw.
        totals[:,6] = totals[:,5]+.1
        other = totals.copy()
        other[:,6] = other[:,5]-.1
        data = {0:{"forward":totals}, 1:{"forward":other}}
        v = pattern.summarize(data,412,119)["paired"]["forward"]["daily_hinge"]
        np.testing.assert_allclose([v["delta_r2"],*v["simultaneous"]],0,atol=1e-12)


if __name__ == "__main__":
    unittest.main()
