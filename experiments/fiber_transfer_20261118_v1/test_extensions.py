from pathlib import Path
import sys
import unittest

import numpy as np
import extensions as ext

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/"fiber_surface_dates_20261118_v1"),str(ROOT/"fiber_daily_pattern_20261118_v1")]
import surface
import pattern
del sys.path[:2]


def role(rng, n, marker):
    d = rng.normal(size=n)
    dose = np.column_stack((d,np.maximum(d,0)))
    x = np.column_stack((np.ones(n),d,rng.normal(size=n)))
    return (2*d*marker + x[:,2] + rng.normal(0,.3,n),x,dose)


class ExtensionsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1390)
        n = 420
        self.held = np.arange(n)%3==0
        self.scores = rng.normal(size=(n,2))
        self.target, self.aux = (role(rng,n,self.scores[:,0]) for _ in range(2))
        self.eco,self.pcs,self.ctx = rng.normal(size=(n,4)),rng.normal(size=(n,5)),rng.uniform(size=(n,2))

    def fit(self,t,a):
        return ext.predictions(surface,pattern,t,a,self.eco,self.pcs,self.scores,self.ctx,~self.held,self.held)

    def test_reference_and_no_auxiliary_holdout_leak(self):
        pred, ref = self.fit(self.target,self.aux)
        np.testing.assert_allclose(pred[:,:2],ref[:,:2])
        np.testing.assert_allclose(pred[:,2]-pred[:,1],.25*(ref[:,5]-ref[:,3]))
        # Alter every opposite-period field of test families, and all held target outcomes.
        aux = tuple(a.copy() for a in self.aux)
        for a in aux:
            a[self.held] += 4000
        target = (self.target[0]+self.held*8000,*self.target[1:])
        other,_ = self.fit(target,aux)
        np.testing.assert_array_equal(pred,other)

    def test_identical_directions(self):
        a,_ = self.fit(self.target,self.target)
        np.testing.assert_allclose(a[:,3],a[:,1])
        np.testing.assert_allclose(a[:,5],a[:,2])
        np.testing.assert_allclose(a[:,6],a[:,4]+a[:,2]-a[:,1])

    def test_robust_correction_limits_response_outliers(self):
        rng = np.random.default_rng(990)
        z = rng.normal(size=(500,4))
        true = np.array([1.,-1.5,.1,.2])
        y = z@true + rng.normal(0,.1,len(z))
        y[z[:,0]>1.5] += 25
        penalty = .01*len(y)*np.eye(4)
        start = np.linalg.solve(z.T@z+penalty,z.T@y)
        robust = ext.huber_ridge(z,y,penalty,start)
        self.assertLess(np.linalg.norm(robust-true),.3*np.linalg.norm(start-true))


if __name__=="__main__":
    unittest.main()
