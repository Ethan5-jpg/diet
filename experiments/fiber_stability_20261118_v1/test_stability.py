import unittest
import numpy as np
import stability as st


class StabilityTests(unittest.TestCase):
    def test_randomization_respects_households(self):
        fam = np.repeat(np.arange(120),1+np.arange(120)%4)
        a = st.partition(fam,27)
        self.assertEqual(set(a),set(range(5)))
        for group in np.unique(fam):
            self.assertEqual(len(set(a[fam==group])),1)
        np.testing.assert_array_equal(a,st.partition(fam,27))
        self.assertFalse(np.array_equal(a,st.partition(fam,28)))

    def test_inner_gate(self):
        loss = np.ones((30,4))
        self.assertEqual(st.choose(loss),0)
        loss[0,0] += 15
        self.assertEqual(st.choose(loss),0)
        loss[:,1:] -= .2
        self.assertEqual(st.choose(loss),1)
        loss[:,3] -= .1
        self.assertEqual(st.choose(loss),3)

    def test_exact_attribution_and_summary(self):
        rng = np.random.default_rng(50)
        n = 500
        fam = np.repeat(np.arange(n//2),2)
        y = rng.normal(size=n)
        fixed = y + rng.normal(0,.5,n)
        s = .1*(y-fixed)
        t = .3*(y-fixed)
        pred = np.column_stack((np.zeros(n),fixed,fixed+s,fixed+t,fixed+s+t))
        a = st.checkpoint(y,0,pred,fam,n//2,0)
        weights = np.vstack((np.ones(n//2),rng.integers(1,3,(99,n//2))))
        res = st.summarize([a]*8,weights,True)
        exp = np.mean((y-fixed)**2-(y-pred[:,4])**2)/np.mean(y**2)
        self.assertAlmostEqual(res["results"]["both"]["delta"],exp)
        self.assertAlmostEqual(sum(res["components"].values()),exp)
        self.assertTrue(res["results"]["both"]["stable"])
        self.assertFalse(res["results"]["gate"]["stable"])


if __name__=="__main__":
    unittest.main()
