import unittest
import numpy as np
from w17_interval_upf_prototype import (design_ranges,moment_ranges,moment_summary,
    confidence_bands,null_feasibility,power_range)


class IntervalChecks(unittest.TestCase):
    def test_polynomial_ranges_include_interior_extrema(self):
        lo,hi=power_range(np.array([-2.,1.]),np.array([3.,2.]),2)
        np.testing.assert_array_equal(lo,[0,1]);np.testing.assert_array_equal(hi,[9,4])

    def test_all_admissible_design_moments_are_enclosed(self):
        r=np.random.default_rng(20261030);n=200
        l=r.uniform(0,.8,n);u=l+r.uniform(0,.2,n)
        c=r.normal(size=(n,3));s=r.normal(size=(n,2));y=r.normal(size=n)
        xl,xu,idx=design_ranges(l,u,c,s,center=.4,delta=.1)
        lower,upper,pairs=moment_ranges(xl,xu,y)
        for t in [0,.1,.5,.9,1]:
            d=l+t*(u-l);x,_,_=design_ranges(d,d,c,s,center=.4,delta=.1)
            exact=np.column_stack([x[:,j]*x[:,k] for j,k in pairs]+[x[:,j]*y for j in range(x.shape[1])])
            self.assertTrue(np.all(exact>=lower-1e-10));self.assertTrue(np.all(exact<=upper+1e-10))

    def test_point_intervals_recover_exact_moments(self):
        r=np.random.default_rng(7);d=r.uniform(size=100);c=r.normal(size=(100,1));s=r.normal(size=(100,1));y=r.normal(size=100)
        xl,xu,_=design_ranges(d,d,c,s,center=.5,delta=.1)
        lo,hi,_=moment_ranges(xl,xu,y);np.testing.assert_array_equal(lo,hi)

    def test_exact_normal_equations_accept_true_null(self):
        g=np.array([[1.,.2],[.2,1.]]);b=np.array([2.,0.]);h=g@b
        self.assertFalse(null_feasibility(g,g,h,h,[1])['null_excluded'])

    def test_exact_normal_equations_exclude_false_null(self):
        g=np.array([[1.,.2],[.2,1.]]);b=np.array([2.,1.]);h=g@b
        self.assertTrue(null_feasibility(g,g,h,h,[1])['null_excluded'])

    def test_conservative_outer_relaxation_never_excludes_feasible_beta(self):
        r=np.random.default_rng(30)
        for _ in range(20):
            a=r.normal(size=(4,4));g=a.T@a;b=r.normal(size=4);b[-1]=0;h=g@b
            self.assertFalse(null_feasibility(g-.1,g+.1,h-.1,h+.1,[3])['null_excluded'])

    def test_budget_exhaustion_is_not_rejection(self):
        g=np.eye(2);h=np.array([1.,1.]);r=null_feasibility(g,g,h,h,[1],max_nodes=0)
        self.assertFalse(r['null_excluded']);self.assertEqual(r['status'],'inconclusive_computation')

    def test_cluster_moment_bands_and_zero_width(self):
        r=np.random.default_rng(13);x=r.normal(size=(180,3));y=r.normal(size=180)
        lo,hi,pairs=moment_ranges(x,x,y);s=moment_summary(lo,hi,np.repeat(np.arange(60),3))
        gl,gu,hl,hu=confidence_bands(s,pairs,3,.05)
        g=x.T@x/180;h=x.T@y/180
        self.assertTrue(np.all(gl<=g));self.assertTrue(np.all(gu>=g))
        self.assertTrue(np.all(hl<=h));self.assertTrue(np.all(hu>=h))


if __name__=='__main__':unittest.main(verbosity=2)
