import tempfile
import unittest
from pathlib import Path
import numpy as np
from w16_estimators import (TrainPCA,CovariateEncoder,block_ridge,ScoreCompression,
    ols_sandwich,wald_block,contrast_interval,stacked_covariance,daily_fixed_effects,holm_fixed_family)


class InterfaceChecks(unittest.TestCase):
    def setUp(self): self.rng=np.random.default_rng(20261030)

    def test_pca_save_order_and_single_row_transform(self):
        x=self.rng.normal(size=(100,5)); names=list('abcde')
        obj=TrainPCA().fit(x,names,range(100),n_components=3)
        np.testing.assert_allclose(obj.transform(x,names).std(axis=0),1)
        np.testing.assert_allclose(obj.transform(x[:1],names)[0],obj.transform(x,names)[0])
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'pca.json';obj.save(f)
            np.testing.assert_allclose(TrainPCA.load(f).transform(x,names),obj.transform(x,names))
        with self.assertRaises(ValueError):obj.transform(x,names[::-1])

    def test_covariate_training_imputation_and_unknown(self):
        data={'age':[10.,20.,np.nan,40.], 'sex':['F','M','F','M'], 'smoke':['no',None,'yes','no']}
        obj=CovariateEncoder().fit(data,range(4),numeric=['age'],categorical=['sex','smoke'])
        self.assertEqual(obj.state['medians']['age'],20)
        self.assertTrue(np.isfinite(obj.transform(data)).all())
        new={key:[val[0]] for key,val in data.items()}
        np.testing.assert_allclose(obj.transform(new)[0],obj.transform(data)[0])
        new['sex']=['unseen']
        with self.assertRaises(ValueError):obj.transform(new)

    def test_ridge_matches_augmented_least_squares(self):
        n=200;u=self.rng.normal(size=n);o=self.rng.normal(size=(n,3));g=self.rng.normal(size=(n,2))
        base=np.column_stack([np.ones(n),u]);y=3+u+2*u*o[:,0]+self.rng.normal(size=n)
        result=block_ridge(base,o,g,u,y,main_penalty=2,interaction_penalty=3)
        x=np.column_stack([base,o,g,u[:,None]*o,u[:,None]*g])
        penalty=np.r_[np.zeros(2),np.full(5,2),np.full(5,3)]
        a=np.vstack([x,np.diag(np.sqrt(penalty))]);b=np.r_[y,np.zeros(len(penalty))]
        np.testing.assert_allclose(result['coef'],np.linalg.lstsq(a,b,rcond=None)[0],atol=1e-10)

    def test_compression_and_no_direction(self):
        z=self.rng.normal(size=(100,4));b=np.array([1.,-2.,0.,0.]);names=list('abcd')
        obj=ScoreCompression().fit(z,b,names,range(100))
        self.assertEqual(obj.state['q'],1)
        self.assertAlmostEqual(float(obj.transform(z,names).std()),1)
        empty=ScoreCompression().fit(z,np.zeros(4),names,range(100))
        self.assertEqual(empty.transform(z,names).shape,(100,0))
        with self.assertRaises(ValueError):obj.fit(z,b,names,range(100),role='confirmation')

    def test_ols_known_contrast_and_covariance_formula(self):
        x=np.column_stack([np.ones(300),self.rng.normal(size=(300,3))]);b=np.array([2.,1.,-1.,.5])
        y=x@b+self.rng.normal(scale=.1,size=300)
        f=ols_sandwich(x,y)
        np.testing.assert_allclose(f['beta'],b,atol=.03)
        a=np.linalg.inv(x.T@x);h=np.sum((x@a)*x,axis=1);r=y-x@f['beta']
        manual=a@(x.T@np.diag((r/(1-h))**2)@x)@a
        np.testing.assert_allclose(f['covariance'],manual,atol=1e-12)
        ci=contrast_interval(f,[0,1,0,0]);self.assertTrue(ci['low']<ci['estimate']<ci['high'])
        self.assertLess(wald_block(f,[1,2])['p'],.001)
        self.assertEqual(wald_block(f,[])['p'],1)

    def test_rank_failure_not_silently_inverted(self):
        x=np.column_stack([np.ones(100),np.ones(100)])
        with self.assertRaises(ValueError):ols_sandwich(x,self.rng.normal(size=100))

    def test_cluster_sandwich_and_stacked_covariance(self):
        n=240;families=np.repeat(np.arange(80),3)
        x=np.column_stack([np.ones(n),self.rng.normal(size=n)])
        y=x@np.array([2.,1.])+np.repeat(self.rng.normal(size=80),3)+self.rng.normal(size=n)
        f=ols_sandwich(x,y,clusters=families);f2=ols_sandwich(x,2*y,clusters=families)
        v=stacked_covariance([f,f2]);self.assertEqual(f['df'],79)
        np.testing.assert_allclose(v[:2,:2],f['covariance'],atol=1e-12)
        np.testing.assert_allclose(v[:2,2:],2*f['covariance'],atol=1e-12)

    def test_daily_fe_matches_explicit_weighted_dummies(self):
        counts=np.tile([3,5,7],20);ids=np.repeat(np.arange(60),counts);n=len(ids)
        x=self.rng.normal(size=(n,2));y=x@np.array([1.,-.5])+np.repeat(self.rng.normal(size=60),counts)+self.rng.normal(size=n)
        f=daily_fixed_effects(x,y,ids,ids//2)
        dummies=np.eye(60)[ids];w=1/counts[ids]
        full=np.column_stack([dummies,x]);a=full*np.sqrt(w)[:,None]
        beta=np.linalg.lstsq(a,y*np.sqrt(w),rcond=None)[0][-2:]
        np.testing.assert_allclose(f['beta'],beta,atol=1e-10)
        for pid in range(60):self.assertAlmostEqual(float(w[ids==pid].sum()),1)

    def test_holm_fixed_slots_including_inactive(self):
        p=[.01,.04,.001,1.0]
        np.testing.assert_allclose(holm_fixed_family(p),[.03,.08,.004,1.])
        with self.assertRaises(ValueError):holm_fixed_family([.01,np.nan])


if __name__=='__main__':unittest.main(verbosity=2)
