"""Necessary independent numerical and fold-isolation checks before real fitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np,pandas as pd
from w19_baseline_core import ridge_fit,predict,choose_one_se,whole_folds,pca_fit,rcs
from preprocessing_v1_20261030 import MicrobiomeCLR,PhenotypeOutliers
r=np.random.default_rng(2026110133);n=120;b=np.column_stack([np.ones(n),r.normal(size=n)]);m=r.normal(size=(n,4));y=b@np.array([2.,.5])+m@np.array([.4,-.2,0,.1])+r.normal(size=n)*.1
tr=np.arange(n)<80
co=ridge_fit(b,m,y,tr,10.)
x=np.column_stack([b[tr],m[tr]]);pen=np.diag(np.r_[np.zeros(2),np.full(4,np.sqrt(10.))]);aug=np.vstack([x,pen]);yy=np.r_[y[tr],np.zeros(6)]
expected=np.linalg.lstsq(aug,yy,rcond=None)[0];np.testing.assert_allclose(np.r_[co[0],co[1]],expected,atol=1e-10)
y2=y.copy();y2[~tr]+=1e6;co2=ridge_fit(b,m,y2,tr,10.);np.testing.assert_array_equal(np.r_[co[0],co[1]],np.r_[co2[0],co2[1]])
pred=predict(b,m,co);np.testing.assert_allclose(pred,np.column_stack([b,m])@expected,atol=1e-10)
families=np.repeat(np.arange(60),2);fold=whole_folds(families,5,44);assert all(len(set(fold[families==v]))==1 for v in set(families));assert np.max(np.bincount(fold))-np.min(np.bincount(fold))<=2
raw=r.dirichlet(np.ones(12),n);mm=MicrobiomeCLR().fit(raw[tr],[str(j) for j in range(12)],np.arange(80),site='gut');raw2=raw.copy();raw2[~tr]=r.dirichlet(np.ones(12)*.1,40);mm2=MicrobiomeCLR().fit(raw2[tr],[str(j) for j in range(12)],np.arange(80),site='gut');assert mm.state==mm2.state
q=PhenotypeOutliers().fit(y[tr],np.arange(80),endpoint='synthetic');q2=PhenotypeOutliers().fit(y2[tr],np.arange(80),endpoint='synthetic');assert q.state==q2.state
pc,state=pca_fit(m,tr,3);np.testing.assert_allclose(pc[tr].mean(axis=0),0,atol=1e-12);np.testing.assert_allclose(pc[tr].std(axis=0),1,atol=1e-12)
i,meta=choose_one_se([[1,1,1],[.99,1.01,.99]],[(5,False,100.),(20,True,10.)]);assert i==0
# Nonlinear basis integration must not equal basis at mean in general.
v=np.array([-2.,2.]);assert abs(rcs(v,[-3,0,3]).mean()-rcs(v.mean(),[-3,0,3]))>1e-4
print('W19_NUMERICAL_AND_FOLD_ISOLATION_CHECKS_PASS 9',flush=True)
