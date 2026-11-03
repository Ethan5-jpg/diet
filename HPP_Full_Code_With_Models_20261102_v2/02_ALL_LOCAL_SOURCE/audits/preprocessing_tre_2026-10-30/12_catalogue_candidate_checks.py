from pathlib import Path
import json
import numpy as np
from upf_catalogue_calibration import point_design,beta_integrated_design,catalogue_draw_design,fit_catalogue_calibration
from w16_estimators import ols_sandwich
Q=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/data_qc/preprocessing_20261030_v1')
r=np.random.default_rng(20261030012);n=240
c=r.normal(size=(n,1));s=r.normal(size=(n,2));lo=r.uniform(.05,.55,n);hi=lo+.15
# Exact-width limit is ordinary regression, including the nonlinear basis.
xp,idx=point_design(lo,c,s)
xe,idx2,_=beta_integrated_design(lo,lo,c,s,a=1,b=3)
np.testing.assert_allclose(xe,xp,atol=1e-12);assert idx==idx2
# Independent dense uniform numerical integration of the entire design.
xe,_,_=beta_integrated_design(lo,hi,c,s,a=1,b=1,quadrature=64)
xd=np.mean([point_design(lo+(hi-lo)*f,c,s)[0] for f in (np.arange(4000)+.5)/4000],axis=0)
np.testing.assert_allclose(xe,xd,atol=2e-8)
# Exact conditional mean for affine columns is not a simulated true exposure.
xb,_,_=beta_integrated_design(lo,hi,c,s,a=1,b=3)
mean=lo+(hi-lo)/4;xm,_=point_design(mean,c,s)
rcscol=5
for j in range(xm.shape[1]):
    if j!=rcscol:np.testing.assert_allclose(xb[:,j],xm[:,j],atol=1e-12)
assert np.max(np.abs(xb[:,rcscol]-xm[:,rcscol]))>1e-3
# A single uncertain FoodID always has one common binary status per draw.
A=np.full((n,1),.15);mu,draw,idx,meta=catalogue_draw_design(lo,A,[.3],c,s,draws=128,seed=12)
dcol=4
np.testing.assert_allclose(mu[:,dcol],(lo+.15*.3-.5)/.1)
deviation=draw[:,:,dcol]-(lo[None,:]-.5)/.1
assert np.all((np.abs(deviation)<1e-10)|(np.abs(deviation-1.5)<1e-10))
assert np.max(np.ptp(deviation,axis=1))<1e-10
# Covariance addition is PSD and cannot silently shrink a contrast variance.
y=.5*lo+.3*c[:,0]+r.normal(size=n);families=np.arange(n)//2
f=fit_catalogue_calibration(mu,draw,y,families)
assert np.linalg.eigvalsh(f['catalogue_covariance']).min()>-1e-10
assert np.linalg.eigvalsh(f['covariance']-f['family_covariance']).min()>-1e-10
# Deterministic labels reduce exactly to the ordinary sandwich interface.
mu,draw,idx,_=catalogue_draw_design(lo,A,[1.],c,s,draws=32,seed=1)
f=fit_catalogue_calibration(mu,draw,y,families);ordinary=ols_sandwich(mu,y,clusters=families)
np.testing.assert_allclose(f['beta'],ordinary['beta'],atol=1e-12)
np.testing.assert_allclose(f['covariance'],ordinary['covariance'],atol=1e-12)
report={'status':'catalogue_candidate_numeric_checks_passed','checks':7,
        'shared_food_labels_verified':True,'nonlinear_basis_integrated_not_mean_imputed':True,
        'statistical_calibration':'pending','real_scientific_Y_reads':0,'real_activation':False}
(Q/'catalogue_candidate_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('CATALOGUE_CANDIDATE_CHECKS',json.dumps(report),flush=True)
