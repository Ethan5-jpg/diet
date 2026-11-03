from pathlib import Path
import json
import numpy as np
from upf_catalogue_calibration import catalogue_draw_design as old_draw
from upf_catalogue_calibration_v2 import catalogue_draw_design,grouped_probability_draws,fit_catalogue_calibration
Q=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/data_qc/preprocessing_20261030_v1')
r=np.random.default_rng(20261030016);n=160;J=20
c=r.normal(size=(n,1));s=r.normal(size=(n,2));low=r.uniform(.05,.5,n)
A=r.dirichlet(np.ones(J),n)*.15;q=r.uniform(.1,.9,J)
x,d,idx,meta=catalogue_draw_design(low,A,q,c,s,draws=64,seed=16)
xo,do,io,mo=old_draw(low,A,q,c,s,draws=64,seed=16)
np.testing.assert_allclose(x,xo,atol=1e-12);np.testing.assert_allclose(d,do,atol=1e-12)
x2,d2,_,m2=catalogue_draw_design(low,A,np.tile(q,(64,1)),c,s,draws=64,seed=16)
np.testing.assert_allclose(x2,x,atol=1e-12);np.testing.assert_allclose(d2,d,atol=1e-12)
assert m2['q_estimation_uncertainty_included']
g=np.array(['g'+str(i) for i in range(30)]);cat=np.array(['c'+str(i%3) for i in range(30)])
y=(np.arange(30)%4==0).astype(float);w=np.ones(30);u=np.array(['c0','c1','c2','unseen'])
p,info=grouped_probability_draws(g,cat,y,w,u,draws=128,seed=16)
aliased,info2=grouped_probability_draws(np.repeat(g,2),np.repeat(cat,2),np.repeat(y,2),np.full(60,.5),u,draws=128,seed=16)
np.testing.assert_allclose(p,aliased,atol=1e-12)
assert info['distinct_reference_groups']==30 and info2['distinct_reference_groups']==30
assert np.all((p>0)&(p<1)) and np.all(p.std(axis=0)>0)
shifted,_=grouped_probability_draws(g,cat,y,w,u,draws=128,seed=16,logodds_shift=np.log(4))
np.testing.assert_allclose(shifted/(1-shifted),4*p/(1-p),atol=1e-10)
# A common probability draw induces common catalogue-level uncertainty.
qp=np.repeat(p[:,[0]],J,axis=1)
x3,d3,idx,m3=catalogue_draw_design(low,A,qp,c,s,draws=128,seed=17)
assert m3['q_estimation_uncertainty_included']
assert np.isfinite(x3).all() and d3.shape==(128,n,x3.shape[1])
f=fit_catalogue_calibration(x3,d3,r.normal(size=n),np.arange(n)//2)
assert np.linalg.eigvalsh(f['covariance']-f['family_covariance']).min()>-1e-10
report={'status':'estimated_probability_numeric_checks_passed','checks':6,
        'source_alias_duplication_does_not_increase_information':True,
        'shared_probability_draws_and_odds_sensitivity_verified':True,
        'statistical_calibration':'pending','new_real_Y_reads':0,'production_ready':False}
(Q/'estimated_probability_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('ESTIMATED_PROBABILITY_CHECKS',json.dumps(report),flush=True)
