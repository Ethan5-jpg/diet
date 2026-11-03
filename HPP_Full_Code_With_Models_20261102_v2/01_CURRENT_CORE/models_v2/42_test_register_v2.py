import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np,json,hashlib
from v2_ridge_core import BlockDesign,direction,score,select_minimum
rng=np.random.default_rng(20261101042);n=230;p=12
z=rng.normal(size=(n,p));u=rng.normal(size=n);base=np.column_stack([np.ones(n),u,rng.normal(size=n)])
y=base@np.array([1.,.4,-.2])+u*(z@rng.normal(size=p))*.2+rng.normal(size=n)*.2
tr=np.arange(n)<170;model=BlockDesign(base,z,u,y,tr);fit=model.fit(.1,1.)
# Compare the entire fitted predictor against the direct block normal equations.
f=np.column_stack([z,u[:,None]*z])/model.scales;xx=np.column_stack([base,f]);pen=np.r_[np.zeros(base.shape[1]),np.full(p,170*.1),np.full(p,170.)]
coef=np.linalg.solve(xx[tr].T@xx[tr]+np.diag(pen),xx[tr].T@y[tr]);np.testing.assert_allclose(xx@coef,fit['prediction'],rtol=1e-10,atol=1e-10)
yb=y.copy();yb[~tr]+=1000;other=BlockDesign(base,z,u,yb,tr).fit(.1,1.)
np.testing.assert_array_equal(other['prediction'],fit['prediction'])
null=model.fit(.1,None);xn=np.column_stack([base,z/model.scales[:p]]);pn=np.r_[np.zeros(3),np.full(p,17.)]
cn=np.linalg.solve(xn[tr].T@xn[tr]+np.diag(pn),xn[tr].T@y[tr]);np.testing.assert_allclose(xn@cn,null['prediction'],rtol=1e-10,atol=1e-10)
d=direction(z,fit['interaction_coef'],tr,np.ones(p));s=score(z,d)
np.testing.assert_allclose(s[tr].mean(),0,atol=1e-12);np.testing.assert_allclose(s[tr].std(),1,atol=1e-12)
np.testing.assert_allclose((s+d['center'])*d['fixed_prediction_amplitude'],z@fit['interaction_coef'],rtol=1e-12,atol=1e-12)
assert direction(z,np.zeros(p),tr,np.ones(p)) is None
cfg=[(5,.1,None),(5,.1,.1),(5,.1,1.),(10,.1,None),(10,.1,.1)]
jn,ji,mm=select_minimum(np.array([[1,1],[1.2,1.2],[1.1,1.1],[2,2],[1,1]]),cfg)
assert jn==0 and ji==2 and mm[jn]<mm[ji] # nonzero candidate kept for association even when null predicts best
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');B=P/'data_qc/w20_simplified_20261101_v2'
L=Path('/home/ec2-user/multisystem_runtime_20261021/w20_simplified_20261101_v2')
assert (L/'41_actual_exit_code').read_text().strip()=='0'
checks={'direct_block_solution':True,'held_Y_invariance':True,'matched_main_only_solution':True,'score_reexpression_retains_amplitude':True,'zero_variance':True,'minimum_MSE_without_null_eligibility_gate':True}
(B/'algorithm_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
(B/'registered_source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
print('V2_ALGORITHM_CHECKS_PASS',json.dumps(checks),flush=True)
