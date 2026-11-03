"""Seal implementation dictionary and test direction algebra before real scores."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib
import numpy as np
from w20_score_core import fit_directions,score_values,low_fit,one_se
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021');L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'36_actual_exit_code').read_text().strip()=='0'
W19=P/'data_qc/w19_discovery_20261101_v1';B=P/'data_qc/w20_discovery_20261101_v1';B.mkdir(exist_ok=False)
old=json.loads((W19/'execution_spec.json').read_text());spec={'version':'W20_discovery_20261101_v1','W19_recipe_sha256':hashlib.sha256((W19/'execution_spec.json').read_bytes()).hexdigest(),
 'seed':20261101020,'confirmation_reads':0,'family_folds':'reuse W19 outer5/inner3','ridge_grid':[[100.,None],[100.,100.],[100.,10.],[10.,100.],[10.,10.]],
 'main_dictionary':'host/core/diet nonlinear/diet-host +20PC per site and first5PC squares; richest registered W19 transparent dictionary, fixed for score step irrespective of baseline wins',
 'direction_dictionary':'all fold-transformed CLR species; retain Z main and u(D)*Z interactions; separate main/interaction ridge',
 'compressed_model':'same host/PCA/square background plus score mains and u(D)*score; training pivoted QR removes exact dependencies',
 'scalar_SVD':'for one endpoint per module, Z*B has one column; its first SVD direction equals normalized coefficient projection up to fixed sign',
 'selection':'inner3 oneSE prefer joint null, then stronger penalties; full-data configuration selected by outer5 candidate losses',
 'stability':old['W20_predeclared_grid']['stability'],'stability_reference':'absolute correlations of each outer heldfold score with full-discovery score on that fold; discovery reproducibility diagnostic, not independent predictive validation',
 'final_q':'full direction selected and >=4/5 outer active and median absolute correlation>=.5; otherwise site q=0; total<=2',
 'OOF_metric_interpretation':'fold-selected compressed predictor; final stability gating evaluated after discovery, not retroactively marketed as unbiased gated-model OOF',
 'formal_p_values':False,'final_support_freeze':'W22','endpoints':old['endpoints'],'exposures':old['exposures']}
(B/'execution_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
r=np.random.default_rng(37);n=160;b=np.column_stack([np.ones(n),r.normal(size=n)]);o=r.normal(size=(n,5));g=r.normal(size=(n,4));u=r.normal(size=n);y=.5*u*o[:,0]+.2*g[:,1]+b[:,1]+r.normal(scale=.2,size=n);tr=np.arange(n)<100
D=fit_directions(b,o,g,u,y,tr,10.,100.)
z=np.column_stack([o,g]);x=np.column_stack([b,z,u[:,None]*z]);pen=np.r_[np.zeros(2),np.full(9,10.),np.full(9,100.)]
co=np.linalg.solve(x[tr].T@x[tr]+np.diag(pen),x[tr].T@y[tr]);vv=co[-9:]
for zz,v,d in [(o,vv[:5],D[0]),(g,vv[5:],D[1])]:
 raw=zz[tr]@v;anchor=np.argmax(np.abs(v));expected=(zz@v-raw.mean())/raw.std()*(1 if v[anchor]>=0 else -1)
 np.testing.assert_allclose(score_values(zz,d),expected,atol=1e-9);np.testing.assert_allclose(score_values(zz,d)[tr].mean(),0,atol=1e-12);np.testing.assert_allclose(score_values(zz,d)[tr].std(),1,atol=1e-12)
y2=y.copy();y2[~tr]+=1e6;D2=fit_directions(b,o,g,u,y2,tr,10.,100.)
for a,c in zip(D,D2):np.testing.assert_array_equal(a['weights'],c['weights'])
f=low_fit(b,o,g,u,y,tr,D);f2=low_fit(b,o,g,u,y2,tr,D);np.testing.assert_array_equal(f['coefficients'],f2['coefficients'])
assert fit_directions(b,o,g,u,y,tr,100.,None)==[None,None]
j,meta=one_se([[1,1,1],[.99,1.01,.99]],[(100.,None),(10.,10.)]);assert j==0
print('W20_DIRECTION_AND_LOW_MODEL_CHECKS_PASS 6',flush=True)
print('W20_EXECUTION_REGISTERED',hashlib.sha256((B/'execution_spec.json').read_bytes()).hexdigest(),flush=True)
