"""Clarify oracle capability using its own matched no-interaction comparator."""
from pathlib import Path
import io,json
import numpy as np,pandas as pd,pyarrow.parquet as pq
from w19_baseline_core import HostEncoder,phenotype,metrics

def evaluate(B,O):
 X=pq.ParquetFile(B/'discovery_covariates_food_TRE_ONLY.parquet').read().to_pandas(ignore_metadata=True);n=len(X)
 C=O/'fold_preprocessing';contexts={}
 for key in ['full']+['fold'+str(i) for i in range(5)]:
  with np.load(io.BytesIO((C/key/'transformed_discovery_TRE_ONLY.npz').read_bytes())) as f:contexts[key]={k:f[k] for k in ['gut_PC','oral_PC','fiber_u','fiber_h','training','supported']}
 with np.load(io.BytesIO((O/'models/D_Glu/fiber_density/full_fit_readback_TRE_ONLY.npz').read_bytes())) as f:eligible=f['training']
 rows=np.flatnonzero(eligible);c=contexts['full'];diet={'u':c['fiber_u'],'h':c['fiber_h']}
 enc=HostEncoder().fit(X,eligible,'fiber_density',diet);bb=np.column_stack([enc.transform(X),c['oral_PC'][:,:5],c['gut_PC'][:,:5]])
 axis=c['gut_PC'][:,0].copy();axis=(axis-np.mean(axis[eligible]))/np.std(axis[eligible]);signal=c['fiber_u']*axis
 scoef=np.linalg.lstsq(bb[eligible],signal[eligible],rcond=None)[0];signal=signal-bb@scoef;signal/=np.std(signal[eligible])
 spec=json.loads((B/'execution_spec.json').read_text());rng=np.random.default_rng(spec['seed']+773)
 unit_noise={v:rng.normal() for v in sorted(X.split_unit_id.unique())};noise=rng.normal(size=n)+.2*X.split_unit_id.map(unit_noise).to_numpy(float);noise/=np.std(noise[eligible]);background=2+.3*(X.age_calendar_years.to_numpy(float)-50)/10
 reports=[];old=pd.read_csv(O/'synthetic_controls.csv')
 for f in [0.,.01,.05]:
  yy=background+noise+np.sqrt(f/(1-f))*signal;ys=np.full(n,np.nan);pm=np.full(n,np.nan);pi=np.full(n,np.nan)
  for o in range(5):
   c=contexts['fold'+str(o)];training=c['training'];values,_=phenotype(yy[rows],rows,training,'D_Glu');y=np.full(n,np.nan);y[rows]=values
   E=X.mean_daily_energy_fiber.to_numpy(float)
   el=X.n_fiber_days.ge(3).to_numpy()&c['supported']&np.isfinite(X.age_calendar_years.to_numpy(float))&X.sex_code.isin([0,1]).to_numpy()&np.isfinite(E)&np.isfinite(c['fiber_u'])&np.isfinite(c['fiber_h'])&np.isfinite(y)
   tr=training&el;va=X.cv_fold.eq(o).to_numpy()&el;diet={'u':c['fiber_u'],'h':c['fiber_h']};enc=HostEncoder().fit(X,tr,'fiber_density',diet)
   base=np.column_stack([enc.transform(X),c['oral_PC'][:,:5],c['gut_PC'][:,:5],axis]);inter=np.column_stack([base,c['fiber_u']*axis])
   cm=np.linalg.lstsq(base[tr],y[tr],rcond=None)[0];ci=np.linalg.lstsq(inter[tr],y[tr],rcond=None)[0]
   pm[va]=base[va]@cm;pi[va]=inter[va]@ci;ys[va]=y[va]
  common=np.isfinite(ys);mm=metrics(ys[common],pm[common]);im=metrics(ys[common],pi[common])
  prior=old.loc[old.control.eq('signal_residual_fraction_'+str(f))].iloc[0]
  assert int(common.sum())==int(prior['n']);np.testing.assert_allclose(im['R2'],prior.oracle_R2,atol=1e-9,rtol=1e-9)
  reports.append({'signal_residual_fraction':f,'n':int(common.sum()),'learned_delta_R2_vs_its_matched_null':float(prior.delta_R2),'oracle_null_R2':mm['R2'],'oracle_interaction_R2':im['R2'],'oracle_delta_R2_vs_its_matched_null':im['R2']-mm['R2'],'old_oracle_delta_field_compared_against_high_dimensional_null':True,'one_seed_not_power':True})
 pd.DataFrame(reports).to_csv(O/'oracle_matched_control_clarification.csv',index=False)
 return reports
