"""Discovery nested-CV main-effect baselines. No confirmation data I/O."""
import json,io,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse,linalg
from preprocessing_v1_20261030 import MicrobiomeCLR,PhenotypeOutliers
from upf_catalogue_calibration_v2 import grouped_probability_draws

def rcs(x,knots):
 a,b,c=np.asarray(knots,float)
 if not a<b<c:raise ValueError('distinct diet spline knots required')
 x=np.asarray(x,float)
 return (np.maximum(x-a,0)**3-(c-a)/(c-b)*np.maximum(x-b,0)**3+(b-a)/(c-b)*np.maximum(x-c,0)**3)/(c-a)**2

def whole_folds(families,k,seed):
 u=np.array(sorted(set(families)));np.random.default_rng(seed).shuffle(u);m={v:i%k for i,v in enumerate(u)}
 return np.array([m[v] for v in families])

def sparse_npz(path):
 with np.load(io.BytesIO(Path(path).read_bytes())) as f:return sparse.csr_matrix((f['data'],f['indices'],f['indptr']),shape=tuple(f['shape']))

def dump_npz(path,**arr):
 b=io.BytesIO();np.savez_compressed(b,**arr)
 with Path(path).open('xb') as f:f.write(b.getvalue())

def pca_fit(z,training,n=20):
 a=np.asarray(z[training],float);mu=a.mean(axis=0);center=a-mu
 val,vec=linalg.eigh(center.T@center,subset_by_index=[max(0,z.shape[1]-n),z.shape[1]-1],check_finite=True,driver='evr')
 order=np.argsort(val)[::-1];val=val[order];vec=vec[:,order];keep=val>max(val.max(),1)*1e-10;vec=vec[:,keep]
 for j in range(vec.shape[1]):
  anchor=np.argmax(np.abs(vec[:,j]));vec[:,j]*=1 if vec[anchor,j]>=0 else -1
 pc=(z-mu)@vec;sc=pc[training].std(axis=0);pc/=sc
 return pc,{'mean':mu.tolist(),'components':vec.tolist(),'scales':sc.tolist(),'eigenvalues':val[keep].tolist()}

class FoldContext:
 def __init__(self,X,oral,gut,taxa,A,reference,unknown,presence,training,seed):
  self.training=np.asarray(training,bool);self.pc={};self.transforms={};self.supported=np.ones(len(X),bool)
  ids=X.loc[training,'participant_id'].tolist()
  for site,raw in [('oral',oral),('gut',gut)]:
   m=MicrobiomeCLR().fit(raw[training],taxa[site],ids,site=site)
   r=m.transform(raw,taxa[site],site=site);support=r['supported'];self.supported&=support
   z=r['values'];z[~support]=0
   pc,ps=pca_fit(z,training&support)
   self.pc[site]=pc;self.transforms[site]={'CLR':m.state,'PCA':ps}
  used=set(presence.loc[presence.input_row.isin(np.flatnonzero(training)),'food_id'])
  ref=reference.loc[reference.food_id.isin(used)].copy();assert len(ref)>50
  ref['weight']=1/ref.groupby('label_group').food_id.transform('size')
  q,meta=grouped_probability_draws(ref.label_group,ref.category,ref.y,ref.weight,unknown.category,draws=1024,seed=seed,shrinkage=10)
  labels=np.random.default_rng(seed+1).random(q.shape)<q
  lo=X.UPF_lower_equalday_fraction.fillna(0).to_numpy(float)
  latent=lo[None,:]+np.asarray(A@labels.T.astype(float)).T
  upfmean=lo+np.asarray(A@q.mean(axis=0)).ravel()
  self.diets={};self.food_metadata={'source':meta,'seed':seed,'reference_FoodIDs':len(ref),'q_predictive_mean':q.mean(axis=0).tolist()}
  for e in ['UPF','fiber_density']:
   valid=training&(X.n_energy_days.ge(3).to_numpy() if e=='UPF' else X.n_fiber_days.ge(3).to_numpy())
   values=latent[:,valid].ravel() if e=='UPF' else X.loc[valid,'fiber_density'].to_numpy(float)
   values=values[np.isfinite(values)];k=np.quantile(values,[.1,.5,.9]);center=float(k[1]);scale=.1 if e=='UPF' else 5.;knots=(k-center)/scale
   if e=='UPF':u=(upfmean-center)/scale;h=rcs((latent-center)/scale,knots).mean(axis=0)
   else:u=(X.fiber_density.to_numpy(float)-center)/scale;h=rcs(u,knots)
   self.diets[e]={'u':u,'h':h,'center':center,'scale':scale,'knots':knots.tolist()}
 def save(self,dest):
  dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
  for site,s in self.transforms.items():(dest/(site+'_transform.json')).write_text(json.dumps(s,allow_nan=False)+'\n')
  (dest/'food_model.json').write_text(json.dumps(self.food_metadata,allow_nan=False)+'\n')
  (dest/'diet_basis.json').write_text(json.dumps({e:{k:v for k,v in s.items() if k not in ['u','h']} for e,s in self.diets.items()},allow_nan=False)+'\n')

def phenotype(y,rows,training,endpoint):
 keep=training[rows]&np.isfinite(y);m=PhenotypeOutliers().fit(y[keep],rows[keep].tolist(),endpoint=endpoint)
 return m.transform(y,endpoint=endpoint)['values'],m.state

class HostEncoder:
 def fit(self,X,training,exposure,diet):
  self.exposure=exposure;self.diet=diet
  num=np.column_stack([X.age_calendar_years,X.sex_code,X['mean_daily_energy_UPF' if exposure=='UPF' else 'mean_daily_energy_fiber']]).astype(float)
  self.mu=np.mean(num[training],axis=0);self.sd=np.std(num[training],axis=0);self.sd[self.sd<1e-12]=1
  self.levels=sorted(set(X.loc[training,'smoking_status'].astype(str))|{'__UNKNOWN__'})
  b,labels=self._raw(X)
  _,r,piv=linalg.qr(b[training],mode='economic',pivoting=True)
  rank=int(np.sum(np.abs(np.diag(r))>max(np.max(np.abs(np.diag(r))),1)*1e-10));self.keep=np.sort(piv[:rank]);self.labels=[labels[i] for i in self.keep]
  return self
 def _raw(self,X):
  num=np.column_stack([X.age_calendar_years,X.sex_code,X['mean_daily_energy_UPF' if self.exposure=='UPF' else 'mean_daily_energy_fiber']]).astype(float)
  c=(num-self.mu)/self.sd;cat=X.smoking_status.astype(str).to_numpy();cat=np.where(np.isin(cat,self.levels),cat,'__UNKNOWN__')
  parts=[np.ones((len(X),1)),c];labels=['intercept','age','sex','energy']
  for level in self.levels[1:]:parts.append((cat==level)[:,None].astype(float));labels.append('smoking=='+level)
  u=self.diet['u'];h=self.diet['h'];parts.extend([u[:,None],h[:,None],(u*c[:,0])[:,None],(u*c[:,1])[:,None]]);labels.extend(['diet_linear','diet_RCS','diet_age','diet_sex'])
  return np.column_stack(parts),labels
 def transform(self,X):return self._raw(X)[0][:,self.keep]
 def state(self):return {'exposure':self.exposure,'numeric_means':self.mu.tolist(),'numeric_scales':self.sd.tolist(),'smoking_levels':self.levels,'keep':self.keep.tolist(),'labels':self.labels}

def configs(model):
 if model=='host_only':return [(0,False,0.)]
 return [(k,sq,lam) for sq in [False,True] for k in [5,10,20] for lam in [100.,10.]]

def main_columns(context,model,config):
 k,square,_=config;sites={'host_only':[],'oral_main':['oral'],'gut_main':['gut'],'oral_gut_main':['oral','gut']}[model]
 cols=[];names=[]
 for site in sites:
  pc=context.pc[site][:,:k];cols.append(pc);names.extend([site+'_PC'+str(j+1) for j in range(pc.shape[1])])
  if square:cols.append(pc[:,:min(5,k)]**2);names.extend([site+'_PC'+str(j+1)+'_square' for j in range(min(5,pc.shape[1]))])
 return (np.column_stack(cols) if cols else np.zeros((len(context.supported),0))),names

def ridge_fit(base,main,y,training,penalty):
 tr=np.asarray(training,bool);b=base[tr];m=main[tr];yy=y[tr]
 assert np.isfinite(b).all() and np.isfinite(m).all() and np.isfinite(yy).all()
 # Residualize penalized features on unpenalized host terms. Penalty is absolute SSE ridge.
 bc=np.linalg.lstsq(b,yy,rcond=None)[0]
 if m.shape[1]:
  proj=np.linalg.lstsq(b,m,rcond=None)[0];res=m-b@proj;coef=np.linalg.solve(res.T@res+penalty*np.eye(m.shape[1]),res.T@(yy-b@bc));bc-=proj@coef
 else:coef=np.zeros(0)
 return bc,coef

def predict(base,main,fit):return base@fit[0]+main@fit[1]

def choose_one_se(fold_losses,candidates):
 a=np.asarray(fold_losses,float);means=a.mean(axis=1);best=int(np.argmin(means));se=float(a[best].std(ddof=1)/np.sqrt(a.shape[1]))
 allowed=[i for i,v in enumerate(means) if v<=means[best]+se]
 selected=min(allowed,key=lambda i:(candidates[i][1],candidates[i][0],-candidates[i][2]))
 return selected,{'mean_MSE':means.tolist(),'best':best,'best_SE':se,'selected':selected,'one_SE_rule':True}

def metrics(y,pred):
 y=np.asarray(y,float);pred=np.asarray(pred,float);e=y-pred;mse=float(np.mean(e*e));den=float(np.sum((y-y.mean())**2))
 return {'n':len(y),'MSE':mse,'RMSE':float(np.sqrt(mse)),'MAE':float(np.mean(np.abs(e))),'R2':float(1-np.sum(e*e)/den) if den>0 else None}
