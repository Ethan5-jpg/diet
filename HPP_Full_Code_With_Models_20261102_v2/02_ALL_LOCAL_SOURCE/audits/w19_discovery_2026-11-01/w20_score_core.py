"""Block-ridge scalar-outcome directions and compressed low-dimensional fits."""
import numpy as np
from scipy import linalg

def fit_directions_grid(base,zoral,zgut,u,y,training,configs,feature_scales=None):
 """Reuse a training Gram matrix across the fixed separate-penalty grid."""
 ix=np.flatnonzero(training);b=base[ix];z=np.column_stack([zoral[ix],zgut[ix]]);uy=u[ix,None]*z
 features=np.column_stack([z,uy]);host_y=np.linalg.lstsq(b,y[ix],rcond=None)[0];host_f=np.linalg.lstsq(b,features,rcond=None)[0]
 residual=features-b@host_f;target=y[ix]-b@host_y;p=z.shape[1];gram0=residual.T@residual;rhs=residual.T@target;po=zoral.shape[1];out=[]
 for main_penalty,interaction_penalty in configs:
  if interaction_penalty is None:out.append([None,None]);continue
  gram=gram0.copy();gram.flat[::len(gram)+1]+=np.r_[np.full(p,main_penalty),np.full(p,interaction_penalty)]
  co=linalg.cho_solve(linalg.cho_factor(gram,lower=True,check_finite=False),rhs,check_finite=False);v=co[p:];directions=[]
  for zz,vv in [(zoral,v[:po]),(zgut,v[po:])]:
   score=zz[ix]@vv;sd=float(score.std())
   if sd<=1e-8 or not np.isfinite(sd):directions.append(None);continue
   sc=np.ones_like(vv) if feature_scales is None else np.asarray(feature_scales[len(directions)],float)
   anchor=int(np.argmax(np.abs(vv/sc)));sign=1. if vv[anchor]>=0 else -1.;w=sign*vv/sd;center=float(np.mean(zz[ix]@w))
   directions.append({'weights':w,'center':center,'raw_score_training_SD':sd,'sign_anchor_index':anchor})
  out.append(directions)
 return out

def fit_directions(base,zoral,zgut,u,y,training,main_penalty,interaction_penalty):
 if interaction_penalty is None:return [None,None]
 return fit_directions_grid(base,zoral,zgut,u,y,training,[(main_penalty,interaction_penalty)])[0]

def score_values(z,direction):
 return None if direction is None else z@direction['weights']-direction['center']

def low_fit(base,oral,gut,u,y,training,directions):
 parts=[base];labels=['base_'+str(j) for j in range(base.shape[1])];sites=[]
 for site,z,d in zip(['oral','gut'],[oral,gut],directions):
  s=score_values(z,d)
  if s is not None:
   parts.extend([s[:,None],(u*s)[:,None]]);labels.extend([site+'_score_main',site+'_diet_score']);sites.append(site)
 x=np.column_stack(parts);tr=np.flatnonzero(training)
 _,rr,piv=linalg.qr(x[tr],mode='economic',pivoting=True);rank=int(np.sum(np.abs(np.diag(rr))>max(np.max(np.abs(np.diag(rr))),1)*1e-10));keep=np.sort(piv[:rank])
 coef=np.linalg.lstsq(x[tr][:,keep],y[tr],rcond=None)[0]
 return {'coefficients':coef,'columns':keep,'labels':[labels[j] for j in keep],'prediction':x[:,keep]@coef,'sites':sites,'rank':rank}

def one_se(loss,cfg):
 a=np.asarray(loss,float);means=a.mean(axis=1);best=int(np.argmin(means));se=float(a[best].std(ddof=1)/np.sqrt(a.shape[1]));allowed=[j for j in range(len(cfg)) if means[j]<=means[best]+se]
 j=min(allowed,key=lambda j:(cfg[j][1] is not None,-cfg[j][1] if cfg[j][1] is not None else 0,-cfg[j][0]))
 return j,{'means':means.tolist(),'best':best,'best_SE':se,'selected':j}

def direction_json(d):
 return None if d is None else {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items()}
