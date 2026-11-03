"""Mean-loss block ridge; retain fitted amplitudes for development predictions."""
import numpy as np
from scipy import linalg

class BlockDesign:
 def __init__(self,base,z,u,y,training):
  tr=np.asarray(training,bool);self.z=z;self.u=u;self.base=base;self.p=z.shape[1];self.n=int(tr.sum())
  _,r,piv=linalg.qr(base[tr],mode='economic',pivoting=True)
  rank=int(np.sum(np.abs(np.diag(r))>max(np.max(np.abs(np.diag(r))),1)*1e-10))
  self.keep=np.sort(piv[:rank]);b=base[tr][:,self.keep]
  f=np.column_stack([z[tr],u[tr,None]*z[tr]])
  self.scales=f.std(axis=0);self.scales[self.scales<=1e-12]=1.
  f=f/self.scales
  assert np.isfinite(b).all() and np.isfinite(f).all() and np.isfinite(y[tr]).all()
  self.by=np.linalg.lstsq(b,y[tr],rcond=None)[0]
  self.projection=np.linalg.lstsq(b,f,rcond=None)[0]
  residual=f-b@self.projection;target=y[tr]-b@self.by
  self.gram=residual.T@residual;self.rhs=residual.T@target

 def fit(self,main_lambda,interaction_lambda):
  p=self.p;ncoef=p if interaction_lambda is None else 2*p
  gram=self.gram[:ncoef,:ncoef].copy()
  penalty=np.full(ncoef,self.n*main_lambda)
  if interaction_lambda is not None:penalty[p:]=self.n*interaction_lambda
  gram.flat[::ncoef+1]+=penalty
  v=linalg.cho_solve(linalg.cho_factor(gram,lower=True,check_finite=False),self.rhs[:ncoef],check_finite=False)
  bc=np.zeros(self.base.shape[1]);bc[self.keep]=self.by-self.projection[:,:ncoef]@v
  original=v/self.scales[:ncoef]
  main=original[:p];interaction=original[p:] if interaction_lambda is not None else np.zeros(p)
  prediction=self.base@bc+self.z@main+self.u*(self.z@interaction)
  return {'base_coef':bc,'main_coef':main,'interaction_coef':interaction,'prediction':prediction,
   'feature_scales':self.scales[:ncoef],'base_rank':len(self.keep),'training_n':self.n}

def direction(z,v,training,clr_scales):
 s=z[training]@v;sd=float(s.std())
 if not np.isfinite(sd) or sd<=1e-12:return None
 anchor=int(np.argmax(np.abs(v/np.asarray(clr_scales))))
 sign=1. if v[anchor]>=0 else -1.
 w=sign*v/sd;center=float((z[training]@w).mean())
 return {'weights':w.tolist(),'center':center,'raw_training_SD':sd,'fixed_prediction_amplitude':sign*sd,'sign_anchor_index':anchor}

def score(z,d):return None if d is None else z@np.asarray(d['weights'])-d['center']

def select_minimum(losses,configs):
 means=np.mean(losses,axis=1)
 nulls=[j for j,c in enumerate(configs) if c[2] is None]
 jnull=min(nulls,key=lambda j:(means[j],configs[j][0],-configs[j][1]))
 chosen_background=configs[jnull][:2]
 candidates=[j for j,c in enumerate(configs) if c[:2]==chosen_background and c[2] is not None]
 jint=min(candidates,key=lambda j:(means[j],-configs[j][2]))
 return jnull,jint,means
