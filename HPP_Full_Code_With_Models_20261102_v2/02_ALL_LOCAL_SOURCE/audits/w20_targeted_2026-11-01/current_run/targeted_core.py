"""Explicit regularized residual projection; no causal or orthogonality claim."""
import numpy as np
from scipy import linalg,stats
class PartialRidge:
 def __init__(self,base,z,tr,lam):
  self.base=base;self.z=z;self.tr=np.asarray(tr,bool);self.n=int(self.tr.sum());self.lam=lam
  _,r,p=linalg.qr(base[self.tr],mode='economic',pivoting=True)
  rank=np.sum(np.abs(np.diag(r))>max(np.max(np.abs(np.diag(r))),1)*1e-10);self.keep=np.sort(p[:rank]);self.b=base[self.tr][:,self.keep]
  self.sd=z[self.tr].std(0);self.sd[self.sd<1e-12]=1
  zz=z[self.tr]/self.sd;self.proj=np.linalg.lstsq(self.b,zz,rcond=None)[0];self.zz=zz-self.b@self.proj
  g=self.zz.T@self.zz;g.flat[::len(g)+1]+=self.n*lam;self.cf=linalg.cho_factor(g,lower=True,check_finite=False)
 def fit(self,target):
  yy=np.asarray(target)[self.tr];bc=np.linalg.lstsq(self.b,yy,rcond=None)[0]
  zz=linalg.cho_solve(self.cf,self.zz.T@(yy-self.b@bc),check_finite=False)
  bc=bc-self.proj@zz;raw=zz/self.sd if zz.ndim==1 else zz/self.sd[:,None]
  bcfull=np.zeros((self.base.shape[1],)+yy.shape[1:]);bcfull[self.keep]=bc
  pred=self.base@bcfull+self.z@raw
  return pred,bcfull,raw
class ResidualVC:
 def __init__(self,base,z,u,y,tr,main_lambda,host_nondiet,modifier):
  self.tr=tr;self.base=base;self.z=z;self.u=u;self.modifier=modifier
  self.null=PartialRidge(base,z,tr,main_lambda);self.nullpred,self.nb,self.nz=self.null.fit(y)
  self.exposure=PartialRidge(host_nondiet,z,tr,1.)
  self.epred,self.eb,self.ez=self.exposure.fit(u)
  self.center=modifier[tr].mean(0);self.W=(u-self.epred)[:,None]*(modifier-self.center)
  self.wp,self.wb,self.wz=self.null.fit(self.W);self.R=self.W-self.wp
  self.sd=self.R[tr].std(0);self.sd[self.sd<1e-12]=1
  rr=self.R[tr]/self.sd;self.gram=rr.T@rr;self.rhs=rr.T@(y-self.nullpred)[tr]
 def fit(self,lam):
  g=self.gram.copy();g.flat[::len(g)+1]+=int(self.tr.sum())*lam
  coef=linalg.cho_solve(linalg.cho_factor(g,lower=True,check_finite=False),self.rhs,check_finite=False)/self.sd
  return {'coef':coef,'prediction':self.nullpred+self.R@coef,'correction':self.R@coef}
def cluster_fit(y,x,w,groups,absorbed=0):
 y=np.asarray(y,float);x=np.asarray(x,float);w=np.asarray(w,float);n,p=x.shape
 assert np.isfinite(y).all() and np.isfinite(x).all() and (w>0).all()
 assert np.linalg.matrix_rank(x*np.sqrt(w[:,None]))==p,'Non-identifiable regression design'
 bread=np.linalg.inv(x.T@(w[:,None]*x));beta=bread@(x.T@(w*y));res=y-x@beta
 uniq,inv=np.unique(np.asarray(groups).astype(str),return_inverse=True);G=len(uniq);assert G>1 and n-absorbed-p>0
 sums=np.zeros((G,p));np.add.at(sums,inv,x*(w*res)[:,None])
 cov=bread@(sums.T@sums)@bread*(G/(G-1))*((n-1)/(n-absorbed-p))
 se=np.sqrt(np.maximum(np.diag(cov),0));crit=stats.t.ppf(.975,G-1)
 return beta,cov,np.column_stack([beta-crit*se,beta+crit*se]),G,res
def checks():
 rng=np.random.default_rng(61101);n=160;b=np.column_stack([np.ones(n),rng.normal(size=(n,2))]);z=rng.normal(size=(n,8));tr=np.arange(n)<120;y=rng.normal(size=n);u=rng.normal(size=n)
 h=PartialRidge(b,z,tr,.2);pred,bc,zc=h.fit(y);D=np.column_stack([b,z/h.sd]);pen=np.diag([0]*b.shape[1]+[tr.sum()*.2]*z.shape[1]);direct=np.linalg.solve(D[tr].T@D[tr]+pen,D[tr].T@y[tr]);np.testing.assert_allclose(pred,D@direct,atol=1e-10)
 vc=ResidualVC(b,z,u,y,tr,.2,b,z);fit=vc.fit(.1);a=vc.R[tr]/vc.sd;cc=np.linalg.solve(a.T@a+tr.sum()*.1*np.eye(z.shape[1]),a.T@(y-vc.nullpred)[tr])/vc.sd;np.testing.assert_allclose(fit['coef'],cc,atol=1e-10)
 # Finite-difference difference of conditional slopes: non-diet exposure nuisance fixed, shared base derivative cancels.
 v=fit['coef'];z1=z[0];z0=z[1];eps=1e-6
 f=lambda uu,zz,ee:(uu-ee)*((zz-vc.center)@v)-(.3*uu+.2*uu**2)
 contrast=((f(.4+eps,z1,.1)-f(.4-eps,z1,.1))-(f(.4+eps,z0,.2)-f(.4-eps,z0,.2)))/(2*eps)
 np.testing.assert_allclose(contrast,(z1-z0)@v,atol=1e-9)
 ids=np.repeat(np.arange(20),4);xx=rng.normal(size=(80,2));yy=rng.normal(size=80);dm=lambda a:a-a.reshape((20,4)+a.shape[1:]).mean(1)[ids]
 bf=np.linalg.lstsq(np.column_stack([np.eye(20)[ids],xx]),yy,rcond=None)[0][-2:];bw=np.linalg.lstsq(dm(xx),dm(yy),rcond=None)[0];np.testing.assert_allclose(bf,bw,atol=1e-12)
 return {'partial_ridge_vs_direct':True,'residual_ridge_vs_direct':True,'continuous_slope_contrast':True,'FE_vs_explicit_dummies':True}
