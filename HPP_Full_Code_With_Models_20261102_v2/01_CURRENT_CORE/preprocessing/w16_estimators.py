"""Synthetic-validated estimator interfaces. No real-data reader or autonomous fit.

Main estimands are low-dimensional projections conditional on frozen scores.
These interfaces do not turn a misspecified projection into a causal estimand.
"""
import numpy as np
from scipy import stats
from preprocessing_v1_20261030 import SavedTransform, _training_units


class TrainPCA(SavedTransform):
    kind = 'hpp_training_pca_v1'

    def fit(self, x, names, ids, *, n_components=1, role='training'):
        x = np.asarray(x, float)
        _training_units(ids, len(x), role)
        if x.ndim != 2 or not np.isfinite(x).all() or len(names) != x.shape[1]:
            raise ValueError('finite named input required')
        mean = x.mean(axis=0)
        u, s, vt = np.linalg.svd(x-mean, full_matrices=False)
        k = min(n_components, int(np.sum(s > 1e-10)), len(vt))
        if k < 1:
            raise ValueError('no PCA direction supported')
        load = vt[:k].T
        for j in range(k):
            if load[np.argmax(np.abs(load[:,j])),j] < 0:
                load[:,j] *= -1
        scores = (x-mean)@load
        self.state = {'kind':self.kind, 'names':list(names), 'mean':mean.tolist(),
                      'loadings':load.tolist(), 'score_sd':scores.std(axis=0).tolist()}
        return self

    def transform(self, x, names):
        if list(names) != self.state['names']:
            raise ValueError('PCA column order changed')
        x = np.asarray(x,float)
        return ((x-self.state['mean'])@np.asarray(self.state['loadings']))/self.state['score_sd']


class CovariateEncoder(SavedTransform):
    kind = 'hpp_training_covariate_encoder_v1'

    def _raw(self, data, fitting=False):
        cols, labels = [], []
        for name in self.state['numeric']:
            a = np.asarray(data[name],float)
            missing = ~np.isfinite(a)
            val = np.where(missing,self.state['medians'][name],a)
            cols.extend([val, missing.astype(float)])
            labels.extend([name, name+'__missing'])
        for name in self.state['categorical']:
            levels = self.state['levels'][name]
            a = np.array(['__MISSING__' if v is None or str(v) in ('nan','<NA>','') else str(v)
                          for v in data[name]])
            if not set(a) <= set(levels):
                raise ValueError('unseen categorical level: '+name)
            for level in levels[1:]:
                cols.append((a==level).astype(float)); labels.append(name+'=='+level)
        return np.column_stack(cols), labels

    def fit(self,data,ids,*,numeric,categorical,role='training'):
        n = len(ids); _training_units(ids,n,role)
        if any(len(v)!=n for v in data.values()):
            raise ValueError('covariate row mismatch')
        medians, levels = {}, {}
        for name in numeric:
            a = np.asarray(data[name],float); finite=a[np.isfinite(a)]
            if len(finite)==0:
                raise ValueError('all-missing required numeric covariate: '+name)
            medians[name] = float(np.median(finite))
        for name in categorical:
            levels[name] = sorted(set('__MISSING__' if v is None or str(v) in ('nan','<NA>','') else str(v)
                                      for v in data[name]))
        self.state={'kind':self.kind,'numeric':list(numeric),'categorical':list(categorical),
                    'medians':medians,'levels':levels}
        raw, labels = self._raw(data)
        means, scales = raw.mean(axis=0), raw.std(axis=0)
        keep=np.flatnonzero(scales>1e-12)
        self.state.update({'raw_labels':labels,'keep':keep.tolist(),
                           'means':means[keep].tolist(),'scales':scales[keep].tolist()})
        return self

    def transform(self,data):
        raw, labels=self._raw(data)
        if labels!=self.state['raw_labels']:
            raise ValueError('encoded column order changed')
        return (raw[:,self.state['keep']]-self.state['means'])/self.state['scales']


def block_ridge(base, oral, gut, u, y, *, main_penalty, interaction_penalty):
    """Unpenalized low-dimensional base; ridge main and D-by-site blocks."""
    base,oral,gut,y=[np.asarray(x,float) for x in (base,oral,gut,y)]
    u=np.asarray(u,float).reshape(-1,1)
    if min(main_penalty,interaction_penalty)<=0 or np.linalg.matrix_rank(base)!=base.shape[1]:
        raise ValueError('positive penalties and full-rank base required')
    x=np.column_stack([base,oral,gut,u*oral,u*gut])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite ridge inputs required')
    p0,po,pg=base.shape[1],oral.shape[1],gut.shape[1]
    penalty=np.r_[np.zeros(p0),np.full(po+pg,main_penalty),np.full(po+pg,interaction_penalty)]
    coef=np.linalg.solve(x.T@x+np.diag(penalty),x.T@y)
    start=p0+po+pg
    return {'coef':coef,'oral_interaction':coef[start:start+po],
            'gut_interaction':coef[start+po:start+po+pg], 'prediction':x@coef}


class ScoreCompression(SavedTransform):
    kind='hpp_site_interaction_prediction_svd_v1'

    def fit(self,z,b,names,ids,*,role='training'):
        z=np.asarray(z,float); b=np.asarray(b,float)
        _training_units(ids,len(z),role)
        if b.ndim==1: b=b[:,None]
        prediction=z@b
        centered=prediction-prediction.mean(axis=0)
        _,s,vt=np.linalg.svd(centered,full_matrices=False)
        if len(s)==0 or s[0]<=1e-10:
            self.state={'kind':self.kind,'q':0,'names':list(names)}
            return self
        weights=b@vt[0]
        if weights[np.argmax(np.abs(weights))]<0: weights=-weights
        score=z@weights; scale=float(score.std())
        if scale<=1e-12: raise ValueError('unsupported score variation')
        self.state={'kind':self.kind,'q':1,'names':list(names),
                    'weights':weights.tolist(),'mean':float(score.mean()),'sd':scale}
        return self

    def transform(self,z,names):
        if list(names)!=self.state['names']: raise ValueError('score feature order changed')
        if self.state['q']==0: return np.empty((len(z),0))
        return ((np.asarray(z)@self.state['weights']-self.state['mean'])/self.state['sd'])[:,None]


def ols_sandwich(x,y,*,clusters=None,weights=None,extra_df=0):
    """HC3 for independent rows; CR1 with cluster-t/F for clustered rows."""
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.ndim!=2 or y.ndim!=1 or len(y)!=len(x): raise ValueError('OLS shape mismatch')
    n,p=x.shape
    if n<=p+extra_df or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('insufficient finite observations')
    w=np.ones(n) if weights is None else np.asarray(weights,float)
    if len(w)!=n or not np.isfinite(w).all() or np.any(w<=0): raise ValueError('positive weights required')
    a=x*np.sqrt(w)[:,None]; b=y*np.sqrt(w)
    if np.linalg.matrix_rank(a)!=p: raise ValueError('rank failure; no generalized inverse')
    bread=np.linalg.inv(a.T@a); beta=bread@(a.T@b)
    residual=y-x@beta
    leverage=np.einsum('ij,jk,ik->i',a,bread,a)
    if np.any(leverage>=1-1e-10): raise ValueError('unit leverage')
    if clusters is None:
        influence=x*(w*residual/(1-leverage))[:,None]
        score_by_cluster={str(i):influence[i] for i in range(n)}
        correction=1.0; df=n-p-extra_df; method='HC3_t'
    else:
        clusters=np.asarray(clusters).astype(str)
        if len(clusters)!=n: raise ValueError('cluster row mismatch')
        unique,inv=np.unique(clusters,return_inverse=True); g=len(unique)
        if g<3: raise ValueError('insufficient independent clusters')
        influence=x*(w*residual)[:,None]
        sums=np.zeros((g,p)); np.add.at(sums,inv,influence)
        correction=g/(g-1)*(n-1)/(n-p-extra_df)
        score_by_cluster={key:row for key,row in zip(unique,sums)}
        df=g-1; method='CR1_cluster_t_F'
    scores=np.array(list(score_by_cluster.values()))
    covariance=correction*bread@(scores.T@scores)@bread
    return {'beta':beta,'covariance':covariance,'bread':bread,'cluster_scores':score_by_cluster,
            'correction':correction,'df':df,'method':method,'n':n,'p':p,
            'max_leverage':float(leverage.max()),'residual':residual}


def wald_block(fit,indices):
    indices=list(indices)
    if not indices: return {'p':1.0,'rank':0,'status':'inactive_q0'}
    beta=fit['beta'][indices]; v=fit['covariance'][np.ix_(indices,indices)]
    r=len(indices)
    if np.linalg.matrix_rank(v)!=r: return {'p':1.0,'rank':0,'status':'inactive_covariance_rank'}
    stat=float(beta@np.linalg.solve(v,beta)/r)
    return {'p':float(stats.f.sf(max(0,stat),r,fit['df'])),'rank':r,'status':'estimable'}


def contrast_interval(fit,contrast,alpha=0.05):
    c=np.asarray(contrast,float); est=float(c@fit['beta'])
    se=float(np.sqrt(max(0,c@fit['covariance']@c)))
    margin=float(stats.t.ppf(1-alpha/2,fit['df'])*se)
    return {'estimate':est,'se':se,'low':est-margin,'high':est+margin,'pointwise_only':True}


def stacked_covariance(fits):
    """Cross-endpoint covariance using shared cluster IDs, without independence assumptions."""
    sizes=[f['p'] for f in fits]; offsets=np.r_[0,np.cumsum(sizes)]
    keys=sorted(set().union(*(set(f['cluster_scores']) for f in fits)))
    influence=np.zeros((len(keys),sum(sizes)))
    for j,f in enumerate(fits):
        for i,key in enumerate(keys):
            if key in f['cluster_scores']:
                influence[i,offsets[j]:offsets[j+1]]=np.sqrt(f['correction'])*(f['bread']@f['cluster_scores'][key])
    return influence.T@influence


def daily_fixed_effects(x,y,person_ids,family_ids):
    x,y=np.asarray(x,float),np.asarray(y,float)
    ids=np.asarray(person_ids).astype(str); family=np.asarray(family_ids).astype(str)
    people,inv,count=np.unique(ids,return_inverse=True,return_counts=True)
    if np.any(count<2): raise ValueError('each retained person needs at least two qualified days')
    for person in people:
        if len(set(family[ids==person]))!=1: raise ValueError('family changes within person')
    sums=np.zeros((len(people),x.shape[1])); np.add.at(sums,inv,x)
    sy=np.zeros(len(people)); np.add.at(sy,inv,y)
    xd=x-sums[inv]/count[inv,None]; yd=y-sy[inv]/count[inv]
    return ols_sandwich(xd,yd,clusters=family,weights=1/count[inv],extra_df=len(people))


def holm_fixed_family(pvalues):
    p=np.asarray(pvalues,float)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('complete fixed family of p-values required, inactive positions use 1')
    order=np.argsort(p,kind='stable'); adjusted=np.empty(len(p))
    adjusted[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return adjusted
