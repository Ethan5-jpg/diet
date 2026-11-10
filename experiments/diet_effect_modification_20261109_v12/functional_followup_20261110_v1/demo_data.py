"""Generated data only; no native HPP reads. Shared by deterministic checks."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from frozen import modules


def folds(groups,k,seed):
    values=np.array(sorted(set(groups)))
    np.random.default_rng(seed).shuffle(values)
    mapping={v:j%k for j,v in enumerate(values)}
    return np.array([mapping[v] for v in groups])


def separated(groups,train,valid):
    if np.any(train & valid) or set(groups[train]) & set(groups[valid]):
        raise ValueError('Synthetic household leak')


class Base:
    def __init__(self,dataset,train,valid):
        self.dataset=dataset
        self.config=dict(contrast=[10,15])
        self.train_rows,self.valid_rows=np.flatnonzero(train),np.flatnonzero(valid)
        self.center=np.median(dataset.index.fiber_density.to_numpy()[train])
        self.scaler=StandardScaler().fit(self.raw(True))
        s=SimpleNamespace(train_rows=self.train_rows,rows=self.valid_rows,yt=dataset.y[train],yv=dataset.y[valid])
        self.role=SimpleNamespace(split=s)
        age=dataset.index.age.to_numpy()
        self.background_t=age[train,None]
        self.background_v=age[valid,None]
        self.pcs={site:(dataset.taxonomy[site][train],dataset.taxonomy[site][valid]) for site in ('oral','gut')}

    def diet(self,training):
        return self.dataset.index.fiber_density.to_numpy()[self.train_rows if training else self.valid_rows]

    def diet_basis(self,training,value=None):
        d=self.diet(training) if value is None else np.full(len(self.diet(training)),value)
        u=(d-self.center)/5.
        return np.column_stack([u,u*u])

    def raw(self,training,value=None):
        rows=self.train_rows if training else self.valid_rows
        age=self.dataset.index.age.to_numpy()[rows]
        diet=self.diet_basis(training,value)
        return np.column_stack([diet,age,age*diet[:,0],self.dataset.index.sex.to_numpy()[rows]])

    def main(self,training=False,value=None):
        raw=self.raw(training,value)
        return np.column_stack([np.ones(len(raw)),self.scaler.transform(raw)])


class Engine:
    def __init__(self,*args):
        pass

    def design(self,dataset,train,valid):
        separated(dataset.index.split_unit_id.astype(str).to_numpy(),train,valid)
        return Base(dataset,train,valid)


def make(config,seed=112,interaction=10.,people=420):
    rng=np.random.default_rng(seed)
    n=people+2
    groups=np.array(['SYNTHETIC_HH_%04d' % (j//2) for j in range(n)])
    diet=rng.uniform(7,24,n)
    latent=rng.normal(size=n)
    age=rng.normal(size=n)
    fields=16
    loadings=np.linspace(-1.3,1.3,fields)
    abundance=np.exp(latent[:,None]*loadings[None,:]+rng.normal(scale=.25,size=(n,fields)))
    present=rng.random((n,fields))<.82
    abundance[~present]=np.nan
    abundance[-2:]=np.nan
    coverage=abundance/(1+abundance)
    y=4*(diet-14)/5.+age+.5*latent+interaction*(diet-14)/5.*latent+rng.normal(scale=.5,size=n)
    index=pd.DataFrame(dict(input_row=np.arange(n),participant_id=np.arange(n)+1,cohort='synthetic',research_stage='synthetic',split_unit_id=groups,fiber_density=diet,age=age,sex=rng.integers(0,2,n)))
    outer=np.stack([folds(groups,config['outer_folds'],config['outer_seed']+j) for j in (0,1)])
    dataset=SimpleNamespace(index=index,y=y,reference=dict(outer=outer,y=np.tile(y,(2,1))),api=SimpleNamespace(folds=folds,separated=separated),taxonomy={site:rng.normal(size=(n,4)) for site in ('oral','gut')})
    counts=np.exp(rng.normal(12,1,n))
    ratio=rng.uniform(.05,.7,n)
    technical=np.column_stack([counts,counts*ratio,ratio])
    available=np.arange(n)<people
    frozen=modules()
    return SimpleNamespace(dataset=dataset,frozen=SimpleNamespace(config=dict(contrast=[10,15]),engine=SimpleNamespace(Engine=Engine),models=frozen.models,features=frozen.features),available=available,
        signals=dict(abundance=abundance,coverage=coverage),technical=technical,labels=['PWY-%d:synthetic' % j for j in range(fields)],
        source_hashes={},versions=dict(synthetic_only=True),audit_text='SYNTHETIC ONLY; no native HPP data.',synthetic_only=True,
        audit=dict(original_people=n,functional_people=people,excluded_whole_candidate_null=2,functional_households=len(set(groups[available])),synthetic_only=True))
