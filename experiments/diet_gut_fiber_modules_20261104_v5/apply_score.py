"""Score unseen samples using the private frozen TRE rule. Requires only NumPy."""
import numpy as np


def score_new(abundance, species, rule):
    """Return centered modification in mg/dL per 10 -> 15. Missing reference taxa are errors."""
    a=np.asarray(abundance,float)
    if a.ndim==1: a=a[None,:]
    if len(set(species))!=len(species) or a.shape[1]!=len(species):
        raise ValueError('Ambiguous abundance columns')
    mapping={s:i for i,s in enumerate(species)}
    if not set(rule['species'])<=set(mapping): raise ValueError('Missing reference taxon')
    a=a[:,[mapping[s] for s in rule['species']]]
    if not np.isfinite(a).all() or (a<0).any(): raise ValueError('Invalid abundance')
    supported=a.sum(1)>0
    value=np.log(np.where(a==0,rule['zero_replacement'],a))@rule['weights']+rule['intercept']
    return np.where(supported,value,np.nan)


