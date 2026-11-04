"""Fold-local scoring formulas. All fitted states are private TRE artifacts."""
import numpy as np
from scipy import linalg
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def pcs(z, training):
    valid = training & np.isfinite(z).all(1)
    mean = z[valid].mean(0)
    a = z[valid] - mean
    d = a.shape[1]
    val, vec = linalg.eigh(a.T@a, subset_by_index=[max(0,d-20),d-1], driver='evr')
    order = np.argsort(val)[::-1]
    vec = vec[:,order[val[order] > max(1,val.max())*1e-10]][:,:10]
    assert vec.shape[1] == 10
    sign = np.sign(vec[np.argmax(abs(vec),axis=0),np.arange(10)])
    vec *= sign
    sc = (a@vec).std(0)
    return (z-mean)@vec/sc, (mean,vec,sc)


def backmap(state, pca, coef, train_center):
    """Return intercept and coefficients on logs of ALL reference species."""
    mean,vec,sc = pca
    w = vec@(coef/sc)
    b = w / np.asarray(state['scales'])
    ref = np.asarray(state['reference_indices'])
    out = np.asarray(state['output_indices'])
    log_weights = np.zeros(len(ref))
    log_weights[out] = b
    log_weights -= log_weights.mean()
    intercept = -train_center - mean@w - np.asarray(state['means'])@b
    return ref, log_weights, intercept


def modules(z, state, training):
    """Positive proportionality of training CLR coordinates; no outcome input."""
    ok = training & np.isfinite(z).all(1)
    cov = np.cov(z[ok]*np.asarray(state['scales']), rowvar=False)
    var = np.diag(cov)
    dist = np.clip(1-2*cov/(var[:,None]+var),0,2)
    np.fill_diagonal(dist,0)
    labels = fcluster(linkage(squareform(dist,checks=False),method='average'),0.5,criterion='distance')
    members = [np.flatnonzero(labels==k) for k in sorted(set(labels))]
    members = [m for m in members if len(m)>=3]
    scores = np.column_stack([z[:,m].mean(1) for m in members]) if members else np.empty((len(z),0))
    return scores,members
