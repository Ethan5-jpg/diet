"""Assumption-dependent integrated-design regression calibration prototype.

Conditional-mean model: E[Y|latent D,C,S,observed records]=X(D,C,S) beta.
Use E[X(D)|observed] as the design, not X(E[D]) for nonlinear basis terms.
No claim of validity without the exposure model and nondifferential-error
assumptions. Shared catalogue draws keep each FoodID label common to people.
The conservative covariance addition is a development candidate; calibration
and effective-food diagnostics are necessary before any real activation.
"""
import numpy as np
from scipy import special,sparse
from w17_interval_upf_rcs import design_ranges
from w16_estimators import ols_sandwich

def point_design(d,c,s,*,center=.5,delta=.1,knots=(-3.2,0.,3.2)):
    x,_,idx=design_ranges(d,d,c,s,center=center,delta=delta,knots=knots)
    return x,idx

def beta_integrated_design(low,high,c,s,*,a,b,quadrature=48,center=.5,delta=.1,knots=(-3.2,0.,3.2)):
    if a<=0 or b<=0 or quadrature<8:raise ValueError('positive beta parameters and adequate quadrature required')
    nodes,weights=special.roots_jacobi(quadrature,b-1,a-1)
    f=(nodes+1)/2;weights=weights/weights.sum()
    low,high=np.asarray(low,float),np.asarray(high,float)
    result=None
    for node,weight in zip(f,weights):
        x,idx=point_design(low+(high-low)*node,c,s,center=center,delta=delta,knots=knots)
        result=weight*x if result is None else result+weight*x
    return result,idx,{'model':'F=(D-L)/(U-L) conditional beta','a':a,'b':b,
                      'quadrature':quadrature,'predicted_exposure_is_true_value':False}

def catalogue_draw_design(lower,unknown_weights,probabilities,c,s,*,draws,seed,
                          center=.5,delta=.1,knots=(-3.2,0.,3.2)):
    """A[i,j] is unknown FoodID j energy share; draw ONE label per FoodID."""
    lower=np.asarray(lower,float);a=sparse.csr_matrix(unknown_weights,dtype=float)
    q=np.asarray(probabilities,float);c=np.asarray(c,float);s=np.asarray(s,float)
    estimated=q.ndim==2
    if estimated and q.shape[0]!=draws:raise ValueError('one probability vector per catalogue draw')
    if q.ndim not in [1,2]:raise ValueError('probability vector or draw matrix required')
    qmean=q.mean(axis=0) if estimated else q
    J=len(qmean)
    if a.shape!=(len(lower),J) or np.any(a.data<0) or not np.isfinite(a.data).all():raise ValueError('nonnegative finite weight matrix')
    upper=lower+np.asarray(a.sum(axis=1)).ravel()
    if np.any(lower<0) or np.any(upper>1+1e-10) or np.any(q<0) or np.any(q>1) or not np.isfinite(q).all():raise ValueError('valid fraction bounds and probabilities')
    if draws<32:raise ValueError('insufficient catalogue draws')
    r=np.random.default_rng(seed);z=(r.random((draws,J))<q).astype(float)
    latent=lower[None,:]+np.asarray(a@z.T).T
    matrices=[]
    for d in latent:
        x,idx=point_design(d,c,s,center=center,delta=delta,knots=knots);matrices.append(x)
    matrices=np.asarray(matrices);mean=matrices.mean(axis=0)
    # All columns except the single RCS basis are affine in D. Use analytic
    # conditional means there so catalogue Monte Carlo error affects only RCS.
    exact,idx=point_design(lower+np.asarray(a@qmean).ravel(),c,s,center=center,delta=delta,knots=knots)
    rcs_column=1+c.shape[1]+s.shape[1]+1
    for j in range(mean.shape[1]):
        if j!=rcs_column:mean[:,j]=exact[:,j]
    contributions=np.asarray(a.sum(axis=0)).ravel()
    uncertain=contributions*np.sqrt(qmean*(1-qmean))
    effective=float(uncertain.sum()**2/np.sum(uncertain**2)) if np.sum(uncertain**2)>0 else None
    return mean,matrices,idx,{'draws':draws,'seed':seed,'FoodIDs':J,
                             'effective_uncertain_FoodIDs_energy':effective,
                             'one_label_per_FoodID_per_draw':True,
                             'q_estimation_uncertainty_included':estimated}

def catalogue_projection(mean_design,draw_designs):
    x=np.asarray(mean_design,float);draw=np.asarray(draw_designs,float)
    bread=np.linalg.inv(x.T@x)
    return np.matmul((draw-draw.mean(axis=0)).transpose(0,2,1),x@bread)

def fit_catalogue_calibration(mean_design,draw_designs,y,families,*,projection=None):
    x=np.asarray(mean_design,float);draw=np.asarray(draw_designs,float)
    if draw.ndim!=3 or draw.shape[1:]!=x.shape:raise ValueError('catalogue design dimensions')
    fit=ols_sandwich(x,y,clusters=families)
    if projection is None:projection=catalogue_projection(x,draw)
    perturbation=np.einsum('mjk,j->mk',projection,fit['beta'])
    catalogue_cov=perturbation.T@perturbation/(len(draw)-1)
    # Keep the family covariance and add the full catalogue component. This
    # duplicates a within-family component rather than subtracting a possibly
    # unstable estimate. Its conservative interpretation is model-dependent.
    fit['family_covariance']=fit['covariance'].copy()
    fit['catalogue_covariance']=catalogue_cov
    fit['covariance']=fit['covariance']+catalogue_cov
    fit['method']='integrated_design_family_plus_shared_catalogue_covariance_prototype'
    fit['exposure_probability_model_must_be_correct']=True
    return fit


def grouped_probability_draws(label_groups,categories,labels,weights,unknown_categories,*,draws,seed,shrinkage=10,logodds_shift=0):
    """Label-provenance Bayesian bootstrap; all aliases share one random weight.

    This propagates source-model estimation uncertainty; exchangeability of
    missing labels and regularization bias remain substantive assumptions.
    """
    groups=np.asarray(label_groups).astype(str);cat=np.asarray(categories).astype(str)
    y=np.asarray(labels,float);w=np.asarray(weights,float);unknown=np.asarray(unknown_categories).astype(str)
    if not (len(groups)==len(cat)==len(y)==len(w)) or np.any(w<=0) or np.any((y!=0)&(y!=1)):raise ValueError('reference arrays')
    gu,gi=np.unique(groups,return_inverse=True);cu,ci=np.unique(cat,return_inverse=True)
    mass=np.zeros((len(gu),len(cu)));positive=np.zeros_like(mass)
    np.add.at(mass,(gi,ci),w);np.add.at(positive,(gi,ci),w*y)
    if not np.allclose(mass.sum(axis=1),1.,atol=1e-10):raise ValueError('one evidence weight per provenance group')
    for j in range(len(gu)):
        if len(set(y[gi==j]))!=1:raise ValueError('conflicting inherited reference labels')
    r=np.random.default_rng(seed);boot=r.gamma(1.,1.,size=(draws,len(gu)))
    totals=boot@mass;success=boot@positive
    gp=(success.sum(axis=1)+1)/(totals.sum(axis=1)+2)
    p=(success+shrinkage*gp[:,None])/(totals+shrinkage)
    lookup={c:i for i,c in enumerate(cu)}
    result=np.column_stack([p[:,lookup[c]] if c in lookup else gp for c in unknown])
    if logodds_shift:
        result=special.expit(special.logit(np.clip(result,1e-10,1-1e-10))+logodds_shift)
    return result,{'distinct_reference_groups':len(gu),'categories':len(cu),'draws':draws,
                   'shared_alias_weights':True,'regularization_bias_corrected':False,
                   'source_model':'grouped Bayesian bootstrap of fixed-shrinkage category probability model'}
