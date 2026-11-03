"""Candidate v2: linear-tailed restricted cubic main effect, three frozen knots.

Uses the same conservative outer-moment interval inference prototype. This
module changes the nuisance dictionary, not the latent-exposure assumptions.
"""
import numpy as np
from w17_interval_upf_prototype import (product_range,power_range,moment_ranges,
    moment_summary,confidence_bands,null_feasibility,conservative_pvalue)

def rcs3(x,knots):
    k=np.asarray(knots,float)
    if k.shape!=(3,) or not np.isfinite(k).all() or not np.all(np.diff(k)>0):
        raise ValueError('three distinct ordered finite training knots required')
    x=np.asarray(x,float);a,b=k[1]-k[0],k[2]-k[1];span=a+b
    middle=(np.maximum(x-k[0],0)**3-span/b*np.maximum(x-k[1],0)**3)/span**2
    # Explicit linear tail avoids cancellation of three large cubic terms.
    at_right=a*(a+2*b)/span;slope=3*a/span
    return np.where(x>=k[2],at_right+slope*(x-k[2]),middle)

def design_ranges(low,high,covariates,scores,*,center,delta,nonlinear=True,knots=(-3.2,0.,3.2)):
    low,high=np.asarray(low,float),np.asarray(high,float)
    c,s=np.asarray(covariates,float),np.asarray(scores,float)
    if delta<=0 or low.shape!=high.shape or np.any(low<0) or np.any(high>1) or np.any(low>high):
        raise ValueError('valid fraction bounds and positive frozen contrast required')
    if c.ndim!=2 or s.ndim!=2 or len(c)!=len(low) or len(s)!=len(low):raise ValueError('design row mismatch')
    if not all(np.isfinite(v).all() for v in [low,high,c,s]):raise ValueError('finite inputs required')
    ulo,uhi=(low-center)/delta,(high-center)/delta
    fixed=np.column_stack([np.ones(len(c)),c,s])
    lower=[fixed,ulo[:,None]];upper=[fixed,uhi[:,None]]
    if nonlinear:
        lower.append(rcs3(ulo,knots)[:,None]);upper.append(rcs3(uhi,knots)[:,None])
    l,u=product_range(ulo[:,None],uhi[:,None],c,c);lower.append(l);upper.append(u)
    nuisance=sum(v.shape[1] for v in lower)
    l,u=product_range(ulo[:,None],uhi[:,None],s,s);lower.append(l);upper.append(u)
    return np.column_stack(lower),np.column_stack(upper),list(range(nuisance,nuisance+s.shape[1]))
