"""Conservative OUTER confidence sets for interval-valued regression covariates.

Prototype, not yet approved for real inference. No midpoint imputation, MAR,
common-sign monotonicity or point-identification assumption is imposed.
Finite-dimensional cluster-t Bonferroni moment bands are asymptotic, not an
exact finite-sample guarantee. W17 calibration is required before activation.

For any admissible latent design X, its population normal equations G beta=h
hold for the OLS projection. Simultaneous bands around pointwise lower/upper
moments contain G,h. Ignoring dependence between moment entries enlarges the
set, so exclusion of a coefficient null from this OUTER set is conservative.
"""
from collections import deque
import time
import numpy as np
from scipy import optimize, stats


def product_range(l1,u1,l2,u2):
    a=np.stack([l1*l2,l1*u2,u1*l2,u1*u2])
    return a.min(axis=0),a.max(axis=0)


def power_range(low,high,power):
    lo=np.minimum(low**power,high**power);hi=np.maximum(low**power,high**power)
    if power%2==0:lo=np.where((low<=0)&(high>=0),0,lo)
    return lo,hi


def design_ranges(low,high,covariates,scores,*,center,delta,nonlinear=True):
    """Frozen polynomial/main/host-interaction dictionary; last columns D*S."""
    low,high=np.asarray(low,float),np.asarray(high,float)
    c,s=np.asarray(covariates,float),np.asarray(scores,float)
    if delta<=0 or low.shape!=high.shape or np.any(low<0) or np.any(high>1) or np.any(low>high):
        raise ValueError('valid fraction bounds and positive frozen contrast required')
    if c.ndim!=2 or s.ndim!=2 or len(c)!=len(low) or len(s)!=len(low):
        raise ValueError('design row mismatch')
    if not all(np.isfinite(a).all() for a in [low,high,c,s]):raise ValueError('finite inputs required')
    ulo,uhi=(low-center)/delta,(high-center)/delta
    fixed=np.column_stack([np.ones(len(c)),c,s])
    lower=[fixed,ulo[:,None]];upper=[fixed,uhi[:,None]]
    if nonlinear:
        for power in (2,3):
            l,u=power_range(ulo,uhi,power);lower.append(l[:,None]);upper.append(u[:,None])
    l,u=product_range(ulo[:,None],uhi[:,None],c,c)
    lower.append(l);upper.append(u)
    nuisance=sum(a.shape[1] for a in lower)
    l,u=product_range(ulo[:,None],uhi[:,None],s,s)
    lower.append(l);upper.append(u)
    return np.column_stack(lower),np.column_stack(upper),list(range(nuisance,nuisance+s.shape[1]))


def moment_ranges(xlow,xhigh,y):
    xl,xu,y=np.asarray(xlow,float),np.asarray(xhigh,float),np.asarray(y,float)
    if xl.shape!=xu.shape or xl.ndim!=2 or len(y)!=len(xl) or np.any(xl>xu):raise ValueError('moment shapes')
    if not all(np.isfinite(v).all() for v in (xl,xu,y)):raise ValueError('finite moments required')
    p=xl.shape[1];lower=[];upper=[];pairs=[]
    for j in range(p):
        for k in range(j,p):
            if j==k:l,u=power_range(xl[:,j],xu[:,j],2)
            else:l,u=product_range(xl[:,j],xu[:,j],xl[:,k],xu[:,k])
            lower.append(l);upper.append(u);pairs.append((j,k))
    for j in range(p):
        l,u=product_range(xl[:,j],xu[:,j],y,y);lower.append(l);upper.append(u)
    return np.column_stack(lower),np.column_stack(upper),pairs


def moment_summary(lower,upper,clusters,weights=None):
    lo,hi=np.asarray(lower,float),np.asarray(upper,float)
    clusters=np.asarray(clusters).astype(str)
    if lo.shape!=hi.shape or len(clusters)!=len(lo):raise ValueError('summary shapes')
    w=np.ones(len(lo)) if weights is None else np.asarray(weights,float)
    if np.any(w<=0) or not np.isfinite(w).all():raise ValueError('positive weights required')
    labels,inv=np.unique(clusters,return_inverse=True);g=len(labels)
    if g<3:raise ValueError('insufficient independent clusters')
    def moments(a):
        mean=np.average(a,axis=0,weights=w)
        centered=w[:,None]*(a-mean)
        sums=np.zeros((g,a.shape[1]));np.add.at(sums,inv,centered)
        se=np.sqrt(g/(g-1)*np.sum(sums**2,axis=0))/w.sum()
        return mean,se
    ml,sl=moments(lo);mu,su=moments(hi)
    return {'lower_mean':ml,'upper_mean':mu,'lower_se':sl,'upper_se':su,'clusters':g,
            'moment_count':lo.shape[1]}


def confidence_bands(summary,pairs,p,alpha):
    if not 0<alpha<1:raise ValueError('alpha between zero and one')
    m=summary['moment_count'];critical=stats.t.ppf(1-alpha/(2*m),summary['clusters']-1)
    low=summary['lower_mean']-critical*summary['lower_se']
    high=summary['upper_mean']+critical*summary['upper_se']
    gl=np.zeros((p,p));gu=np.zeros((p,p))
    for index,(j,k) in enumerate(pairs):
        gl[j,k]=gl[k,j]=low[index];gu[j,k]=gu[k,j]=high[index]
    return gl,gu,low[len(pairs):],high[len(pairs):]


def null_feasibility(gl,gu,hl,hu,zero_indices,*,max_nodes=4096,time_limit=5.0):
    """Global sign-complementarity search; any unresolved solver node => inconclusive.

Write beta=positive-negative. LP relaxations allow both parts >0; branching
sets one part to zero. All accepted solutions have sign complementarity.
Only exhaustion with every node infeasible certifies null exclusion.
"""
    gl,gu,hl,hu=[np.asarray(x,float) for x in (gl,gu,hl,hu)]
    p=len(hl)
    if gl.shape!=(p,p) or gu.shape!=(p,p) or hu.shape!=(p,) or np.any(gl>gu) or np.any(hl>hu):
        raise ValueError('valid outer moment bands required')
    # Enlarge bands for numerical feasibility tolerances; never tighten them.
    eps=1e-8*(1+max(np.max(np.abs(hl)),np.max(np.abs(hu))))
    a=np.vstack([np.column_stack([gl,-gu]),np.column_stack([-gu,gl])])
    rhs=np.r_[hu+eps,-hl+eps]
    root=[(0,None)]*(2*p)
    for j in zero_indices:root[j]=(0,0);root[p+j]=(0,0)
    queue=deque([root]);nodes=0;start=time.monotonic()
    while queue:
        if nodes>=max_nodes or time.monotonic()-start>time_limit:
            return {'status':'inconclusive_computation','nodes':nodes,'null_excluded':False}
        bounds=queue.pop();nodes+=1
        result=optimize.linprog(np.ones(2*p),A_ub=a,b_ub=rhs,bounds=bounds,method='highs')
        if result.status==2:continue
        if not result.success:
            return {'status':'inconclusive_solver','nodes':nodes,'null_excluded':False}
        plus,minus=result.x[:p],result.x[p:]
        overlap=np.minimum(plus,minus)
        j=int(np.argmax(overlap))
        if overlap[j]<=1e-8:
            beta=plus-minus
            actual_low=gl@np.maximum(beta,0)-gu@np.maximum(-beta,0)
            actual_high=gu@np.maximum(beta,0)-gl@np.maximum(-beta,0)
            if np.any(actual_low>hu+2*eps) or np.any(actual_high<hl-2*eps):
                return {'status':'inconclusive_numeric_check','nodes':nodes,'null_excluded':False}
            return {'status':'null_compatible','nodes':nodes,'null_excluded':False,'witness':beta}
        left,right=list(bounds),list(bounds)
        left[j]=(0,0);right[p+j]=(0,0)
        queue.append(left);queue.append(right)
    return {'status':'null_excluded_outer_set','nodes':nodes,'null_excluded':True}


def conservative_pvalue(summary,pairs,p,zero_indices,*,alpha_grid,max_nodes=4096,time_limit=5.0):
    """Upper-rounded test inversion; computational failure can only inflate p."""
    grid=sorted(set(float(x) for x in alpha_grid))
    results={};left,right=0,len(grid)
    while left<right:
        mid=(left+right)//2;alpha=grid[mid]
        bands=confidence_bands(summary,pairs,p,alpha)
        result=null_feasibility(*bands,zero_indices,max_nodes=max_nodes,time_limit=time_limit)
        results[str(alpha)]={k:v for k,v in result.items() if k!='witness'}
        if result['null_excluded']:right=mid
        else:left=mid+1
    return {'p_upper':grid[left] if left<len(grid) else 1.0,'evaluations':results,
            'interpretation':'outer-set projection test; not a point estimate or exact finite-sample guarantee'}
