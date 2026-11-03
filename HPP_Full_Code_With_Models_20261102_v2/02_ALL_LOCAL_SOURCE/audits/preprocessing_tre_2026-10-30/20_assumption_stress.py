"""Complete 55-position synthetic learner/inference program, no real Y.

Shared FoodID labels generate BOTH training and confirmation outcomes. Scores
are selected on training outcomes with family CV. This specifically checks the
dependence not covered by earlier outcome-independent score component tests.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json,resource,time
import numpy as np
from scipy import sparse,special,stats
from w17_interval_upf_rcs import rcs3
from w16_estimators import ols_sandwich,wald_block,contrast_interval,daily_fixed_effects,holm_fixed_family
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(2400,2400))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1';L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'16_actual_exit_code').read_text().strip()=='0'
spec={'version':'assumption_stress_full55_v1','seed':20261030220,'training_n':800,'confirmation_n':800,
      'null_repetitions':400,'power_repetitions':100,'source_label_groups':640,'source_categories':20,
      'uncertain_FoodIDs':500,'catalogue_draws':256,'lambda_grid':[10,100,'infinite_zero_direction'],
      'family_CV_folds':3,'score_max_dimension_per_site':1,'q0_training_SD_threshold':1e-8,
      'complete_family_positions':55,'daily_formal_exposure':'fiber',
      'null_scenarios':['source_exchangeability_violation','omitted_cubic_microbiome_main'],
      'power_gamma_per10pp':[.03,.30],'score_main_dictionary':'all 3 synthetic transformed features per site plus their squares; redundant score mains omitted',
      'microbiome_scope':'synthetic training-standardized post-transform features; tests learner/inference dependence, not new real-data fit',
      'gate_FWER':'one-sided Wilson95 upper <=.08 per fixed scenario',
      'gate_zero_coverage':'descriptive component coverage>=.92 when active; zero active count must be reported',
      'gate_strong_signal':'joint UPF power Wilson95 lower>=.75; both-site direction availability reported',
      'real_Y_reads':0,'real_activation':False,'old_component_results_not_overwritten':True}
spec.update({'power_repetitions':0,'power_gamma_per10pp':[], 'gate_FWER':'not an acceptance gate: structural zero does not imply projection zero under misspecification','gate_zero_coverage':'not evaluated for the misspecified projection target','gate_strong_signal':'not evaluated','purpose':'stress source exchangeability and omitted nonlinear main effects; preserve all outcomes'})
sp=Q/'assumption_stress_full55_spec_v1.json';assert not sp.exists();sp.write_text(json.dumps(spec,indent=2)+'\n')
print('FULL55_LEARNER_SPEC_SEALED',hashlib.sha256(sp.read_bytes()).hexdigest(),flush=True)
seeds=iter(np.random.SeedSequence(spec['seed']).spawn(10000))
def rng():return np.random.default_rng(next(seeds))
def wilson(k,n,upper=True):
    z=stats.norm.ppf(.95);v=k/n;den=1+z*z/n
    m=(v+z*z/(2*n))/den;w=z*np.sqrt(v*(1-v)/n+z*z/(4*n*n))/den
    return float(m+w if upper else m-w)
def fixed(c,m):return np.column_stack([np.ones(len(c)),c,m,m*m])
def nuis(z,u,h,c):return np.column_stack([z,u,h,u*c[:,0]])
def train_directions(z,u,h,c,m,y):
    """CV chooses each outcome's penalty, including a zero-direction candidate."""
    b=nuis(z,u,h,c);inter=u[:,None]*m;x=np.column_stack([b,inter]);p=b.shape[1]
    n=len(x);fold=(np.arange(n)//2)%3;loss=np.zeros((3,y.shape[1]))
    def fit(a,t,k):
        if k==2:
            co=np.zeros((a.shape[1],t.shape[1]));co[:p]=np.linalg.lstsq(a[:,:p],t,rcond=None)[0];return co
        penalty=np.r_[np.zeros(p),np.full(6,[10.,100.][k])]
        return np.linalg.solve(a.T@a+np.diag(penalty),a.T@t)
    for f in range(3):
        tr=fold!=f;va=~tr
        for k in range(3):
            co=fit(x[tr],y[tr],k);loss[k]+=np.mean((y[va]-x[va]@co)**2,axis=0)
    selected=np.argmin(loss,axis=0);allco=[fit(x,y,k) for k in range(3)]
    result=[]
    for endpoint,k in enumerate(selected):
        vectors=[]
        for sl in [slice(0,3),slice(3,6)]:
            v=allco[k][p:,endpoint][sl];scale=float(np.std(m[:,sl]@v,ddof=0))
            vectors.append(v/scale if scale>spec['q0_training_SD_threshold'] else None)
        result.append((vectors,int(k)))
    return result
def dictionary(z,u,h,c,m,directions):
    b=nuis(z,u,h,c);scores=[];sites={};cols=[]
    for site,(sl,v) in enumerate(zip([slice(0,3),slice(3,6)],directions)):
        if v is None:sites[site]=[];continue
        s=m[:,sl]@v;sites[site]=[b.shape[1]+len(cols)];scores.append(s);cols.append(u*s)
    x=np.column_stack([b]+cols);score=np.column_stack(scores) if scores else np.zeros((len(z),0))
    return x,sites,score
def inference(z,u,h,c,m,directions,y,groups,draw_u=None,draw_h=None):
    x,sites,scores=dictionary(z,u,h,c,m,directions)
    f=ols_sandwich(x,y,clusters=groups)
    if draw_u is not None:
        k=z.shape[1];beta=f['beta'];slope=beta[k]+beta[k+2]*c[:,0]
        for site,v in enumerate(directions):
            if v is not None:
                sl=slice(0,3) if site==0 else slice(3,6)
                slope=slope+beta[sites[site][0]]*(m[:,sl]@v)
        pred=draw_u*slope[None,:]+draw_h*beta[k+1]
        perturb=(pred-pred.mean(axis=0))@x@f['bread']
        f['covariance']=f['covariance']+perturb.T@perturb/(len(perturb)-1)
    allidx=sites[0]+sites[1]
    if f['max_leverage']>.2:return [1.,1.,1.],[],sites,f
    p=[wald_block(f,idx)['p'] if idx else 1. for idx in [allidx,sites[0],sites[1]]]
    coverage=[contrast_interval(f,np.eye(x.shape[1])[j]) for j in allidx]
    return p,coverage,sites,f
def daily_null(r):
    people=180;counts=r.integers(3,13,people);ids=np.repeat(np.arange(people),counts);family=ids//2
    n=len(ids);d=r.normal(size=n);host=r.normal(size=people)[ids];so=r.normal(size=people)[ids];sg=r.normal(size=people)[ids]
    x=np.column_stack([d,rcs3(d,[-1.28155,0,1.28155]),d*host,d*so,d*sg]);ar=[];shared=r.normal(scale=.4,size=(people//2,12))
    for i,k in enumerate(counts):
        e=r.normal(size=k)
        for t in range(1,k):e[t]=.6*e[t-1]+np.sqrt(.64)*e[t]
        ar.extend(e+shared[i//2,:k])
    y=.3*d+.2*d*d+.3*d*host+r.normal(size=people)[ids]+np.asarray(ar)
    return wald_block(daily_fixed_effects(x,y,ids,family),[3,4])['p']
def generate(r,case,gamma=0):
    nt=spec['training_n'];nc=spec['confirmation_n'];n=nt+nc
    c=r.normal(size=(n,1));o=r.normal(size=(n,3));rho=.95 if case=='correlated_sites' else .3
    g=rho*o+np.sqrt(1-rho*rho)*r.normal(size=(n,3));m=np.column_stack([o,g])
    m=(m-m[:nt].mean(axis=0))/m[:nt].std(axis=0,ddof=0)
    z=fixed(c,m);cats=20;J=spec['uncertain_FoodIDs'];M=spec['catalogue_draws']
    qcat=r.beta(2,5,cats);fc=r.integers(cats,size=J);nref=120 if case=='weak_source_support' else spec['source_label_groups']
    rc=r.integers(cats,size=nref);ry=(r.random(nref)<qcat[rc]).astype(float);onehot=np.eye(cats)[rc]
    boot=r.gamma(1.,1.,size=(M,nref));total=boot@onehot;positive=boot@(onehot*ry[:,None]);gp=(positive.sum(axis=1)+1)/(total.sum(axis=1)+2)
    qposterior=(positive+10*gp[:,None])/(total+10);qp=qposterior[:,fc]
    v=(qcat[fc]-qcat.mean())/max(qcat.std(),.05)+.5*r.normal(size=J)
    selection=r.gumbel(size=(n,J))+.7*m[:,[0]]*v[None,:]
    chosen=np.argpartition(selection,-3,axis=1)[:,-3:]
    width=.03+.24*special.expit(.6*m[:,0]+.4*c[:,0]+r.normal(size=n))
    portions=r.dirichlet([1.,1.,1.],n)*width[:,None]
    A=sparse.csr_matrix((portions.ravel(),(np.repeat(np.arange(n),3),chosen.ravel())),shape=(n,J))
    lo=r.uniform(.02,.55,n);truep=special.expit(special.logit(qcat[fc])+2.5*np.tanh(v)) if case=='source_exchangeability_violation' else qcat[fc];trueZ=(r.random(J)<truep).astype(float);d=lo+np.asarray(A@trueZ).ravel()
    catalogueZ=(r.random((M,J))<qp).astype(float);draw=lo[None,:]+np.asarray(A@catalogueZ.T).T
    du=(draw-.5)/.1;dh=rcs3(du,[-3.2,0,3.2]);u=(lo+np.asarray(A@qp.mean(axis=0)).ravel()-.5)/.1;h=dh.mean(axis=0)
    trueu=(d-.5)/.1;trueh=rcs3(trueu,[-3.2,0,3.2]);f=r.normal(size=n);fh=rcs3(f,[-1.28155,0,1.28155])
    mu=.2*c[:,0]+.3*m[:,0]-.2*m[:,4]+.4*trueu+.1*trueh+.15*f+.1*f*f+.2*f*c[:,0]
    if case=='host_interaction_only':mu+=.5*trueu*c[:,0]
    if case=='nonlinear_microbiome_main':mu+=.25*m[:,0]**2+.2*m[:,4]**2
    mu+=gamma*trueu*m[:,0]
    if case=='omitted_cubic_microbiome_main':mu+=.5*(m[:,0]**3-3*m[:,0])
    shared=r.normal(size=(n//2,1));noise=.5*shared[np.arange(n)//2]+np.sqrt(.75)*r.normal(size=(n,9))
    y=mu[:,None]+noise
    return c,m,z,u,h,du,dh,f,fh,y

reports=[]
for case in spec['null_scenarios']:
    start=time.monotonic();rej=activeU=activeF=leverage=0
    for rep in range(spec['null_repetitions']):
        r=rng();c,m,z,u,h,du,dh,f,fh,y=generate(r,case)
        nt=spec['training_n'];tr=slice(0,nt);te=slice(nt,None);groups=np.arange(spec['confirmation_n'])//2
        us=train_directions(z[tr],u[tr],h[tr],c[tr],m[tr],y[tr]);fs=train_directions(z[tr],f[tr],fh[tr],c[tr],m[tr],y[tr])
        pv=[]
        for j in range(9):
            pu,ci,sites,fit=inference(z[te],u[te],h[te],c[te],m[te],us[j][0],y[te,j],groups,du[:,te],dh[:,te])
            pv.extend(pu);activeU+=bool(sites[0] or sites[1]);leverage+=fit['max_leverage']>.2
            pf,ci,sites,fit=inference(z[te],f[te],fh[te],c[te],m[te],fs[j][0],y[te,j],groups)
            pv.extend(pf);activeF+=bool(sites[0] or sites[1]);leverage+=fit['max_leverage']>.2
        pv.append(daily_null(r));assert len(pv)==55
        rej+=bool(np.any(holm_fixed_family(pv)<=.05))
        if (rep+1)%100==0:print('ASSUMPTION_STRESS_PROGRESS',case,rep+1,flush=True)
    v={'scenario':case,'repetitions':spec['null_repetitions'],
       'any_rejection_with_zero_structural_diet_interaction':rej/spec['null_repetitions'],
       'rejection_Wilson95_upper':wilson(rej,spec['null_repetitions']),
       'active_UPF_endpoint_fraction':activeU/(9*spec['null_repetitions']),
       'active_fiber_endpoint_fraction':activeF/(9*spec['null_repetitions']),
       'high_leverage_inactive_fits':leverage,'seconds':time.monotonic()-start,
       'projection_null_guaranteed':False,
       'interpretation':'assumption failure diagnostic, not a coverage claim for a nonzero projection coefficient'}
    reports.append(v);print('ASSUMPTION_STRESS_RESULT',json.dumps(v),flush=True)
result={'spec':spec,'scenarios':reports,'new_real_Y_reads':0,'production_activated':False,
        'real_missing_classification_mechanism_verified':False,
        'interpretation':'Passing correct-model simulations never proves robustness to these failures; conditional association tests cannot establish biological effect modification.'}
(Q/'assumption_stress_full55_v1.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('ASSUMPTION_STRESS_COMPLETE',flush=True)
