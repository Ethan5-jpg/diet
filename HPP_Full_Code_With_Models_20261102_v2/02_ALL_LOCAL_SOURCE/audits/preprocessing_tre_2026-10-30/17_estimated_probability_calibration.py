"""Fixed synthetic evaluation of model-dependent UPF regression calibration."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json,resource,time
import numpy as np
from scipy import special,stats
from upf_catalogue_calibration_v2 import (grouped_probability_draws,point_design,catalogue_draw_design,catalogue_projection,
                                      fit_catalogue_calibration)
from w16_estimators import wald_block,contrast_interval,holm_fixed_family
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(1800,1800))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'16_actual_exit_code').read_text().strip()=='0'
spec={'version':'estimated_probability_calibration_v1','seed':20261030117,'n':800,'null_repetitions':400,
      'power_repetitions':100,'catalogue_draws':256,'exposure_bound_width_fraction':.15,
      'power_gamma_per10pp':.30,'family_positions':55,'active_UPF_positions':27,
      'other_28_positions':'p=1 in this component experiment; not complete W17 acceptance',
      'scenarios':[{'name':'estimated_640_reference_normal','foods':500,'reference_groups':640,'noise':'normal','q_logodds_shift':0},
                   {'name':'estimated_640_reference_heavy','foods':500,'reference_groups':640,'noise':'heavy','q_logodds_shift':0},
                   {'name':'estimated_120_reference_weak','foods':500,'reference_groups':120,'noise':'normal','q_logodds_shift':0},
                   {'name':'MNAR_unknown_odds_x4','foods':500,'reference_groups':640,'noise':'normal','q_logodds_shift':float(np.log(4))}],
      'null_family_gate':'one-sided Wilson95 upper <=.08, development check only',
      'component_coverage_gate':'.92 descriptive minimum; components dependent within repetition',
      'operational_effective_food_gate':50,'max_leverage_gate':.2,
      'probability_parameters':'estimated from distinct reference groups; grouped Bayesian-bootstrap uncertainty shared across FoodIDs',
      'reference_categories':20,'reference_probability_DGP':'Beta(2,5)',
      'probability_fit_shrinkage':10,'varying_unknown_width':'.03+.24*expit(.6*S_oral+.4*C+normal noise)',
      'q_logodds_shift':'true unresolved-food odds relative to reference; analysis assumes zero; MNAR sensitivity, not a gate',
      'scope':'frozen outcome-independent scores; correct conditional mean; learned-score pipeline NOT validated; missingness violation reported separately',
      'selection':'all fixed scenarios, no seed selection or outcome-driven actual model choice',
      'new_real_Y_reads':0,'real_activation':False}
sp=Q/'estimated_probability_simulation_spec_v1.json';assert not sp.exists()
sp.write_text(json.dumps(spec,indent=2)+'\n')
print('CATALOGUE_SIM_SPEC_SEALED',hashlib.sha256(sp.read_bytes()).hexdigest(),flush=True)
seeds=iter(np.random.SeedSequence(spec['seed']).spawn(10000))
def rng():return np.random.default_rng(next(seeds))
def wilson(k,n,upper=True):
    z=stats.norm.ppf(.95);v=k/n;den=1+z*z/n
    middle=(v+z*z/(2*n))/den;width=z*np.sqrt(v*(1-v)/n+z*z/(4*n*n))/den
    return float(middle+width if upper else middle-width)
def inputs(r,n,J,shift,nreference):
    c=r.normal(size=(n,1));scores=r.normal(size=(n,2));cats=spec['reference_categories']
    probabilities=r.beta(2,5,cats);foodcat=r.integers(cats,size=J)
    refcat=r.integers(cats,size=nreference);refy=(r.random(nreference)<probabilities[refcat]).astype(float)
    qdraw,_=grouped_probability_draws(np.arange(nreference).astype(str),refcat.astype(str),refy,np.ones(nreference),
        foodcat.astype(str),draws=spec['catalogue_draws'],seed=int(r.integers(2**32)),shrinkage=10)
    qtrue=special.expit(special.logit(probabilities[foodcat])+shift)
    v=(probabilities[foodcat]-probabilities.mean())/max(probabilities.std(),.05)+.5*r.normal(size=J)
    logits=.7*scores[:,0,None]*v[None,:];logits-=logits.max(axis=1,keepdims=True)
    selection=np.exp(logits);selection/=selection.sum(axis=1,keepdims=True)
    chosen=np.asarray([r.choice(J,3,replace=False,p=z) for z in selection])
    width=.03+.24*special.expit(.6*scores[:,0]+.4*c[:,0]+r.normal(size=n))
    portions=r.dirichlet([1.,1.,1.],n)*width[:,None];A=np.zeros((n,J))
    for k in range(3):A[np.arange(n),chosen[:,k]]=portions[:,k]
    lower=r.uniform(.02,.55,n);truth=(r.random(J)<qtrue).astype(float)
    d=lower+A@truth;xt,idx=point_design(d,c,scores)
    mean,draw,idx,meta=catalogue_draw_design(lower,A,qdraw,c,scores,draws=spec['catalogue_draws'],
        seed=int(r.integers(2**32)),center=.5,delta=.1)
    assert meta['q_estimation_uncertainty_included']
    projection=catalogue_projection(mean,draw)
    return c,scores,xt,mean,draw,projection,idx,meta
def beta(p,idx,gamma=0.):
    b=np.zeros(p);b[:4]=[.1,.2,.3,-.2];b[4]=.4;b[5]=.1;b[6]=.1;b[idx[0]]=gamma
    return b
reports=[]
for case in spec['scenarios']:
    start=time.monotonic();family_reject=unprotected_reject=covered=coverage_n=inactive=0
    coefficient_errors=[];eff=[];maxh=[]
    for rep in range(spec['null_repetitions']):
        r=rng();n=spec['n'];groups=np.arange(n)//2
        c,s,xt,x,draw,projection,idx,meta=inputs(r,n,case['foods'],case['q_logodds_shift'],case['reference_groups'])
        b=beta(x.shape[1],idx);mu=xt@b;eff.append(meta['effective_uncertain_FoodIDs_energy'])
        shared=r.normal(size=n//2);pvalues=[];naive_p=[]
        for endpoint in range(9):
            noise=.5*shared[groups]+np.sqrt(.75)*r.normal(size=n)
            if case['noise']=='heavy':noise=(.5*shared[groups]+np.sqrt(.45)*r.standard_t(5,n))*(1+.4*np.abs(c[:,0]))
            fit=fit_catalogue_calibration(x,draw,mu+noise,groups,projection=projection)
            maxh.append(fit['max_leverage'])
            if fit['max_leverage']>spec['max_leverage_gate']:
                pvalues.extend([1.,1.,1.]);naive_p.extend([1.,1.,1.]);inactive+=1;continue
            pvalues.extend(wald_block(fit,j)['p'] for j in [idx,[idx[0]],[idx[1]]])
            naive=dict(fit);naive['covariance']=fit['family_covariance']
            naive_p.extend(wald_block(naive,j)['p'] for j in [idx,[idx[0]],[idx[1]]])
            for j in idx:
                ci=contrast_interval(fit,np.eye(x.shape[1])[j]);covered+=ci['low']<=0<=ci['high'];coverage_n+=1
                coefficient_errors.append(float(fit['beta'][j]))
        pvalues.extend([1.]*28);naive_p.extend([1.]*28)
        assert len(pvalues)==55
        family_reject+=bool(np.any(holm_fixed_family(pvalues)<=.05))
        unprotected_reject+=bool(np.any(holm_fixed_family(naive_p)<=.05))
        if (rep+1)%100==0:print('CATALOGUE_NULL_PROGRESS',case['name'],rep+1,flush=True)
    report={'scenario':case,'repetitions':spec['null_repetitions'],
            'UPF_component_family_rejection_rate':family_reject/spec['null_repetitions'],
            'FWER_Wilson95_upper':wilson(family_reject,spec['null_repetitions']),
            'ignoring_shared_catalogue_family_rejection_rate':unprotected_reject/spec['null_repetitions'],
            'component_coverage':covered/coverage_n if coverage_n else None,
            'mean_interaction_error':float(np.mean(coefficient_errors)) if coefficient_errors else None,
            'interaction_error_RMSE':float(np.sqrt(np.mean(np.square(coefficient_errors)))) if coefficient_errors else None,
            'effective_uncertain_foods_median':float(np.median(eff)),
            'effective_foods_ge50_repetition_fraction':float(np.mean(np.asarray(eff)>=50)),
            'max_leverage_median':float(np.median(maxh)),'inactive_fit_count':inactive,'seconds':time.monotonic()-start,
            'coverage_MC_note':'component coverage descriptive; family rejection has independent repetition units'}
    report['correct_model_FWER_gate']=report['FWER_Wilson95_upper']<=.08 if case['q_logodds_shift']==0 else None
    report['correct_model_coverage_gate']=report['component_coverage'] is not None and report['component_coverage']>=.92 if case['q_logodds_shift']==0 else None
    reports.append(report);print('CATALOGUE_NULL_CALIBRATION',json.dumps(report),flush=True)
power=[]
for J,nref,shift in [(500,640,0.),(500,120,0.),(500,640,float(np.log(4)))]:
    rejected=covered=0;errors=[];widths=[]
    for rep in range(spec['power_repetitions']):
        r=rng();n=spec['n'];groups=np.arange(n)//2
        c,s,xt,x,draw,projection,idx,meta=inputs(r,n,J,shift,nref)
        b=beta(x.shape[1],idx,spec['power_gamma_per10pp']);y=xt@b+.5*r.normal(size=n//2)[groups]+np.sqrt(.75)*r.normal(size=n)
        fit=fit_catalogue_calibration(x,draw,y,groups,projection=projection)
        p=wald_block(fit,idx)['p'];rejected+=p<=.05/55
        ci=contrast_interval(fit,np.eye(x.shape[1])[idx[0]])
        covered+=ci['low']<=b[idx[0]]<=ci['high'];errors.append(float(fit['beta'][idx[0]]-b[idx[0]]));widths.append((ci['high']-ci['low'])/2)
    pr={'foods':J,'reference_groups':nref,'unknown_vs_reference_logodds_shift':shift,'repetitions':spec['power_repetitions'],'gamma_per10pp':spec['power_gamma_per10pp'],
        'joint_power_at_Bonferroni_family_threshold':rejected/spec['power_repetitions'],
        'power_Wilson95_lower':wilson(rejected,spec['power_repetitions'],False),
        'pointwise95_coverage':covered/spec['power_repetitions'],'mean_gamma_error':float(np.mean(errors)),
        'median_CI_halfwidth':float(np.median(widths))}
    power.append(pr);print('CATALOGUE_POWER_CALIBRATION',json.dumps(pr),flush=True)
result={'spec':spec,'null_scenarios':reports,'power':power,'new_real_Y_reads':0,
        'estimated_q_component_gates_passed':all(x['correct_model_FWER_gate'] and x['correct_model_coverage_gate'] for x in reports if x['scenario']['q_logodds_shift']==0),
        'production_ready':False,'fitted_q_component_uncertainty_evaluated':True,'full_W17_acceptance':False,
        'interpretation':'estimated-probability component calibration conditional on fixed scores; no real outcomes, full W17 or learned-score uncertainty acceptance'}
(Q/'estimated_probability_calibration_v1.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('CATALOGUE_CALIBRATION_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['spec','null_scenarios']}),flush=True)
