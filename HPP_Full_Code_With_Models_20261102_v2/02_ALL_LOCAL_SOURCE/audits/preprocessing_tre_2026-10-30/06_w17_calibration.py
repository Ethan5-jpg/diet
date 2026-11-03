"""Precommitted synthetic calibration; no real outcomes or fitted real scores."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib, json, time
import numpy as np
from scipy import stats
from w16_estimators import ols_sandwich,wald_block,contrast_interval,daily_fixed_effects,holm_fixed_family
from w17_interval_upf_prototype import design_ranges,moment_ranges,moment_summary,confidence_bands,null_feasibility

P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
Q=P/'data_qc/preprocessing_20261030_v1'
L=Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'05_actual_exit_code').read_text().strip()=='0'
spec={'version':'W17_calibration_v1','seed':20261030017,'point_repetitions':400,
      'interval_complete_family_repetitions':100,'nominal_family_alpha':.05,
      'candidate_complete_family_positions':55,'formal_daily':'fiber only; UPF daily not activated',
      'point_scenarios':['nonlinear_main','host_interaction_only','correlated_sites','selection_heteroskedasticity'],
      'interval_widths_fraction':[0.,.03,.15],'point_n':800,'interval_n':700,
      'family_rate_gate':'one-sided Wilson95 upper <=0.08 (development tolerance, not exact-level proof)',
      'coverage_gate':'descriptive component coverage >=0.90; within-repetition components are dependent',
      'interval_power_repetitions':100,'interval_power_effect_per10pp':0.30,
      'interval_null_test':'conservative raw p either alpha/55 after outer-set exclusion, or 1',
      'LP_max_nodes':1024,'LP_time_limit_seconds_per_null':.25,
      'computation_failure':'return p=1; record uncertainty, never rejection',
      'high_leverage_gate':.2,'real_scientific_Y_reads':0,
      'clinical_minimum_effect':'not supplied; standardized precision is not a clinical threshold',
      'selection_rule':'report all fixed scenarios; no favorable seed selection or unlogged rerun'}
spec_path=Q/'simulation_spec_v1.json'
assert not spec_path.exists()
spec_path.write_text(json.dumps(spec,indent=2)+'\n')
print('W17_SPEC_SEALED',hashlib.sha256(spec_path.read_bytes()).hexdigest(),flush=True)
root=np.random.SeedSequence(spec['seed']);seeds=iter(root.spawn(10000))
def rng():return np.random.default_rng(next(seeds))
def wilson(k,n,upper=True):
    z=stats.norm.ppf(.95);v=k/n;den=1+z*z/n
    middle=(v+z*z/(2*n))/den
    width=z*np.sqrt(v*(1-v)/n+z*z/(4*n*n))/den
    return float(middle+width if upper else middle-width)

def person_design(r,n,case):
    families=np.arange(n)//2
    c=r.normal(size=n);oral=.6*c+np.sqrt(.64)*r.normal(size=n)
    rho=.95 if case=='correlated_sites' else .3
    gut=rho*oral+np.sqrt(1-rho*rho)*r.normal(size=n)
    d=r.normal(size=n)
    if case=='selection_heteroskedasticity':d=(r.chisquare(3,n)-3)/np.sqrt(6)
    x=np.column_stack([np.ones(n),d,d*d,d**3,c,oral,gut,d*c,d*oral,d*gut])
    mean=.3*d+.2*d*d+.25*c+.3*oral-.2*gut
    if case=='nonlinear_main':mean+=.2*oral**2
    if case=='host_interaction_only':mean+=.5*d*c
    if case=='selection_heteroskedasticity':
        selected=d>-.5
        x,mean,families,d=x[selected],mean[selected],families[selected],d[selected]
    return x,mean,families,d

def daily_null(r):
    people=180;counts=r.integers(3,13,people);ids=np.repeat(np.arange(people),counts)
    n=len(ids);family=ids//2
    d=r.normal(size=n);host=r.normal(size=people)[ids]
    so=r.normal(size=people)[ids];sg=r.normal(size=people)[ids]
    x=np.column_stack([d,d*d,d**3,d*host,d*so,d*sg])
    ar=[]
    shared=r.normal(scale=.4,size=(people//2,12))
    for i,k in enumerate(counts):
        e=r.normal(size=k)
        for t in range(1,k):e[t]=.6*e[t-1]+np.sqrt(1-.36)*e[t]
        ar.extend(e+shared[i//2,:k])
    y=.3*d+.2*d*d+.3*d*host+r.normal(size=people)[ids]+np.asarray(ar)
    fit=daily_fixed_effects(x,y,ids,family)
    return wald_block(fit,[4,5])['p']

point_reports=[]
for case in spec['point_scenarios']:
    rejects=covered=coverage_n=inactive=0;widths=[]
    for rep in range(spec['point_repetitions']):
        r=rng();x,mean,groups,d=person_design(r,spec['point_n'],case)
        shared=r.normal(size=len(np.unique(groups)))
        _,inv=np.unique(groups,return_inverse=True)
        pvals=[]
        for endpoint in range(9):
            noise=.5*shared[inv]+np.sqrt(.75)*r.normal(size=len(d))
            if case=='selection_heteroskedasticity':noise*=1+.3*np.abs(d)
            fit=ols_sandwich(x,mean+noise,clusters=groups)
            if fit['max_leverage']>spec['high_leverage_gate']:
                pvals.extend([1.,1.,1.]);inactive+=1;continue
            pvals.extend(wald_block(fit,idx)['p'] for idx in [[8,9],[8],[9]])
            for j in [8,9]:
                ci=contrast_interval(fit,np.eye(x.shape[1])[j])
                covered+=ci['low']<=0<=ci['high'];coverage_n+=1
                widths.append((ci['high']-ci['low'])/2)
        pvals.extend([1.]*27)  # interval-UPF slots remain occupied in this component check
        pvals.append(daily_null(r))
        assert len(pvals)==55
        rejects+=bool(np.any(holm_fixed_family(pvals)<=.05))
    report={'scenario':case,'repetitions':spec['point_repetitions'],'FWER':rejects/spec['point_repetitions'],
            'FWER_Wilson95_upper':wilson(rejects,spec['point_repetitions']),
            'coverage':covered/coverage_n,'coverage_Wilson95_lower':wilson(covered,coverage_n,False),
            'CI_halfwidth_median':float(np.median(widths)),'inactive_endpoint_fits':inactive,
            'coverage_MC_units_note':'component coverage correlated within repetition; binomial limit descriptive only'}
    # Use independent per-repetition family results as the inferential-error development gate.
    report['passes_FWER_gate']=report['FWER_Wilson95_upper']<=.08
    report['passes_descriptive_coverage_gate']=report['coverage']>=.90
    point_reports.append(report)
    print('POINT_CALIBRATION',json.dumps(report),flush=True)

# A known misspecification counterexample: scientific no-interaction != zero projection coefficient.
miss=[]
rho=.6;truth=2*rho/(1+rho*rho)
for rep in range(200):
    r=rng();n=1000;d=r.normal(size=n);s=rho*d+np.sqrt(1-rho*rho)*r.normal(size=n)
    x=np.column_stack([np.ones(n),d,s,d*s]);y=d*d+r.normal(size=n)
    f=ols_sandwich(x,y);ci=contrast_interval(f,[0,0,0,1])
    miss.append([wald_block(f,[3])['p']<.05,ci['low']<=truth<=ci['high']])
miss_report={'projection_interaction_truth':truth,'scientific_zero_rejection_rate':float(np.mean(np.array(miss)[:,0])),
             'projection_truth_coverage':float(np.mean(np.array(miss)[:,1])),
             'interpretation':'expected scientific misinterpretation under misspecified main effect, not calibration against a false projection null'}
print('MISSPECIFICATION_COUNTEREXAMPLE',json.dumps(miss_report),flush=True)

interval_reports=[]
for width in spec['interval_widths_fraction']:
    rejects=indeterminate=excluded=total_nodes=0;start=time.monotonic()
    for rep in range(spec['interval_complete_family_repetitions']):
        r=rng();n=spec['interval_n'];groups=np.arange(n)//2
        c=r.normal(size=(n,1));scores=r.normal(size=(n,2));latent=r.uniform(.1,.9,n)
        yshared=r.normal(size=n//2)
        pvals=[]
        for endpoint in range(9):
            y=.4*latent+.2*c[:,0]+.3*scores[:,0]-.2*scores[:,1]+.5*yshared[groups]+r.normal(size=n)
            # Bounds may depend on Y: containment is required, MAR is not assumed.
            location=1/(1+np.exp(-np.clip(y,-20,20)))
            lower=np.maximum(0,latent-width*location)
            upper=np.minimum(1,latent+width*(1-location))
            xl,xu,interaction=design_ranges(lower,upper,c,scores,center=.5,delta=.1,nonlinear=True)
            ml,mu,pairs=moment_ranges(xl,xu,y)
            summary=moment_summary(ml,mu,groups)
            bands=confidence_bands(summary,pairs,xl.shape[1],.05/55)
            for null in [interaction,[interaction[0]],[interaction[1]]]:
                outcome=null_feasibility(*bands,null,max_nodes=spec['LP_max_nodes'],time_limit=spec['LP_time_limit_seconds_per_null'])
                total_nodes+=outcome['nodes'];excluded+=outcome['null_excluded']
                indeterminate+=outcome['status'].startswith('inconclusive')
                pvals.append(.05/55 if outcome['null_excluded'] else 1.)
            # A separate scalar fiber-like exposure, with the same frozen score interface.
            f=r.normal(size=n)
            x=np.column_stack([np.ones(n),f,f*f,f**3,c[:,0],scores,f*c[:,0],f[:,None]*scores])
            fit=ols_sandwich(x,y,clusters=groups)
            pvals.extend(wald_block(fit,idx)['p'] for idx in [[8,9],[8],[9]])
        pvals.append(daily_null(r));assert len(pvals)==55
        rejects+=bool(np.any(holm_fixed_family(pvals)<=.05))
        if (rep+1)%25==0:print('INTERVAL_PROGRESS',width,rep+1,flush=True)
    report={'interval_width_fraction':width,'repetitions':spec['interval_complete_family_repetitions'],
            'complete_55_position_FWER':rejects/spec['interval_complete_family_repetitions'],
            'FWER_Wilson95_upper':wilson(rejects,spec['interval_complete_family_repetitions']),
            'UPF_null_exclusions':excluded,'computational_inconclusive':indeterminate,'LP_nodes':total_nodes,
            'seconds':time.monotonic()-start}
    report['passes_FWER_gate']=report['FWER_Wilson95_upper']<=.08
    interval_reports.append(report)
    print('INTERVAL_CALIBRATION',json.dumps(report),flush=True)

# Prespecified strong-alternative stress check. These are synthetic, not clinical effect sizes.
power_reports=[]
for width in spec['interval_widths_fraction']:
    rejected=inconclusive=0
    for rep in range(spec['interval_power_repetitions']):
        r=rng();n=spec['interval_n'];groups=np.arange(n)//2
        c=r.normal(size=(n,1));scores=r.normal(size=(n,2));latent=r.uniform(.1,.9,n)
        y=.4*latent+.2*c[:,0]+.3*scores[:,0]-.2*scores[:,1]
        y+=spec['interval_power_effect_per10pp']*(latent-.5)/.1*scores[:,0]
        y+=.5*r.normal(size=n//2)[groups]+r.normal(size=n)
        location=1/(1+np.exp(-np.clip(y,-20,20)))
        lower=np.maximum(0,latent-width*location);upper=np.minimum(1,latent+width*(1-location))
        xl,xu,interaction=design_ranges(lower,upper,c,scores,center=.5,delta=.1,nonlinear=True)
        ml,mu,pairs=moment_ranges(xl,xu,y)
        bands=confidence_bands(moment_summary(ml,mu,groups),pairs,xl.shape[1],.05/55)
        result=null_feasibility(*bands,interaction,max_nodes=spec['LP_max_nodes'],time_limit=spec['LP_time_limit_seconds_per_null'])
        rejected+=result['null_excluded'];inconclusive+=result['status'].startswith('inconclusive')
    pr={'interval_width_fraction':width,'repetitions':spec['interval_power_repetitions'],
        'strong_effect_per10pp':spec['interval_power_effect_per10pp'],
        'Bonferroni_family_threshold_rejection_rate':rejected/spec['interval_power_repetitions'],
        'power_Wilson95_lower':wilson(rejected,spec['interval_power_repetitions'],False),
        'computational_inconclusive':inconclusive}
    power_reports.append(pr);print('INTERVAL_POWER_STRESS_CHECK',json.dumps(pr),flush=True)

# Precision comparison under explicit assumptions, not a claim about clinical relevance or true signals.
precision=[]
for n in [400,800,1200,1600,2000]:
    widths=[]
    for rep in range(100):
        r=rng();x,mean,groups,d=person_design(r,n,'correlated_sites')
        y=mean+r.normal(size=len(mean));fit=ols_sandwich(x,y,clusters=groups)
        ci=contrast_interval(fit,np.eye(x.shape[1])[8]);widths.append((ci['high']-ci['low'])/2*1.349)
    precision.append({'confirmation_n':n,'site_score_correlation':.95,'score_contrast':'normal IQR=1.349',
                      'Y_scale':'synthetic residual SD=1; not a clinical threshold',
                      'median_pointwise95_halfwidth':float(np.median(widths))})
report={'spec':spec,'point_calibration':point_reports,'misspecification':miss_report,
        'interval_calibration':interval_reports,'precision_comparison':precision,
        'all_development_gates_passed':all(x['passes_FWER_gate'] and x['passes_descriptive_coverage_gate'] for x in point_reports)
                                     and all(x['passes_FWER_gate'] for x in interval_reports),
        'interval_power_stress_checks':power_reports,'interval_power_validated':False,'real_scientific_Y_reads':0,
        'interpretation':'synthetic calibration only; no real-data efficacy or independent statistical review claimed'}
(Q/'W17_calibration_v1.json').write_text(json.dumps(report,indent=2)+'\n')
print('W17_CALIBRATION_REPORT',json.dumps({k:v for k,v in report.items() if k not in ['spec','point_calibration','interval_calibration']}),flush=True)
