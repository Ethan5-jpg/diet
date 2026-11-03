from pathlib import Path
import subprocess,sys
D=Path(__file__).parent
s=(D/'13_catalogue_model_calibration.py').read_text()
s=s.replace('from upf_catalogue_calibration import (','from upf_catalogue_calibration_v2 import (grouped_probability_draws,')
s=s.replace("(L/'12_actual_exit_code')","(L/'16_actual_exit_code')")
s=s.replace('shared_catalogue_calibration_v1','estimated_probability_calibration_v1').replace('20261030013','20261030117')
s=s.replace("{'name':'correct_100_foods_normal','foods':100,'noise':'normal','q_logodds_shift':0}","{'name':'estimated_640_reference_normal','foods':500,'reference_groups':640,'noise':'normal','q_logodds_shift':0}")
s=s.replace("{'name':'correct_500_foods_heavy_heteroskedastic','foods':500,'noise':'heavy','q_logodds_shift':0}","{'name':'estimated_640_reference_heavy','foods':500,'reference_groups':640,'noise':'heavy','q_logodds_shift':0}")
s=s.replace("{'name':'correct_20_foods_weak_catalogue','foods':20,'noise':'normal','q_logodds_shift':0}","{'name':'estimated_120_reference_weak','foods':500,'reference_groups':120,'noise':'normal','q_logodds_shift':0}")
s=s.replace("{'name':'misspecified_q_odds_x4','foods':100,'noise':'normal','q_logodds_shift':float(np.log(4))}","{'name':'MNAR_unknown_odds_x4','foods':500,'reference_groups':640,'noise':'normal','q_logodds_shift':float(np.log(4))}")
s=s.replace("'probability_parameters':'known in this experiment; fitted-q uncertainty NOT validated'", "'probability_parameters':'estimated from distinct reference groups; grouped Bayesian-bootstrap uncertainty shared across FoodIDs',\n      'reference_categories':20,'reference_probability_DGP':'Beta(2,5)',\n      'probability_fit_shrinkage':10,'varying_unknown_width':'.03+.24*expit(.6*S_oral+.4*C+normal noise)',\n      'q_logodds_shift':'true unresolved-food odds relative to reference; analysis assumes zero; MNAR sensitivity, not a gate'")
s=s.replace('catalogue_simulation_spec_v1.json','estimated_probability_simulation_spec_v1.json').replace('catalogue_calibration_v1.json','estimated_probability_calibration_v1.json')
a=s.index('def inputs(');b=s.index('def beta(',a)
s=s[:a]+'''def inputs(r,n,J,shift,nreference):
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
''' + s[b:]
s=s.replace("inputs(r,n,case['foods'],case['q_logodds_shift'])", "inputs(r,n,case['foods'],case['q_logodds_shift'],case['reference_groups'])")
s=s.replace('for J in [100,500]:','for J,nref,shift in [(500,640,0.),(500,120,0.),(500,640,float(np.log(4)))]:')
s=s.replace('inputs(r,n,J,0)','inputs(r,n,J,shift,nref)')
s=s.replace("pr={'foods':J,", "pr={'foods':J,'reference_groups':nref,'unknown_vs_reference_logodds_shift':shift,")
s=s.replace('known_q_component_gates_passed','estimated_q_component_gates_passed')
s=s.replace("'production_ready':False,'fitted_q_uncertainty_validated':False", "'production_ready':False,'fitted_q_component_uncertainty_evaluated':True")
s=s.replace("'scope':'correct conditional mean dictionary and non-differential classification; q misspecification reported separately'", "'scope':'frozen outcome-independent scores; correct conditional mean; learned-score pipeline NOT validated; missingness violation reported separately'")
s=s.replace("'interpretation':'conditional synthetic development evaluation; does not establish NOVA probabilities or validate real outcomes'", "'interpretation':'estimated-probability component calibration conditional on fixed scores; no real outcomes, full W17 or learned-score uncertainty acceptance'")
compile(s,'17_estimated_probability_calibration.py','exec')
(D/'17_estimated_probability_calibration.py').write_text(s)
subprocess.run([sys.executable,str(D/'prepare_transfer.py'),'17_estimated_probability_calibration.py','17'],check=True)
