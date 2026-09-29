"""Same-person HC3 models and paired outcome differences with fixed shared scales."""
import numpy as np
import pandas as pd
from hppmeal.common import require
from hppmeal.models import bh_planned
from seven_diet_meals.inputs import BASE
from seven_diet_meals.models import encode, linear_fit, sample_hash
from response_followup.support import SETTINGS, OUTCOMES
from .metrics import METRICS

TARGETS = ['late_primary','full_comparator','late_minus_full']


def paired_sample(person, setting, measure, metric):
    d = person.loc[person.setting.eq(setting) & person.candidate_basic].copy()
    d = d.dropna(subset=['early_'+measure,'late_'+metric,'full_'+metric]+BASE)
    require(not d.participant_id.duplicated().any(), 'Duplicate person in model')
    return d.sort_values('participant_id').reset_index(drop=True)


def fit_target(d, measure, metric, target, scale):
    x = d['early_'+measure].to_numpy(float)
    X = encode(d, BASE)
    X['early_meal_z'] = (x-scale['x_mean'])/scale['x_sd']
    late = d['late_'+metric].to_numpy(float); full = d['full_'+metric].to_numpy(float)
    if target=='late_primary': y = (late-scale['y_mean'])/scale['y_sd']
    elif target=='full_comparator': y = (full-scale['y_mean'])/scale['y_sd']
    else: y = (late-full)/scale['y_sd']
    require(len(d)>=50,'Fewer than50 paired people')
    require(len(np.unique(x))>=3 and len(np.unique(y))>=3,'Fewer than3 distinct X/Y values')
    coef, info = linear_fit(y,X,'early_meal_z')
    focal = next(c for c in coef if c['term']=='early_meal_z')
    return focal, info, coef


def analyze(person, progress=print):
    rows, scales, coefficients, memberships = [], [], [], []
    for measure in OUTCOMES:
        for metric in METRICS:
            reference = paired_sample(person,'source_primary',measure,metric)
            scale = dict(meal_measure=measure,cgm=metric,reference='source_primary paired cohort; full-grid Y and early-meal X',
                reference_people=len(reference),reference_person_sha256=sample_hash(reference.participant_id),
                x_mean=reference['early_'+measure].mean(),x_sd=reference['early_'+measure].std(ddof=1),
                y_mean=reference['full_'+metric].mean(),y_sd=reference['full_'+metric].std(ddof=1))
            scales.append(scale)
            scale_ok = len(reference)>=50 and all(np.isfinite(scale[k]) for k in ['x_mean','x_sd','y_mean','y_sd']) and scale['x_sd']>1e-10 and scale['y_sd']>1e-10
            for setting in SETTINGS:
                d = paired_sample(person,setting,measure,metric)
                require(set(d.participant_id).issubset(set(reference.participant_id)),'Joint paired cohort not subset of reference')
                digest = sample_hash(d.participant_id)
                for pid in d.participant_id: memberships.append(dict(setting=setting,meal_measure=measure,cgm=metric,participant_id=pid))
                fitted = {}
                for target in TARGETS:
                    row = dict(setting=setting,meal_measure=measure,cgm=metric,target=target,people=len(d),
                        meals=int(d.early_meals.sum()),person_sha256=digest,status='NOT_ESTIMATED',reason='',
                        estimate=np.nan,se=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,q=np.nan,
                        x_reference_sd=scale['x_sd'],y_reference_sd=scale['y_sd'],
                        raw_y_per_reference_xSD=np.nan,raw_y_ci_low=np.nan,raw_y_ci_high=np.nan,
                        unit='new full-grid reference Y SD per reference SD of early personal meal measure',
                        family=target+'_24',planned_family_size=24)
                    try:
                        require(scale_ok,'Insufficient/degenerate source_primary reference scale')
                        coef, info, allcoef = fit_target(d,measure,metric,target,scale)
                        row.update(coef); row.update(info); row['status']='ESTIMATED'
                        row.update(raw_y_per_reference_xSD=row['estimate']*scale['y_sd'],
                            raw_y_ci_low=row['ci_low']*scale['y_sd'],raw_y_ci_high=row['ci_high']*scale['y_sd'])
                        fitted[target] = row['estimate']
                        coefficients.extend([dict(c,setting=setting,meal_measure=measure,cgm=metric,target=target) for c in allcoef])
                    except (ValueError,np.linalg.LinAlgError) as e:
                        row['reason'] = str(e)
                    rows.append(row)
                if len(fitted)==3:
                    require(np.isclose(fitted['late_minus_full'],fitted['late_primary']-fitted['full_comparator'],rtol=1e-8,atol=1e-9),'Paired-difference model does not equal coefficient difference')
                progress('MODELS {} {} {}: primary {}'.format(setting,measure,metric,rows[-3]['status']))
    result = pd.DataFrame(rows)
    for target in TARGETS:
        ix = result.target.eq(target)
        require(int(ix.sum())==24,'Incomplete planned24 family')
        result.loc[ix,'q'] = bh_planned(result.loc[ix,'p'])
    require(len(result)==72,'Incomplete24 primary+24 comparator+24 paired-difference rows')
    require(result.groupby(['setting','meal_measure','cgm']).person_sha256.nunique().eq(1).all(),'Comparators use different participants')
    cols = ['setting','meal_measure','cgm','target','term','estimate','se','ci_low','ci_high','p']
    return dict(models=result,scales=pd.DataFrame(scales),coefficients=pd.DataFrame(coefficients,columns=cols),
        model_members=pd.DataFrame(memberships,columns=['setting','meal_measure','cgm','participant_id']))
