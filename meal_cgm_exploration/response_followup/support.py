"""Outcome-blind support and linear transformations of frozen B interactions."""
import numpy as np
import pandas as pd
from hppmeal.common import require
from seven_diet_meals.inputs import DIETS
from seven_diet_meals.models import features,prepare,sample_hash

SETTINGS=['source_primary','joint']
OUTCOMES=['mean_increment','peak_rise_0_120','positive_mean']
MIN_ANCHOR_PEOPLE=50
MIN_CROSSING_PEOPLE=30
BAND_G=5.0


def weighted_quantile(values, weights, probs):
    x=np.asarray(values,float);w=np.asarray(weights,float)
    require(len(x)>0 and np.isfinite(x).all() and np.isfinite(w).all() and (w>0).all(),'Invalid weighted distribution')
    order=np.argsort(x,kind='mergesort');x=x[order];w=w[order]
    # Collapse ties, then interpolate mid-CDF. Weight per person sums to one.
    ux,inv=np.unique(x,return_inverse=True);uw=np.bincount(inv,weights=w)
    mid=(np.cumsum(uw)-uw/2)/uw.sum()
    return np.interp(probs,mid,ux,left=ux[0],right=ux[-1])


def quantiles(d,field,probs):
    w=1/d.groupby('participant_id').meal_id.transform('size').to_numpy(float)
    return weighted_quantile(d[field],w,probs)


def validate_members(sample,old_members,model_rows):
    require(not sample.meal_id.duplicated().any() and not old_members.meal_id.duplicated().any(),'Duplicate membership')
    require(set(zip(sample.participant_id,sample.meal_id))==set(zip(old_members.participant_id,old_members.meal_id)), 'Frozen participant-meal membership differs')
    ph=sample_hash(pd.Series(sample.participant_id.unique()));mh=sample_hash(sample.meal_id)
    for r in model_rows:
        require(r['person_sha256']==ph and r['meal_sha256']==mh,'Frozen model membership hash differs')
        require(int(r['people'])==sample.participant_id.nunique() and int(r['meals'])==len(sample),'Frozen model counts differ')
    return ph,mh


def support_for(sample,diet):
    person=sample.drop_duplicates('participant_id')
    row=dict(people=len(person),meals=len(sample),support_status='INSUFFICIENT_SUPPORT',reason='',
             score_low=np.nan,score_high=np.nan,energy_low=np.nan,energy_high=np.nan,carb_low=np.nan,carb_high=np.nan,
             common_carb_low=np.nan,common_carb_high=np.nan,
             low_score_lowC_people=np.nan,low_score_highC_people=np.nan,high_score_lowC_people=np.nan,high_score_highC_people=np.nan,
             low_score_crossers=np.nan,high_score_crossers=np.nan,
             low_score_energy_people=np.nan,high_score_energy_people=np.nan,
             low_score_energy_meals=np.nan,high_score_energy_meals=np.nan)
    descriptive=[]
    if sample.empty:
        row['reason']='Empty frozen sample';return row,descriptive,[]
    for field in ['carbohydrate_g','calories_kcal']:
        v=quantiles(sample,field,[.1,.25,.5,.75,.9])
        descriptive.append(dict(field=field,weight='equal_person_within_full_B_sample',p10=v[0],p25=v[1],median=v[2],p75=v[3],p90=v[4]))
    sv=np.quantile(person[diet],[.1,.25,.5,.75,.9]);row.update(score_low=sv[1],score_high=sv[3])
    descriptive.append(dict(field='original_diet_Z',weight='one_per_person',p10=sv[0],p25=sv[1],median=sv[2],p75=sv[3],p90=sv[4]))
    if row['score_high']<=row['score_low']:
        row['reason']='Score IQR is zero (ties); no arbitrary score contrast';return row,descriptive,[]
    groups=[sample.loc[sample[diet]<=row['score_low']].copy(),sample.loc[sample[diet]>=row['score_high']].copy()]
    energy=[quantiles(g,'calories_kcal',[.25,.75]) for g in groups]
    row['energy_low']=max(e[0] for e in energy);row['energy_high']=min(e[1] for e in energy)
    # A shared observed energy value is valid isocaloric support, even with zero interval width.
    if row['energy_high']<row['energy_low']:
        row['reason']='No overlap of group central meal energy IQRs';return row,descriptive,[]
    groups=[g.loc[g.calories_kcal.between(row['energy_low'],row['energy_high'])].copy() for g in groups]
    for label,g in zip(['low_score','high_score'],groups):
        row[label+'_energy_people']=g.participant_id.nunique();row[label+'_energy_meals']=len(g)
    if any(g.participant_id.nunique()<MIN_ANCHOR_PEOPLE or len(g)<100 for g in groups):
        row['reason']='Too few people/meals inside common energy band';return row,descriptive,[]
    carb=[quantiles(g,'carbohydrate_g',[.1,.9]) for g in groups]
    row['common_carb_low']=max(v[0] for v in carb);row['common_carb_high']=min(v[1] for v in carb)
    if row['common_carb_high']<=row['common_carb_low']:
        row['reason']='No overlap of central carbohydrate ranges';return row,descriptive,[]
    pooled=pd.concat(groups).loc[lambda x:x.carbohydrate_g.between(row['common_carb_low'],row['common_carb_high'])]
    if pooled.empty:
        row['reason']='No observed meals in common carbohydrate/energy support';return row,descriptive,[]
    row['carb_low'],row['carb_high']=quantiles(pooled,'carbohydrate_g',[.25,.75])
    if row['carb_high']-row['carb_low']<=2*BAND_G:
        row['reason']='Carbohydrate contrast <=10g; endpoint support bands would overlap';return row,descriptive,[]
    for label,g in zip(['low_score','high_score'],groups):
        hits=[]
        for clabel,value in [('lowC',row['carb_low']),('highC',row['carb_high'])]:
            band=g.loc[g.carbohydrate_g.between(max(row['common_carb_low'],value-BAND_G),min(row['common_carb_high'],value+BAND_G))]
            ids=set(band.participant_id);hits.append(ids)
            row[label+'_'+clabel+'_people']=len(ids);row[label+'_'+clabel+'_meals']=len(band)
        row[label+'_crossers']=len(hits[0]&hits[1])
    if min(row[k] for k in ['low_score_lowC_people','low_score_highC_people','high_score_lowC_people','high_score_highC_people'])<MIN_ANCHOR_PEOPLE:
        row['reason']='Fewer than 50 people in at least one score/endpoint cell'
    elif min(row['low_score_crossers'],row['high_score_crossers'])<MIN_CROSSING_PEOPLE:
        row['reason']='Fewer than 30 within-person crossers in at least one score group'
    else:row.update(support_status='SUPPORTED',reason='Common energy/carbohydrate support screen passed; not full covariate positivity proof')
    grid=[]
    for c in np.linspace(row['carb_low'],row['carb_high'],21):
        counts=[g.loc[g.carbohydrate_g.between(max(row['common_carb_low'],c-BAND_G),min(row['common_carb_high'],c+BAND_G))].participant_id.nunique() for g in groups]
        grid.append(dict(carb_g=float(c),low_score_people=counts[0],high_score_people=counts[1],grid_supported=min(counts)>=MIN_ANCHOR_PEOPLE))
    return row,descriptive,grid


def scale_interval(model,factor):
    require(np.isfinite(factor),'Nonfinite contrast multiplier')
    v=[float(model['ci_low'])*factor,float(model['ci_high'])*factor]
    return float(model['estimate'])*factor,min(v),max(v)


def analyze(meals,people,tables,progress=print):
    data=features(meals);supports=[];descriptions=[];contrasts=[];curves=[];checks=[]
    for setting in SETTINGS:
        pool=data if setting=='source_primary' else data.loc[data.joint]
        for diet in DIETS:
            sample,_=prepare(pool,people,diet,'primary_available','B')
            old=tables['members'].loc[lambda x:x.setting.eq(setting)&x.diet.eq(diet)&x.branch.eq('B')]
            rows=[]
            for outcome in OUTCOMES:
                table=tables['positive'] if outcome=='positive_mean' else tables['original']
                selected=table.loc[lambda x:x.setting.eq(setting)&x.diet.eq(diet)&x.branch.eq('B')&x.outcome.eq(outcome)&x.model.eq('carb_interaction')]
                require(len(selected)==1,'Missing/duplicate frozen B specification')
                rows.append(selected.iloc[0].to_dict())
            ph,mh=validate_members(sample,old,rows)
            checks.append(dict(setting=setting,diet=diet,people=sample.participant_id.nunique(),meals=len(sample),person_sha256=ph,meal_sha256=mh,status='MATCHED'))
            sup,desc,grid=support_for(sample,diet);sup.update(setting=setting,diet=diet);supports.append(sup)
            descriptions.extend([dict(r,setting=setting,diet=diet) for r in desc])
            for outcome,model in zip(OUTCOMES,rows):
                qcol='q' if outcome=='positive_mean' else 'q_family14'
                row=dict(setting=setting,diet=diet,outcome=outcome,people=model['people'],meals=model['meals'],
                    original_beta=model['estimate'],original_ci_low=model['ci_low'],original_ci_high=model['ci_high'],
                    original_q=model[qcol],q_family_size=21 if outcome=='positive_mean' else 14,
                    q_family='positive_21' if outcome=='positive_mean' else ('primary_14' if outcome=='mean_increment' else 'peak_secondary_14'),
                    coefficient_source='05_positive_models.csv' if outcome=='positive_mean' else '02_original_models.csv',
                    score_delta=sup['score_high']-sup['score_low'],carb_low=sup['carb_low'],carb_high=sup['carb_high'],
                    estimate=np.nan,ci_low=np.nan,ci_high=np.nan,unit='mg/dL difference in relative carbohydrate response changes',
                    status='NOT_REPORTED',reason=sup['reason'])
                if model['status']=='ESTIMATED' and sup['support_status']=='SUPPORTED':
                    factor=row['score_delta']*(row['carb_high']-row['carb_low'])/10
                    row['estimate'],row['ci_low'],row['ci_high']=scale_interval(model,factor)
                    row.update(status='SUPPORTED_CONTRAST',reason='Transformed frozen coefficient; pointwise CI conditional on observed support anchors; q retained')
                    for g in grid:
                        factor=row['score_delta']*(g['carb_g']-row['carb_low'])/10
                        beta,lo,hi=scale_interval(model,factor)
                        curves.append(dict(g,setting=setting,diet=diet,outcome=outcome,estimate=beta if g['grid_supported'] else np.nan,ci_low=lo if g['grid_supported'] else np.nan,ci_high=hi if g['grid_supported'] else np.nan))
                elif model['status']!='ESTIMATED':row['reason']='Frozen model not estimated: '+str(model.get('reason',''))
                contrasts.append(row)
            progress('SUPPORT {} {}: {}'.format(setting,diet,sup['support_status']))
    require(len(contrasts)==42 and len(supports)==14,'Incomplete prespecified follow-up')
    return dict(support=pd.DataFrame(supports),descriptive=pd.DataFrame(descriptions),contrasts=pd.DataFrame(contrasts),
                curves=pd.DataFrame(curves,columns=['setting','diet','outcome','carb_g','low_score_people','high_score_people','grid_supported','estimate','ci_low','ci_high']),membership=pd.DataFrame(checks))
