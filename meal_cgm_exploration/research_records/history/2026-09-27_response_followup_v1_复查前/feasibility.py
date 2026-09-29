"""Provisional disjoint-date counts, not newly fitted CGM associations."""
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd
from hppmeal.common import require,read_csv,norm
from hppmeal.cgm import prepare_connections
from nutrition_bridge.analysis import bridge_sample
from seven_diet_meals.models import features
from .support import SETTINGS,OUTCOMES


def bool_column(values,label):
    t=norm(values).str.lower()
    require(t.isin(['true','false','1','0']).all(),'Invalid boolean: '+label)
    return t.isin(['true','1'])


def split_boundary(t,timezone):
    require(len(t)>0 and np.isfinite(t).all(),'Missing/invalid connection timestamps')
    local=pd.to_datetime([t[0],t[-1]],unit='s',utc=True).tz_convert(timezone)
    first,last=[pd.Timestamp(x.date()) for x in local]
    span=(last-first).days+1
    cut=first+pd.Timedelta(days=int(np.ceil(span/2.0)))
    epoch=cut.tz_localize(timezone,ambiguous=False,nonexistent='shift_forward').timestamp()
    return str(cut.date()),epoch,span


def last_response_reading(t,start,post_minutes=120):
    # Exactly the frozen response() right endpoint convention, including interpolation bracket.
    rel=(np.asarray(t,float)-start)/60
    right=int(np.searchsorted(rel,post_minutes))
    require(right<len(t),'Original response endpoint no longer bracketed')
    return float(t[right])


def consecutive_support(t,g,good_dates,timezone,warmup):
    """Count same local 15-minute clock bins on adjacent days; NOT the iglu MODD algorithm."""
    local=pd.to_datetime(t,unit='s',utc=True).tz_convert(timezone)
    valid=np.isfinite(g)&(t>=t[0]+warmup*60)
    frame=pd.DataFrame({'date':local.strftime('%Y-%m-%d'),'bin':local.hour*4+local.minute//15,'valid':valid})
    bins={day:set(z.loc[z.valid,'bin'].astype(int)) for day,z in frame.loc[frame.date.isin(good_dates)].groupby('date')}
    pairs=0;total=0
    for day,items in bins.items():
        nextday=str((pd.Timestamp(day)+pd.Timedelta(days=1)).date())
        if nextday in bins:
            n=len(items & bins[nextday]);total+=n;pairs+=int(n>=48)
    return pairs,total


def person_windows(group,t,g,days,timezone,cgm_config):
    cut,epoch,span=split_boundary(t,timezone)
    reference=days.loc[days.reference_day]
    second=reference.loc[reference.local_date.ge(cut)]
    good=second.loc[second.coverage.ge(.9)]
    end_read=np.array([last_response_reading(t,x,cgm_config['post_minutes']) for x in group.start])
    early=group.start.to_numpy()<epoch
    # Frozen quality check establishes the original meal timezone; do not reinterpret it.
    aligned=group.meal_zone_ok.to_numpy(bool)
    x=group.loc[early&aligned&(end_read<epoch)].copy()
    allpositive=bool(len(x)>0 and x.positive_mean.notna().all())
    xok=len(x)>=5 and x.local_date.nunique()>=3
    yok=len(good)>=3
    pair_count,matched=consecutive_support(t,g,set(good.local_date),timezone,cgm_config['warmup_minutes'])
    latest=float(end_read[early&aligned&(end_read<epoch)].max()) if len(x) else np.nan
    require(not len(x) or latest<epoch,'Meal/CGM read overlap')
    row=dict(split_local_date=cut,split_epoch=epoch,calendar_span_days=span,
        first_reference_days=int(reference.local_date.lt(cut).sum()),second_reference_days=len(second),
        first_logged_reference_days=int(reference.loc[reference.local_date.lt(cut),'date_recorded'].sum()),
        second_logged_reference_days=int(second.date_recorded.sum()),second_good90_days=len(good),
        first_qualified_meals=len(x),first_meal_dates=x.local_date.nunique(),
        first_positive_complete=allpositive,first_meals_ok=xok,second_days_ok=yok,candidate_basic=bool(xok and yok),
        adjacent_day_pairs_48bins=pair_count,matched_clock_bins=matched,
        candidate_MODD_proxy=bool(xok and yok and pair_count>=2),
        excluded_meals_crossing_cut=int((early&aligned&(end_read>=epoch)).sum()),
        excluded_first_meals_timezone=int((early&~aligned).sum()),
        last_first_window_read=latest,no_overlap=True,
        dst_second_days=int(second.expected_bins.ne(96).sum()))
    return row,x


def analyze(run,paths,meals,people,cgm,tables,progress=print):
    data=features(meals);base={};members={}
    for setting in SETTINGS:
        p,_,m=bridge_sample(data,people,cgm,setting)
        old=tables['bridge_members'].loc[lambda x:x.setting.eq(setting)]
        require(set(zip(m.participant_id,m.meal_id,m.connection_id))==set(zip(old.participant_id,old.meal_id,old.connection_id)), 'Frozen bridge membership differs')
        members[setting]=m
        base[setting]=data.loc[data.meal_id.isin(m.meal_id)].copy()
    timing=read_csv(paths['quality_meals'])
    timing['meal_zone_ok']=bool_column(timing.meal_zone_ok,'meal_zone_ok')
    for setting in SETTINGS:
        base[setting]=base[setting].merge(timing[['meal_id','meal_zone_ok']],on='meal_id',validate='one_to_one')
    days=read_csv(paths['quality_days'])
    for f in ['reference_day','date_recorded']:
        days[f]=bool_column(days[f],f)
    for f in ['coverage','expected_bins']:
        days[f]=pd.to_numeric(days[f],errors='raise')
    require(not days.duplicated(['participant_id','connection_id','local_date']).any(),'Duplicate frozen daily coverage')
    ids=sorted(set().union(*[set(m.participant_id) for m in members.values()]))
    groups={s:{pid:z for pid,z in base[s].groupby('participant_id')} for s in SETTINGS}
    day_groups={(pid,cid):z for (pid,cid),z in days.groupby(['participant_id','connection_id'])}
    records=[];xmembers=[]
    timezone=run.c['time']['local_timezone']
    # No WAL/SHM sidecars; no SQL mutations, even temporary indices.
    with closing(sqlite3.connect(paths['cache'].as_uri()+'?mode=ro&immutable=1',uri=True)) as con:
        con.execute('PRAGMA query_only=ON')
        require(con.execute('PRAGMA quick_check').fetchone()[0]=='ok','Original cache integrity failure')
        for i,pid in enumerate(ids,1):
            allmeals=pd.concat([groups[s][pid] for s in SETTINGS if pid in groups[s]])
            cons=allmeals.connection_id.unique()
            require(len(cons)==1,'Bridge participant has multiple matched CGM connections')
            cid=str(cons[0])
            raw=pd.read_sql_query('SELECT connection_id,t,glucose,valid FROM cgm WHERE participant_id=? AND cohort=? AND research_stage=? AND connection_id=? ORDER BY t',con,
                                  params=[pid,run.c['cohort'],run.c['research_stage'],cid])
            connections,_=prepare_connections(raw)
            require(cid in connections and (pid,cid) in day_groups,'Matched original connection/date coverage unavailable')
            t,g=connections[cid]
            for setting in SETTINGS:
                if pid not in groups[setting]:continue
                row,x=person_windows(groups[setting][pid],t,g,day_groups[(pid,cid)],timezone,run.c['cgm'])
                row.update(participant_id=pid,connection_id=cid,setting=setting);records.append(row)
                xmembers.append(x[['participant_id','meal_id','connection_id']].assign(setting=setting))
            if i%100==0 or i==len(ids):progress('DISJOINT FEASIBILITY: {}/{} people checked'.format(i,len(ids)))
    person=pd.DataFrame(records)
    require(len(person)>0,'No original bridge members for feasibility')
    private_members=pd.concat(xmembers,ignore_index=True)
    summary=[];specs=[];balance=[];distributions=[]
    for setting in SETTINGS:
        a=person.loc[person.setting.eq(setting)]
        for stage,mask in [
            ('original_bridge_members',np.ones(len(a),bool)),('first_window_5meals_3days',a.first_meals_ok),
            ('second_window_3full_days_coverage90',a.second_days_ok),('both_basic_requirements',a.candidate_basic),
            ('both_and_positive_complete',a.candidate_basic&a.first_positive_complete),
            ('both_and_MODD_clock_proxy',a.candidate_MODD_proxy),
            ('both_MODD_proxy_and_positive',a.candidate_MODD_proxy&a.first_positive_complete)]:
            summary.append(dict(setting=setting,stage=stage,people=int(np.sum(mask))))
        for measure in OUTCOMES:
            for metric in ['Mean','CV','MODD','SDw']:
                mask=a.candidate_MODD_proxy if metric=='MODD' else a.candidate_basic
                if measure=='positive_mean':mask=mask&a.first_positive_complete
                specs.append(dict(setting=setting,meal=measure,cgm=metric,potential_people=int(mask.sum()),
                    status='PROVISIONAL_COUNTS_ONLY',definition_contract='WINDOW_RECOMPUTATION_NOT_YET_CONFIRMED'))
        for field in ['calendar_span_days','first_reference_days','second_reference_days','first_meal_dates','first_qualified_meals','second_good90_days','adjacent_day_pairs_48bins','first_logged_reference_days','second_logged_reference_days']:
            v=a[field].quantile([.1,.5,.9]);distributions.append(dict(setting=setting,field=field,p10=v.iloc[0],median=v.iloc[1],p90=v.iloc[2]))
        for field in ['excluded_meals_crossing_cut','excluded_first_meals_timezone','dst_second_days']:
            distributions.append(dict(setting=setting,field=field,total=int(a[field].sum()),affected_people=int(a[field].gt(0).sum())))
        old=tables['bridge_people'].loc[lambda x:x.setting.eq(setting)].copy()
        compare=old.merge(a[['participant_id','candidate_basic']],on='participant_id',validate='one_to_one')
        require(len(compare)==len(old)==len(a),'Selection comparator membership mismatch')
        for group,flag in [('candidate_retained',compare.candidate_basic),('excluded',~compare.candidate_basic)]:
            sub=compare.loc[flag]
            for field in ['age_years','bmi','meals','dates']+OUTCOMES:
                values=pd.to_numeric(sub[field],errors='raise')
                balance.append(dict(setting=setting,group=group,field='original_'+field,people=len(sub),valid=int(values.notna().sum()),mean=values.mean(),sd=values.std(),median=values.median()))
            for field in ['sex','cgm_device_type']:
                for value in sorted(compare[field].dropna().unique()):
                    balance.append(dict(setting=setting,group=group,field=field+'='+str(value),people=len(sub),valid=len(sub),mean=sub[field].eq(value).mean(),sd=np.nan,median=np.nan))
    return dict(feasibility=pd.DataFrame(summary),specifications=pd.DataFrame(specs),window_distributions=pd.DataFrame(distributions),selection=pd.DataFrame(balance),
                person_windows=person,first_window_members=private_members)
