"""Reproduce frozen cuts/membership and compute new window phenotypes read-only."""
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd
from hppmeal.common import require, read_csv
from hppmeal.cgm import prepare_connections
from seven_diet_meals.inputs import BASE
from seven_diet_meals.models import features
from nutrition_bridge.analysis import bridge_sample, safe_corr
from quality_timecourse.coverage import daily_coverage
from response_followup.feasibility import person_windows, bool_column
from response_followup.support import SETTINGS, OUTCOMES
from .metrics import METRICS, LEGACY, grid_days, summarize_window


def verify_window(current, old):
    for field, value in current.items():
        require(field in old, 'Missing prior window field: ' + field)
        if isinstance(value, str): ok = value == old[field]
        elif isinstance(value, (bool, np.bool_)): ok = value == old[field]
        else: ok = bool(np.isclose(value, old[field], rtol=0, atol=1e-5, equal_nan=True))
        require(ok, 'Frozen window/count changed: ' + field)


def build(run, paths, meals, people, cgm, tables, progress=print):
    cfg = run.c['cgm']; timezone = run.c['time']['local_timezone']
    require(cfg['sampling_minutes'] == 15 and cfg['post_minutes'] == 120, 'Requires frozen15-minute/120-minute contract')
    data = features(meals); pools = {}; start_checks = []
    timing = read_csv(paths['quality_meals']); timing['meal_zone_ok'] = bool_column(timing.meal_zone_ok,'meal_zone_ok')
    for setting in SETTINGS:
        _, _, m = bridge_sample(data, people, cgm, setting)
        old = tables['bridge_members'].loc[lambda x:x.setting.eq(setting)]
        cols = ['participant_id','meal_id','connection_id']
        require(set(map(tuple,m[cols].to_numpy())) == set(map(tuple,old[cols].to_numpy())), 'Original bridge assignment differs')
        d = data.loc[data.meal_id.isin(m.meal_id)].copy()
        d = d.merge(timing[['meal_id','meal_zone_ok']], on='meal_id', validate='one_to_one')
        pools[setting] = {pid:g for pid,g in d.groupby('participant_id')}
        start_checks.append(dict(setting=setting, original_bridge_people=m.participant_id.nunique(), original_meals=len(m)))
    require(set(pools['joint']).issubset(pools['source_primary']), 'Joint is not a subset of source bridge')
    days = read_csv(paths['quality_days'])
    for col in ['reference_day','date_recorded','full_day']: days[col] = bool_column(days[col], col)
    for col in ['coverage','expected_bins','valid_bins']: days[col] = pd.to_numeric(days[col], errors='raise')
    day_lookup = {key:d for key,d in days.groupby(['participant_id','connection_id'])}
    oldw = tables['prior_windows'].set_index(['setting','participant_id'])
    prior_members = tables['prior_first_members']
    require(set(oldw.index) == {(s,p) for s in SETTINGS for p in pools[s]}, 'Prior window participant set differs')
    expected_first = {(s,p):set(zip(d.meal_id,d.connection_id)) for (s,p),d in prior_members.groupby(['setting','participant_id'])}
    cgm_by = cgm.set_index('participant_id')
    persons = []; audit_days = []; first_members = []; current_flow = []; read_checks = []
    ids = sorted(pools['source_primary'])
    with closing(sqlite3.connect(paths['cache'].as_uri()+'?mode=ro&immutable=1',uri=True)) as con:
        con.execute('PRAGMA query_only=ON')
        require(con.execute('PRAGMA quick_check').fetchone()[0] == 'ok', 'Cache integrity failed')
        for i,pid in enumerate(ids,1):
            original = pools['source_primary'][pid]; connections = original.connection_id.unique()
            require(len(connections) == 1, 'Multiple source connections')
            cid = str(connections[0])
            raw = pd.read_sql_query('SELECT connection_id,t,glucose,valid FROM cgm WHERE participant_id=? AND cohort=? AND research_stage=? AND connection_id=? ORDER BY t',con,
                params=[pid,run.c['cohort'],run.c['research_stage'],cid])
            cons,_ = prepare_connections(raw); require(cid in cons, 'Frozen CGM connection unavailable')
            t,g = cons[cid]; dd = day_lookup[(pid,cid)]
            fresh = daily_coverage({cid:(t,g)},timezone,15,cfg['warmup_minutes']).set_index('local_date').sort_index()
            previous = dd.set_index('local_date').sort_index()
            require(fresh.index.equals(previous.index), 'Frozen coverage dates differ')
            for col in ['expected_bins','valid_bins','coverage','full_day']:
                require(np.allclose(fresh[col].to_numpy(float),previous[col].to_numpy(float),rtol=0,atol=1e-12), 'Frozen coverage differs: '+col)
            allowed = set(dd.loc[dd.reference_day & dd.coverage.ge(.9),'local_date'])
            current_original,_ = person_windows(original,t,g,dd,timezone,cfg)
            cut = current_original['split_local_date']; epoch = current_original['split_epoch']
            grid_audit, grids = grid_days(t,g,allowed,timezone,cfg['warmup_minutes'])
            late_grids = {day:v for day,v in grids.items() if day >= cut}
            late, whole = summarize_window(late_grids), summarize_window(grids)
            late_days = [d for d in grid_audit if d['local_date'] >= cut and d['grid_day_eligible']]
            earliest = min((d['first_support'] for d in late_days),default=np.nan)
            require(not late_days or earliest >= epoch, 'Late CGM uses a pre-cut reading')
            for d in grid_audit: audit_days.append(dict(d,participant_id=pid,connection_id=cid,window='late' if d['local_date']>=cut else 'early'))
            for setting in SETTINGS:
                if pid not in pools[setting]: continue
                group = pools[setting][pid]
                require(set(group.connection_id)=={cid}, 'Setting connection differs')
                w,x = person_windows(group,t,g,dd,timezone,cfg)
                verify_window(w,oldw.loc[(setting,pid)])
                require(set(zip(x.meal_id,x.connection_id)) == expected_first.get((setting,pid),set()), 'Frozen first-window meal assignment differs')
                if len(x): first_members.append(x[['participant_id','meal_id','connection_id']].assign(setting=setting))
                require(not len(x) or not late_days or w['last_first_window_read'] < earliest, 'Early meal and late grid read overlap')
                row = dict(participant_id=pid,connection_id=cid,setting=setting,**w)
                row.update(people.loc[pid,BASE].to_dict())
                row.update(early_meals=len(x),early_meal_days=x.local_date.nunique(),late_first_support=earliest,
                    early_meals_on_full_grid_dates=int(x.local_date.isin(grids).sum()),
                    legacy_recipe_equivalence='NOT_ASSUMED_NEW_GRID_RECIPE')
                for measure in OUTCOMES:
                    row['early_'+measure] = float(x[measure].mean()) if len(x) and x[measure].notna().all() else np.nan
                    row['original_'+measure] = float(group[measure].mean()) if group[measure].notna().all() else np.nan
                for metric in METRICS:
                    row['late_'+metric] = late[metric]; row['full_'+metric] = whole[metric]
                    row['legacy_'+metric] = cgm_by.loc[pid,LEGACY[metric]]
                for label,v in [('late',late),('full',whole)]:
                    for field in ['days','finite_points','MODD15_pairs','MODD15_matches','dst_days','basic_reason','MODD15_reason']:
                        row[label+'_'+field] = v[field]
                persons.append(row)
                read_checks.append(dict(setting=setting,participant_id=pid,connection_id=cid,cut_epoch=epoch,
                    early_last_read=w['last_first_window_read'],late_first_read=earliest,no_shared_reading=True))
            if i % 100 == 0 or i == len(ids): progress('WINDOW REBUILD: {}/{} people'.format(i,len(ids)))
    person = pd.DataFrame(persons)
    for setting in SETTINGS:
        d = person.loc[person.setting.eq(setting)]
        masks = dict(original_bridge_members=np.ones(len(d),bool),first_window_5meals_3days=d.first_meals_ok,
            second_window_3full_days_coverage90=d.second_days_ok,both_basic_requirements=d.candidate_basic,
            both_and_positive_complete=d.candidate_basic & d.first_positive_complete,both_and_MODD_clock_proxy=d.candidate_MODD_proxy,
            both_MODD_proxy_and_positive=d.candidate_MODD_proxy & d.first_positive_complete)
        for name,mask in masks.items():
            prior = tables['prior_flow'].loc[lambda z:z.setting.eq(setting)&z.stage.eq(name)]
            require(len(prior)==1 and int(prior.iloc[0].people)==int(np.sum(mask)), 'Prior feasibility summary count differs')
            current_flow.append(dict(setting=setting,stage=name,people=int(np.sum(mask)),status='MATCHED'))
    require(len(current_flow)==14, 'Incomplete window count verification')
    return dict(person_features=person,daily_grid_audit=pd.DataFrame(audit_days),
        first_members=pd.concat(first_members,ignore_index=True) if first_members else pd.DataFrame(columns=['participant_id','meal_id','connection_id','setting']),
        read_separation=pd.DataFrame(read_checks),prior_count_checks=pd.DataFrame(current_flow),original_membership_checks=pd.DataFrame(start_checks))


def descriptive(person):
    flow, concordance, selection = [], [], []
    for setting in SETTINGS:
        d = person.loc[person.setting.eq(setting)]
        for metric in METRICS:
            keep = d.candidate_basic & d['late_'+metric].notna() & d['full_'+metric].notna()
            flow.append(dict(setting=setting,cgm=metric,original_people=len(d),prior_basic=int(d.candidate_basic.sum()),
                prior_MODD_proxy=int(d.candidate_MODD_proxy.sum()),metric_ready_people=int(keep.sum()),
                first_meals_p50=d.loc[keep,'early_meals'].median(),late_days_p50=d.loc[keep,'late_days'].median(),
                late_pairs_p50=d.loc[keep,'late_MODD15_pairs'].median(),late_grid_points_p50=d.loc[keep,'late_finite_points'].median()))
            pair = d[['full_'+metric,'legacy_'+metric]].dropna()
            concordance.append(dict(setting=setting,cgm=metric,people=len(pair),
                spearman=safe_corr(pair.iloc[:,0],pair.iloc[:,1]),pearson=safe_corr(pair.iloc[:,0],pair.iloc[:,1],'pearson'),
                comparison='new full-grid raw metric vs old full-period cleaned Z; correlation does not prove equality'))
            for name,flag in [('retained',keep),('excluded',~keep)]:
                sub = d.loc[flag]
                for field in ['age_years','bmi','early_meals','early_meal_days']+['original_'+m for m in OUTCOMES]:
                    v = sub[field]
                    selection.append(dict(setting=setting,cgm=metric,group=name,field=field,people=len(sub),valid=int(v.notna().sum()),mean=v.mean(),median=v.median(),sd=v.std()))
    return dict(flow=pd.DataFrame(flow),legacy_concordance=pd.DataFrame(concordance),selection=pd.DataFrame(selection))
