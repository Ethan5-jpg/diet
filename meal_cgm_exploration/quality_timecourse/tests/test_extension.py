"""Synthetic edge cases and independent identities; no real participant data."""
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from hppmeal.common import Run, sha256, write_json
from hppmeal.cgm import prepare_connections, match_meal
from seven_diet_meals.tests.test_analysis import synthetic, make_source
from seven_diet_meals.models import run_models as original_models
from seven_diet_meals.inputs import PACKAGE
from seven_diet_meals.run_analysis import main as seven_main
from quality_timecourse.coverage import daily_coverage, split_curve, attach_quality, DAY_COLS, TIME_COLS, build
from quality_timecourse.inputs import load_diary, inspect, bundle
from quality_timecourse.analysis import run_models, reproduce, adjusted_comparison
from quality_timecourse.run_analysis import main


def test_split_piecewise_linear_exact_independent_integral():
    # y-base = t/10, so means are 3 and9 even with no observation exactly at60.
    x = np.array([0, 13, 29, 47, 63, 80, 101, 120.])
    early, late, diff = split_curve((x, 100 + x/10, 100))
    assert (early, late, diff) == pytest.approx((3, 9, 6))
    assert (early + late) / 2 == pytest.approx(np.trapz(x/10, x)/120)


@pytest.mark.parametrize('date,bins', [('2024-03-31', 92), ('2024-10-27', 100), ('2024-04-02', 96)])
def test_dst_day_denominator(date, bins):
    a = pd.Timestamp(date, tz='Europe/Berlin')
    b = (pd.Timestamp(date) + pd.Timedelta(days=1)).tz_localize('Europe/Berlin')
    t = np.arange(a.timestamp(), b.timestamp(), 900)
    d = daily_coverage({'sensor': (t, np.ones(len(t))*100)}, 'Europe/Berlin')
    assert len(d) == 1 and d.expected_bins.iloc[0] == bins
    assert d.full_day.iloc[0] and d.coverage.iloc[0] == 1


def test_internal_absent_date_kept_and_boundaries_excluded():
    a = pd.Timestamp('2024-01-01', tz='UTC').timestamp()
    t = np.r_[np.arange(a+3600, a+86400, 900), np.arange(a+2*86400, a+3*86400-3600, 900)]
    d = daily_coverage({'s': (t, np.ones(len(t))*100)}, 'UTC')
    assert d.full_day.tolist() == [False, True, False]
    assert d.valid_bins.tolist()[1] == 0
    assert d.coverage.tolist()[1] == 0


def test_duplicate_conflict_and_slot_counts_not_inflated():
    a = pd.Timestamp('2024-01-01', tz='UTC').timestamp()
    t = np.arange(a, a+86400, 900)
    raw = pd.DataFrame(dict(connection_id='s', t=t, glucose=100., valid=1))
    raw = pd.concat([raw, raw.iloc[[0]], raw.iloc[[1]].assign(glucose=200), raw.iloc[[2]].assign(t=a+1801)], ignore_index=True)
    conn, stats = prepare_connections(raw)
    d = daily_coverage(conn, 'UTC')
    assert d.valid_bins.iloc[0] == 95
    assert stats[0]['conflicting_timepoints'] == 1


def attach_fixture():
    dates = pd.date_range('2024-01-01', periods=10).strftime('%Y-%m-%d')
    days = pd.DataFrame([dict(participant_id=p, connection_id=p+'s', local_date=d, full_day=True, expected_bins=96, valid_bins=96, coverage=1.) for p in ['001','002'] for d in dates])
    diary = pd.DataFrame([dict(participant_id='001', local_date=d, food_items=3, diary_zone='UTC') for d in dates[:8]])
    meals = pd.DataFrame([dict(participant_id=p, meal_id=p+str(i), connection_id=p+'s', local_date=d) for p in ['001','002'] for i,d in enumerate(dates)])
    times = pd.DataFrame([dict(meal_id=m, meal_zone_ok=True, early_mean=1.,late_mean=2.,late_minus_early=1.) for m in meals.meal_id])
    return meals, days, times, diary


def test_unknown_diary_not_zero_intake_and_log80_boundary():
    meals, days, times, diary = attach_fixture()
    d, day, people = attach_quality(meals, days, times, diary, 'UTC')
    info=people.set_index('participant_id')
    assert info.loc['001','logging_fraction'] == .8
    assert info.loc['001','log80_person']
    assert np.isnan(info.loc['002','logging_fraction'])
    assert d.joint.sum()==8 and not d.loc[d.participant_id.eq('002'),'aligned'].any()
    assert len(d)==20 # No fabricated meals, no deleted original observations.


def test_overlapping_connection_day_excluded_without_pooling():
    meals, days, times, diary = attach_fixture()
    days = pd.concat([days, days.iloc[[0]].assign(connection_id='other')])
    d, day, people = attach_quality(meals, days, times, diary, 'UTC')
    assert not d.loc[d.meal_id.eq('0010'),'aligned'].iloc[0]
    assert people.set_index('participant_id').loc['001','reference_days']==9


def test_timezone_mismatch_excludes_day_but_keeps_source():
    meals, days, times, diary = attach_fixture()
    diary.loc[0,'diary_zone']='Asia/Jerusalem'
    d, _, _ = attach_quality(meals, days, times, diary, 'UTC')
    assert d.source_primary.all() and not d.loc[0,'aligned']


def augmented_fixture(n=90):
    meals, people = synthetic(n, meals_each=18)
    meals['source_primary']=True
    meals['diary_person_present']=True
    meals['diary_date_present']=True
    meals['actual_diary_dates']=6
    for s in ['aligned','cgm90','log80','joint']:
        meals[s]=True
    # Selected people lose their first date; excluded people lose every date.
    excluded=meals.participant_id.isin(['000000','000001','000002'])
    for s in ['aligned','cgm90','log80','joint']:
        meals.loc[excluded | meals.local_date.eq('2024-01-01'),s]=False
    meals['early_mean']=meals.mean_increment + .6*meals.participant_id.map(people.AMED) + np.sin(meals.start)
    meals['late_mean']=2*meals.mean_increment-meals.early_mean
    meals['late_minus_early']=meals.late_mean-meals.early_mean
    previous=original_models(meals,people,lambda _:None)[0]
    return meals, people, previous


def test_fixed_families_memberships_and_linear_time_contrast():
    meals, people, previous=augmented_fixture()
    result=run_models(meals,people,previous,lambda _:None)
    assert len(result['models'])==252 and len(result['timecourse'])==42
    assert result['models'].status.eq('ESTIMATED').all()
    assert result['timecourse'].status.eq('ESTIMATED').all()
    joint=result['models'].query("setting=='joint'").set_index(['diet','branch','outcome','model'])
    matched=result['models'].query("setting=='joint_people_all_meals'").set_index(['diet','branch','outcome','model'])
    assert joint.person_sha256.equals(matched.person_sha256)
    assert (joint.meals < matched.meals).all()
    for _, d in result['timecourse'].groupby(['setting','diet']):
        b=d.set_index('outcome').estimate
        assert b.late_minus_early==pytest.approx(b.late_mean-b.early_mean,abs=1e-10)
    # Check original feature origin is not reset after first-day filtering.
    original=result['models'].query("setting=='source_primary'")
    assert len(reproduce(original,previous))==42
    assert all(result['models'].groupby(['setting','family']).size()==14)
    assert all(result['timecourse'].groupby('setting').size()==21)


def test_empty_quality_samples_keep_all_specs_and_headers():
    meals,people,previous=augmented_fixture(n=12)
    for s in ['aligned','cgm90','log80','joint']:
        meals[s]=False
    result=run_models(meals,people,previous,lambda _:None)
    assert len(result['models'])==252 and result['models'].status.eq('NOT_ESTIMATED').all()
    assert len(result['timecourse'])==42
    assert 'term' in result['coefficients'] and 'term' in result['time_coefficients']
    assert result['joint_membership'].empty and 'meal_id' in result['joint_membership']


def test_reproduction_rejects_changed_members_and_coefficient():
    meals,people,previous=augmented_fixture(n=12)
    current=previous.query("setting=='primary_available'").copy()
    current['setting']='source_primary'
    current.loc[current.index[0],'person_sha256']='wrong'
    with pytest.raises(ValueError,match='baseline mismatch'):
        reproduce(current,previous)
    current=previous.query("setting=='primary_available'").copy()
    current.loc[current.index[0],'estimate']=42
    with pytest.raises(ValueError,match='coefficients'):
        reproduce(current,previous)


def diary_run(tmp_path):
    c=json.loads((PACKAGE/'config/default.json').read_text())
    return Run(tmp_path,tmp_path/'out',c)


def diary_rows():
    return pd.DataFrame([dict(participant_id='001',cohort='10k',research_stage='00_00_visit',collection_date='2024-01-01',food_items=3,total_logging_days=1,timezone='asia/jerusalem')])


def test_diary_ids_case_and_totals_audit(tmp_path):
    run=diary_run(tmp_path)
    p=tmp_path/'diary.csv'
    d=diary_rows(); d['total_logging_days']=14; d.to_csv(p,index=False)
    result=load_diary(run,p)
    assert result.participant_id.iloc[0]=='001' and result.diary_zone.iloc[0]=='Asia/Jerusalem'
    assert pd.read_csv(run.out/'tables/01_diary_audit.csv').reported_total_disagrees_people.iloc[0]==1


@pytest.mark.parametrize('problem', ['duplicate','nonmidnight','invalid_date','negative_items','missing_id','fractional_items'])
def test_diary_contract_rejects_errors(tmp_path,problem):
    run=diary_run(tmp_path); p=tmp_path/'diary.csv'; d=diary_rows()
    if problem=='duplicate': d=pd.concat([d,d])
    if problem=='nonmidnight': d['collection_date']='2024-01-01 13:00:00'
    if problem=='invalid_date': d['collection_date']='2024-02-30'
    if problem=='negative_items': d['food_items']=-1
    if problem=='missing_id': d['participant_id']=''
    if problem=='fractional_items': d['food_items']=3.5
    d.to_csv(p,index=False)
    with pytest.raises(ValueError): load_diary(run,p)


def make_curve_source(tmp_path, n=100):
    root, source = make_source(tmp_path,n=n,meals_each=30)
    original=json.loads((source/'manifest.json').read_text())
    c=original['config']; tz=c['time']['local_timezone']
    meals=pd.read_csv(source/'private/02_meals.csv',dtype={'participant_id':str})
    _, people=synthetic(n,30)
    a=pd.Timestamp('2024-01-01',tz=tz).timestamp()
    t=np.arange(a,a+12*86400,900)
    con=sqlite3.connect(source/'private/cache.sqlite')
    con.execute('CREATE TABLE cgm(participant_id TEXT,cohort TEXT,research_stage TEXT,connection_id TEXT,t REAL,glucose REAL,valid INT)')
    con.execute('CREATE TABLE food(participant_id TEXT,cohort TEXT,research_stage TEXT,t REAL,tz TEXT)')
    con.execute('CREATE INDEX cgm_key ON cgm(participant_id,cohort,research_stage,connection_id,t)')
    con.execute('CREATE INDEX food_key ON food(participant_id,cohort,research_stage,t)')
    rng=np.random.RandomState(725); diary=[]
    for i,(pid, group) in enumerate(meals.groupby('participant_id',sort=True)):
        g=100+rng.normal(0,2,len(t)) + rng.normal(0,4)
        sensor='sensor_'+pid
        for idx,m in group.iterrows():
            local=pd.to_datetime(m.start,unit='s',utc=True).tz_convert(tz)
            meals.loc[idx,'local_date']=str(local.date())
            meals.loc[idx,'local_hour']=local.hour+local.minute/60+local.second/3600
            rel=(t-m.start)/60
            select=(rel>=0)&(rel<=120)
            amp=15+4*people.loc[pid,'AHEI']+.2*m.carbohydrate_g+rng.normal(0,2)
            g[select]+=amp*np.sin(np.pi*rel[select]/120)**2
            con.execute('INSERT INTO food VALUES(?,?,?,?,?)',(pid,'10k','00_00_visit',m.start,tz))
        valid=np.ones(len(t),dtype=int)
        if i%4==0:
            missing=(t>a+900)&(t<a+6*3600)
            g[missing]=np.nan;valid[missing]=0
        connections={sensor:(t,g)}
        for idx,m in group.iterrows():
            response,curve=match_meal(connections,m.start,c['cgm'])
            assert response['cgm_valid']
            for field in ['baseline_glucose','pre_slope_per_min','iauc_net_0_120','peak_rise_0_120']:
                meals.loc[idx,field]=response[field]
            meals.loc[idx,'connection_id']=sensor
        con.executemany('INSERT INTO cgm VALUES(?,?,?,?,?,?,?)',[(pid,'10k','00_00_visit',sensor,float(ts),None if not np.isfinite(gl) else float(gl),int(v)) for ts,gl,v in zip(t,g,valid)])
        for j in range(12 if i%5 else 8):
            diary.append(dict(participant_id=pid,cohort='10k',research_stage='00_00_visit',collection_date='2024-01-{:02d}'.format(j+1),food_items=12,total_logging_days=12 if i%5 else 8,timezone=tz.lower()))
    con.commit();con.close()
    meals.to_csv(source/'private/02_meals.csv',index=False)
    original['status']['audit']['cache_sha256']=sha256(source/'private/cache.sqlite')
    original['status']['build']['meals_sha256']=sha256(source/'private/02_meals.csv')
    write_json(source/'manifest.json',original)
    p=root/'Transfer/diet_logging/diet_logging.csv';p.parent.mkdir(parents=True)
    pd.DataFrame(diary).to_csv(p,index=False)
    assert seven_main(['--run-dir',str(source),'--data-root',str(root),'--output-root',str(tmp_path/'seven')])==0
    seven=next((tmp_path/'seven').glob('seven_diets_*'))
    return root,source,seven


@pytest.fixture(scope='module')
def curve_source(tmp_path_factory):
    return make_curve_source(tmp_path_factory.mktemp('quality synthetic'))


def test_end_to_end_source_readonly_and_complete(curve_source,tmp_path):
    root,source,seven=curve_source
    pinned={str(f):sha256(f) for folder in [source,seven] for f in folder.rglob('*') if f.is_file()}
    assert main(['--meal-run',str(source),'--seven-run',str(seven),'--data-root',str(root),'--output-root',str(tmp_path/'results')])==0
    out=next((tmp_path/'results').glob('quality_timecourse_*'))
    state=json.loads((out/'manifest.json').read_text())['status']['quality_timecourse']
    assert state['status']=='COMPLETE' and state['estimated_models']==294
    assert len(pd.read_csv(out/'tables/03_model_reproduction.csv'))==42
    times=pd.read_csv(out/'private/03_meal_quality_timecourse.csv')
    assert len(times)==3000 and times.joint.sum() < len(times)
    assert (out/'figures/01_quality_timecourse.png').is_file()
    assert 'SYNTHETIC' in (out/'SUMMARY.md').read_text()
    for f,digest in pinned.items(): assert sha256(f)==digest
    for f in (out/'tables').glob('*.csv'):
        assert 'participant_id' not in pd.read_csv(f,nrows=0).columns
    assert main(['--show',str(out)])==0


def test_inputs_pin_scores_cache_and_target_run(curve_source,tmp_path):
    root,source,seven=curve_source
    out=tmp_path/'verify';out.mkdir()
    _,_,paths,expected,_=inspect(source,seven,root,out)
    assert str(paths['cache']) in expected
    wrong=root/'missing meal'
    with pytest.raises((ValueError,FileNotFoundError)):
        inspect(wrong,seven,root,out)


def test_bundle_pins_original_code():
    b=bundle()
    assert str(PACKAGE/'seven_diet_meals/models.py') in b
    assert str(PACKAGE/'hppmeal/cgm.py') in b


def test_epoch_roundtrip_timezone_match_and_conflict():
    from quality_timecourse.coverage import meal_zone_at_start
    food=pd.DataFrame({'t':[1700000000.123456], 'tz':['Asia/Jerusalem']})
    assert meal_zone_at_start(food,1700000000.1234562)=='Asia/Jerusalem'
    with pytest.raises(ValueError,match='timezone'):
        meal_zone_at_start(pd.concat([food,food.assign(tz='UTC')]),1700000000.123456)


def test_cgm_cutoff_and_minimum_days_are_separate_filters():
    meals,days,times,diary=attach_fixture()
    days.loc[0,'coverage']=.9
    days.loc[1,'coverage']=.899999
    d,_,_=attach_quality(meals,days,times,diary,'UTC')
    assert d.loc[0,'cgm90'] and not d.loc[1,'cgm90']
    assert d.loc[1,'log80'] and not d.loc[1,'joint']
    # Six reference dates cannot satisfy the minimum, even if all are logged.
    days=days.loc[days.local_date.le('2024-01-06')]
    d,_,info=attach_quality(meals,days,times,diary,'UTC')
    assert not info.log80_person.any() and not d.joint.any()


def test_model_time_origin_not_reset(monkeypatch):
    import quality_timecourse.analysis as analysis
    from seven_diet_meals.models import features
    meals,people,previous=augmented_fixture(n=12)
    expected=features(meals).set_index('meal_id').observation_day
    original=analysis.attempt
    counts=[]
    def checked(sample,diet,setting,branch,outcome,model):
        if len(sample):
            assert np.allclose(sample.observation_day,expected.loc[sample.meal_id])
            if setting=='joint': counts.append(len(sample))
        return original(sample,diet,setting,branch,outcome,model)
    monkeypatch.setattr(analysis,'attempt',checked)
    analysis.run_models(meals,people,previous,lambda _:None)
    assert counts


def test_no_estimable_models_report_and_plot_are_readable(tmp_path):
    from quality_timecourse.reporting import write_results,figures,summary
    meals,people,previous=augmented_fixture(n=12)
    for s in ['aligned','cgm90','log80','joint']: meals[s]=False
    result=run_models(meals,people,previous,lambda _:None)
    run=diary_run(tmp_path);run.c['data_kind']='SYNTHETIC'
    write_results(run,result,previous);figures(run,result)
    summary(run,dict(status='COMPLETE_WITH_LIMITATIONS'),result)
    for p in (run.out/'tables').glob('*.csv'): pd.read_csv(p)
    assert 'NOT_ESTIMATED' in (run.out/'SUMMARY.md').read_text()


def test_changed_input_and_nonempty_wal_rejected(curve_source,tmp_path):
    root,source,seven=curve_source
    out=tmp_path/'verify';out.mkdir()
    spec=json.loads((seven/'analysis_spec.json').read_text())
    path=Path(spec['sources']['new3']);original=path.read_bytes()
    try:
        path.write_bytes(original+b'\n')
        with pytest.raises(ValueError,match='differs from seven-diet'):
            inspect(source,seven,root,out)
    finally:
        path.write_bytes(original)
    wal=source/'private/cache.sqlite-wal'
    try:
        wal.write_bytes(b'not an empty WAL')
        with pytest.raises(ValueError,match='sidecar'):
            inspect(source,seven,root,out)
    finally:
        wal.unlink()


def test_readonly_wal_mode_cache_does_not_create_sidecars(curve_source,tmp_path):
    # SQLite can create sidecars even on mode=ro; immutable snapshot must not.
    import shutil
    root,source,seven=curve_source
    cache=tmp_path/'snapshot.sqlite'
    shutil.copy2(source/'private/cache.sqlite',cache)
    con=sqlite3.connect(cache);con.execute('PRAGMA journal_mode=WAL');con.close()
    assert not Path(str(cache)+'-wal').exists()
    from seven_diet_meals.inputs import inspect as old_inspect,load
    root,c,paths,_,_=old_inspect(source,root,PACKAGE/'seven_diet_meals/sources.json')
    run=Run(root,tmp_path/'run',c)
    meals,people=load(run,paths)
    paths['cache']=cache
    diary=load_diary(run,root/'Transfer/diet_logging/diet_logging.csv')
    before=sha256(cache)
    build(run,meals,paths,diary,lambda _:None)
    assert sha256(cache)==before
    assert not Path(str(cache)+'-wal').exists() and not Path(str(cache)+'-shm').exists()


def test_launcher_in_uploaded_copy_any_working_directory(curve_source,tmp_path):
    import shutil,subprocess,sys
    root,source,seven=curve_source
    upload=tmp_path/'uploaded project'
    upload.mkdir()
    for folder in ['quality_timecourse','seven_diet_meals','hppmeal','scripts']:
        shutil.copytree(PACKAGE/folder,upload/folder,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    import os
    env=dict(os.environ,HPP_MEAL_PYTHON=sys.executable)
    proc=subprocess.run(['bash',str(upload/'quality_timecourse/run_quality_timecourse.sh'),'--meal-run',str(source),'--seven-run',str(seven),'--data-root',str(root),'--stage','check','--output-root',str(tmp_path/'output')],cwd=str(tmp_path),env=env,capture_output=True,text=True)
    assert proc.returncode==0,proc.stdout+proc.stderr
    assert 'CHECK_PASS' in proc.stdout
