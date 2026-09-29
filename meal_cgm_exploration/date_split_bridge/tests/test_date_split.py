"""Analytic scientific checks and explicit synthetic end-to-end validation."""
import ast
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from hppmeal.common import sha256
from hppmeal.models import bh_planned
from seven_diet_meals.inputs import BASE,CATEGORIES
from seven_diet_meals.models import encode
from date_split_bridge.metrics import day_grid,grid_days,summarize_window,METRICS
from date_split_bridge.build import verify_window
from date_split_bridge.models import analyze,paired_sample
from date_split_bridge.run_analysis import main
from date_split_bridge import VERSION


def series(days=4):
    t0=pd.Timestamp('2024-01-01',tz='UTC').timestamp()
    t=t0+np.arange(days*96)*900
    g=100+10*(np.arange(len(t))//96)+(np.arange(len(t))%96)
    return t,g.astype(float),['2024-01-{:02d}'.format(i+1) for i in range(days)]


def test_metrics_match_closed_form_regular_grid():
    t,g,dates=series();audit,grids=grid_days(t,g,dates,'UTC',0);r=summarize_window(grids)
    assert all(d['finite_grid_points']==96 for d in audit)
    assert r['Mean']==pytest.approx(162.5)
    assert r['SDw']==pytest.approx(np.sqrt(96*97/12))
    expected_var=(4*95*(96*97/12)+96*500)/(384-1)
    assert r['CV']==pytest.approx(100*np.sqrt(expected_var)/162.5)
    assert r['MODD15']==pytest.approx(10) and r['MODD15_pairs']==3 and r['MODD15_matches']==288


def test_constant_glucose_metrics_zero_not_missing():
    t,g,dates=series();_,grids=grid_days(t,np.ones(len(t))*100,dates,'UTC',0);r=summarize_window(grids)
    assert [r[x] for x in METRICS]==[100,0,0,0]


def test_gap_interpolation_bounded_and_invalid_reading_not_skipped():
    t,g,dates=series(1)
    keep=np.ones(len(t),bool);keep[[2,3]]=False
    _,v=day_grid(t[keep],g[keep],dates[0],'UTC',t[0]);assert np.allclose(v[[2,3]],g[[2,3]])
    keep[4]=False
    _,v=day_grid(t[keep],g[keep],dates[0],'UTC',t[0]);assert np.isnan(v[[2,3,4]]).all()
    bad=g.copy();bad[2]=np.nan
    _,v=day_grid(t,bad,dates[0],'UTC',t[0]);assert np.isnan(v[2]) and v[3]==g[3]


def test_no_extrapolation_cross_date_or_warmup_borrowing():
    t,g,dates=series();t=t+300
    _,v=day_grid(t,g,dates[1],'UTC',t[0]);assert np.isnan(v[0])
    a,_=day_grid(t,g,dates[1],'UTC',t[0]);cut=pd.Timestamp(dates[1],tz='UTC').timestamp()
    assert a['first_support']>=cut and a['last_support']<cut+86400
    a,v=day_grid(t,g,dates[0],'UTC',t[0]+1800);assert np.isnan(v[:3]).all() and a['first_support']>=t[0]+1800


def test_early_values_cannot_affect_late_metrics():
    t,g,dates=series(8);cut=pd.Timestamp(dates[4],tz='UTC').timestamp()
    _,first=grid_days(t,g,dates[4:],'UTC',0)
    changed=g.copy();changed[t<cut]*=2
    _,second=grid_days(t,changed,dates[4:],'UTC',0)
    assert summarize_window(first)==summarize_window(second)


@pytest.mark.parametrize('start,stop,days,expected',[('2024-03-30','2024-04-03',['2024-03-30','2024-03-31','2024-04-01','2024-04-02'],92),('2024-10-26','2024-10-30',['2024-10-26','2024-10-27','2024-10-28','2024-10-29'],100)])
def test_dst_actual_day_duration_and_modd_does_not_bridge(start,stop,days,expected):
    t=np.arange(pd.Timestamp(start,tz='Europe/Berlin').timestamp(),pd.Timestamp(stop,tz='Europe/Berlin').timestamp(),900)
    audit,grids=grid_days(t,100+np.arange(len(t))*.01,days,'Europe/Berlin',0)
    assert audit[1]['expected_grid_points']==expected
    r=summarize_window(grids);assert r['days']==4 and np.isfinite(r['SDw'])
    assert r['MODD15_pairs']==1 and np.isnan(r['MODD15'])


def test_nonadjacent_days_not_paired_and_two_dates_insufficient():
    t,g,dates=series(5);_,grids=grid_days(t,g,[dates[0],dates[2],dates[4]],'UTC',0)
    r=summarize_window(grids);assert r['MODD15_pairs']==0 and np.isnan(r['MODD15']) and np.isfinite(r['Mean'])
    r=summarize_window({d:grids[d] for d in [dates[0],dates[2]]});assert all(np.isnan(r[k]) for k in METRICS)


def test_grid_coverage_cannot_turn_sparse_date_into_eligible_date():
    t,g,dates=series(1);g[:15]=np.nan
    audit,grids=grid_days(t,g,dates,'UTC',0)
    assert audit[0]['grid_coverage']<.9 and not grids


def test_reproduced_cut_counts_and_missing_values_checked():
    row=dict(split_local_date='2024-01-05',split_epoch=1704412800.,first_qualified_meals=8,no_overlap=True,last_first_window_read=np.nan)
    verify_window(row,dict(row))
    with pytest.raises(ValueError,match='first_qualified_meals'):verify_window(row,dict(row,first_qualified_meals=9))
    with pytest.raises(ValueError,match='split_epoch'):verify_window(row,dict(row,split_epoch=row['split_epoch']+.01))


def synthetic_persons(n=220):
    rng=np.random.default_rng(90428);d=pd.DataFrame({'participant_id':['p{:04d}'.format(i) for i in range(n)]})
    for k in BASE:d[k]=rng.normal(size=n)
    for k in CATEGORIES:d[k]=rng.choice(['a','b','c'],size=n)
    d['candidate_basic']=True;d['early_meals']=13
    x=rng.normal(size=(n,3))
    for j,name in enumerate(['mean_increment','peak_rise_0_120','positive_mean']):d['early_'+name]=x[:,j]*3+10
    for j,metric in enumerate(METRICS):
        shared=x[:,0]+rng.normal(size=n)
        d['full_'+metric]=100+shared*3+j
        d['late_'+metric]=100+shared*2+rng.normal(size=n)+j
    return pd.concat([d.assign(setting='source_primary'),d.iloc[:170].assign(setting='joint')],ignore_index=True)


def test_all_specs_shared_reference_scales_and_paired_HC3():
    person=synthetic_persons();result=analyze(person,lambda _:None);m=result['models']
    assert len(m)==72 and m.status.eq('ESTIMATED').all() and len(result['scales'])==12
    assert m.groupby(['meal_measure','cgm']).x_reference_sd.nunique().eq(1).all()
    assert m.groupby(['meal_measure','cgm']).y_reference_sd.nunique().eq(1).all()
    for _,d in m.groupby(['setting','meal_measure','cgm']):
        v=d.set_index('target');assert v.loc['late_minus_full','estimate']==pytest.approx(v.loc['late_primary','estimate']-v.loc['full_comparator','estimate'])
        assert d.person_sha256.nunique()==1 and d.people.nunique()==1
    # Independently compute paired-outcome HC3 sandwich, not independent-SE addition.
    d=paired_sample(person,'source_primary','mean_increment','Mean')
    s=result['scales'].loc[lambda x:x.meal_measure.eq('mean_increment')&x.cgm.eq('Mean')].iloc[0]
    X=encode(d,BASE);X['early_meal_z']=(d.early_mean_increment-s.x_mean)/s.x_sd;X.insert(0,'intercept',1.)
    A=X.to_numpy(float);y=(d.late_Mean-d.full_Mean).to_numpy()/s.y_sd
    bread=np.linalg.inv(A.T@A);beta=bread@A.T@y;e=y-A@beta;h=np.sum((A@bread)*A,axis=1)
    scores=A*(e/(1-h))[:,None];cov=bread@scores.T@scores@bread
    r=m.loc[m.setting.eq('source_primary')&m.meal_measure.eq('mean_increment')&m.cgm.eq('Mean')&m.target.eq('late_minus_full')].iloc[0]
    assert r.se==pytest.approx(np.sqrt(cov[-1,-1]),rel=1e-10)
    assert r.raw_y_per_reference_xSD==pytest.approx(r.estimate*s.y_sd)


def test_failed_slots_stay_in_family_and_joint_cannot_change_reference():
    d=synthetic_persons();d['early_mean_increment']=1
    result=analyze(d,lambda _:None);m=result['models']
    assert len(m)==72 and m.loc[m.meal_measure.eq('mean_increment'),'status'].eq('NOT_ESTIMATED').all()
    for _,f in m.groupby('target'):assert np.allclose(f.q,bh_planned(f.p),equal_nan=True)
    assert bh_planned([.01,np.nan])[0]==pytest.approx(.02)
    d=synthetic_persons();d.loc[d.setting.eq('joint'),'participant_id']='foreign'+d.loc[d.setting.eq('joint'),'participant_id']
    with pytest.raises(ValueError,match='subset'):analyze(d,lambda _:None)


def test_python38_and_shell_syntax():
    root=Path(__file__).resolve().parents[1]
    for p in root.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,8))
    assert subprocess.run(['bash','-n',str(root/'run_date_split_bridge.sh')]).returncode==0


@pytest.mark.parametrize('status',['FAILED','RUNNING','CHECK_PASS','missing'])
def test_latest_never_hides_unsuccessful_run(tmp_path,status):
    from date_split_bridge.reporting import show
    for i,s in [(1,'COMPLETE_WITH_LIMITATIONS'),(2,status)]:
        folder=tmp_path/('date_split_bridge_20260928T00000{}Z'.format(i));folder.mkdir()
        if s!='missing':(folder/'manifest.json').write_text(json.dumps({'status':{'date_split_bridge':{'status':s,'version':VERSION}}}))
    with pytest.raises(ValueError,match='[Ll]atest'):show('latest',tmp_path,brief=True)


def test_end_to_end_synthetic_frozen_followup(tmp_path,capsys):
    value=os.environ.get('HPP_DATE_SPLIT_SYNTHETIC_RUN')
    if not value:pytest.skip('Set HPP_DATE_SPLIT_SYNTHETIC_RUN to completed synthetic response_followup2026.09.27.2')
    source=Path(value).resolve();mf=json.loads((source/'manifest.json').read_text())
    assert mf['data_kind']=='SYNTHETIC','Never use real data for tests'
    tracked={x['path']:sha256(x['path']) for x in mf['inputs']}
    tracked.update({str(source/k):sha256(source/k) for k in mf['status']['response_followup']['result_sha256']})
    assert main(['--followup-run',str(source),'--output-root',str(tmp_path/'results')])==0
    out=next((tmp_path/'results').glob('date_split_bridge_*'));m=pd.read_csv(out/'tables/models.csv')
    assert len(m)==72 and len(m.loc[m.target.eq('late_primary')])==24 and m.status.eq('ESTIMATED').all()
    state=json.loads((out/'manifest.json').read_text())['status']['date_split_bridge']
    assert state['status']=='COMPLETE_WITH_LIMITATIONS' and state['prior_count_checks']==14
    for k,digest in tracked.items():assert sha256(k)==digest
    for f in (out/'tables').glob('*.csv'):assert 'participant_id' not in pd.read_csv(f,nrows=0).columns
    p=pd.read_csv(out/'private/person_features.csv')
    assert p.loc[p.early_meals.gt(0)&p.late_first_support.notna(),'last_first_window_read'].lt(p.loc[p.early_meals.gt(0)&p.late_first_support.notna(),'late_first_support']).all()
    assert len(list((out/'pages').glob('*.txt')))==3
    assert max(len(p.read_text().splitlines()) for p in (out/'pages').glob('*.txt'))<=32
    assert max(len(line) for p in (out/'pages').glob('*.txt') for line in p.read_text().splitlines())<200
    assert main(['--show',str(out),'--brief'])==0
    for i in [1,2,3]:assert main(['--show',str(out),'--page',str(i)])==0
    assert not any(Path(x+s).exists() for x in tracked if x.endswith('cache.sqlite') for s in ['-wal','-shm','-journal'])
    capsys.readouterr();(out/'pages/02.txt').write_text('changed')
    assert main(['--show',str(out),'--brief'])==2
    assert 'KEY ' not in capsys.readouterr().out


def test_followup_tampering_rejected_without_mutating_source(tmp_path):
    from response_followup.inputs import verify_outputs
    p=tmp_path/'table.csv';p.write_text('a\n1\n');state={'result_sha256':{'table.csv':sha256(p)}}
    verify_outputs(tmp_path,state);p.write_text('a\n2\n')
    with pytest.raises(ValueError,match='changed/missing'):verify_outputs(tmp_path,state)


def older_success(output):
    old=output/'date_split_bridge_20000101T000000_000000Z';(old/'pages').mkdir(parents=True)
    hashes={}
    for i in [1,2,3]:
        p=old/'pages/0{}.txt'.format(i);p.write_text('SYNTHETIC_OLDER_RESULT\n')
        hashes[str(p.relative_to(old))]=sha256(p)
    (old/'manifest.json').write_text(json.dumps(dict(data_kind='SYNTHETIC',status=dict(date_split_bridge=dict(
        version=VERSION,status='COMPLETE_WITH_LIMITATIONS',result_sha256=hashes)))))
    return old


@pytest.mark.parametrize('failure',['missing_run','malformed_followup','missing_nutrition','malformed_quality'])
def test_early_preflight_failure_cannot_show_old_success(tmp_path,capsys,failure):
    output=tmp_path/'outputs';old=older_success(output);old_hash=sha256(old/'manifest.json')
    source=tmp_path/'followup';nutrition=tmp_path/'nutrition';quality=tmp_path/'quality'
    if failure!='missing_run':
        source.mkdir()
        (source/'manifest.json').write_text('{' if failure=='malformed_followup' else json.dumps({'root':str(tmp_path)}))
        (source/'analysis_spec.json').write_text(json.dumps({'nutrition_run':str(nutrition)}))
    if failure=='malformed_quality':
        nutrition.mkdir();quality.mkdir()
        (nutrition/'analysis_spec.json').write_text(json.dumps({'quality_run':str(quality)}))
        (quality/'analysis_spec.json').write_text('{')
    assert main(['--followup-run',str(source),'--output-root',str(output)])==2
    latest=sorted(output.glob('date_split_bridge_*'))[-1]
    assert latest!=old and (latest/'ERROR.txt').is_file()
    m=json.loads((latest/'manifest.json').read_text())
    assert m['data_kind']=='UNVERIFIED' and m['status']['date_split_bridge']['status']=='FAILED'
    assert sha256(old/'manifest.json')==old_hash
    capsys.readouterr()
    assert main(['--show','latest','--brief','--output-root',str(output)])==2
    assert 'SYNTHETIC_OLDER_RESULT' not in capsys.readouterr().out


def test_protected_output_not_modified_even_on_early_failure(tmp_path):
    existing=tmp_path/'old_run';existing.mkdir();f=existing/'manifest.json';f.write_text('{}')
    before={str(p.relative_to(tmp_path)):sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    assert main(['--followup-run',str(tmp_path/'missing'),'--output-root',str(existing/'nested')])==2
    after={str(p.relative_to(tmp_path)):sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    assert before==after and not (existing/'nested').exists()


def test_output_root_is_file_returns_original_error_without_secondary_traceback(tmp_path,capsys):
    f=tmp_path/'not_a_directory';f.write_text('keep')
    assert main(['--followup-run',str(tmp_path/'missing'),'--output-root',str(f)])==2
    assert f.read_text()=='keep' and 'DATE_SPLIT_BRIDGE ERROR:' in capsys.readouterr().err


def test_empty_eligible_cohort_retains_all_planned_rows():
    p=synthetic_persons();p['candidate_basic']=False
    result=analyze(p,lambda _:None);m=result['models']
    assert len(m)==72 and m.people.eq(0).all() and m.status.eq('NOT_ESTIMATED').all()
    assert m.q.isna().all() and m.estimate.isna().all() and result['model_members'].empty


def test_uploaded_launcher_from_different_working_directory(tmp_path):
    import shutil
    source=Path(__file__).resolve().parents[1]
    project=tmp_path/'uploaded project';upload=project/'date_split_bridge'
    # Copy runtime packages only, with no access through cwd or PYTHONPATH.
    for name in ['date_split_bridge','response_followup','nutrition_bridge','quality_timecourse','seven_diet_meals','hppmeal']:
        shutil.copytree(source.parent/name,project/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    output=tmp_path/'results';older_success(output)
    env=os.environ.copy();env.pop('PYTHONPATH',None);env['HPP_MEAL_PYTHON']=os.sys.executable
    result=subprocess.run(['bash',str(upload/'run_date_split_bridge.sh'),'--show','latest','--brief','--output-root',str(output)],
        cwd=str(tmp_path),env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.count('SYNTHETIC_OLDER_RESULT')==3
