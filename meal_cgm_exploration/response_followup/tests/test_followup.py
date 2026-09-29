"""Synthetic scientific invariants and end-to-end frozen-input checks."""
import ast
import json
import os
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from hppmeal.common import sha256
from quality_timecourse.coverage import daily_coverage
from response_followup.support import weighted_quantile,quantiles,support_for,scale_interval,validate_members
from response_followup.feasibility import split_boundary,last_response_reading,person_windows,consecutive_support
from response_followup.inputs import verify_outputs
from response_followup.run_analysis import main


def meal_distribution(n=240):
    return pd.DataFrame([dict(participant_id='p{:04d}'.format(i),meal_id='m{}_{}'.format(i,j),
         AHEI=float(i),carbohydrate_g=float(c),calories_kcal=450+10*(j%7))
        for i in range(n) for j,c in enumerate(np.linspace(10,130,49))])


def test_equal_person_quantiles_not_dominated_by_logging_frequency():
    d=pd.DataFrame({'participant_id':['a']+['b']*100,'meal_id':['a']+['b'+str(i) for i in range(100)],'x':[0]+[100]*100})
    assert quantiles(d,'x',[.5])[0]==pytest.approx(50)
    assert weighted_quantile([0,0,100],[.5,.5,1],[.5])[0]==pytest.approx(50)


def test_support_independent_of_outcome_and_central_energy_check():
    d=meal_distribution();a,_,_=support_for(d,'AHEI')
    assert a['support_status']=='SUPPORTED'
    assert a['low_score_crossers']>=30 and a['high_score_crossers']>=30
    assert a['carb_high']-a['carb_low']>10
    e=d.assign(mean_increment=np.arange(len(d))*10000,positive_mean=-999)
    b,_,_=support_for(e,'AHEI')
    assert a==b
    d.loc[d.AHEI>=180,'calories_kcal']+=10000
    c,_,_=support_for(d,'AHEI')
    assert c['support_status']=='INSUFFICIENT_SUPPORT' and 'energy' in c['reason']


def test_score_ties_and_no_within_person_crossing_are_not_extrapolated():
    d=meal_distribution();d['AHEI']=0
    a,_,_=support_for(d,'AHEI');assert 'zero' in a['reason']
    d=meal_distribution(400)
    d['carbohydrate_g']=d.participant_id.str[1:].astype(int)%100+20.0
    a,_,_=support_for(d,'AHEI')
    assert a['support_status']=='INSUFFICIENT_SUPPORT' and a['low_score_crossers']==0


def test_contrast_scale_matches_algebra_and_reverses_negative_interval():
    model=dict(estimate=-.07,ci_low=-.11,ci_high=-.03)
    factor=1.6*(80-30)/10
    beta,lo,hi=scale_interval(model,factor)
    assert (beta,lo,hi)==pytest.approx((-.56,-.88,-.24))
    # Independent four predictions; all common carbohydrate/energy terms cancel.
    def y(z,c):return 3+2*c+.4*c*c+model['estimate']*z*c/10
    assert (y(.8,80)-y(.8,30))-(y(-.8,80)-y(-.8,30))==pytest.approx(beta)
    assert scale_interval(model,-2)==pytest.approx((.14,.06,.22))


def test_membership_checks_pairs_not_only_marginal_hashes():
    from seven_diet_meals.models import sample_hash
    d=pd.DataFrame({'participant_id':['001','002'],'meal_id':['a','b']})
    r=dict(person_sha256=sample_hash(d.participant_id),meal_sha256=sample_hash(d.meal_id),people=2,meals=2)
    validate_members(d,d.copy(),[r])
    changed=d.copy();changed['meal_id']=['b','a']
    with pytest.raises(ValueError,match='membership differs'):validate_members(d,changed,[r])


def test_actual_interpolation_bracket_crossing_is_excluded():
    start=pd.Timestamp('2024-01-01',tz='UTC').timestamp()
    t=np.arange(start,start+8*86400,900)+300  # readings at 00:05, 00:20 etc
    g=np.ones(len(t))*100
    cut,epoch,span=split_boundary(t,'UTC')
    assert (cut,span)==('2024-01-05',8)
    # Meal nominal endpoint 23:58, but interpolation needs Jan5 00:05.
    meal_start=epoch-2*3600-120
    assert meal_start+7200<epoch and last_response_reading(t,meal_start)>epoch
    z=pd.DataFrame(dict(start=[meal_start,epoch-86400-4*3600],meal_zone_ok=True,local_date=['2024-01-04','2024-01-03'],positive_mean=[1,2]))
    days=daily_coverage({'s':(t,g)},'UTC').assign(reference_day=lambda d:d.full_day,date_recorded=True)
    row,x=person_windows(z,t,g,days,'UTC',dict(post_minutes=120,warmup_minutes=0))
    assert row['excluded_meals_crossing_cut']==1 and len(x)==1
    assert row['last_first_window_read']<epoch and row['no_overlap']


def test_split_calendar_days_handles_dst_not_fixed_86400_seconds():
    a=pd.Timestamp('2024-03-29',tz='Europe/Berlin');b=pd.Timestamp('2024-04-02',tz='Europe/Berlin')
    t=np.arange(a.timestamp(),b.timestamp(),900)
    cut,epoch,span=split_boundary(t,'Europe/Berlin')
    assert cut=='2024-03-31' and span==4
    assert epoch==pd.Timestamp('2024-03-31',tz='Europe/Berlin').timestamp()


def test_modd_proxy_requires_adjacent_days_not_any_two_days():
    start=pd.Timestamp('2024-01-01',tz='UTC').timestamp()
    t=np.arange(start,start+4*86400,900);g=np.ones(len(t))*100
    assert consecutive_support(t,g,{'2024-01-01','2024-01-03'},'UTC',0)==(0,0)
    assert consecutive_support(t,g,{'2024-01-01','2024-01-02','2024-01-03'},'UTC',0)==(2,192)


def test_frozen_output_tamper_fails(tmp_path):
    f=tmp_path/'result.txt';f.write_text('original')
    s={'result_sha256':{'result.txt':sha256(f)}}
    verify_outputs(tmp_path,s);f.write_text('changed')
    with pytest.raises(ValueError,match='changed/missing'):verify_outputs(tmp_path,s)


def test_python38_and_shell_syntax():
    import subprocess
    folder=Path(__file__).resolve().parents[1]
    for f in folder.rglob('*.py'):
        tree=ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,8))
        for node in ast.walk(tree):
            if isinstance(node,ast.AnnAssign):assert not isinstance(node.annotation,ast.Subscript)
    assert subprocess.run(['bash','-n',str(folder/'run_response_followup.sh')]).returncode==0


def test_end_to_end_existing_synthetic_snapshot(tmp_path,capsys):
    value=os.environ.get('HPP_FOLLOWUP_SYNTHETIC_RUN')
    if not value:pytest.skip('Set HPP_FOLLOWUP_SYNTHETIC_RUN to verified completed synthetic nutrition snapshot')
    source=Path(value).resolve();mf=json.loads((source/'manifest.json').read_text())
    assert mf['data_kind']=='SYNTHETIC', 'Never use real data for this test'
    tracked={x['path']:sha256(x['path']) for x in mf['inputs']}
    tracked.update({str(source/k):sha256(source/k) for k in mf['status']['nutrition_bridge']['result_sha256']})
    assert main(['--nutrition-run',str(source),'--output-root',str(tmp_path/'results')])==0
    out=next((tmp_path/'results').glob('response_followup_*'))
    state=json.loads((out/'manifest.json').read_text())['status']['response_followup']
    assert state['status']=='COMPLETE_WITH_LIMITATIONS' and state['membership_checks']==14
    assert len(pd.read_csv(out/'tables/contrasts.csv'))==42
    assert len(pd.read_csv(out/'tables/specifications.csv'))==24
    assert pd.read_csv(out/'private/person_windows.csv').no_overlap.all()
    for f in (out/'tables').glob('*.csv'):assert 'participant_id' not in pd.read_csv(f,nrows=0).columns
    for name,digest in tracked.items():assert sha256(name)==digest
    for page in range(1,4):assert main(['--show',str(out),'--page',str(page)])==0
    assert len(list((out/'pages').glob('*.txt')))==3
    assert main(['--show',str(out),'--brief'])==0
    assert 'DATA=SYNTHETIC' in (out/'pages/01.txt').read_text()
    assert max(len(f.read_text().splitlines()) for f in (out/'pages').glob('*.txt'))<=34
    assert max(len(line) for f in (out/'pages').glob('*.txt') for line in f.read_text().splitlines())<190
    assert not any(Path(x+suffix).exists() for x in tracked if x.endswith('cache.sqlite') for suffix in ['-wal','-shm','-journal'])
    bad=out/'pages/02.txt';bad.write_text('tampered')
    assert main(['--show',str(out),'--page','2'])==2


def test_all42_supported_transformations_preserve_frozen_q(tmp_path):
    from seven_diet_meals.tests.test_analysis import synthetic
    from seven_diet_meals.inputs import DIETS
    from seven_diet_meals.models import prepare,features,sample_hash
    from response_followup.support import analyze,SETTINGS,OUTCOMES
    from response_followup.reporting import figures
    from hppmeal.common import Run
    meals,people=synthetic(400,49)
    idx=meals.groupby('participant_id').cumcount()
    meals['carbohydrate_g']=10+idx*2.5
    meals['calories_kcal']=450+10*(idx%7)
    meals['joint']=True;meals['positive_mean']=np.maximum(meals.mean_increment,0)+1
    data=features(meals);members=[];original=[];positive=[]
    for setting in SETTINGS:
        for diet in DIETS:
            d,_=prepare(data,people,diet,'primary_available','B')
            members.append(d[['participant_id','meal_id']].assign(setting=setting,diet=diet,branch='B'))
            for outcome in OUTCOMES:
                row=dict(setting=setting,diet=diet,branch='B',outcome=outcome,model='carb_interaction',status='ESTIMATED',
                    person_sha256=sample_hash(pd.Series(d.participant_id.unique())),meal_sha256=sample_hash(d.meal_id),
                    people=d.participant_id.nunique(),meals=len(d),estimate=-.07,ci_low=-.11,ci_high=-.03,q_family14=.06,q=.04)
                (positive if outcome=='positive_mean' else original).append(row)
    result=analyze(meals,people,dict(members=pd.concat(members),original=pd.DataFrame(original),positive=pd.DataFrame(positive)),lambda _:None)
    c=result['contrasts'];assert len(c)==42 and c.status.eq('SUPPORTED_CONTRAST').all()
    assert len(result['curves'])==42*21
    assert c.loc[c.outcome.eq('positive_mean'),'original_q'].eq(.04).all()
    assert c.loc[~c.outcome.eq('positive_mean'),'original_q'].eq(.06).all()
    assert (c.ci_low<c.estimate).all() and (c.estimate<c.ci_high).all()
    assert result['curves'].groupby(['setting','diet','outcome']).first().estimate.eq(0).all()
    run=Run(tmp_path,tmp_path/'synthetic_figures',{'data_kind':'SYNTHETIC'})
    figures(run,result)
    assert len(list((run.out/'figures').glob('*.png')))==2
    from response_followup.reporting import effect_cell,number_text
    assert effect_cell(c.iloc[0]).startswith(number_text(c.iloc[0].estimate)+'[')
    assert 'q=0.06' in effect_cell(c.iloc[0])



def test_constant_common_energy_is_valid_support():
    d=meal_distribution();d['calories_kcal']=500.0
    row,_,_=support_for(d,'AHEI')
    assert row['energy_low']==row['energy_high']==500
    assert row['support_status']=='SUPPORTED'
    assert row['low_score_crossers']>=30 and row['high_score_crossers']>=30


def test_early_rejection_preserves_unchecked_counts_as_missing():
    from response_followup.reporting import endpoint_minimum
    row,_,_=support_for(meal_distribution(80),'AHEI')
    assert row['reason']=='Too few people/meals inside common energy band'
    assert row['low_score_energy_people']==20 and row['high_score_energy_people']==20
    assert np.isnan(row['low_score_lowC_people']) and np.isnan(row['low_score_crossers'])
    assert endpoint_minimum(row,['low_score_crossers','high_score_crossers'])=='NA'
    legacy=dict(row,low_score_crossers=0,high_score_crossers=0)
    assert endpoint_minimum(legacy,['low_score_crossers','high_score_crossers'])=='NA'
    # A measured zero is distinct from an unperformed check.
    measured=dict(legacy,reason='Fewer than 30 within-person crossers in at least one score group')
    assert endpoint_minimum(measured,['low_score_crossers','high_score_crossers'])=='0'


@pytest.mark.parametrize('new_state',['FAILED','RUNNING','CHECK_PASS','missing_manifest'])
def test_latest_does_not_skip_newer_unsuccessful_run(tmp_path,new_state):
    from response_followup.reporting import resolve_run
    first=tmp_path/'response_followup_20260927T000000Z';first.mkdir()
    (first/'manifest.json').write_text(json.dumps({'status':{'response_followup':{'status':'COMPLETE'}}}))
    later=tmp_path/'response_followup_20260927T010000Z';later.mkdir()
    if new_state!='missing_manifest':
        (later/'manifest.json').write_text(json.dumps({'status':{'response_followup':{'status':new_state}}}))
    with pytest.raises(ValueError,match='latest run'):resolve_run('latest',tmp_path)
    assert resolve_run(str(first),tmp_path)[0]==first


def test_legacy_brief_verified_and_readonly(tmp_path,capsys):
    import shutil
    from response_followup.reporting import show,BRIEF_TABLES
    value=os.environ.get('HPP_FOLLOWUP_LEGACY_RUN')
    if not value:pytest.skip('Optional old completed synthetic response_followup v1 run')
    source=Path(value);m=json.loads((source/'manifest.json').read_text())
    assert m['data_kind']=='SYNTHETIC'
    tracked={str(f):sha256(f) for f in source.rglob('*') if f.is_file()}
    show(source,source.parent,brief=True)
    output=capsys.readouterr().out
    assert output.count('KEY ')==3 and '旧版结果只读压缩显示' in output
    assert all(sha256(Path(p))==h for p,h in tracked.items())
    # Copy only needed tables and change one, never change the original result.
    altered=tmp_path/'oldcopy';(altered/'tables').mkdir(parents=True)
    shutil.copy2(source/'manifest.json',altered/'manifest.json')
    for key in BRIEF_TABLES:shutil.copy2(source/'tables'/(key+'.csv'),altered/'tables'/(key+'.csv'))
    with (altered/'tables/support.csv').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='changed/missing'):show(altered,altered.parent,brief=True)
    assert not capsys.readouterr().out
