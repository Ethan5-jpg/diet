"""Synthetic arithmetic, scientific sample, provenance and CLI regression tests."""
import ast
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hppmeal.common import Run, sha256, write_json
from seven_diet_meals.inputs import record_expected
from seven_diet_meals import models as dm
from gut_meal_bridge_v2.tests.test_bridge import fixture as gut_fixture
from gut_meal_bridge_v2 import inputs as gi, analysis as ga
from window_qc_followup.coverage import (EXPECTED, CADENCE, DURATION, slots, diet_days, longest_run,
                                        attach, quality_mask, build)
from window_qc_followup.inputs import windows, metadata_scope, inspect, bundle
from window_qc_followup.analysis import (aggregate_gut, reproduce, diet_fdr, gut_fdr, settings)
from window_qc_followup.run_analysis import main, protect_output
from window_qc_followup.reporting import show

PACKAGE = Path(__file__).resolve().parents[2]
MODULE = PACKAGE / "window_qc_followup"
START = pd.Timestamp("2024-01-01", tz="UTC").timestamp()


def readings(n=EXPECTED, start=START):
    return pd.DataFrame(dict(t=start + np.arange(n)*CADENCE, glucose=100., valid=1))


def diary_food(n=14, start=START, timezone="UTC"):
    date = pd.to_datetime(start, unit="s", utc=True).tz_convert(timezone).date()
    dates = [str((pd.Timestamp(date)+pd.Timedelta(days=i)).date()) for i in range(n)]
    diary = pd.DataFrame(dict(local_date=dates, food_items=1, diary_zone=timezone))
    times = [pd.Timestamp(d, tz=timezone).timestamp()+12*3600 for d in dates]
    food = pd.DataFrame(dict(t=times, tz=timezone))
    return food, diary


def test_complete_fixed_denominator_and_duplicates():
    raw = readings()
    raw = pd.concat([raw, raw.iloc[[0]], raw.iloc[[20]]], ignore_index=True)
    row, mask = slots(raw, START)
    assert row["expected_slots"] == 1344 and row["valid_slots"] == 1344
    assert row["missing_slots"] == row["longest_missing_minutes"] == 0
    assert row["duplicate_excess_rows"] == 2 and mask.all()


def test_short_record_counts_tail_not_observed_span_only():
    row, mask = slots(readings(7*96), START)
    assert row["cgm_missing_rate"] == .5
    assert row["trailing_missing_slots"] == 672
    assert row["longest_missing_minutes"] == 7*1440


def test_absent_day_and_invalid_conflict_separate():
    raw = readings().drop(index=range(96, 192))
    raw.loc[0, "valid"] = 0
    raw = pd.concat([raw, raw.iloc[[2]].assign(glucose=200)], ignore_index=True)
    row, mask = slots(raw, START)
    assert row["absent_slots"] == 96
    assert row["invalid_or_conflict_slots"] == 2
    assert row["conflicting_slots"] == 1 and row["missing_slots"] == 98
    assert row["internal_missing_slots"] == 97 and not mask[0]


def test_off_grid_noise_cannot_create_coverage():
    raw = readings()
    raw.loc[1, "t"] += 400
    raw.loc[2, "glucose"] = np.inf
    raw = pd.concat([raw, pd.DataFrame([dict(t=START+DURATION, glucose=100., valid=1)])])
    row, _ = slots(raw, START)
    assert row["valid_slots"] == 1342 and row["off_grid_or_unassignable_rows"] == 1
    assert row["rows_after_window"] == 1


def test_declared_activation_exposes_leading_missing():
    raw = readings().iloc[2:]
    planned, _ = slots(raw, START)
    proxy, _ = slots(raw, raw.t.min())
    assert planned["leading_missing_slots"] == 2 and planned["trailing_missing_slots"] == 0
    assert proxy["leading_missing_slots"] == 0 and proxy["trailing_missing_slots"] == 2


def test_activation_and_sensor_sampling_clock_have_different_phase():
    raw=readings(start=START+8*60)
    row,_=slots(raw,START)
    assert row["valid_slots"]==1344 and row["cadence_phase_seconds"]==480


def test_all_missing_and_longest_run():
    row, _ = slots(readings(0), START)
    assert row["missing_slots"] == 1344 and row["cgm_missing_rate"] == 1
    assert row["whole_window_missing_slots"] == 1344
    assert row["leading_missing_slots"] == row["trailing_missing_slots"] == row["internal_missing_slots"] == 0
    assert longest_run([]) == 0 and longest_run([1,1,0,1]) == 2


def test_strict_threshold_boundary():
    q = pd.DataFrame(dict(status=["AUDITED"]*4, connection_unambiguous=True, unresolved_diet_time=False,
                          cgm_missing_rate=[.049, .05, .051, 0], diet_missing_day_rate=[0,0,0,.05]))
    assert quality_mask(q, .05, .05).tolist() == [True, False, False, False]
    assert settings()[0] == ("qc_primary", .05, .05, False)
    assert settings(.1)[0] == ("qc_primary", .05, .1, False)


@pytest.mark.parametrize("n,passes5,passes10,passes20", [(14,True,True,True),(13,False,True,True),(12,False,False,True),(11,False,False,False)])
def test_diet_14_day_denominator(n, passes5, passes10, passes20):
    food, diary = diary_food(n)
    row, d = diet_days(food, diary, START, "UTC")
    assert len(d) == 14 and row["diet_recorded_days"] == n
    assert row["diet_missing_day_rate"] == pytest.approx((14-n)/14)
    assert (row["diet_missing_day_rate"] < .05) == passes5
    assert (row["diet_missing_day_rate"] < .10) == passes10
    assert (row["diet_missing_day_rate"] < .20) == passes20


def test_diet_presence_requires_both_sources_and_timezone():
    food, diary = diary_food()
    diary.loc[0,"food_items"] = 0
    diary.loc[1,"diary_zone"] = "Asia/Jerusalem"
    food.loc[2,"tz"] = "bad_timezone"
    row, _ = diet_days(food, diary, START, "UTC")
    assert row["diet_recorded_days"] == 11
    assert row["diet_source_disagreement_days"] == 1 and row["diet_timezone_problem_days"] == 2


@pytest.mark.parametrize("date", ["2024-03-25", "2024-10-21"])
def test_elapsed_time_cgm_denominator_stays_1344_across_dst(date):
    start = pd.Timestamp(date+" 06:00", tz="Europe/Berlin").timestamp()
    row, _ = slots(readings(start=start), start)
    assert row["valid_slots"] == 1344
    food, diary = diary_food(start=start, timezone="Europe/Berlin")
    diet, days = diet_days(food, diary, start, "Europe/Berlin")
    assert len(days) == 14 and diet["diet_recorded_days"] == 14
    assert diet["diet_first_day_partial_in_cgm"]
    assert diet["diet_full_cgm_days"] == 13


def test_explicit_diet_start_and_partial_days():
    food, diary = diary_food()
    row, d = diet_days(food, diary, START+6*3600, "UTC", "2024-01-02")
    assert row["diet_window_source"] == "DECLARED_DATE"
    assert row["diet_recorded_days"] == 13 and row["diet_first_date"] == "2024-01-02"


def test_plan_identity_timezone_and_duplicates(tmp_path):
    config = dict(cohort="10k", research_stage="00_00_visit")
    path = tmp_path / "windows.csv"
    frame = pd.DataFrame([dict(participant_id="000001", cohort="10k", research_stage="00_00_visit",
                               connection_id="001", planned_start="2024-01-01T00:00:00Z")])
    frame.to_csv(path, index=False)
    parsed = windows(path, config)
    assert parsed.participant_id.iloc[0] == "000001" and parsed.connection_id.iloc[0] == "001"
    assert parsed.window_start.iloc[0] == START
    frame.planned_start = "2024-01-01 00:00:00"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="timezone-aware"):
        windows(path, config)
    frame.planned_start = "2024-01-01T00:00:00Z"
    pd.concat([frame,frame]).to_csv(path,index=False)
    with pytest.raises(ValueError, match="duplicate"):
        windows(path, config)


def test_no_filename_identity_in_cgm_metadata(tmp_path):
    path = tmp_path / "metadata.csv"
    pd.DataFrame(dict(cgm_time_series=["cgm__10k__123.csv"], collection_timestamp=["2024-01-01"])).to_csv(path,index=False)
    d, status = metadata_scope(path, dict(cohort="10k", research_stage="00_00_visit"))
    assert d.empty and "MISSING_IDENTITY" in status


def test_meal_window_boundaries_and_multiple_connections():
    q = pd.DataFrame([dict(participant_id="001",connection_id="s", status="AUDITED",anchor_source="X",
        window_start=START,window_end=START+DURATION,cgm_missing_rate=0,diet_missing_day_rate=0,
        diet_first_date="2024-01-01",diet_last_date="2024-01-14",unresolved_diet_time=False,connection_unambiguous=True)])
    m = pd.DataFrame(dict(participant_id="001",connection_id="s", meal_id=["a","b","c","d","e"],
        start=[START+1799,START+1800,START+DURATION-7201,START+DURATION-7200,START+DURATION-7199],
        local_date=["2024-01-01","2024-01-01","2024-01-14","2024-01-14","2024-01-14"]))
    a = attach(m,q)
    assert quality_mask(a,.05,.05,meal=True).tolist() == [False,True,True,False,False]
    q.connection_unambiguous = False
    assert not quality_mask(attach(m,q),.2,.2,meal=True).any()


def make_fixture(tmp_path, n=110, shorten=False):
    """Create all synthetic historical manifests AND a full scoped source cache."""
    root, meal, nutrition, _, _, _ = gut_fixture(tmp_path,n)
    old = json.loads((meal/"manifest.json").read_text())
    config = old["config"]
    tz = config["time"]["local_timezone"]
    start = pd.Timestamp("2024-01-01",tz=tz).timestamp()
    cache = meal/"private/cache.sqlite"
    diary_rows = []
    with sqlite3.connect(cache) as con:
        con.execute("CREATE TABLE cgm(participant_id TEXT,cohort TEXT,research_stage TEXT,connection_id TEXT,t REAL,glucose REAL,valid INT)")
        con.execute("CREATE TABLE food(participant_id TEXT,cohort TEXT,research_stage TEXT,t REAL,tz TEXT)")
        con.execute("CREATE TABLE diet_time_issues(participant_id TEXT,cohort TEXT,research_stage TEXT)")
        con.execute("CREATE INDEX cgm_key ON cgm(participant_id,cohort,research_stage,connection_id,t)")
        con.execute("CREATE INDEX food_key ON food(participant_id,cohort,research_stage,t)")
        for i in range(n):
            pid = "{:06d}".format(i)
            count = 7*96 if shorten else EXPECTED
            con.executemany("INSERT INTO cgm VALUES(?,?,?,?,?,?,?)", [(pid,"10k","00_00_visit",pid+"_connection",start+j*CADENCE,100.,1) for j in range(count)])
            for j in range(14):
                date = str((pd.Timestamp("2024-01-01")+pd.Timedelta(days=j)).date())
                t = pd.Timestamp(date,tz=tz).timestamp()+12*3600
                con.execute("INSERT INTO food VALUES(?,?,?,?,?)",(pid,"10k","00_00_visit",t,tz))
                diary_rows.append(dict(participant_id=pid,cohort="10k",research_stage="00_00_visit",collection_date=date,
                    food_items=1,total_logging_days=14,timezone=tz))
    old["status"]["audit"]["cache_sha256"] = sha256(cache)
    write_json(meal/"manifest.json",old)
    diary = root/"Transfer/diet_logging/diet_logging.csv"
    diary.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(diary_rows).to_csv(diary,index=False)
    root2,c,paths,pinned,_,mapping = gi.inspect(meal,nutrition,root,PACKAGE/"gut_meal_bridge_v2/sources.json",PACKAGE/"seven_diet_meals/sources.json")
    scratch = Run(root,tmp_path/"scratch",c)
    meals,people = gi.load(scratch,paths,mapping)
    seven = root/"meal_cgm_exploration/outputs/seven_fixture"
    srun = Run(root,seven,c)
    record_expected(srun,pinned)
    result = dm.run_models(meals,people,lambda _:None)[0]
    srun.table(result,"all_models.csv")
    srun.status["seven_diets"] = dict(status="COMPLETE")
    srun.save()
    gut = root/"meal_cgm_exploration/outputs/gut_fixture"
    grun = Run(root,gut,c)
    record_expected(grun,pinned)
    p,m,_,_ = ga.prepare(meals,people)
    result = ga.analyze(p,m,lambda _:None)["models"]
    grun.table(result,"models.csv")
    grun.status["gut_meal_bridge_v2"] = dict(status="COMPLETE",result_sha256={"tables/models.csv":sha256(gut/"tables/models.csv")})
    grun.save()
    return root,meal,nutrition,seven,gut,meals,people


def args_for(fixture, output, extra=()):
    root,meal,nutrition,seven,gut,_,_ = fixture
    return ["--data-root",str(root),"--meal-run",str(meal),"--nutrition-run",str(nutrition),
            "--seven-run",str(seven),"--gut-run",str(gut),"--output-root",str(output)] + list(extra)


def test_end_to_end_models_provenance_pages_and_no_source_changes(tmp_path,capsys):
    f = make_fixture(tmp_path)
    hashes = {p:sha256(p) for p in f[0].rglob("*") if p.is_file()}
    output = tmp_path/"results"
    assert main(args_for(f,output)) == 0
    run = next(output.iterdir())
    d = pd.read_csv(run/"tables/diet_models.csv")
    g = pd.read_csv(run/"tables/gut_models.csv")
    assert len(d)==210 and len(g)==144
    assert d.status.eq("ESTIMATED").all() and g.status.eq("ESTIMATED").all()
    assert g.groupby("setting").person_sha256.nunique().eq(1).all()
    assert g.groupby("setting").meal_sha256.nunique().eq(1).all()
    assert len(pd.read_csv(run/"tables/03_diet_reproduction.csv"))==42
    assert len(pd.read_csv(run/"tables/03_gut_reproduction.csv"))==48
    balance=pd.read_csv(run/"tables/05_selection_balance.csv")
    gut_fields=set(balance.loc[balance.module.eq("gut"),"field"])
    assert {"age_years","bmi","hPDI","NOVA4","T023","T046","T068","T096"}.issubset(gut_fields)
    assert any(f.startswith("sex=") for f in gut_fields)
    assert all(sha256(p)==h for p,h in hashes.items())
    pages = list((run/"pages").glob("*.txt"))
    assert len(pages)==4 and all("SYNTHETIC" in p.read_text() for p in pages)
    assert all("000001" not in p.read_text() for p in pages)
    assert main(["--show","latest","--output-root",str(output),"--brief"])==0
    assert "KEY 4/4" in capsys.readouterr().out
    manifest = json.loads((run/"manifest.json").read_text())
    state = manifest["status"]["window_qc_followup"]
    assert state["reproduction"].startswith("PASS")
    spec = json.loads((run/"analysis_spec.json").read_text())
    assert spec["expected_slots"]==1344 and spec["data_kind"]=="SYNTHETIC"


def test_empty_quality_keeps_all_specs_and_partial_status(tmp_path):
    f = make_fixture(tmp_path,n=70,shorten=True)
    output=tmp_path/"results"
    assert main(args_for(f,output))==2
    run=next(output.iterdir())
    for table,count in [("diet_models.csv",168),("gut_models.csv",96)]:
        d=pd.read_csv(run/"tables"/table)
        d=d.loc[d.setting.str.startswith("qc")]
        assert len(d)==count and d.status.eq("NOT_ESTIMATED").all()
        assert d.p.isna().all() and d.filter(regex="^q_").isna().all().all()
    state=json.loads((run/"manifest.json").read_text())["status"]["window_qc_followup"]
    assert state["status"]=="PARTIAL"
    assert main(["--show",str(run),"--page","3"])==0


def test_source_effect_mismatch_stops_new_models(tmp_path):
    f=make_fixture(tmp_path,n=75)
    path=f[3]/"tables/all_models.csv"
    d=pd.read_csv(path,dtype={"person_sha256":str,"meal_sha256":str})
    row=d.index[d.setting.eq("primary_available")][0]
    d.loc[row,"estimate"] += 1
    d.to_csv(path,index=False)
    output=tmp_path/"results"
    assert main(args_for(f,output))==2
    run=next(output.iterdir())
    state=json.loads((run/"manifest.json").read_text())["status"]["window_qc_followup"]
    assert state["status"]=="FAILED" and "coefficients differ" in state["reason"]
    assert not (run/"tables/diet_models.csv").exists()
    assert main(["--show","latest","--output-root",str(output)])==2


def test_cache_hash_mismatch_and_output_protection(tmp_path):
    f=make_fixture(tmp_path,n=60)
    with (f[1]/"private/cache.sqlite").open("ab") as stream:
        stream.write(b"changed")
    output=tmp_path/"results"
    assert main(args_for(f,output,["--stage","check"]))==2
    state=json.loads((next(output.iterdir())/"manifest.json").read_text())["status"]["window_qc_followup"]
    assert "cache missing or changed" in state["reason"]
    with pytest.raises(ValueError,match="existing source"):
        protect_output(f[1]/"nested",f[0],[f[1]])
    with pytest.raises(ValueError,match="Inside Data"):
        protect_output(f[0]/"Transfer/nested",f[0],[])


def test_audit_without_models_and_plan_unmatched_not_fallback(tmp_path):
    f=make_fixture(tmp_path,n=60)
    plan=tmp_path/"plan.csv"
    pd.DataFrame([dict(participant_id="000001",cohort="10k",research_stage="00_00_visit",connection_id="000001_connection",
        planned_start="2024-01-01T00:00:00+02:00")]).to_csv(plan,index=False)
    output=tmp_path/"audit_results"
    assert main(args_for(f,output,["--stage","audit","--window-starts",str(plan)]))==0
    run=next(output.iterdir())
    q=pd.read_csv(run/"private/connection_quality.csv",dtype={"participant_id":str})
    assert q.anchor_source.eq("DECLARED_PLAN").sum()==1
    assert q.status.eq("DECLARED_WINDOW_UNMATCHED").sum()==59
    assert not q.anchor_source.eq("FIRST_RECORDED_POINT_PROXY").any()
    assert q.raw_rows.eq(EXPECTED).all()
    unmatched=q.loc[q.status.eq("DECLARED_WINDOW_UNMATCHED")]
    assert unmatched.cgm_missing_rate.isna().all() and unmatched.diet_missing_day_rate.isna().all()
    population=pd.read_csv(run/"tables/01_coverage_population.csv")
    assert population.no_cgm_people.iloc[0]==0
    assert population.connections_without_auditable_window.iloc[0]==59
    assert not (run/"tables/diet_models.csv").exists()


def test_aggregate_preserves_original_date_origin_and_means(tmp_path):
    f=make_fixture(tmp_path,n=70)
    meals,people=f[-2:]
    p,m,_,_=ga.prepare(meals,people)
    data=dm.features(meals)
    pool=data.loc[data.local_date.ne("2024-01-01")]
    new,member,_=aggregate_gut(data,p,m,pool)
    expected=data.loc[data.meal_id.isin(member.meal_id)].groupby("participant_id").observation_day.mean()
    assert np.allclose(new.set_index("participant_id").observation_day,expected.reindex(new.participant_id))
    assert (new.observation_day > 1).all()
    assert (new.carb_10_squared > new.carb_10**2).all()


def test_fdr_planned_denominator_includes_ne():
    rows=[]
    for taxon in gi.contract()["taxa"]:
        for outcome in gi.contract()["meal_outcomes"]+gi.contract()["reference_outcomes"]:
            rows.append(dict(setting="qc_primary",taxon=taxon["id"],outcome=outcome,p=.01,
                status="ESTIMATED",person_sha256="same",meal_sha256="same"))
    d=pd.DataFrame(rows)
    d.loc[0,"p"]=np.nan
    d.loc[0,"status"]="NOT_ESTIMATED"
    corrected=gut_fdr(d)
    assert pd.isna(corrected.loc[0,"q_family"])
    primary=corrected.loc[corrected.family.eq("net_primary") & corrected.p.notna()]
    assert np.allclose(primary.q_family,.01*4/3)


def test_bundle_and_python38_syntax():
    assert bundle()
    for path in MODULE.glob("*.py"):
        ast.parse(path.read_text(),filename=str(path),feature_version=(3,8))


def test_raw_audit_metadata_only_people_and_sensors_not_stitched(tmp_path):
    config=json.loads((PACKAGE/"config/default.json").read_text())
    config["data_kind"]="SYNTHETIC"
    config["time"]["local_timezone"]="UTC"
    cache=tmp_path/"cache.sqlite"
    with sqlite3.connect(cache) as con:
        con.execute("CREATE TABLE cgm(participant_id TEXT,cohort TEXT,research_stage TEXT,connection_id TEXT,t REAL,glucose REAL,valid INT)")
        con.execute("CREATE TABLE food(participant_id TEXT,cohort TEXT,research_stage TEXT,t REAL,tz TEXT)")
        con.execute("CREATE TABLE diet_time_issues(participant_id TEXT,cohort TEXT,research_stage TEXT)")
        for cid,begin,end in [("s1",0,EXPECTED//2),("s2",EXPECTED//2,EXPECTED)]:
            con.executemany("INSERT INTO cgm VALUES(?,?,?,?,?,?,?)",[("001","10k","00_00_visit",cid,START+j*CADENCE,100.,1) for j in range(begin,end)])
        con.execute("INSERT INTO food VALUES(?,?,?,?,?)",("003","10k","00_00_visit",START+12*3600,"UTC"))
        con.execute("INSERT INTO diet_time_issues VALUES(?,?,?)",("003","10k","00_00_visit"))
    diary=pd.DataFrame(dict(participant_id=["003"],local_date=["2024-01-01"],food_items=1,diary_zone="UTC"))
    metadata=pd.DataFrame(dict(participant_id=["002"],cohort="10k",research_stage="00_00_visit",connection_id="missing"))
    run=Run(tmp_path,tmp_path/"out",config)
    q=build(run,dict(cache=cache),diary,None,metadata,lambda _:None)
    sensor=q.loc[q.participant_id.eq("001")]
    assert len(sensor)==2 and sensor.cgm_missing_rate.eq(.5).all()
    assert not sensor.connection_unambiguous.any()
    assert not quality_mask(q,.2,.2).any()
    absent=q.loc[q.participant_id.isin(["002","003"])]
    assert absent.raw_rows.eq(0).all() and absent.cgm_missing_rate.isna().all()
    audit=pd.read_csv(run.out/"tables/01_coverage_population.csv")
    assert audit.people.iloc[0]==3 and audit.no_cgm_people.iloc[0]==2
    plan=pd.DataFrame([dict(participant_id="002",connection_id="missing",window_start=START,diet_start_date=None)])
    second=Run(tmp_path,tmp_path/"out_planned",config)
    q2=build(second,dict(cache=cache),diary,plan,metadata,lambda _:None)
    known_empty=q2.loc[q2.participant_id.eq("002")].iloc[0]
    assert known_empty.status=="AUDITED" and known_empty.cgm_missing_rate==1.
    assert known_empty.whole_window_missing_slots==EXPECTED
    recorded_unmatched=q2.loc[q2.participant_id.eq("001")]
    assert recorded_unmatched.raw_rows.eq(EXPECTED//2).all()
    assert recorded_unmatched.cgm_missing_rate.isna().all()
    assert pd.read_csv(second.out/"tables/01_coverage_population.csv").no_cgm_people.iloc[0]==2
