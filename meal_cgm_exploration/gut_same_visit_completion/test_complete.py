import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hppmeal.models import bh_planned
from gut_same_visit_completion.complete import (MEAL, REFERENCE, SETTINGS, TAXA,
    bh, create, digest, family, load_full, load_preview, qnum, validate_models)


def fixture(tmp_path, ne=False):
    source=tmp_path/"original"
    (source/"tables").mkdir(parents=True)
    rows,samples=[],[]
    for i,s in enumerate(SETTINGS):
        n=100-i*10
        samples.append(dict(setting=s,people=n,meals=n*25,person_sha256="synthetic_person_hash_"+s,
            meal_sha256="synthetic_meal_hash_"+s,meals_p50=25,dates_p50=11))
        for j,t in enumerate(TAXA):
            for k,o in enumerate(MEAL+REFERENCE):
                p=.001 if j==0 else .3+j*.1
                b=.2-j*.1
                rows.append(dict(setting=s,taxon=t,outcome=o,status="ESTIMATED",reason="",
                    people=n,meals=n*25,person_sha256=samples[-1]["person_sha256"],meal_sha256=samples[-1]["meal_sha256"],
                    estimate=b,se=.1,ci_low=b-.2,ci_high=b+.2,p=p,family=family(o),
                    unit="mg/dL per original CLR-Z" if o in MEAL else "original whole-CGM Z per original CLR-Z",
                    adjustment="age_years;sex"))
    d=pd.DataFrame(rows)
    if ne:
        d.loc[0,"status"]="NOT_ESTIMATED"
        d.loc[0,"reason"]="Synthetic constant outcome"
        d.loc[0,["estimate","se","ci_low","ci_high","p"]]=np.nan
    for _,ix in d.groupby(["setting","family"]).groups.items():
        d.loc[ix,"q_family"]=bh_planned(d.loc[ix,"p"])
        d.loc[ix,"planned_family_size"]=len(ix)
    d["q_meal12"]=np.nan
    for s in SETTINGS:
        ix=d.index[d.setting.eq(s)&d.outcome.isin(MEAL)]
        d.loc[ix,"q_meal12"]=bh_planned(d.loc[ix,"p"])
    sample=pd.DataFrame(samples)
    d.to_csv(source/"tables/models.csv",index=False)
    sample.to_csv(source/"tables/sample.csv",index=False)
    pd.DataFrame([dict(stage="eligible",people=100,meals=2500)]).to_csv(source/"tables/02_sample_flow.csv",index=False)
    pd.DataFrame([dict(setting=s,people=100-i*10,known_date=90-i*10,unknown_date=10) for i,s in enumerate(SETTINGS)]).to_csv(source/"tables/timing.csv",index=False)
    pd.DataFrame([dict(setting=SETTINGS[0],field="mean_increment",people=100,mean=1.,sd=.5,q25=.5,median=1.,q75=1.5)]).to_csv(source/"tables/descriptive.csv",index=False)
    hashes={str(p.relative_to(source)):digest(p) for p in (source/"tables").glob("*.csv")}
    manifest=dict(data_kind="SYNTHETIC",status=dict(gut_meal_bridge_v2=dict(status="COMPLETE_WITH_LIMITATIONS",version="2026.09.29.2",estimated=47 if ne else 48,result_sha256=hashes)))
    (source/"manifest.json").write_text(json.dumps(manifest))
    return source,d,sample


def test_bh_known_values_and_ne_denominator():
    np.testing.assert_allclose(bh([.01,.04,.03,np.nan]),[.04,.05333333333333334,.05333333333333334,np.nan],equal_nan=True)


def test_complete_specs_and_original_fdr(tmp_path):
    source,_,_=fixture(tmp_path)
    frames,pins,kind,preview=load_full(source)
    assert len(frames["models"])==48 and kind=="SYNTHETIC" and not preview
    assert len(pins)==6


@pytest.mark.parametrize("change",["drop","duplicate","q","membership"])
def test_bad_spec_q_or_membership_rejected(tmp_path,change):
    _,d,s=fixture(tmp_path)
    if change=="drop": d=d.iloc[1:]
    if change=="duplicate": d=pd.concat([d.iloc[:47],d.iloc[[0]]])
    if change=="q": d.loc[0,"q_family"] = .99
    if change=="membership": d.loc[0,"person_sha256"]="wrong_membership"
    with pytest.raises(ValueError): validate_models(d,s)


def test_ne_stays_in_planned_families(tmp_path):
    source,_,_=fixture(tmp_path,ne=True)
    d=load_full(source)[0]["models"]
    assert len(d)==48 and d.status.eq("NOT_ESTIMATED").sum()==1
    assert pd.isna(d.loc[0,"q_family"])


def test_changed_file_rejected(tmp_path):
    source,_,_=fixture(tmp_path)
    with (source/"tables/models.csv").open("a") as f: f.write("\n")
    with pytest.raises(ValueError,match="changed"): load_full(source)


def test_full_export_read_only_no_individual_fields(tmp_path):
    source,_,_=fixture(tmp_path)
    before={str(p):digest(p) for p in source.rglob("*") if p.is_file()}
    args=SimpleNamespace(source=source,preview_json=None,output_root=tmp_path/"exports")
    out=create(args)
    m=json.loads((out/"manifest.json").read_text())
    assert m["status"]=="COMPLETE_EXPORT" and m["data_kind"]=="SYNTHETIC" and m["available_models"]==48
    assert (out/"figures/Figure2_whole_CGM_reference.pdf").is_file()
    for p in (out/"tables").glob("*.csv"):
        content=p.read_text()
        assert "participant_id" not in content and "synthetic_person_hash" not in content
    assert before=={str(p):digest(p) for p in source.rglob("*") if p.is_file()}
    args.output_root=out
    with pytest.raises(ValueError,match="existing run"): create(args)


def test_cannot_write_inside_source(tmp_path):
    source,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match="outside"):
        create(SimpleNamespace(source=source,preview_json=None,output_root=source/"new"))


def test_preview_preserves_unknown_values_and_is_labelled(tmp_path):
    _,d,sample=fixture(tmp_path)
    codes=dict(zip(MEAL+REFERENCE,["net","pre","peak","Mean","CV","SDw"]))
    selected=d.loc[d.outcome.isin(MEAL)|d.setting.eq(SETTINGS[0])].copy()
    selected=selected[["setting","taxon","outcome","people","estimate","ci_low","ci_high","q_family","planned_family_size"]]
    selected.loc[selected.outcome.isin(REFERENCE),["ci_low","ci_high"]]=np.nan
    selected["outcome"]=selected.outcome.map(codes)
    selected=selected.rename(columns=dict(people="N",planned_family_size="family_size"))
    body=dict(data_label_reported="SYNTHETIC",models=json.loads(selected.to_json(orient="records")),
        samples=json.loads(sample.drop(columns=["person_sha256","meal_sha256"]).to_json(orient="records")))
    source=tmp_path/"screenshots.json"
    source.write_text(json.dumps(body))
    before=digest(source)
    out=create(SimpleNamespace(source=None,preview_json=source,output_root=tmp_path/"previews"))
    state=json.loads((out/"manifest.json").read_text())
    public=pd.read_csv(out/"tables/all_available_models.csv")
    missing=pd.read_csv(out/"tables/missing_local_specs.csv")
    assert state["status"]=="PREVIEW_ONLY" and state["data_kind"]=="SYNTHETIC_SCREENSHOT_ROUNDED"
    assert len(public)==36 and len(missing)==12
    assert public[["p","se","q_meal12"]].isna().all().all()
    assert public.loc[public.outcome.isin(REFERENCE),["ci_low","ci_high"]].isna().all().all()
    assert (out/"figures/Figure1_meal_associations.pdf").is_file()
    assert not (out/"figures/Figure2_whole_CGM_reference.pdf").exists()
    assert "SYNTHETIC_SCREENSHOT_ROUNDED" in (out/"Methods_en.md").read_text()
    assert digest(source)==before
    body["models"][0]["N"]=999
    source.write_text(json.dumps(body))
    with pytest.raises(ValueError,match="sample count mismatch"): load_preview(source)


def test_underflow_does_not_invent_a_probability_bound():
    assert qnum(0)=="0*"


def test_original_pipeline_output_compatibility(tmp_path):
    from gut_meal_bridge_v2.tests.test_bridge import fixture as original_fixture, run_args
    from gut_meal_bridge_v2.run_analysis import main as original_main
    files=original_fixture(tmp_path)
    runs=tmp_path/"original_runs"
    assert original_main(run_args(files,runs))==0
    source=next(runs.iterdir())
    out=create(SimpleNamespace(source=source,preview_json=None,output_root=tmp_path/"completed"))
    original=pd.read_csv(source/"tables/models.csv",float_precision="round_trip").set_index(["setting","taxon","outcome"]).sort_index()
    exported=pd.read_csv(out/"tables/all_available_models.csv",float_precision="round_trip").set_index(["setting","taxon","outcome"]).sort_index()
    fields=["estimate","se","ci_low","ci_high","p","q_family","q_meal12","residual_df","inference_df"]
    pd.testing.assert_frame_equal(original[fields],exported[fields],check_exact=True)
    assert len(exported)==48 and exported.status.eq("ESTIMATED").all()
