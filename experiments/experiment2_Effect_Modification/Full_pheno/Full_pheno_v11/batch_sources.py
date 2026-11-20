"""Discover and link quantitative phenotypes and food categories in TRE."""

from collections import OrderedDict
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from diet_inventory import consume, read_inventory
from food_features import DAY, KEY, SOURCES, normalise, projected
from new_endpoints import RAW, read_new_outcomes, source_paths as prior_paths
from run_expanded_grid import diet_catalog
from run_v2 import require


EXACT = {
    "anthropometrics": ("anthropometrics/anthropometrics", r"^(?:height|weight|bmi|.*circumference|waist_to_hip_ratio)$"),
    "blood_pressure": ("blood_pressure/blood_pressure", r"^(?:lying|sitting|standing).*blood_pressure_"),
    "body_composition": ("body_composition/body_composition", r"^(?:body_comp_|total_scan_)"),
    "bone_density": ("bone_density/bone_density", r"^(?:body_|bone_|bmd_|total_|.*(?:bmd|bmc|t_score|z_score|tissue_percent_fat))"),
    "carotid": ("carotid_ultrasound/carotid_ultrasound", r"^imt_(?:left|right)$"),
    "liver_ultrasound": ("liver_ultrasound/liver_ultrasound_aggregated", r"^(?:elasticity|velocity|viscosity|attenuation|dispersion|speed_of_sound)_"),
    "grip": ("hand_grip/hand_grip", r"^(?:finger|hand)_grip_strength_"),
    "retinal_vascular": ("fundus/microvasculature", r"(?:left|right)$"),
    "serum_lipidomics": ("serum_lipidomics/serum_lipidomics", r".+"),
    "nightingale": ("nightingale_metabolomics/nightingale_metabolomics", r".+"),
}
CURATED = {
    "adiposity": ("abdominal_adiposity", r"(?:vat_sat_ratio|vat_index|sat_mass|visceral_adipose_index)$"),
    "renal": ("ckd", r"(?:egfr|blood_bun|blood_creatinine|urine_acr|urine_microalbumin|urine_creatinine)$"),
    "glycemic": ("diabetes", r"(?:hba1c|fpg|ealc|hbgi|mage)$"),
    "lipid_profile": ("hyperlipidemia", r"_value$"),
    "liver_indices": ("mafld", r"(?:index|score|fli|hsi|fib4|imaging|cap|alt|ast)$"),
}


def paths(root):
    sources={key:RAW/(v[0]+".parquet") for key,v in EXACT.items()}
    sources.update({key:RAW/"curated_phenotypes"/(v[0]+".parquet") for key,v in CURATED.items()})
    sources.update(
        nightingale_tags=RAW/"nightingale_metabolomics/tags_per_biomarker.parquet",
        population=RAW/"population/population.parquet",
        cgm_indices=RAW/"cgm/iglu.parquet",
        cgm_qc=root/"../analysis/shared/cgm_visit_phenotypes_TRE_ONLY.parquet",
        blood=RAW/"blood_tests/blood_tests.parquet",
        ecg=RAW/"ecg/ecg.parquet",
        ecg_qc=RAW/"ecg/ecg_qc.parquet",
        sleep=RAW/"sleep/sleep.parquet",
        sleep_hrv=RAW/"sleep/hrv.parquet",
    )
    sources.update({"v10_"+k:v for k,v in prior_paths(root).items()})
    return {key:path.resolve(strict=True) for key,path in sources.items()}


def quantitative(path, pattern):
    pfile=pq.ParquetFile(path)
    fields=[f.name for f in pfile.schema_arrow if
            (pa.types.is_floating(f.type) or pa.types.is_integer(f.type))
            and f.name not in {*KEY,"array_index"} and re.search(pattern,f.name,re.I)]
    return fields,pfile.metadata.num_rows


def index_for(study):
    index=pd.DataFrame(dict(zip(KEY,study.keys,strict=True)))
    index["input_row"]=np.arange(study.n)
    return index,index.participant_id.astype("int64").tolist()


def features_by_visit(data, fields, n, repeat=False):
    z=data[["input_row",*fields]].copy()
    numeric=z[fields].apply(pd.to_numeric,errors="coerce")
    z[fields]=numeric.where(np.isfinite(numeric))
    group=z.groupby("input_row")[fields]
    if repeat:
        out=group.median()
        ambiguous=pd.Series(0,index=fields,dtype=int)
    else:
        count=group.nunique()
        out=group.first().where(count<=1)
        ambiguous=(count>1).sum(0)
    return out.reindex(range(n)),ambiguous


def read_outcomes(path_map, study):
    import protocol as p

    index, ids=index_for(study)
    outcomes=OrderedDict((k, study.outcome(k).y) for k in p.ENDPOINTS)
    families,domains={}, {k:"original" for k in p.ENDPOINTS}
    audited={}

    def collect(name, fields, frame, raw, rule, repeat=False, detail=None):
        if not fields:
            audited[name]=dict(raw_rows=raw,linked_people=frame.input_row.nunique(),measured_columns=0,rule=rule)
            return
        x,conflicts=features_by_visit(frame,fields,study.n,repeat)
        for k in fields:
            label=name+":"+k
            outcomes[label]=x[k].to_numpy(float)
            domains[label]=name
        audited[name]=dict(raw_rows=raw, linked_people=frame.input_row.nunique(), measured_columns=len(fields), rule=rule,
                           finite_people={k:int(x[k].notna().sum()) for k in fields},
                           overlaps_with_conflicting_values={k:int(c) for k,c in conflicts.items() if c})
        if detail is not None:
            audited[name]["qc"]=detail

    for name,(_, pattern) in EXACT.items():
        path=path_map[name]
        fields,raw=quantitative(path,pattern)
        keys=[*KEY,"array_index"] if name=="nightingale" else KEY
        frame=projected(path,[*keys,*fields],ids).merge(index,on=KEY,how="inner",validate="many_to_one")
        remark=None
        if name=="nightingale":
            tag_path=path_map["nightingale_tags"]
            schema=set(pq.ParquetFile(tag_path).schema_arrow.names)
            tagged={k:k+"_tag" for k in fields if k+"_tag" in schema}
            require(len(tagged)==len(fields),"V11_NIGHTINGALE_MISSING_ASSAY_TAG")
            tags=projected(tag_path,[*keys,*tagged.values()],ids)
            frame=frame.merge(tags,on=keys,how="inner",validate="one_to_one")
            flags=frame[list(tagged.values())].notna().to_numpy()
            frame[fields]=frame[fields].mask(flags)
            remark=dict(flagged_values=int(flags.sum()),rule="exclude all nonempty assay QC tags")
        collect(name,fields,frame,raw,"exact discovery visit",detail=remark)

    qc_keys=[*KEY,"connection_id"]
    cgm=projected(path_map["cgm_qc"],[*qc_keys,"point_qc_verified","phenotype_eligible"],ids)
    cgm=cgm.loc[cgm.point_qc_verified.eq(True)&cgm.phenotype_eligible.eq(True),qc_keys]
    cgm["connection_id"]=cgm.connection_id.astype("string")
    require(not cgm.duplicated(KEY).any(),"V11_CGM_MULTIPLE_CERTIFIED_CONNECTIONS")
    fields,raw=quantitative(path_map["cgm_indices"],r"^cgm_(?!mean$)")
    frame=projected(path_map["cgm_indices"],[*qc_keys,*fields],ids)
    frame["connection_id"]=frame.connection_id.astype("string")
    frame=frame.merge(cgm,on=qc_keys,validate="many_to_one").merge(index,on=KEY,validate="many_to_one")
    collect("cgm_indices",fields,frame,raw,"exact discovery visit and point-QC-selected connection")

    rec_keys=[*KEY,"array_index"]
    ecg_quality="MDC_ECG_LEAD_II__ecg_quality"
    qc=projected(path_map["ecg_qc"],[*rec_keys,ecg_quality],ids)
    qc=qc.loc[qc[ecg_quality].eq("Excellent"),rec_keys]
    fields,raw=quantitative(path_map["ecg"],r"^(?:hr_bpm|[a-z_]+_ms|[pqt]+_axis|[a-z]+_mv_.*)$")
    ecg=projected(path_map["ecg"],[*rec_keys,*fields],ids)
    ecg=ecg.merge(qc,on=rec_keys,validate="one_to_one").merge(index,on=KEY,validate="many_to_one")
    collect("ecg",fields,ecg,raw,"exact discovery visit, recording match, excellent lead II quality")

    fields,raw=quantitative(path_map["sleep"],r"^(?:ahi(?:_|$)|odi(?:_|$)|.*spo2|.*saturation|.*heart_rate)")
    quality=["quality_score_spo2","quality_score_heart_rate"]
    fields=[k for k in fields if not k.startswith("quality_score")]
    sleep=projected(path_map["sleep"],[*KEY,*quality,*fields],ids).merge(index,on=KEY,validate="many_to_one")
    for k in fields:
        score=quality[1 if "heart_rate" in k else 0]
        sleep[k]=sleep[k].where(sleep[score].between(80.,100.))
    collect("sleep",fields,sleep,raw,"exact discovery visit, channel quality 80–100, median across retained nights",repeat=True)
    stages=("night","nrem","rem","wake")
    quality={k:f"hrv_quality_score_during_{k}" for k in stages}
    fields,raw=quantitative(path_map["sleep_hrv"],r"^hrv_(?!quality_).+_during_(?:night|nrem|rem|wake)$")
    hrv=projected(path_map["sleep_hrv"],[*KEY,*quality.values(),*fields],ids).merge(index,on=KEY,validate="many_to_one")
    for k in fields:
        hrv[k]=hrv[k].where(hrv[quality[k.rsplit("_",1)[1]]].between(80.,100.))
    collect("sleep_hrv",fields,hrv,raw,"exact discovery visit, stage quality 80–100, median across retained nights",repeat=True)

    days=normalise(projected(path_map["v10_food_days"],DAY+["energy_eligible","energy_kcal"],ids)).merge(
        index,on=KEY,validate="many_to_one")
    first=days.loc[days.energy_eligible.eq(True)&days.energy_kcal.gt(0.)].groupby("input_row").date.min()
    fields,raw=quantitative(path_map["blood"],r"^bt__.+_float_value$")
    blood=projected(path_map["blood"],[*KEY[:2],"collection_date",*fields],ids).merge(
        index[[*KEY[:2],"input_row"]],on=KEY[:2],validate="many_to_one")
    lag=(blood.input_row.map(first)-pd.to_datetime(blood.collection_date,errors="coerce").dt.normalize()).dt.days
    blood=blood.loc[lag.between(1,365)].assign(lag=lag.loc[lag.between(1,365)]).sort_values("lag")
    for k in fields:
        val=pd.to_numeric(blood[k],errors="coerce")
        blood[k]=val.where(np.isfinite(val)&val.ge(0.))
        if k=="bt__hba1c_float_value":
            blood[k]=blood[k].where(blood[k].between(3.,18.))
        valid=blood[k].notna()
        earliest=blood.loc[valid].groupby("input_row").lag.min()
        blood[k]=blood[k].where(blood.lag.eq(blood.input_row.map(earliest)))
    collect("blood",fields,blood,raw,"latest valid draw up to 365 days before first eligible food day")

    population=projected(path_map["population"],KEY[:2],ids).drop_duplicates()
    safe=population.loc[~population.duplicated("participant_id",keep=False)]
    curated_index=index.merge(safe,on=KEY[:2],validate="one_to_one")
    for name,(_,pattern) in CURATED.items():
        fields,raw=quantitative(path_map[name],pattern)
        keys=KEY if "cohort" in pq.ParquetFile(path_map[name]).schema_arrow.names else [KEY[0],KEY[2]]
        frame=projected(path_map[name],[*keys,*fields],ids).merge(curated_index[[*keys,"input_row"]],on=keys,validate="many_to_one")
        collect(name,fields,frame,raw,"exact discovery visit with unique population cohort")

    previous,_,_=read_new_outcomes({k.removeprefix("v10_"):v for k,v in path_map.items() if k.startswith("v10_")},study)
    for name,out in previous.items():
        if name.startswith("cgm_") or name.startswith("seated_"):
            outcomes[name]=out.y
            domains[name]="v10_derived"
    for k,z in outcomes.items():
        families[k]=int(np.unique(study.family[np.isfinite(z)]).size)
    kept={}
    duplicates={}
    for k,z in outcomes.items():
        key=np.nan_to_num(z,nan=0.).astype("<f8").tobytes(), np.packbits(np.isfinite(z)).tobytes()
        if key in kept and families[k]>0:
            duplicates[k]=kept[key]
        else:
            kept[key]=k
    excluded={k:"fewer than 200 observed discovery households" for k,n in families.items() if n<200 and k not in p.ENDPOINTS}
    excluded.update({k:"exact duplicate of "+v for k,v in duplicates.items() if k not in p.ENDPOINTS})
    ready={k:z for k,z in outcomes.items() if k not in excluded}
    return ready, dict(sources=audited,observed_families=families,excluded=excluded,domains=domains)


def read_diets(root,study):
    values,audit,path_map=read_inventory(root,study)
    recipes=diet_catalog(values,audit)
    index,ids=index_for(study)
    day=normalise(projected(path_map["day"],DAY+["energy_eligible","energy_kcal"],ids)).merge(
        index,on=KEY,how="inner",validate="many_to_one")
    day=day.loc[day.energy_eligible.eq(True)&day.energy_kcal.gt(0.)].copy()
    ev=normalise(projected(path_map["events"],DAY+["food_category","qc_primary_eligible","calories_kcal_clean"],ids))
    ev=ev.loc[ev.qc_primary_eligible.eq(True)].merge(day[DAY],on=DAY,validate="many_to_one")
    ev["category"]=ev.food_category.astype("string").fillna("").str.lower().str.strip()
    ev["kcal"]=ev.calories_kcal_clean.where(ev.calories_kcal_clean.ge(0.),0.)
    counts=ev.groupby(DAY).kcal.sum().rename("event_kcal")
    daily=day.merge(counts,on=DAY,validate="one_to_one")
    daily=daily.loc[np.isclose(daily.energy_kcal,daily.event_kcal,rtol=.005,atol=.1)].copy()
    require(len(daily)==audit["matched_days"]["event_energy"],"V11_CATEGORY_DAYS_NOT_RECONCILED")
    pivot=ev.loc[ev.category.ne("")].groupby([*DAY,"category"]).kcal.sum().unstack("category",fill_value=0.)
    cats=list(pivot.columns)
    daily=daily.merge(pivot.reset_index(),on=DAY,how="left",validate="one_to_one")
    daily[cats]=daily[cats].fillna(0.).div(daily.energy_kcal,axis=0)
    panel=consume(daily,cats,study.numeric["n_energy_days"],study.n)
    stats={k:dict(measured=int(np.isfinite(z).sum()),nonzero=int((np.isfinite(z)&(z>0.)).sum()))
           for k,z in zip(cats,panel.T,strict=True)}
    comparable=[x for _,_,_,x in recipes]
    extra, duplicates=[], {}
    for k,z in zip(cats,panel.T,strict=True):
        prior=next((nm for (nm,*_),x in zip(recipes,comparable,strict=True)
                    if np.allclose(x,z,rtol=0.,atol=1e-12,equal_nan=True)),None)
        if prior is None:
            extra.append(("category:"+k,"UPF",z))
        else:
            duplicates[k]=prior
    base=[(name,kind,val) for name,_,kind,val in recipes]
    original=[("fiber_density","fiber_density",study.numeric["fiber_density"]),
              ("UPF","UPF",study.backgrounds[0].archived["UPF"][:,0])]
    all_recipes=original+base+extra
    require(len({d[0] for d in all_recipes})==len(all_recipes),"V11_DUPLICATE_DIET_LABEL")
    return all_recipes,dict(previous=audit,category_inventory=stats,duplicate_categories=duplicates,
                            energy_days=len(daily)),path_map


def inventory():
    output={}
    for directory in sorted(path for path in RAW.iterdir() if path.is_dir()):
        output[directory.name]={}
        for path in sorted(directory.glob("*.parquet")):
            f=pq.ParquetFile(path)
            output[directory.name][path.name]=dict(rows=f.metadata.num_rows,columns=len(f.schema_arrow))
    return output
