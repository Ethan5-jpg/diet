"""Inventory and align additional numeric outcomes to discovery identities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from food_features import DAY, KEY, SOURCES, normalise, projected
from run_v2 import require
from source import Outcome


RAW = Path("/home/ec2-user/studies/hpp_datasets")
CGM = "../analysis/shared/cgm_visit_qualified_daily_TRE_ONLY.parquet"
BLOOD = ("hba1c", "glucose", "triglycerides", "cholesterol", "total_cholesterol", "hdl_cholesterol",
         "ldl_cholesterol", "non_hdl_cholesterol", "alt", "ast", "ggt", "alkaline_phosphatase",
         "albumin", "total_bilirubin", "direct_bilirubin", "creatinine", "uric_acid", "crp",
         "hs_crp", "insulin", "ferritin", "tsh")
ANTHRO = ("bmi", "weight", "waist_circumference", "hip_circumference", "neck_circumference",
          "waist_to_hip_ratio")
BODY = ("body_comp_total_tissue_percent_fat", "body_comp_total_lean_mass",
        "body_comp_trunk_fat_mass", "body_comp_trunk_tissue_percent_fat", "body_comp_trunk_lean_mass",
        "body_comp_android_fat_mass", "body_comp_android_tissue_percent_fat",
        "body_comp_gynoid_fat_mass", "body_comp_gynoid_tissue_percent_fat")
PRESSURE = {f"seated_{name}": (f"sitting_blood_pressure_{name}", f"sitting_second_blood_pressure_{name}")
            for name in ("systolic", "diastolic", "pulse_rate")}


def source_paths(root):
    paths = {name:RAW/name/(name+".parquet") for name in
             ("blood_tests", "anthropometrics", "body_composition", "blood_pressure")}
    paths.update(cgm=root/CGM, food_days=root/SOURCES["day"])
    return {name:path.resolve(strict=True) for name, path in paths.items()}


def aligned(series, n):
    return series.reindex(range(n)).to_numpy(float)


def distinct_measurements(frame, name):
    """An identity with conflicting values receives no outcome for that measurement."""
    groups = frame[["input_row", name]].dropna().groupby("input_row")[name]
    counts = groups.nunique()
    use = counts[counts==1].index
    return groups.first().reindex(use), int((counts>1).sum())


def scoped(path, keys, fields, index, ids):
    raw = projected(path, [*keys, *fields], ids)
    return raw.merge(index[[*keys, "input_row"]], on=keys, how="inner", validate="many_to_one")


def numeric(x, bounds=(0., np.inf)):
    val = pd.to_numeric(x, errors="coerce")
    lo, hi = bounds
    return val.where(np.isfinite(val) & val.between(lo, hi))


def read_new_outcomes(paths, study):
    index = pd.DataFrame(dict(zip(KEY, study.keys, strict=True)))
    index["input_row"] = np.arange(study.n)
    ids = index.participant_id.astype("int64").tolist()
    result, audit, missing = {}, {}, {}

    def add(name, values, unit, source, **detail):
        arr = aligned(values, study.n)
        result[name] = Outcome(arr, unit)
        audit[name] = dict(source=source, unit=unit, observed=int(np.isfinite(arr).sum()), **detail)

    def visit(source, mappings):
        path = paths[source]
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        good = {name:tuple(c for c in fields if c in schema) for name, fields in mappings.items()}
        fields = sorted(set(c for fs in good.values() for c in fs))
        missing[source] = sorted(name for name, fs in good.items() if not fs)
        frame = scoped(path, KEY, fields, index, ids)
        for name, cols in good.items():
            if not cols:
                continue
            values = pd.concat([numeric(frame[c]).where(lambda x:x>0.) for c in cols], axis=1)
            frame[name] = values.mean(axis=1)
            obs, conflicts = distinct_measurements(frame, name)
            add(name, obs, "source units", source, conflicting=conflicts, rule="exact discovery visit")

    visit("anthropometrics", {k:(k,) for k in ANTHRO})
    visit("body_composition", {k.removeprefix("body_comp_"):(k,) for k in BODY})
    visit("blood_pressure", PRESSURE)

    days = normalise(scoped(paths["food_days"], KEY, ["date", "energy_eligible", "energy_kcal"], index, ids))
    first = days.loc[days.energy_eligible.eq(True) & days.energy_kcal.gt(0.)].groupby("input_row").date.min()
    blood_schema = set(pq.ParquetFile(paths["blood_tests"]).schema_arrow.names)
    assays = {k:f"bt__{k}_float_value" for k in BLOOD if f"bt__{k}_float_value" in blood_schema}
    missing["blood_tests"] = sorted(set(BLOOD)-assays.keys())
    require("hba1c" in assays, "V10_HBA1C_FIELD_MISSING")
    blood = scoped(paths["blood_tests"], KEY[:2], ["collection_date", *assays.values()], index, ids)
    blood["date"] = pd.to_datetime(blood.collection_date, errors="coerce").dt.normalize()
    blood["lag"] = (blood.input_row.map(first)-blood.date).dt.days
    blood = blood.loc[blood.lag.between(1,365)].copy()
    for name, column in assays.items():
        blood[column] = numeric(blood[column], (3.,18.) if name=="hba1c" else (0.,np.inf))
        valid = blood.loc[blood[column].notna()]
        latest = valid.loc[valid.lag.eq(valid.groupby("input_row").lag.transform("min"))]
        obs, conflicts = distinct_measurements(latest, column)
        add("blood_"+name, obs, "percent" if name=="hba1c" else "source units", "blood_tests",
            conflicting=conflicts, rule="latest valid blood draw 1-365 days before first eligible food day")

    names = ["glucose_mean_mg_dL", "glucose_sd_mg_dL", "glucose_CV_pct"]
    flags = ["person_phenotype_selected_connection", "point_qc_verified", "cgm_day_eligible"]
    cgm = normalise(scoped(paths["cgm"], KEY, ["date", "unit", *names, *flags], index, ids))
    cgm = cgm.loc[cgm[flags].eq(True).all(axis=1) & cgm.unit.eq("mg/dL")].copy()
    require(not cgm.duplicated(["input_row", "date"]).any(), "V10_CGM_MULTIPLE_DAY_RECORDS")
    for name, field, unit, op in (
        ("cgm_within_day_sd", names[1], "mg/dL", "mean"),
        ("cgm_within_day_cv", names[2], "percent", "mean"),
        ("cgm_between_day_sd", names[0], "mg/dL", "std"),
    ):
        cgm[field] = numeric(cgm[field])
        group = cgm.groupby("input_row")[field]
        enough = group.count() >= 3
        add(name, group.agg(op).loc[enough], unit, "cgm", rule="at least three QC days on selected connection")
    return result, audit, missing
