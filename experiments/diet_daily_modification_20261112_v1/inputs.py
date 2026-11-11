"""Read the existing discovery cohort and matched diet/CGM days inside TRE."""

from hashlib import sha256
import importlib.util
from io import BytesIO
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


KEY = ["participant_id", "cohort", "research_stage"]
DAY = KEY + ["date"]
INDEX = KEY + ["input_row", "final_role", "split_unit_id", "age_calendar_years", "sex_code"]
FIBER = "fiber_density_g_per_1000kcal"
MACROS = [name + "_complete_sum" for name in ("carbohydrate_g", "lipid_g", "protein_g")]
ENERGY_ANCHOR = "calories_kcal_complete_sum"
EXP2 = "experiment2_effect_modification_20261111_v1_101"


def digest(path):
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_read(path):
    return json.loads(path.read_bytes())


def integer_ids(table):
    values = table["participant_id"].to_pylist()
    dtype = table.schema.field("participant_id").type
    if pa.types.is_floating(dtype):
        exact = 2 ** {16: 11, 32: 24, 64: 53}[dtype.bit_width]
        require(all(v is None or (np.isfinite(v) and abs(v) < exact and v.is_integer())
                    for v in values), "Floating participant keys cannot be normalized losslessly")
    else:
        require(pa.types.is_integer(dtype), "Unexpected participant ID storage")
    return [None if v is None else int(v) for v in values]


def frame(table):
    result = table.drop(KEY).to_pandas(ignore_metadata=True)
    result[KEY[0]] = pd.Series(integer_ids(table), dtype=object)
    for name in KEY[1:]:
        result[name] = pd.Series(table[name].to_pylist(), dtype=object)
    return result


def unique(frame, keys):
    require(frame[keys].notna().all().all() and not frame.duplicated(keys).any(),
            "Missing or duplicate identity/date")


def projected_day(path, columns, people):
    schema = pq.ParquetFile(path, pre_buffer=False).schema_arrow
    require(set(DAY + columns).issubset(schema.names), f"Missing required fields in {path.name}")
    groups = people.groupby(KEY[1:], observed=True, sort=False)
    filters = [[(KEY[0], "in", group[KEY[0]].tolist()), (KEY[1], "=", cohort),
                (KEY[2], "=", visit)] for (cohort, visit), group in groups]
    data = frame(pq.read_table(path, columns=DAY + columns, filters=filters,
                               pre_buffer=False, use_threads=False))
    scoped = data.merge(people[KEY], on=KEY, how="inner", validate="many_to_one")
    require(len(scoped) == len(data), "Projected input contains out-of-scope identities")
    scoped["date"] = pd.to_datetime(scoped.date.astype("string").str[:10], format="%Y-%m-%d",
                                    errors="coerce")
    # The upstream nutrient registry can contain undated records. They cannot enter a date join.
    scoped = scoped.loc[scoped.date.notna()].copy()
    unique(scoped, DAY)
    return scoped


def assemble(people, food, cgm, nutrients, min_days):
    for column in ("energy_eligible", "fiber_eligible"):
        require(food[column].dropna().isin([True, False]).all(), "Unexpected dietary QC flag")
    require(cgm.base_eligible.dropna().isin([True, False]).all(), "Unexpected CGM QC flag")
    energy_ok = food.energy_eligible.eq(True)
    require(food.loc[energy_ok, "energy_kcal"].between(500, 4000).all(),
            "Daily energy values contradict upstream eligibility")
    good = energy_ok & food.fiber_eligible.eq(True) & np.isfinite(food[FIBER]) & food[FIBER].ge(0)
    joint = food.loc[good].merge(nutrients, on=DAY, how="left", validate="one_to_one")
    require(np.allclose(joint.energy_kcal, joint[ENERGY_ANCHOR], rtol=1e-8, atol=1e-4),
            "Dietary source energy anchors disagree")
    macros = joint[MACROS].to_numpy(float)
    kcal = macros * np.array([4.0, 9.0, 4.0])
    total = kcal.sum(axis=1)
    valid = np.isfinite(kcal).all(axis=1) & (kcal >= 0).all(axis=1) & (total > 0)
    joint = joint.loc[valid].copy()
    joint["carb"] = 100 * kcal[valid, 0] / total[valid]
    joint["protein"] = 100 * kcal[valid, 2] / total[valid]
    joint["fiber"] = joint[FIBER]
    valid_cgm = cgm.base_eligible.eq(True) & np.isfinite(cgm.Y_value)
    cgm = cgm.loc[valid_cgm, DAY + ["Y_value"]]
    daily = joint.merge(cgm, on=DAY, validate="one_to_one").merge(
        people[INDEX], on=KEY, validate="many_to_one"
    )
    require(daily.age_calendar_years.notna().all() and daily.sex_code.isin([0, 1]).all(),
            "Unexpected host covariate support")
    n_days = daily.groupby("input_row").date.transform("size")
    daily = daily.loc[n_days >= min_days].sort_values(["input_row", "date"]).reset_index(drop=True)
    require(not daily.empty, "No participants with enough aligned diet/CGM days")
    return daily


def read(package, cfg):
    inputs = package / "inputs/discovery_v2"
    days = package / "inputs/daily"
    exp2 = package / "experiment2_Effect_Modification/v1/runs_TRE_ONLY" / EXP2
    complete = json_read(exp2 / "COMPLETE.json")
    registration = exp2 / "registration.json"
    require(digest(registration) == complete["registration"], "Experiment 2 registration changed")
    sources = json_read(registration)["sources"]
    index_path = inputs / "discovery_covariates_food_TRE_ONLY.parquet"
    reference = package / "experiments/diet_nested_stability_20261103_v4/results/TRE_ONLY/final_predictions.npz"
    prep_path = package / "preprocessing/code/preprocessing_v1_20261030.py"
    food_path = days / "food_days_discovery_TRE_ONLY.parquet"
    nutrient_path = package.parent / "analysis/shared/food_day_qc_TRE_ONLY.parquet"
    cgm_path = days / "cgm_days_discovery_TRE_ONLY.parquet"
    raw_paths = {s: inputs / (s + "_raw_TRE_ONLY.npz") for s in ("oral", "gut")}
    taxa_paths = {s: inputs / (s + "_taxa.json") for s in raw_paths}
    registered = [index_path, reference, prep_path, food_path, nutrient_path,
                  *raw_paths.values(), *taxa_paths.values()]
    fingerprints = {str(p): digest(p) for p in [*registered, cgm_path, registration]}
    require(all(fingerprints[str(p)] == sources.get(str(p)) for p in registered),
            "One or more Experiment 2 registered inputs changed")
    index = frame(pq.ParquetFile(index_path, pre_buffer=False).read(columns=INDEX, use_threads=False))
    unique(index, KEY)
    require(len(index) == cfg["expected_discovery_rows"] and index.final_role.eq("discovery").all()
            and np.array_equal(index.input_row, np.arange(len(index))),
            "Unexpected discovery cohort or input order")
    with np.load(BytesIO(reference.read_bytes()), allow_pickle=False) as data:
        eligible = np.isfinite(data["y"])
        folds = data["outer"]
    require(eligible.shape == folds.shape == (2, len(index))
            and np.array_equal(eligible[0], eligible[1]), "Inconsistent reference folds")
    people = index.loc[eligible[0]].copy()
    require(len(people) == cfg["expected_people"]
            and people.split_unit_id.notna().all()
            and people.split_unit_id.nunique() == cfg["expected_households"],
            "Frozen cohort size or household allocation changed")
    for repeat in folds:
        require(set(repeat[eligible[0]]) == set(range(5))
                and people.assign(fold=repeat[eligible[0]]).groupby("split_unit_id").fold.nunique().eq(1).all(),
                "Household leakage or unexpected reference fold")
    food = projected_day(food_path, ["energy_eligible", "fiber_eligible", "energy_kcal", FIBER], people)
    cgm_schema = pq.ParquetFile(cgm_path, pre_buffer=False).schema_arrow.names
    optional = [name for name in ("base_eligible", "Y_unit") if name in cgm_schema]
    cgm = projected_day(cgm_path, ["Y_value", *optional], people)
    if "Y_unit" in cgm:
        require(set(cgm.Y_unit.dropna().unique()) == {"mg/dL"}, "Unexpected CGM unit")
    if "base_eligible" not in cgm:
        cgm["base_eligible"] = True
    nutrients = projected_day(nutrient_path, [ENERGY_ANCHOR, *MACROS], people)
    daily = assemble(people, food, cgm, nutrients, cfg["min_common_days"])
    raw, taxa = {}, {}
    for site, path in raw_paths.items():
        taxa[site] = json_read(taxa_paths[site])
        with np.load(BytesIO(path.read_bytes()), allow_pickle=False) as data:
            raw[site] = data["abundance"]
        require(raw[site].shape == (len(index), len(taxa[site])), "Taxon array alignment changed")
    spec = importlib.util.spec_from_file_location("hpp_frozen_clr", prep_path)
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    return daily, index, raw, taxa, folds, prep.MicrobiomeCLR, fingerprints
