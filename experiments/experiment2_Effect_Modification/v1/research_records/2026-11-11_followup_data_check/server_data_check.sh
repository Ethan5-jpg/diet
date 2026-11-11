# Paste this entire block into the HPP server terminal.
(
set -e
cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiment2_Effect_Modification/v1
IFS= read -r hpp_audit_python < ../../experiments/diet_effect_modification_20261109_v12/.python_path
"$hpp_audit_python" -B -X pycache_prefix=/dev/null -u - <<'PY'
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


HERE = Path.cwd().resolve()
HANDOFF = HERE.parent.parent
PROJECT = HANDOFF.parent
RUN_ID = "experiment2_effect_modification_20261111_v1_101"
RUN = HERE / "runs_TRE_ONLY" / RUN_ID
KEY = ["participant_id", "cohort", "research_stage"]
DAY = KEY + ["date"]
FIBER = "fiber_density_g_per_1000kcal"


class AuditStop(ValueError):
    pass


def require(ok, label):
    if not ok:
        raise AuditStop(label)


def json_file(path):
    return json.loads(path.read_bytes())


def sha(path):
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def emit(label, **values):
    print(label, json.dumps(values, separators=(",", ":")), flush=True)


def keys(frame):
    for name in KEY:
        frame[name] = frame[name].astype("string").str.strip()
    return frame


def calendar(series):
    # Use the source calendar date without shifting the time zone.
    return pd.to_datetime(series.astype("string").str.slice(0, 10), format="%Y-%m-%d", errors="coerce")


def unique(frame, columns, label):
    require(frame[columns].notna().all().all() and not frame.duplicated(columns).any(), label)


def quantiles(series):
    if series.empty:
        return None
    return [round(float(x), 2) for x in series.quantile([0.25, 0.50, 0.75])]


def cohort():
    cfg = json_file(HERE / "config.json")
    require(cfg["version"] == RUN_ID, "RUN_ID_CHANGED")
    marker = json_file(RUN / "COMPLETE.json")
    reg_path = RUN / "registration.json"
    require(sha(reg_path) == marker["registration"], "RUN_REGISTRATION_CHANGED")
    reg = json_file(reg_path)
    index_path = HANDOFF / "inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet"
    require(sha(index_path) == reg["sources"].get(str(index_path)), "DISCOVERY_INDEX_CHANGED")
    index = pq.ParquetFile(index_path, pre_buffer=False).read(
        columns=KEY + ["input_row", "final_role", "split_unit_id"], use_threads=False
    ).to_pandas(ignore_metadata=True)
    keys(index)
    unique(index, KEY, "DISCOVERY_KEYS_INVALID")
    require(index.final_role.eq("discovery").all() and
            np.array_equal(index.input_row.to_numpy(), np.arange(cfg["expected_discovery_rows"])),
            "DISCOVERY_ORDER_OR_ROLE_CHANGED")
    masks = []
    for repeat in range(cfg["outer_repeats"]):
        parts = []
        for fold in range(cfg["outer_folds"]):
            receipt_path = RUN / ("r%d_f%d.json" % (repeat, fold))
            require(sha(receipt_path) == marker["folds"].get(receipt_path.name), "FOLD_RECEIPT_CHANGED")
            receipt = json_file(receipt_path)
            require((receipt["registration"], receipt["repeat"], receipt["fold"]) ==
                    (marker["registration"], repeat, fold), "FOLD_IDENTITY_CHANGED")
            payload = receipt_path.with_suffix(".npz").read_bytes()
            require(sha256(payload).hexdigest() == receipt["arrays_sha"], "FOLD_ARRAYS_CHANGED")
            with np.load(BytesIO(payload), allow_pickle=False) as arrays:
                parts.append(arrays["rows"])
        row = np.concatenate(parts)
        require(np.issubdtype(row.dtype, np.integer) and len(np.unique(row)) == len(row)
                and len(row) == cfg["expected_people"] and np.all((row >= 0) & (row < len(index))),
                "FROZEN_ROWS_INVALID")
        masks.append(np.sort(row))
    require(all(np.array_equal(masks[0], m) for m in masks[1:]), "FROZEN_REPEATS_DIFFER")
    fixed = index.iloc[masks[0]].copy()
    require(fixed.split_unit_id.notna().all() and
            fixed.split_unit_id.nunique() == cfg["expected_households"], "HOUSEHOLD_COUNTS_CHANGED")
    emit("COHORT", people=len(fixed), households=int(fixed.split_unit_id.nunique()), run=RUN_ID)
    return fixed


def daily_rows(name, cols, rule, fixed):
    path = HANDOFF / "inputs/daily" / (name + "_days_discovery_TRE_ONLY.parquet")
    src = ds.dataset(path, format="parquet")
    has_cgm_flag = name == "cgm" and "base_eligible" in src.schema.names
    if has_cgm_flag:
        rule &= ds.field("base_eligible") == True
    df = src.to_table(columns=cols, filter=rule, use_threads=False).to_pandas(ignore_metadata=True)
    df = keys(df).merge(fixed[KEY], on=KEY, validate="many_to_one")
    df["date"] = calendar(df.date)
    unique(df, DAY, "INVALID_OR_DUPLICATE_" + name.upper() + "_DAY")
    return df, has_cgm_flag


def daily(fixed):
    field = ds.field
    food, _ = daily_rows("food", DAY + [FIBER],
                         (field("energy_eligible") == True) & (field("fiber_eligible") == True)
                         & pc.is_finite(field("energy_kcal")) & (field("energy_kcal") > 0)
                         & pc.is_finite(field(FIBER)), fixed)
    # Test daily outcome availability in Arrow and return only the qualifying keys.
    cgm, used_cgm_flag = daily_rows("cgm", DAY, pc.is_finite(field("Y_value")), fixed)
    same = food.merge(cgm, on=DAY, validate="one_to_one")
    count = same.groupby("participant_id", observed=True)[FIBER].agg(["size", "std", "min", "max"])
    enough = count[count["size"] >= 3]
    emit("DAILY", food_days=len(food), food_people=int(food.participant_id.nunique()),
         cgm_days=len(cgm), cgm_people=int(cgm.participant_id.nunique()), cgm_base_flag_used=used_cgm_flag,
         matched_days=len(same), matched_people=len(count),
         matched_households=int(same.merge(fixed[KEY + ["split_unit_id"]], on=KEY,
                                           validate="many_to_one").split_unit_id.nunique()),
         people_ge3=int((count["size"] >= 3).sum()), people_ge5=int((count["size"] >= 5).sum()),
         people_ge7=int((count["size"] >= 7).sum()), people_ge10=int((count["size"] >= 10).sum()))
    emit("FIBER_WITHIN", units="g/1000kcal", people_with_ge3_days=len(enough),
         within_person_SD_Q25_Q50_Q75=quantiles(enough["std"]),
         people_with_range_ge5=int(((enough["max"] - enough["min"]) >= 5).sum()))
    return same[DAY]


def dates(fixed, same):
    path = PROJECT / "analysis/shared/sample_registry_TRE_ONLY.parquet"
    lock = json_file(PROJECT / "config/w13_sample_selection_lock.json")
    require(sha(path) == lock["sample_registry_sha256"], "SAMPLE_REGISTRY_CHANGED")
    col = "descriptive_sample_calendar_date"
    reg = pq.ParquetFile(path, pre_buffer=False).read(
        columns=KEY + ["site", "sample_lock_status", col], use_threads=False
    ).to_pandas(ignore_metadata=True)
    reg = keys(reg).merge(fixed[KEY], on=KEY, validate="many_to_one")
    reg = reg.loc[reg.sample_lock_status.eq("locked_W13_independent_full_source_verification_passed")]
    unique(reg, KEY + ["site"], "DUPLICATE_LOCKED_SPECIMEN")
    for site in ("oral", "gut"):
        sub = reg.loc[reg.site.eq(site), KEY + [col]].copy()
        sub[col] = calendar(sub[col])
        paired = same.merge(sub.dropna(subset=[col]), on=KEY, validate="many_to_one")
        paired["gap"] = (paired.date - paired[col]).dt.days.abs()
        nearest = paired.groupby("participant_id", observed=True).gap.min()
        near_days = paired.loc[paired.gap <= 30].groupby("participant_id", observed=True).size()
        emit("SAMPLE_DATES", site=site, locked_people=len(sub), dated_people=int(sub[col].notna().sum()),
             dated_with_matched_day=len(nearest), nearest_abs_days_Q25_Q50_Q75=quantiles(nearest),
             nearest_le7=int((nearest <= 7).sum()), nearest_le30=int((nearest <= 30).sum()),
             ge3_matched_days_within30=int((near_days >= 3).sum()))


def meals(fixed):
    roots = [Path("/home/ec2-user/Desktop/HPP/Data/meal_cgm_exploration"),
             Path("/home/ec2-user/Desktop/HPP/meal_cgm_exploration")]
    rel = "outputs/meal_20260924T143630_752200Z/private/02_meals.csv"
    path = next((root / rel for root in roots if (root / rel).is_file()), None)
    if path is None:
        emit("MEAL", status="historical_prepared_file_not_at_known_paths")
        return
    required = KEY + ["meal_id", "local_date", "analysis_ready", "dietary_fiber_g", "calories_kcal"]
    columns = pd.read_csv(path, nrows=0).columns
    if not set(required).issubset(columns):
        emit("MEAL", status="required_fields_unavailable", missing=sorted(set(required) - set(columns)))
        return
    flag = "time_semantics_verified"
    use = required + ([flag] if flag in columns else [])
    parts = []
    for chunk in pd.read_csv(path, usecols=use, dtype="string", chunksize=50000):
        scoped = keys(chunk).merge(fixed[KEY], on=KEY, validate="many_to_one")
        ready = scoped.analysis_ready.str.lower().str.strip().isin(["true", "1"])
        parts.append(scoped.loc[ready])
    if not parts:
        emit("MEAL", status="prepared_file_empty")
        return
    ready = pd.concat(parts, ignore_index=True)
    unique(ready, ["meal_id"], "DUPLICATE_OR_MISSING_READY_MEAL_ID")
    ready["local_date"] = calendar(ready.local_date)
    fiber = pd.to_numeric(ready.dietary_fiber_g, errors="coerce").to_numpy(float, na_value=np.nan)
    energy = pd.to_numeric(ready.calories_kcal, errors="coerce").to_numpy(float, na_value=np.nan)
    complete = ready.loc[np.isfinite(fiber) & (fiber >= 0) & np.isfinite(energy) & (energy > 0)
                         & ready.local_date.notna()]
    person = complete.groupby("participant_id", observed=True).local_date.agg(["size", "nunique"])
    timing = int(complete[flag].str.lower().str.strip().isin(["true", "1"]).sum()) if flag in columns else None
    emit("MEAL", status="historical_build", ready_meals=len(ready), ready_people=int(ready.participant_id.nunique()),
         fiber_complete_meals=len(complete), fiber_complete_people=len(person),
         people_ge5_meals_ge3_days=int(((person["size"] >= 5) & (person["nunique"] >= 3)).sum()),
         people_ge10_meals_ge3_days=int(((person["size"] >= 10) & (person["nunique"] >= 3)).sum()),
         complete_meals_marked_time_verified=timing)


def main():
    print("EXP2_DATA_CHECK | discovery aggregates", flush=True)
    stage = "cohort"
    try:
        fixed = cohort()
        stage = "daily"
        same = daily(fixed)
        stage = "sample_dates"
        dates(fixed, same)
        stage = "prepared_meals"
        meals(fixed)
    except AuditStop as exc:
        emit("EXP2_DATA_CHECK_STOP", stage=stage, reason=str(exc))
        sys.exit(1)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        emit("EXP2_DATA_CHECK_STOP", stage=stage, error=type(exc).__name__)
        sys.exit(1)
    print("EXP2_DATA_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
PY
)
