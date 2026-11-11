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
import math
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
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


def native_frame(table):
    # Use Arrow scalars for keys so pandas cannot turn nullable int64 into float64.
    values = table["participant_id"].to_pylist()
    storage = table.schema.field("participant_id").type
    if pa.types.is_floating(storage):
        precision = {16: 11, 32: 24, 64: 53}[storage.bit_width]
        limit = 2 ** precision
        require(all(value is None or (
            math.isfinite(value) and abs(value) < limit and float(value).is_integer()
        ) for value in values), "FLOAT_ID_NOT_LOSSLESS")
        values = [None if value is None else int(value) for value in values]
    else:
        require(pa.types.is_integer(storage), "PARTICIPANT_ID_STORAGE_UNSUPPORTED")
    frame = table.drop(KEY).to_pandas(ignore_metadata=True)
    frame["participant_id"] = pd.Series(values, index=frame.index, dtype=object)
    for column in KEY[1:]:
        frame[column] = pd.Series(table[column].to_pylist(), index=frame.index, dtype=object)
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
    with pq.ParquetFile(index_path, pre_buffer=False) as parquet:
        index_table = parquet.read(
            columns=KEY + ["input_row", "final_role", "split_unit_id", "n_fiber_days", "fiber_density"],
            use_threads=False,
        )
    emit("DISCOVERY_ID", storage=str(index_table.schema.field("participant_id").type))
    index = native_frame(index_table)
    unique(index, KEY, "DISCOVERY_KEYS_INVALID")
    require(index.participant_id.is_unique, "DISCOVERY_PARTICIPANT_IDS_NOT_UNIQUE")
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
    return fixed, reg["sources"]


DAILY_DIR = HANDOFF / "inputs/daily"


def snapshot(name, sources=None):
    path = DAILY_DIR / (name + "_TRE_ONLY.parquet")
    if sources is not None:
        require(sha(path) == sources.get(str(path)), "REGISTERED_FOOD_SNAPSHOT_CHANGED")
    with pq.ParquetFile(path, pre_buffer=False) as parquet:
        emit("FILE", table=name, rows=parquet.metadata.num_rows,
             id_type=str(parquet.schema_arrow.field("participant_id").type))
    return path


def qualified(path, kind, fixed):
    dataset = ds.dataset(path, format="parquet")
    field = ds.field
    if kind == "food":
        rule = (field("energy_eligible") == True) & (field("fiber_eligible") == True)
        rule &= pc.is_finite(field("energy_kcal")) & (field("energy_kcal") > 0) & pc.is_finite(field(FIBER))
        columns = DAY + [FIBER]
    else:
        rule = pc.is_finite(field("Y_value"))
        if "base_eligible" in dataset.schema.names:
            rule &= field("base_eligible") == True
        columns = DAY
    selected = native_frame(dataset.to_table(columns=columns, filter=rule, use_threads=False))
    scoped = selected.merge(fixed[KEY], on=KEY, validate="many_to_one")
    scoped["date"] = calendar(scoped.date)
    unique(scoped, DAY, "INVALID_OR_DUPLICATE_" + kind.upper() + "_DAY")
    emit("VALID_DAYS", table=kind, eligible_rows=len(selected),
         matched_rows=len(scoped), matched_people=int(scoped.participant_id.nunique()))
    return scoped


def food_anchor(food, fixed):
    observed = food.groupby(KEY, observed=True)[FIBER].agg(food_days="size", food_mean="mean").reset_index()
    compared = fixed.merge(observed, on=KEY, how="left", validate="one_to_one")
    days_agree = compared.food_days.fillna(0).eq(compared.n_fiber_days.fillna(0))
    means_agree = np.isclose(compared.food_mean.to_numpy(float), compared.fiber_density.to_numpy(float),
                             rtol=1e-8, atol=1e-6, equal_nan=True)
    emit("FOOD_ANCHOR", fixed_people=len(fixed), day_count_agrees=int(days_agree.sum()),
         mean_fiber_agrees=int(means_agree.sum()))
    require(days_agree.all() and means_agree.all(), "FROZEN_FOOD_ANCHOR_MISMATCH")


def w20_overlap(same, fixed):
    path = snapshot("common_days_no_scores")
    with pq.ParquetFile(path, pre_buffer=False) as parquet:
        history = native_frame(parquet.read(columns=DAY, use_threads=False))
    history = history.merge(fixed[KEY], on=KEY, validate="many_to_one")
    history["date"] = calendar(history.date)
    unique(history, DAY, "INVALID_OR_DUPLICATE_W20_DAY")
    both = history.merge(same[DAY], on=DAY, validate="one_to_one")
    emit("W20_COMMON", in_fixed_people=int(history.participant_id.nunique()),
         in_fixed_days=len(history), recovered_days=len(both))


def daily_report(same, fixed, source):
    stats = same.groupby("participant_id", observed=True)[FIBER].agg(["size", "std", "min", "max"])
    households = same.merge(fixed[KEY + ["split_unit_id"]], on=KEY, validate="many_to_one")
    enough = stats[stats["size"] >= 3]
    emit("SAME_DAY", source=source, days=len(same), people=len(stats),
         households=int(households.split_unit_id.nunique()),
         days_per_person_Q25_Q50_Q75=quantiles(stats["size"]),
         ge3=int((stats["size"] >= 3).sum()), ge5=int((stats["size"] >= 5).sum()),
         ge7=int((stats["size"] >= 7).sum()), ge10=int((stats["size"] >= 10).sum()))
    emit("FIBER_WITHIN", units="g/1000kcal", people_with_ge3_days=len(enough),
         within_person_SD_Q25_Q50_Q75=quantiles(enough["std"]),
         people_with_range_ge5=int(((enough["max"] - enough["min"]) >= 5).sum()))


def dates(fixed, same):
    path = PROJECT / "analysis/shared/sample_registry_TRE_ONLY.parquet"
    lock = json_file(PROJECT / "config/w13_sample_selection_lock.json")
    require(sha(path) == lock["sample_registry_sha256"], "SAMPLE_REGISTRY_CHANGED")
    col = "descriptive_sample_calendar_date"
    with pq.ParquetFile(path, pre_buffer=False) as parquet:
        reg = native_frame(parquet.read(
            columns=KEY + ["site", "sample_lock_status", col], use_threads=False,
        ))
    reg = reg.merge(fixed[KEY], on=KEY, validate="many_to_one")
    reg = reg.loc[reg.sample_lock_status.eq("locked_W13_independent_full_source_verification_passed")]
    unique(reg, KEY + ["site"], "DUPLICATE_LOCKED_SPECIMEN")
    for site in ("oral", "gut"):
        sub = reg.loc[reg.site.eq(site), KEY + [col]].copy()
        sub[col] = calendar(sub[col])
        paired = same.merge(sub.dropna(subset=[col]), on=KEY, validate="many_to_one")
        paired["signed_gap"] = (paired.date - paired[col]).dt.days
        paired["gap"] = paired.signed_gap.abs()
        nearest = paired.groupby("participant_id", observed=True).gap.min()
        near_days = paired.loc[paired.gap <= 30].groupby("participant_id", observed=True).size()
        later_days = paired.loc[paired.signed_gap.between(1, 30)].groupby("participant_id", observed=True).size()
        emit("SAMPLE_DATES", site=site, locked_people=len(sub), dated_people=int(sub[col].notna().sum()),
             dated_with_matched_day=len(nearest), nearest_abs_days_Q25_Q50_Q75=quantiles(nearest),
             nearest_le7=int((nearest <= 7).sum()), nearest_le30=int((nearest <= 30).sum()),
             ge3_matched_days_within30=int((near_days >= 3).sum()),
             ge3_days_1_to_30_after_registry_date=int((later_days >= 3).sum()))


def main():
    print("EXP2_DAILY_LINK | discovery aggregates", flush=True)
    stage = "cohort"
    try:
        fixed, sources = cohort()
        stage = "food"
        food = qualified(snapshot("food_days_discovery", sources), "food", fixed)
        food_anchor(food, fixed)
        stage = "cgm"
        cgm = qualified(snapshot("cgm_days_discovery"), "cgm", fixed)
        stage = "same_day"
        same = food.merge(cgm, on=DAY, validate="one_to_one")
        daily_report(same, fixed, "handoff_native_keys")
        stage = "w20"
        w20_overlap(same, fixed)
        stage = "sample_dates"
        dates(fixed, same[DAY])
    except AuditStop as exc:
        emit("EXP2_DAILY_LINK_STOP", stage=stage, reason=str(exc))
        sys.exit(1)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        emit("EXP2_DAILY_LINK_STOP", stage=stage, error=type(exc).__name__)
        sys.exit(1)
    print("EXP2_DAILY_LINK_DONE", flush=True)


if __name__ == "__main__":
    main()
PY
)
