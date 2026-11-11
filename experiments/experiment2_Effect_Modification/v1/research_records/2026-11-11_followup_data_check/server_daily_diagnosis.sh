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



DAILY_DIR = HANDOFF / "inputs/daily"
FOOD_SOURCE = PROJECT / "data_qc/preprocessing_20261030_v1/food_qc_v1/food_daily_primary_TRE_ONLY.parquet"
CGM_SOURCE = PROJECT / "analysis/Daily_CGM/fiber_density/base_table_TRE_ONLY.parquet"


def typed_scope(schema, fixed):
    """Filter canonical tables to exact discovery person/visit keys before reading them."""
    field = ds.field
    pid_type = schema.field("participant_id").type
    filters = []
    for (cohort_value, stage), group in fixed.groupby(KEY[1:], observed=True, sort=False):
        ids = group.participant_id.tolist()
        pid = field("participant_id")
        if pa.types.is_integer(pid_type):
            require(group.participant_id.str.fullmatch(r"0|[1-9][0-9]*").all(), "NONINTEGER_DISCOVERY_ID")
            ids = [int(value) for value in ids]
        elif pa.types.is_floating(pid_type):
            require(group.participant_id.str.fullmatch(r"0|[1-9][0-9]*").all(), "NONINTEGER_DISCOVERY_ID")
            limit = 2 ** (24 if pid_type.bit_width == 32 else 53)
            require(all(int(value) < limit for value in ids), "FLOAT_ID_PRECISION")
            ids = [float(value) for value in ids]
        else:
            pid = pid.cast(pa.string())
        filters.append(pid.isin(ids) & (field("cohort").cast(pa.string()) == cohort_value)
                       & (field("research_stage").cast(pa.string()) == stage))
    combined = filters[0]
    for other in filters[1:]:
        combined = combined | other
    return combined


def source_keys(frame, pid_type):
    if pa.types.is_floating(pid_type):
        # Physical numeric IDs can be stored as whole-valued floats in a Parquet table.
        pid = frame.participant_id.to_numpy(dtype=float, na_value=np.nan)
        limit = 2 ** (24 if pid_type.bit_width == 32 else 53)
        exact = np.isfinite(pid) & (pid >= 0) & (pid < limit) & (pid == np.floor(pid))
        frame["participant_id"] = pd.Series(np.where(exact, pid, np.nan), index=frame.index).astype("Int64")
    return keys(frame)


def inventory(path, label, manifest):
    if not path.is_file():
        emit("FILE", table=label, status="missing")
        return None
    pf = pq.ParquetFile(path, pre_buffer=False)
    try:
        n, schema = pf.metadata.num_rows, pf.schema_arrow
    finally:
        pf.close()
    rel = str(path.relative_to(HANDOFF)) if HANDOFF in path.parents else None
    emit("FILE", table=label, rows=n, manifest_rows=manifest.get(rel),
         id_type=str(schema.field("participant_id").type) if "participant_id" in schema.names else None)
    return schema


def qualify(path, label, kind, fixed, manifest, canonical=False):
    schema = inventory(path, label, manifest)
    if schema is None:
        return None
    available = set(schema.names)
    required = set(DAY + (["energy_eligible", "fiber_eligible", "energy_kcal", FIBER] if kind == "food"
                          else ["Y_value"]))
    if not required <= available:
        emit("STAGES", table=label, missing=sorted(required - available))
        return None
    dataset = ds.dataset(path, format="parquet")
    scope = typed_scope(schema, fixed) if canonical else None
    pid_type = schema.field("participant_id").type
    identity = dataset.to_table(columns=KEY, filter=scope, use_threads=False).to_pandas(ignore_metadata=True)
    # Report each match stage on the discovery-only snapshot or on the typed raw scope.
    literal = int(identity.participant_id.astype("string").str.strip().isin(fixed.participant_id).sum())
    identity = source_keys(identity, pid_type)
    compound = identity.merge(fixed[KEY], on=KEY, validate="many_to_one")
    field = ds.field
    if kind == "food":
        rule = (field("energy_eligible") == True) & (field("fiber_eligible") == True)
        rule &= pc.is_finite(field("energy_kcal")) & (field("energy_kcal") > 0) & pc.is_finite(field(FIBER))
        cols = DAY + [FIBER]
    else:
        rule = pc.is_finite(field("Y_value"))
        if "base_eligible" in available:
            rule &= field("base_eligible") == True
        cols = DAY
    selected = dataset.to_table(columns=cols, filter=rule if scope is None else scope & rule,
                                use_threads=False).to_pandas(ignore_metadata=True)
    matched = source_keys(selected, pid_type).merge(fixed[KEY], on=KEY, validate="many_to_one")
    matched["date"] = calendar(matched.date)
    unique(matched, DAY, "INVALID_OR_DUPLICATE_DAY_" + label.upper())
    emit("STAGES", table=label, scanned_keys=len(identity), literal_id_rows=literal,
         composite_key_rows=len(compound), passes_flags=len(selected), valid_matched_rows=len(matched))
    return matched


def prepared(fixed, manifest):
    label = "prepared_common"
    path = DAILY_DIR / "common_days_no_scores_TRE_ONLY.parquet"
    schema = inventory(path, label, manifest)
    if schema is None:
        return None
    if not set(DAY + [FIBER]) <= set(schema.names):
        emit("STAGES", table=label, missing=sorted(set(DAY + [FIBER]) - set(schema.names)))
        return None
    with pq.ParquetFile(path, pre_buffer=False) as pf:
        table = pf.read(columns=DAY + [FIBER], use_threads=False).to_pandas(ignore_metadata=True)
    matched = source_keys(table, schema.field("participant_id").type).merge(fixed[KEY], on=KEY,
                                                                           validate="many_to_one")
    matched["date"] = calendar(matched.date)
    unique(matched, DAY, "INVALID_OR_DUPLICATE_PREPARED_DAY")
    require(np.isfinite(matched[FIBER]).all(), "NONFINITE_PREPARED_FIBER")
    emit("STAGES", table=label, valid_matched_rows=len(matched), people=int(matched.participant_id.nunique()))
    return matched


def combine(food, cgm):
    if food is None or cgm is None or food.empty or cgm.empty:
        return None
    paired = food.merge(cgm, on=DAY, validate="one_to_one")
    return paired if len(paired) else None


def daily_report(same, fixed, source):
    stats = same.groupby("participant_id", observed=True)[FIBER].agg(["size", "std", "min", "max"])
    households = same.merge(fixed[KEY + ["split_unit_id"]], on=KEY, validate="many_to_one")
    enough = stats[stats["size"] >= 3]
    emit("SAME_DAY", source=source, days=len(same), people=len(stats),
         households=int(households.split_unit_id.nunique()),
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



def main():
    print("EXP2_DAILY_DIAG | discovery aggregates", flush=True)
    stage = "cohort"
    try:
        fixed = cohort()
        manifest_path = HANDOFF / "manifests/derived_daily_sources.json"
        manifest = {x["relative_path"]: x["rows"] for x in json_file(manifest_path)} if manifest_path.is_file() else {}
        stage = "handoff_tables"
        food = qualify(DAILY_DIR / "food_days_discovery_TRE_ONLY.parquet", "handoff_food", "food", fixed, manifest)
        cgm = qualify(DAILY_DIR / "cgm_days_discovery_TRE_ONLY.parquet", "handoff_cgm", "cgm", fixed, manifest)
        ready = prepared(fixed, manifest)
        same, source = combine(food, cgm), "handoff_exact"
        if same is None:
            stage = "canonical_tables"
            food = qualify(FOOD_SOURCE, "canonical_food", "food", fixed, manifest, canonical=True)
            cgm = qualify(CGM_SOURCE, "canonical_cgm", "cgm", fixed, manifest, canonical=True)
            same, source = combine(food, cgm), "canonical_exact"
        if same is None and ready is not None and not ready.empty:
            same, source = ready, "prepared_common_already_eligible"
        if same is None:
            emit("SAME_DAY", status="no_verified_join")
        else:
            daily_report(same, fixed, source)
            stage = "sample_dates"
            dates(fixed, same[DAY])
    except AuditStop as exc:
        emit("EXP2_DAILY_DIAG_STOP", stage=stage, reason=str(exc))
        sys.exit(1)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        emit("EXP2_DAILY_DIAG_STOP", stage=stage, error=type(exc).__name__)
        sys.exit(1)
    print("EXP2_DAILY_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
PY
)
