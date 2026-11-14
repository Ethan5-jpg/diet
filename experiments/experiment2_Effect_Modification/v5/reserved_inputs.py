"""Load reserved-role predictors with W20 alignment and V10 context rules."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path
import stat

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


KEY = ["participant_id", "cohort", "research_stage"]
HOUSEHOLD = "split_unit_id"
DAY = KEY + ["date"]
SITES = ("oral", "gut")
SAMPLE_LOCK = "locked_W13_independent_full_source_verification_passed"
HOST = ["age_calendar_years", "sex_code", "smoking_current_status_label"]
FOOD = DAY + ["energy_kcal", "fiber_g", "energy_eligible", "fiber_eligible"]
DENSITY = "fiber_density_g_per_1000kcal"
CONTEXT_PATH = "experiments/diet_model_context_20261108_v10/context.py"
W13_PUBLICATION = "data_qc/w13_microbiome_qc_2026-10-29/final_artifact_manifest.json"
PREPROCESS_PREFLIGHT = "data_qc/preprocessing_20261030_v1/preflight_TRE_ONLY.json"
W20_RECEIPTS = "data_qc/w20_simplified_20261101_v2/access_receipts"
W20_RECEIPT_NAME = {"oral": "004.json", "gut": "005.json"}
W20_POINTER = "config/w20_simplified_20261101_v2/active_execution.json"
W20_SPEC = "data_qc/w20_simplified_20261101_v2/execution_spec.json"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def digest(path):
    hasher = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _normalize_ids(values, float_bits=64):
    integers = []
    limit = 2 ** {16: 11, 32: 24, 64: 53}[float_bits]
    for value in values:
        if isinstance(value, (float, np.floating)):
            require(math.isfinite(value) and abs(value) < limit and float(value).is_integer(),
                    "NONEXACT_PARTICIPANT_ID")
        else:
            require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
                    "INVALID_PARTICIPANT_ID")
        integer = int(value)
        require(-(2**63) <= integer < 2**63, "PARTICIPANT_ID_OUTSIDE_INT64")
        integers.append(integer)
    return np.asarray(integers, dtype=np.int64)


def _check_keys(frame, key, code):
    require(not frame[key].isna().any().any() and not frame.duplicated(key).any(), code)


def _arrow_frame(table):
    storage = table.schema.field(KEY[0]).type
    require(pa.types.is_integer(storage) or pa.types.is_floating(storage), "UNSUPPORTED_PARTICIPANT_ID_STORAGE")
    bits = storage.bit_width if pa.types.is_floating(storage) else 64
    ids = _normalize_ids(table[KEY[0]].to_pylist(), bits)
    # Avoid a pandas float round trip for nullable Arrow integer fields.
    frame = table.drop([KEY[0]]).to_pandas(ignore_metadata=True)
    frame.insert(0, KEY[0], ids)
    require(not frame[KEY].isna().any().any(), "NULL_PARTICIPANT_KEY")
    return frame


def project_population(path, columns, population, unique_by=None):
    """Project explicit columns for a frozen baseline roster using native key types."""
    require(set(KEY) <= set(columns) and len(columns) == len(set(columns)), "INVALID_PROJECTION_COLUMNS")
    _check_keys(population, KEY, "INVALID_PROJECTION_POPULATION")
    require(len(population) > 0 and population.cohort.eq("10k").all()
            and population.research_stage.eq("00_00_visit").all(), "INVALID_PROJECTION_SCOPE")
    ids = _normalize_ids(population.participant_id.tolist())
    source = ds.dataset(path, format="parquet")
    require(set(columns) <= set(source.schema.names), "MISSING_PROJECTED_FIELDS")
    storage = source.schema.field(KEY[0]).type
    require(pa.types.is_integer(storage) or pa.types.is_floating(storage), "UNSUPPORTED_PARTICIPANT_ID_STORAGE")
    bits = storage.bit_width if pa.types.is_floating(storage) else 64
    values = ids.astype(f"float{bits}") if pa.types.is_floating(storage) else ids.tolist()
    native_ids = pa.array(values, type=storage)
    require(np.array_equal(_normalize_ids(native_ids.to_pylist(), bits), ids), "NONEXACT_SOURCE_ID_FILTER")
    filters = (ds.field(KEY[0]).isin(native_ids) & (ds.field("cohort") == "10k")
               & (ds.field("research_stage") == "00_00_visit"))
    table = source.scanner(columns=columns, filter=filters, use_threads=False, batch_size=1024).to_table()
    frame = _arrow_frame(table)
    require(frame.participant_id.isin(ids).all() and frame.cohort.eq("10k").all()
            and frame.research_stage.eq("00_00_visit").all(), "PROJECTION_SCOPE_MISMATCH")
    if unique_by is not None:
        _check_keys(frame, unique_by, "INVALID_PROJECTED_KEYS")
    return frame


def _role_roster(path, pointer, discovery, hashes):
    reference = json.loads(pointer.read_text(encoding="utf-8"))
    require(reference.get("roles_sha256") == hashes[path], "RESERVED_ROLE_HASH_MISMATCH")
    columns = KEY + [HOUSEHOLD, "final_role"]
    roles = _arrow_frame(pq.ParquetFile(path, pre_buffer=False).read(columns=columns, use_threads=False))
    _check_keys(roles, columns[:-1], "INVALID_FROZEN_ROLE_KEYS")
    require(roles.participant_id.is_unique and not roles[columns].isna().any().any()
            and roles.cohort.eq("10k").all() and roles.research_stage.eq("00_00_visit").all()
            and set(roles.final_role) == {"discovery", "confirmation_reserved"}
            and roles.groupby(HOUSEHOLD).final_role.nunique().max() == 1, "INVALID_FROZEN_ROLE_ASSIGNMENT")
    archived = discovery.index[KEY + [HOUSEHOLD, "final_role"]].copy()
    archived[KEY[0]] = _normalize_ids(archived.participant_id.tolist())
    _check_keys(archived, KEY, "INVALID_DISCOVERY_ROLE_KEYS")
    matched = roles.loc[roles.final_role.eq("discovery"), columns].merge(
        archived, on=columns, how="outer", validate="one_to_one", indicator=True)
    require(matched["_merge"].eq("both").all(), "DISCOVERY_AND_FROZEN_ROSTER_MISMATCH")
    return roles.loc[roles.final_role.eq("confirmation_reserved")].reset_index(drop=True)


def _calendar(frame, *, drop_missing=False):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date.astype("string").str.slice(0, 10), errors="raise")
    return frame.loc[frame.date.notna()].copy() if drop_missing else frame


def _food_inputs(path, roles):
    daily = _calendar(project_population(path, FOOD + [DENSITY], roles))
    _check_keys(daily, DAY, "INVALID_RESERVED_FOOD_DAYS")
    flags = ["energy_eligible", "fiber_eligible"]
    require(all(daily[name].dropna().isin([True, False]).all() for name in flags), "INVALID_RESERVED_FOOD_FLAGS")
    energy_ok = daily.energy_eligible.eq(True).fillna(False)
    fiber_ok = daily.fiber_eligible.eq(True).fillna(False)
    require(energy_ok.all() and daily.energy_kcal.between(500, 4000).all(), "INVALID_PRIMARY_FOOD_ENERGY")
    eligible = daily.loc[fiber_ok]
    energy, fiber, density = (eligible[c].to_numpy(float, na_value=np.nan) for c in ("energy_kcal", "fiber_g", DENSITY))
    require(np.isfinite(fiber).all() and (fiber >= 0).all() and np.isfinite(density).all()
            and np.allclose(density, 1000 * fiber / energy, rtol=1e-8, atol=1e-6), "INVALID_RESERVED_FIBER_DENSITY")
    # W20's named UPF energy columns also describe the primary energy-day subset.
    food = daily.groupby(KEY, as_index=False, observed=True).agg(
        n_energy_days=("date", "size"), mean_daily_energy_UPF=("energy_kcal", "mean"))
    summary = eligible.groupby(KEY, as_index=False, observed=True).agg(
        n_fiber_days=("date", "size"), fiber_density=(DENSITY, "mean"),
        mean_daily_energy_fiber=("energy_kcal", "mean"))
    return food.merge(summary, on=KEY, how="left", validate="one_to_one"), daily


def _microbiome_inputs(registry_path, matrix_paths, roles, discovery):
    columns = KEY + ["site", "sample_id"]
    registry = project_population(registry_path, columns + ["sample_lock_status"], roles,
                                  unique_by=KEY + ["site"])
    _check_keys(registry, ["sample_id"], "INVALID_LOCKED_SAMPLE_IDS")
    require(registry.site.isin(SITES).all() and registry.sample_lock_status.eq(SAMPLE_LOCK).all(),
            "INVALID_RESERVED_SAMPLE_LOCK")
    raw, taxa, audit = {}, {}, {}
    for site, path in matrix_paths.items():
        schema = pq.ParquetFile(path, pre_buffer=False).schema_arrow.names
        native_taxa = [name for name in schema if name.startswith(("k__", "s__")) or "|s__" in name]
        taxa[site] = list(discovery.taxa[site])
        require(taxa[site] == native_taxa and len(native_taxa) == len(set(native_taxa)) and len(native_taxa) > 1,
                "RESERVED_TAXON_DICTIONARY_MISMATCH")
        table = project_population(path, columns + taxa[site], roles, unique_by=KEY)
        _check_keys(table, ["sample_id"], "INVALID_SELECTED_SAMPLE_IDS")
        require(table.site.eq(site).all(), "INVALID_SELECTED_MATRIX_SITE")
        locked = registry.loc[registry.site.eq(site), KEY + ["sample_id"]]
        matched = locked.merge(table[KEY + ["sample_id"]], on=KEY + ["sample_id"],
                               how="outer", validate="one_to_one", indicator=True)
        require(matched["_merge"].eq("both").all(), "RESERVED_LOCKED_SAMPLE_MATRIX_MISMATCH")
        matrix = roles[KEY].merge(table[KEY + taxa[site]], on=KEY, how="left", validate="one_to_one")
        raw[site] = matrix[taxa[site]].to_numpy(float, na_value=np.nan)
        valid = (np.isfinite(raw[site]).all(1) & (raw[site] >= 0).all(1)
                 & np.isclose(raw[site].sum(1), 1, rtol=0, atol=1e-3))
        raw[site][~valid] = np.nan
        audit[site] = {"locked_rows": len(locked), "usable_profiles": int(valid.sum()),
                       "unusable_locked_profiles": len(locked) - int(valid.sum())}
    return raw, taxa, audit


def _context_module(path):
    spec = spec_from_file_location("hpp_v5_archived_context", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_registered_inputs(package, food_manifest, food_path, registry_lock, registry, hashes):
    manifest = json.loads(food_manifest.read_text(encoding="utf-8"))
    food_records = [row for row in manifest if row.get("relative_path") == "inputs/daily/food_days_discovery_TRE_ONLY.parquet"]
    require(len(food_records) == 1 and food_records[0].get("source") == str(food_path)
            and food_records[0].get("source_sha256") == hashes[food_path], "PRIMARY_FOOD_SOURCE_DIFFERS_FROM_W20_HANDOFF")
    historical_food = package / food_records[0]["relative_path"]
    require(digest(historical_food) == food_records[0].get("sha256"), "DISCOVERY_FOOD_PROJECTION_MISMATCH")

    lock = json.loads(registry_lock.read_text(encoding="utf-8"))
    require(lock.get("sample_registry") == str(registry) and lock.get("sample_registry_sha256") == hashes[registry]
            and lock.get("apply_same_lock_across_outcome_modules") is True
            and lock.get("no_reselection_on_outcome_window_or_effect") is True, "W13_SAMPLE_REGISTRY_LOCK_MISMATCH")


def _read_receipt(path, hashes):
    payload = path.read_bytes()
    hashes[path] = sha256(payload).hexdigest()
    return json.loads(payload)


def _match_artifact(record, path, hashes):
    require(record.get("path") == str(path) and record.get("sha256") == hashes[path]
            and record.get("bytes") == path.stat().st_size, "W13_PREDICTOR_SOURCE_HASH_MISMATCH")


def _receipt_exists(path):
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        return False
    require(stat.S_ISREG(mode), "INVALID_MICROBIOME_RECEIPT_FILE")
    return True


def _preflight_covers_sources(receipt, registry_lock, sources, hashes):
    """Use the historical file hashes even if the optional snapshot list is partial."""
    require(receipt.get("status") == "metadata_preflight_complete"
            and receipt.get("new_scientific_Y_reads") == 0, "INVALID_HISTORICAL_PREPROCESS_PREFLIGHT")
    snapshots = receipt.get("snapshots", [])
    require(isinstance(snapshots, list), "INVALID_HISTORICAL_PREFLIGHT_SNAPSHOTS")
    lock_records = [entry for entry in snapshots if entry.get("path") == str(registry_lock)]
    require(len(lock_records) <= 1, "HISTORICAL_PREFLIGHT_DUPLICATE_LOCK")
    if lock_records and lock_records[0].get("sha256") is not None:
        _match_artifact(lock_records[0], registry_lock, hashes)

    inputs = receipt.get("inputs", {})
    require(isinstance(inputs, dict), "INVALID_HISTORICAL_PREFLIGHT_INPUTS")
    complete = True
    for path in sources:
        record = inputs.get(path.name, {})
        require(isinstance(record, dict), "INVALID_HISTORICAL_PREFLIGHT_SOURCE")
        # Validate any recorded facts even when another record is incomplete.
        for field, expected in (("path", str(path)), ("sha256", hashes[path]), ("bytes", path.stat().st_size)):
            actual = record.get(field)
            if actual is not None:
                require(actual == expected, "W13_PREDICTOR_SOURCE_HASH_MISMATCH")
        complete &= record.get("exists") is True and all(
            record.get(field) is not None for field in ("path", "sha256", "bytes"))
    return complete


def _check_matrix_provenance(project, registry_lock, registry, matrices, hashes):
    publication = project / W13_PUBLICATION
    preflight = project / PREPROCESS_PREFLIGHT
    sources = (registry, *matrices.values())
    manifests = [(publication, "w13_publication_manifest")]
    if project.parent.name == "diet_gut_cgm_v2" and project.parent.parent.name == "studies":
        runtime = project.parents[2] / "multisystem_runtime_20261021" / publication.parent.name / publication.name
        manifests.append((runtime, "w13_runtime_manifest"))
    for receipt_path, label in manifests:
        if not _receipt_exists(receipt_path):
            continue
        records = _read_receipt(receipt_path, hashes)
        for path in sources:
            exact = [record for record in records if record.get("path") == str(path)]
            require(len(exact) == 1, "W13_PREDICTOR_REGISTRATION_MISMATCH")
            _match_artifact(exact[0], path, hashes)
        return label
    if _receipt_exists(preflight):
        receipt = _read_receipt(preflight, hashes)
        if _preflight_covers_sources(receipt, registry_lock, sources, hashes):
            return "archived_preprocessing_preflight"
    provisional = publication.with_name("provisional_artifact_manifest.json")
    if _receipt_exists(provisional):
        records = _read_receipt(provisional, hashes)
        for path in matrices.values():
            old_path = provisional.with_name(path.name)
            exact = [record for record in records if record.get("path") == str(old_path)]
            require(len(exact) == 1 and exact[0].get("sha256") == hashes[path]
                    and exact[0].get("bytes") == path.stat().st_size, "W13_PROVISIONAL_MATRIX_HASH_MISMATCH")
        return "archived_w13_provisional_manifest"
    receipt_paths = {site: project / W20_RECEIPTS / W20_RECEIPT_NAME[site] for site in matrices}
    spec = project / W20_SPEC
    require(all(_receipt_exists(path) for path in (spec, *receipt_paths.values())),
            "W13_MATRIX_PROVENANCE_UNAVAILABLE")
    pointer = _read_receipt(project / W20_POINTER, hashes)
    design = _read_receipt(spec, hashes)
    require(pointer.get("execution_spec") == str(spec) and pointer.get("execution_spec_sha256") == hashes[spec]
            and design.get("version") == "W19_W20_simplified_20261101_v2"
            and isinstance(design.get("roles_sha256"), str), "W20_MATRIX_ROLE_RECEIPT_MISMATCH")
    for site, path in matrices.items():
        receipt = _read_receipt(receipt_paths[site], hashes)
        require(receipt.get("path") == str(path) and receipt.get("source_sha256") == hashes[path]
                and receipt.get("purpose") == "v2_discovery_raw_fold_inputs" and receipt.get("contains_Y") is False
                and receipt.get("filter_before_materialization") == "sealed discovery IDs,10k,00_00_visit"
                and receipt.get("roles_sha256") == design["roles_sha256"], "W20_MATRIX_SOURCE_HASH_MISMATCH")
    return "archived_w20_matrix_access_receipts"


def _build_context(module, nutrients_path, host, daily, index, eligible):
    population = index.loc[eligible].copy()
    nutrients = project_population(nutrients_path, DAY + module.ANCHORS +
                                   [name + "_complete_sum" for name in module.NUTRIENTS], population)
    extra = module.aggregate(daily.merge(population[KEY], on=KEY, validate="many_to_one"),
                             _calendar(nutrients, drop_missing=True),
                             host.merge(population[KEY], on=KEY, validate="one_to_one"), population)
    for name, values in extra["blocks"].items():
        aligned = np.full((len(index), values.shape[1]), np.nan)
        aligned[eligible] = values
        extra["blocks"][name] = aligned
    alcohol = np.full(len(index), None, dtype=object)
    alcohol[eligible] = extra["alcohol"]
    extra["alcohol"] = alcohol
    return extra


@dataclass(frozen=True)
class ReservedInputs:
    index: pd.DataFrame
    raw: dict
    taxa: dict
    extra: dict
    predictor_eligible: np.ndarray
    hashes: dict
    audit: dict

    def verify_sources(self):
        require(all(digest(path) == expected for path, expected in self.hashes.items()),
                "RESERVED_PREDICTOR_SOURCE_CHANGED")


def load_reserved(root, discovery):
    """Return an aligned predictor-only roster and a model-independent eligibility mask."""
    package = Path(root).resolve().parent.parent
    project = package.parent
    shared = project / "analysis/shared"
    role_path = package / "inputs/discovery_v2/roles_v2_full_TRE_ONLY.parquet"
    pointer = project / W20_POINTER
    food_path = project / "data_qc/preprocessing_20261030_v1/food_qc_v1/food_daily_primary_TRE_ONLY.parquet"
    food_manifest = package / "manifests/derived_daily_sources.json"
    host_path = shared / "covariate_registry_TRE_ONLY.parquet"
    nutrients_path = shared / "food_day_qc_TRE_ONLY.parquet"
    context_path = package / CONTEXT_PATH
    registry_path = shared / "sample_registry_TRE_ONLY.parquet"
    registry_lock = project / "config/w13_sample_selection_lock.json"
    matrices = {site: shared / f"{site}_selected_abundance_TRE_ONLY.parquet" for site in SITES}
    sources = (role_path, pointer, food_path, food_manifest, host_path, nutrients_path, context_path,
               registry_path, registry_lock, *matrices.values())
    hashes = {path: digest(path) for path in sources}
    historical = getattr(discovery, "source_hashes", {})
    require(all(historical.get(path, value) == value for path, value in hashes.items()),
            "RESERVED_CONTEXT_DIFFERS_FROM_DISCOVERY_SOURCE")
    _check_registered_inputs(package, food_manifest, food_path, registry_lock, registry_path, hashes)
    microbiome_provenance = _check_matrix_provenance(project, registry_lock, registry_path, matrices, hashes)
    print("EXP2_V5_MICROBIOME_PROVENANCE", json.dumps({"source": microbiome_provenance}), flush=True)
    roles = _role_roster(role_path, pointer, discovery, hashes)
    context = _context_module(context_path)
    food, daily = _food_inputs(food_path, roles)
    host = project_population(host_path, KEY + HOST + context.LIFE + [context.ALCOHOL], roles, unique_by=KEY)
    index = roles.merge(host[KEY + HOST], on=KEY, how="left", validate="one_to_one")
    index = index.merge(food, on=KEY, how="left", validate="one_to_one")
    for name in HOST[:2]:
        index[name] = pd.to_numeric(index[name], errors="coerce")
    index["smoking_status"] = index.smoking_current_status_label.astype("string").fillna("__UNKNOWN__").replace("", "__UNKNOWN__")
    index["input_row"] = np.arange(len(index))
    raw, taxa, sites = _microbiome_inputs(registry_path, matrices, index, discovery)
    fiber_ok = (index.n_fiber_days.ge(3).fillna(False).to_numpy(dtype=bool)
                & np.isfinite(index.fiber_density.to_numpy(float, na_value=np.nan))
                & np.isfinite(index.mean_daily_energy_fiber.to_numpy(float, na_value=np.nan)))
    host_ok = (np.isfinite(index.age_calendar_years.to_numpy(float, na_value=np.nan))
               & index.sex_code.isin([0, 1]).to_numpy(dtype=bool))
    eligible = fiber_ok & host_ok & np.isfinite(raw["oral"]).all(1) & np.isfinite(raw["gut"]).all(1)
    require(eligible.any(), "NO_USABLE_RESERVED_PREDICTOR_PROFILES")
    extra = _build_context(context, nutrients_path, host, daily, index, eligible)
    # The historical key name is retained inside V10; report the actual reserved role here.
    context_audit = dict(extra["audit"])
    context_audit["reserved_context_people"] = context_audit.pop("discovery_people")
    extra["audit"] = context_audit
    result = ReservedInputs(index=index, raw=raw, taxa=taxa, extra=extra, predictor_eligible=eligible,
                            hashes=hashes, audit={"role_people": len(index),
                            "role_households": int(index[HOUSEHOLD].nunique()),
                            "fiber_ge3_people": int(fiber_ok.sum()), "host_usable_people": int(host_ok.sum()),
                            "predictor_eligible_people": int(eligible.sum()),
                            "predictor_eligible_households": int(index.loc[eligible, HOUSEHOLD].nunique()),
                            "microbiome": sites, "microbiome_provenance": microbiome_provenance,
                            "context": context_audit})
    result.verify_sources()
    return result
