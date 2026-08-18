"""Read-only audit for the approved seven-component HPP modified AHEI.

This step deliberately does not create a score.  It freezes the protocol,
checks event/population/shared AMED inputs, and inventories optional lookup
tables so unavailable nutrients can never be silently interpreted as zero.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-18-ahei-input-audit-v5"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR.parent

DEFAULT_EVENTS_CSV = (
    DATA_DIR / "Transfer" / "diet_logging" / "diet_logging_events.csv"
)
DEFAULT_RAW_EVENTS_CSV = (
    DATA_DIR / "Transfer" / "diet_logging" / "raw_diet_logging_events.csv"
)
DEFAULT_POPULATION_CSV = (
    DATA_DIR / "Transfer" / "population" / "population.csv"
)
LEGACY_POPULATION_CSV = (
    DATA_DIR / "Transfer" / "diet_logging" / "population.csv"
)
DEFAULT_AMED_SHARED_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "amed"
    / "amed_participant_component_intakes.csv"
)
DEFAULT_TRANSFER_DIR = DATA_DIR / "Transfer"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "00_input_audit" / "ahei"
DEFAULT_COHORT = "10k"
DEFAULT_RESEARCH_STAGE = "00_00_visit"

SCORE_PROTOCOL_NAME = "modified_AHEI_2010_7_component"
SCORE_COMPONENT_COUNT = 7
SCORE_RAW_MIN = 0.0
SCORE_RAW_MAX = 70.0
EXCLUDED_SCORE_COMPONENTS = ("trans_fat", "pufa", "sodium")
STANDARD_DRINK_G = 14.0
ALCOHOL_NONDRINKER_SCORE = 2.5
HPP_KCAL_PER_SERVING = {
    "vegetables": 30.0,
    "fruit": 70.0,
    "ssb_plus_fruit_juice": 70.0,
    "red_plus_processed_meat": 220.0,
}
NUTS_LEGUMES_G_PER_SERVING = 28.35
WHOLE_GRAIN_PROXY_VARIABLE = "whole_grain_proxy_g"
FULL_MAPPING_MISSING_LABEL_FOOD_ID_REFERENCE_COUNT = 368

HPP_EXPECTED_COUNTS = {
    "target_participant_count": 9737,
    "excluded_missing_label_food_id_count": 333,
    "excluded_missing_labels_rows": 8052,
    "excluded_missing_label_participant_count": 2663,
    "participant_day_count_preserved": 125374,
}

LABEL_COLUMNS = ["short_food_name", "product_name", "food_category"]
EVENT_KEY_COLUMNS = [
    "participant_id",
    "cohort",
    "research_stage",
    "collection_date",
    "food_id",
    *LABEL_COLUMNS,
]

KEYWORD_GROUPS = {
    "serving_or_portion": (
        "serving",
        "portion",
    ),
    "whole_grain": (
        "whole_grain",
        "whole grain",
        "wholegrain",
        "whole_wheat",
    ),
    "trans_fat": (
        "trans_fat",
        "trans fat",
        "transfat",
        "fatty acids, total trans",
    ),
    "pufa": (
        "pufa",
        "polyunsaturated",
        "poly_unsaturated",
    ),
    "nutrient": (
        "nutrient",
        "nutrition",
        "nutritional",
    ),
}

CANDIDATE_COLUMNS = [
    "path",
    "column_count",
    "has_food_id",
    "has_participant_id",
    "long_format_nutrient_schema",
    "matched_keyword_groups",
    "matched_columns",
    "likely_ahei_lookup",
    "header_error",
]


def normalize_text(series: pd.Series) -> pd.Series:
    """Strip text and represent blank strings as missing."""
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def parse_boolean(series: pd.Series, column_name: str) -> pd.Series:
    """Parse common boolean spellings without converting missing values."""
    normalized = normalize_text(series).str.casefold()
    true_values = {"true", "t", "1", "yes", "y"}
    false_values = {"false", "f", "0", "no", "n"}
    unknown = normalized.notna() & ~normalized.isin(true_values | false_values)
    if unknown.any():
        values = sorted(normalized.loc[unknown].astype(str).unique())
        raise ValueError(
            "{}含无法识别的布尔值：{}".format(
                column_name, ", ".join(values[:10])
            )
        )
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[normalized.isin(true_values)] = True
    result.loc[normalized.isin(false_values)] = False
    return result


def validate_columns(
    columns: Iterable[str], required: Iterable[str], source_name: str
) -> None:
    """Raise a helpful error when an input table lacks required fields."""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(
            "{}缺少字段：{}".format(source_name, ", ".join(missing))
        )


def resolve_population_csv(path: Path) -> Path:
    """Prefer the server layout while supporting the local snapshot layout."""
    path = Path(path)
    if path.is_file():
        return path
    if path == DEFAULT_POPULATION_CSV and LEGACY_POPULATION_CSV.is_file():
        return LEGACY_POPULATION_CSV
    return path


def find_keyword_hits(columns: Iterable[str]) -> List[Dict[str, str]]:
    """Return one record for every lookup requirement matched by a header."""
    normalized_columns = [str(column).strip().casefold() for column in columns]
    hits = []
    for group, keywords in KEYWORD_GROUPS.items():
        matched = []
        for original, normalized in zip(columns, normalized_columns):
            if any(keyword in normalized for keyword in keywords):
                matched.append(str(original))
        if matched:
            hits.append(
                {
                    "keyword_group": group,
                    "matched_columns": "|".join(sorted(set(matched))),
                }
            )
    return hits


def has_long_format_nutrient_schema(columns: Iterable[str]) -> bool:
    """Identify generic long nutrient tables as unverified candidates."""
    normalized = {
        str(column).strip().casefold().replace(" ", "_")
        for column in columns
    }
    name_fields = {
        "nutrient",
        "nutrient_name",
        "nutrition_name",
        "component_name",
    }
    value_fields = {"amount", "value", "nutrient_value", "quantity"}
    unit_fields = {"unit", "units", "measurement_unit"}
    join_fields = {"food_id", "participant_id", "registrationcode"}
    return bool(
        normalized & name_fields
        and normalized & value_fields
        and normalized & unit_fields
        and normalized & join_fields
    )


def _read_header(path: Path) -> List[str]:
    suffix = path.suffix.casefold()
    if suffix in {".tsv", ".txt"}:
        table = pd.read_csv(path, sep="\t", nrows=0, encoding="utf-8-sig")
    else:
        table = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    return table.columns.astype(str).tolist()


def scan_candidate_sources(
    root: Path,
    excluded_paths: Set[Path],
    file_limit: int,
) -> Tuple[pd.DataFrame, int, bool]:
    """Scan CSV/TSV headers only; zero file_limit means an unlimited scan."""
    root = Path(root)
    excluded = {Path(path).resolve() for path in excluded_paths}
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".csv", ".tsv"}
    )
    rows = []
    scanned = 0
    limit_reached = False
    for path in paths:
        if path.resolve() in excluded:
            continue
        if file_limit > 0 and scanned >= file_limit:
            limit_reached = True
            break
        scanned += 1
        try:
            columns = _read_header(path)
            hits = find_keyword_hits(columns)
            groups = sorted({hit["keyword_group"] for hit in hits})
            matched_columns = sorted(
                {
                    column
                    for hit in hits
                    for column in hit["matched_columns"].split("|")
                    if column
                }
            )
            long_schema = has_long_format_nutrient_schema(columns)
            normalized = {
                str(column).strip().casefold().replace(" ", "_")
                for column in columns
            }
            has_food_id = "food_id" in normalized
            has_participant_id = bool(
                normalized & {"participant_id", "registrationcode"}
            )
            likely = bool(
                (has_food_id or has_participant_id)
                and (groups or long_schema)
            )
            rows.append(
                {
                    "path": str(path),
                    "column_count": len(columns),
                    "has_food_id": has_food_id,
                    "has_participant_id": has_participant_id,
                    "long_format_nutrient_schema": long_schema,
                    "matched_keyword_groups": "|".join(groups),
                    "matched_columns": "|".join(matched_columns),
                    "likely_ahei_lookup": likely,
                    "header_error": "",
                }
            )
        except Exception as error:  # pragma: no cover - server inventory guard
            rows.append(
                {
                    "path": str(path),
                    "column_count": 0,
                    "has_food_id": False,
                    "has_participant_id": False,
                    "long_format_nutrient_schema": False,
                    "matched_keyword_groups": "",
                    "matched_columns": "",
                    "likely_ahei_lookup": False,
                    "header_error": "{}: {}".format(
                        type(error).__name__, str(error)
                    ),
                }
            )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS), scanned, limit_reached


def candidate_count_for_group(candidates: pd.DataFrame, group: str) -> int:
    """Count header candidates for a requirement, including generic long tables."""
    if candidates.empty:
        return 0
    groups = candidates.get(
        "matched_keyword_groups", pd.Series("", index=candidates.index)
    ).fillna("").astype(str)
    direct = groups.str.split("|").map(lambda values: group in values)
    long_schema = candidates.get(
        "long_format_nutrient_schema",
        pd.Series(False, index=candidates.index),
    ).fillna(False).astype(bool)
    likely = candidates.get(
        "likely_ahei_lookup", pd.Series(False, index=candidates.index)
    ).fillna(False).astype(bool)
    if group in {"whole_grain", "trans_fat", "pufa"}:
        return int((likely & (direct | long_schema)).sum())
    return int((likely & direct).sum())


def _field_valid(series: pd.Series, field: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if field == "weight_g":
        return numeric.gt(0).fillna(False)
    return numeric.ge(0).fillna(False)


def aggregate_event_chunk(
    chunk: pd.DataFrame,
    cohort: str,
    research_stage: str,
    coverage_fields: List[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Aggregate one event chunk while preserving the pre-exclusion index."""
    required = [*EVENT_KEY_COLUMNS, *coverage_fields]
    validate_columns(chunk.columns, required, "diet_logging_events")
    chunk = chunk.loc[:, required].copy()
    for column in [
        "participant_id",
        "cohort",
        "research_stage",
        "collection_date",
        "food_id",
        *LABEL_COLUMNS,
    ]:
        chunk[column] = normalize_text(chunk[column])
    selected = chunk["cohort"].eq(cohort) & chunk["research_stage"].eq(
        research_stage
    )
    chunk = chunk.loc[selected].copy()
    missing_keys = chunk[["participant_id", "collection_date"]].isna().any(axis=1)
    if missing_keys.any():
        raise ValueError(
            "目标事件有 {:,} 行缺少 participant_id/collection_date。".format(
                int(missing_keys.sum())
            )
        )

    all_labels_missing = chunk[LABEL_COLUMNS].isna().all(axis=1)
    retained = ~all_labels_missing
    participant_day_keys = set(
        zip(chunk["participant_id"].astype(str), chunk["collection_date"].astype(str))
    )
    retained_participant_day_keys = set(
        zip(
            chunk.loc[retained, "participant_id"].astype(str),
            chunk.loc[retained, "collection_date"].astype(str),
        )
    )
    excluded_food_ids = set(
        chunk.loc[all_labels_missing, "food_id"].dropna().astype(str)
    )
    excluded_participants = set(
        chunk.loc[all_labels_missing, "participant_id"].dropna().astype(str)
    )

    grouped = chunk.groupby("participant_id", sort=False, dropna=False)
    partial = grouped.size().rename("event_count").to_frame()
    partial["retained_event_count"] = retained.groupby(
        chunk["participant_id"]
    ).sum().astype("int64")
    partial["original_day_count"] = grouped["collection_date"].nunique()
    retained_days = (
        chunk.loc[retained]
        .groupby("participant_id")["collection_date"]
        .nunique()
    )
    partial["retained_day_count"] = retained_days.reindex(
        partial.index, fill_value=0
    )
    for field in coverage_fields:
        valid = retained & _field_valid(chunk[field], field)
        partial["{}_valid_event_count".format(field)] = valid.groupby(
            chunk["participant_id"]
        ).sum().reindex(partial.index, fill_value=0).astype("int64")
    partial = partial.reset_index()

    metrics = {
        "selected_rows": len(chunk),
        "retained_selected_rows": int(retained.sum()),
        "excluded_missing_labels_rows": int(all_labels_missing.sum()),
        "excluded_missing_label_food_ids": excluded_food_ids,
        "excluded_missing_label_participant_ids": excluded_participants,
        "participant_day_keys": participant_day_keys,
        "retained_participant_day_keys": retained_participant_day_keys,
    }
    return partial, metrics


def build_event_coverage(
    participant_summary: pd.DataFrame, coverage_fields: List[str]
) -> pd.DataFrame:
    """Summarize valid-event and strict all-retained-event completeness."""
    rows = []
    retained_counts = participant_summary["retained_event_count"]
    for field in coverage_fields:
        valid = participant_summary["{}_valid_event_count".format(field)]
        strict = retained_counts.gt(0) & valid.eq(retained_counts)
        rows.append(
            {
                "field": field,
                "valid_event_count": int(valid.sum()),
                "retained_event_count": int(retained_counts.sum()),
                "strictly_complete_participant_count": int(strict.sum()),
                "participant_count": int(len(participant_summary)),
            }
        )
    return pd.DataFrame(rows)


def audit_amed_shared_inputs(
    path: Path,
    target_participant_ids: Set[str],
    cohort: str,
    research_stage: str,
) -> Dict[str, object]:
    """Audit the frozen AMED participant alcohol and total-energy outputs."""
    required = [
        "participant_id",
        "cohort",
        "research_stage",
        "alcohol_complete",
        "mean_daily_alcohol_g",
        "total_alcohol_unresolved_events",
        "energy_complete",
        "mean_daily_energy_kcal",
    ]
    if not Path(path).is_file():
        raise FileNotFoundError("找不到 AMED 共享摄入表：{}".format(path))
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    validate_columns(header.columns, required, "AMED 共享摄入表")
    data = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=required,
        dtype={"participant_id": "string", "cohort": "string", "research_stage": "string"},
        low_memory=False,
    )
    for column in ["participant_id", "cohort", "research_stage"]:
        data[column] = normalize_text(data[column])
    data = data.loc[
        data["cohort"].eq(cohort)
        & data["research_stage"].eq(research_stage)
        & data["participant_id"].isin(target_participant_ids)
    ].copy()
    if data["participant_id"].duplicated().any():
        raise ValueError("AMED 共享摄入表在目标范围内存在重复 participant_id。")
    alcohol_complete = parse_boolean(data["alcohol_complete"], "alcohol_complete")
    energy_complete = parse_boolean(data["energy_complete"], "energy_complete")
    alcohol = pd.to_numeric(data["mean_daily_alcohol_g"], errors="coerce")
    unresolved = pd.to_numeric(
        data["total_alcohol_unresolved_events"], errors="coerce"
    )
    energy = pd.to_numeric(data["mean_daily_energy_kcal"], errors="coerce")
    alcohol_source_complete = (
        alcohol_complete.eq(True)
        & alcohol.notna()
        & alcohol.ge(0)
        & unresolved.eq(0)
    )
    energy_source_complete = (
        energy_complete.eq(True) & energy.notna() & energy.gt(0)
    )
    both = alcohol_source_complete & energy_source_complete
    return {
        "target_participant_count": len(target_participant_ids),
        "matched_participant_count": int(len(data)),
        "alcohol_complete_participant_count": int(alcohol_complete.eq(True).sum()),
        "alcohol_g_source_complete_count": int(alcohol_source_complete.sum()),
        "energy_complete_participant_count": int(energy_source_complete.sum()),
        "energy_source_complete_count": int(energy_source_complete.sum()),
        "amed_shared_input_complete_count": int(both.sum()),
        "standard_drink_g": STANDARD_DRINK_G,
        "standard_drink_g_parameter_status": "verified_and_frozen_hpp",
        "drinks_per_day_formula": "mean_daily_alcohol_g / 14.0",
    }


def audit_population_sex(
    path: Path, target_participant_ids: Set[str], cohort: str
) -> Dict[str, int]:
    """Check one valid binary sex value for each target participant."""
    required = ["participant_id", "cohort", "sex"]
    if not Path(path).is_file():
        raise FileNotFoundError("找不到 population.csv：{}".format(path))
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    validate_columns(header.columns, required, "population")
    data = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=required,
        dtype={"participant_id": "string", "cohort": "string"},
        low_memory=False,
    )
    data["participant_id"] = normalize_text(data["participant_id"])
    data["cohort"] = normalize_text(data["cohort"])
    data = data.loc[
        data["cohort"].eq(cohort)
        & data["participant_id"].isin(target_participant_ids)
    ].copy()
    sex = normalize_text(data["sex"])
    valid = sex.str.casefold().isin(
        {"0", "1", "female", "male", "f", "m", "woman", "man"}
    )
    valid_by_participant = valid.groupby(data["participant_id"]).all()
    return {
        "target_participant_count": len(target_participant_ids),
        "sex_matched_count": int(data["participant_id"].nunique()),
        "valid_sex_matched_count": int(valid_by_participant.sum()),
    }


def build_component_availability(
    main_columns: List[str],
    raw_columns: List[str],
    candidates: pd.DataFrame,
    amed_metrics: Dict[str, object],
    sex_metrics: Dict[str, int],
) -> pd.DataFrame:
    """Record readiness for all ten HPP-observed AHEI components."""
    del main_columns, raw_columns, candidates
    target = int(amed_metrics.get("target_participant_count", 0))
    alcohol_ready = (
        int(amed_metrics.get("alcohol_g_source_complete_count", 0)) == target
        and int(sex_metrics.get("valid_sex_matched_count", 0)) == target
    )
    rows = [
        (
            "vegetables",
            True,
            "protocol_defined_mapping_pending",
            "映射后按30 kcal/serving换算，0至5 servings/day线性计分。",
            target,
        ),
        (
            "fruit",
            True,
            "protocol_defined_mapping_pending",
            "完整水果按70 kcal/serving换算；果汁进入不利组件。",
            target,
        ),
        (
            "whole_grains",
            True,
            "proxy_protocol_approved_mapping_pending",
            "whole_grain_proxy_g = 明确全谷物映射事件的有效或批准填补 weight_g；不声称为严格干重。",
            target,
        ),
        (
            "ssb_plus_fruit_juice",
            True,
            "protocol_defined_mapping_pending",
            "含糖饮料与果汁合并，按70 kcal/serving换算。",
            target,
        ),
        (
            "nuts_plus_legumes",
            True,
            "protocol_defined_mapping_pending",
            "坚果和豆类有效重量合并，28.35 g/serving。",
            target,
        ),
        (
            "red_plus_processed_meat",
            True,
            "protocol_defined_mapping_pending",
            "红肉和加工肉按220 kcal/serving换算。",
            target,
        ),
        (
            "trans_fat",
            False,
            "excluded_by_user_protocol",
            "无可靠营养素来源；整项排除，不填补且不按0分处理。",
            0,
        ),
        (
            "pufa",
            False,
            "excluded_by_user_protocol",
            "无可靠营养素来源；整项排除，不填补且不按0分处理。",
            0,
        ),
        (
            "sodium",
            False,
            "excluded_by_user_protocol",
            "按已确认mAHEI-7协议整项排除；保留覆盖QC，不填补、不算十分位也不按0分处理。",
            0,
        ),
        (
            "alcohol",
            True,
            "strictly_available" if alcohol_ready else "shared_input_incomplete",
            "使用AMED纯乙醇g/day，除以14 g/standard drink得到drinks/day；不饮酒者得2.5分。",
            int(amed_metrics.get("alcohol_g_source_complete_count", 0)),
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "ahei_component",
            "included_in_final_score",
            "availability_status",
            "reason",
            "current_eligible_participant_count",
        ],
    )


def _combine_partials(partials: List[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(partials, ignore_index=True)
    numeric_columns = [
        column for column in combined.columns if column != "participant_id"
    ]
    return (
        combined.groupby("participant_id", as_index=False, sort=True)[numeric_columns]
        .sum()
    )


def _qc_rows(section: str, metrics: Dict[str, object]) -> List[Dict[str, object]]:
    return [
        {"section": section, "metric": key, "value": value}
        for key, value in metrics.items()
        if not isinstance(value, (set, dict, list, tuple))
    ]


def audit_ahei_inputs(
    events_csv: Path = DEFAULT_EVENTS_CSV,
    raw_events_csv: Path = DEFAULT_RAW_EVENTS_CSV,
    population_csv: Path = DEFAULT_POPULATION_CSV,
    alcohol_csv: Path = DEFAULT_AMED_SHARED_CSV,
    transfer_dir: Path = DEFAULT_TRANSFER_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    chunk_size: int = 200000,
    candidate_file_limit: int = 0,
    cohort: str = DEFAULT_COHORT,
    research_stage: str = DEFAULT_RESEARCH_STAGE,
    enforce_expected_counts: bool = False,
) -> Tuple[Path, Path, Path, Path, Path]:
    """Run the complete read-only audit and write auditable CSV summaries."""
    population_csv = resolve_population_csv(population_csv)
    paths_to_check = [events_csv, raw_events_csv, population_csv, alcohol_csv]
    for path in paths_to_check:
        if not Path(path).is_file():
            raise FileNotFoundError("找不到输入文件：{}".format(path))
    event_header = pd.read_csv(events_csv, encoding="utf-8-sig", nrows=0)
    raw_header = pd.read_csv(raw_events_csv, encoding="utf-8-sig", nrows=0)
    coverage_fields = ["weight_g", "calories_kcal", "sodium_mg"]
    validate_columns(
        event_header.columns,
        [*EVENT_KEY_COLUMNS, *coverage_fields],
        "diet_logging_events",
    )

    partials = []
    totals = {
        "selected_rows": 0,
        "retained_selected_rows": 0,
        "excluded_missing_labels_rows": 0,
    }
    excluded_food_ids: Set[str] = set()
    excluded_participants: Set[str] = set()
    participant_day_keys: Set[Tuple[str, str]] = set()
    retained_day_keys: Set[Tuple[str, str]] = set()
    for chunk in pd.read_csv(
        events_csv,
        encoding="utf-8-sig",
        usecols=[*EVENT_KEY_COLUMNS, *coverage_fields],
        dtype={column: "string" for column in EVENT_KEY_COLUMNS},
        chunksize=chunk_size,
        low_memory=False,
    ):
        partial, metrics = aggregate_event_chunk(
            chunk, cohort, research_stage, coverage_fields
        )
        if not partial.empty:
            partials.append(partial)
        for metric in totals:
            totals[metric] += int(metrics[metric])
        excluded_food_ids.update(metrics["excluded_missing_label_food_ids"])
        excluded_participants.update(
            metrics["excluded_missing_label_participant_ids"]
        )
        participant_day_keys.update(metrics["participant_day_keys"])
        retained_day_keys.update(metrics["retained_participant_day_keys"])
    if not partials:
        raise ValueError("目标 cohort/research_stage 没有饮食事件。")
    participants = _combine_partials(partials)
    target_ids = set(participants["participant_id"].astype(str))
    event_coverage = build_event_coverage(participants, coverage_fields)
    event_metrics = dict(totals)
    event_metrics.update(
        {
            "target_participant_count": len(target_ids),
            "excluded_missing_label_food_id_count": len(excluded_food_ids),
            "excluded_missing_label_participant_count": len(excluded_participants),
            "participant_day_count_preserved": len(participant_day_keys),
            "retained_participant_day_count": len(retained_day_keys),
            "participant_days_with_only_excluded_events": len(
                participant_day_keys - retained_day_keys
            ),
        }
    )

    if enforce_expected_counts:
        mismatches = []
        for metric, expected in HPP_EXPECTED_COUNTS.items():
            observed = event_metrics.get(metric)
            if observed != expected:
                mismatches.append(
                    "{}={} (expected {})".format(metric, observed, expected)
                )
        if mismatches:
            raise ValueError("HPP baseline contract mismatch: " + "; ".join(mismatches))

    amed_metrics = audit_amed_shared_inputs(
        alcohol_csv, target_ids, cohort, research_stage
    )
    sex_metrics = audit_population_sex(population_csv, target_ids, cohort)
    excluded_scan_paths = {
        Path(events_csv),
        Path(raw_events_csv),
        Path(population_csv),
        Path(alcohol_csv),
    }
    candidates, scanned, limit_reached = scan_candidate_sources(
        Path(transfer_dir), excluded_scan_paths, candidate_file_limit
    )
    availability = build_component_availability(
        event_header.columns.astype(str).tolist(),
        raw_header.columns.astype(str).tolist(),
        candidates,
        amed_metrics,
        sex_metrics,
    )

    protocol_metrics = {
        "script_version": SCRIPT_VERSION,
        "score_protocol_name": SCORE_PROTOCOL_NAME,
        "score_component_count": SCORE_COMPONENT_COUNT,
        "score_raw_min": SCORE_RAW_MIN,
        "score_raw_max": SCORE_RAW_MAX,
        "excluded_score_components": "|".join(EXCLUDED_SCORE_COMPONENTS),
        "standard_drink_g": STANDARD_DRINK_G,
        "alcohol_nondrinker_score": ALCOHOL_NONDRINKER_SCORE,
        "whole_grain_proxy_variable": WHOLE_GRAIN_PROXY_VARIABLE,
        "full_mapping_missing_label_food_id_reference_count": FULL_MAPPING_MISSING_LABEL_FOOD_ID_REFERENCE_COUNT,
        "vegetables_kcal_per_serving": HPP_KCAL_PER_SERVING["vegetables"],
        "fruit_kcal_per_serving": HPP_KCAL_PER_SERVING["fruit"],
        "ssb_plus_fruit_juice_kcal_per_serving": HPP_KCAL_PER_SERVING["ssb_plus_fruit_juice"],
        "red_plus_processed_meat_kcal_per_serving": HPP_KCAL_PER_SERVING["red_plus_processed_meat"],
        "nuts_legumes_g_per_serving": NUTS_LEGUMES_G_PER_SERVING,
    }
    scan_metrics = {
        "scanned_file_count": scanned,
        "file_limit": candidate_file_limit,
        "file_limit_reached": limit_reached,
        "likely_candidate_count": int(
            candidates["likely_ahei_lookup"].fillna(False).astype(bool).sum()
        ) if not candidates.empty else 0,
    }
    qc = pd.DataFrame(
        _qc_rows("protocol", protocol_metrics)
        + _qc_rows("events", event_metrics)
        + _qc_rows("amed_shared", amed_metrics)
        + _qc_rows("population", sex_metrics)
        + _qc_rows("lookup_scan", scan_metrics)
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    availability_path = output_dir / "ahei_component_availability.csv"
    event_coverage_path = output_dir / "ahei_event_field_coverage.csv"
    participant_path = output_dir / "ahei_participant_input_coverage.csv"
    candidate_path = output_dir / "ahei_lookup_candidates.csv"
    qc_path = output_dir / "ahei_input_audit_qc.csv"
    availability.to_csv(availability_path, index=False, encoding="utf-8-sig")
    event_coverage.to_csv(event_coverage_path, index=False, encoding="utf-8-sig")
    participants.to_csv(participant_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    return (
        availability_path,
        event_coverage_path,
        participant_path,
        candidate_path,
        qc_path,
    )


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--events-csv", type=Path, default=DEFAULT_EVENTS_CSV)
    parser.add_argument("--raw-events-csv", type=Path, default=DEFAULT_RAW_EVENTS_CSV)
    parser.add_argument("--population-csv", type=Path, default=DEFAULT_POPULATION_CSV)
    parser.add_argument("--alcohol-csv", type=Path, default=DEFAULT_AMED_SHARED_CSV)
    parser.add_argument("--transfer-dir", type=Path, default=DEFAULT_TRANSFER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--candidate-file-limit", type=int, default=0)
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--research-stage", default=DEFAULT_RESEARCH_STAGE)
    parser.add_argument("--enforce-expected-counts", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    written = audit_ahei_inputs(
        events_csv=arguments.events_csv,
        raw_events_csv=arguments.raw_events_csv,
        population_csv=arguments.population_csv,
        alcohol_csv=arguments.alcohol_csv,
        transfer_dir=arguments.transfer_dir,
        output_dir=arguments.output_dir,
        chunk_size=arguments.chunk_size,
        candidate_file_limit=arguments.candidate_file_limit,
        cohort=arguments.cohort,
        research_stage=arguments.research_stage,
        enforce_expected_counts=arguments.enforce_expected_counts,
    )
    for output_path in written:
        print(output_path)
