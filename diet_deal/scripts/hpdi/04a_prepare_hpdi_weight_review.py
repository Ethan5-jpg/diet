"""用其他参与者相同 food_id 的等权中位数填补 hPDI 事件重量。"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-14-hpdi-weight-review-v2"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR.parent
DEFAULT_INPUT_CSV = DATA_DIR / "Transfer" / "diet_logging" / "diet_logging_events.csv"
DEFAULT_MAPPING_CSV = (
    PROJECT_DIR / "outputs" / "03_score_mapping" / "hpdi" / "hpdi_food_id_mapping.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "04_score_intakes" / "hpdi"
DEFAULT_COHORT = "10k"
DEFAULT_RESEARCH_STAGE = "00_00_visit"

EVENT_KEY_COLUMNS = [
    "source_event_row",
    "participant_id",
    "cohort",
    "research_stage",
    "collection_timestamp",
    "food_id",
]
LABEL_COLUMNS = ["short_food_name", "product_name", "food_category"]
REQUIRED_EVENT_COLUMNS = [
    "participant_id",
    "cohort",
    "research_stage",
    "logging_day",
    "collection_timestamp",
    "collection_date",
    "food_id",
    *LABEL_COLUMNS,
    "weight_g",
]
REQUIRED_MAPPING_COLUMNS = ["food_id", "hpdi_component", "mapping_status"]
RESOLUTION_COLUMNS = [
    *EVENT_KEY_COLUMNS,
    "resolution_status",
    "resolved_weight_g",
    "review_notes",
]
VALID_RESOLUTION_STATUSES = {"pending", "unresolved", "use_resolved_weight"}
AUTOMATIC_IMPUTATION_METHOD = (
    "median_of_other_participants_same_food_id_within_person_medians"
)


def normalize_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def validate_columns(columns: List[str], required: List[str], source_name: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{source_name}缺少字段：{', '.join(missing)}")


def load_mapping(mapping_csv: Path) -> pd.DataFrame:
    if not mapping_csv.is_file():
        raise FileNotFoundError(f"找不到 hPDI 映射表：{mapping_csv}")
    header = pd.read_csv(mapping_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(
        header.columns.tolist(), REQUIRED_MAPPING_COLUMNS, "hPDI 映射表"
    )
    mapping = pd.read_csv(
        mapping_csv,
        encoding="utf-8-sig",
        usecols=REQUIRED_MAPPING_COLUMNS,
        dtype={"food_id": "string"},
        low_memory=False,
    )
    mapping["food_id"] = normalize_text(mapping["food_id"])
    if mapping["food_id"].isna().any() or mapping["food_id"].duplicated().any():
        raise ValueError("hPDI 映射表的 food_id 必须完整且唯一。")
    return mapping


def prepare_event_chunk(
    chunk: pd.DataFrame,
    cohort: str,
    research_stage: str,
) -> pd.DataFrame:
    """规范事件键并筛选目标基线数据；保留 CSV 原始行号。"""
    prepared = chunk.copy()
    prepared["source_event_row"] = prepared.index.to_series().astype("int64") + 2
    for column in [
        "participant_id",
        "cohort",
        "research_stage",
        "collection_timestamp",
        "food_id",
        *LABEL_COLUMNS,
    ]:
        prepared[column] = normalize_text(prepared[column])
    selected = prepared["cohort"].eq(cohort) & prepared["research_stage"].eq(
        research_stage
    )
    return prepared.loc[selected].copy()


def scan_missing_weight_events(
    input_csv: Path,
    mapping: pd.DataFrame,
    cohort: str,
    research_stage: str,
    chunk_size: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """第一遍扫描：收集已映射、但 weight_g 缺失或为负的事件。"""
    partials: List[pd.DataFrame] = []
    metrics = {
        "input_rows": 0,
        "selected_rows": 0,
        "mapped_missing_weight_events": 0,
    }
    reader = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        usecols=REQUIRED_EVENT_COLUMNS,
        dtype={
            column: "string"
            for column in [
                "participant_id",
                "cohort",
                "research_stage",
                "collection_timestamp",
                "food_id",
            ]
        },
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        metrics["input_rows"] += len(chunk)
        selected = prepare_event_chunk(chunk, cohort, research_stage)
        metrics["selected_rows"] += len(selected)
        if selected.empty:
            continue
        selected = selected.merge(
            mapping, on="food_id", how="left", validate="many_to_one"
        )
        original_weight = selected["weight_g"]
        weight = pd.to_numeric(original_weight, errors="coerce")
        missing_weight = weight.isna() | weight.lt(0)
        target = selected["mapping_status"].eq("mapped") & missing_weight
        if target.any():
            review = selected.loc[
                target,
                [
                    *EVENT_KEY_COLUMNS,
                    "logging_day",
                    "collection_date",
                    *LABEL_COLUMNS,
                    "hpdi_component",
                    "weight_g",
                ],
            ].copy()
            review = review.rename(columns={"weight_g": "original_weight_g"})
            partials.append(review)
            metrics["mapped_missing_weight_events"] += len(review)
        print(
            f"缺重量扫描第 {chunk_number} 块：累计目标事件 "
            f"{metrics['mapped_missing_weight_events']:,} 条"
        )
    if not partials:
        columns = [
            *EVENT_KEY_COLUMNS,
            "logging_day",
            "collection_date",
            *LABEL_COLUMNS,
            "hpdi_component",
            "original_weight_g",
        ]
        return pd.DataFrame(columns=columns), metrics
    missing = pd.concat(partials, ignore_index=True)
    if missing["source_event_row"].duplicated().any():
        raise ValueError("缺重量复核表出现重复 source_event_row。")
    return missing, metrics


def collect_food_id_donors(
    input_csv: Path,
    target_food_ids: Set[str],
    cohort: str,
    research_stage: str,
    chunk_size: int,
) -> pd.DataFrame:
    """第二遍扫描：收集目标 food_id 的参与者和正重量事件。"""
    columns = ["participant_id", "food_id", "weight_g"]
    if not target_food_ids:
        return pd.DataFrame(columns=columns)
    partials: List[pd.DataFrame] = []
    reader = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        usecols=REQUIRED_EVENT_COLUMNS,
        dtype={
            column: "string"
            for column in [
                "participant_id",
                "cohort",
                "research_stage",
                "collection_timestamp",
                "food_id",
            ]
        },
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        selected = prepare_event_chunk(chunk, cohort, research_stage)
        selected = selected.loc[selected["food_id"].isin(target_food_ids)].copy()
        if selected.empty:
            continue
        selected["weight_g"] = pd.to_numeric(
            selected["weight_g"], errors="coerce"
        )
        donor = selected.loc[selected["weight_g"].gt(0), columns].copy()
        if not donor.empty:
            partials.append(donor)
        print(
            f"供体扫描第 {chunk_number} 块：累计正重量供体事件 "
            f"{sum(len(item) for item in partials):,} 条"
        )
    if not partials:
        return pd.DataFrame(columns=columns)
    return pd.concat(partials, ignore_index=True)


def build_participant_food_donors(donors: pd.DataFrame) -> pd.DataFrame:
    """把重复供体事件折叠为每位参与者每个 food_id 的一个中位数。"""
    output_columns = [
        "food_id",
        "participant_id",
        "donor_event_count",
        "participant_food_weight_min_g",
        "participant_food_weight_median_g",
        "participant_food_weight_max_g",
    ]
    validate_columns(
        donors.columns.tolist(),
        ["participant_id", "food_id", "weight_g"],
        "重量供体表",
    )
    prepared = donors.copy()
    prepared["participant_id"] = normalize_text(prepared["participant_id"])
    prepared["food_id"] = normalize_text(prepared["food_id"])
    prepared["weight_g"] = pd.to_numeric(prepared["weight_g"], errors="coerce")
    valid = (
        prepared["participant_id"].notna()
        & prepared["food_id"].notna()
        & prepared["weight_g"].gt(0)
    )
    prepared = prepared.loc[valid].copy()
    if prepared.empty:
        return pd.DataFrame(columns=output_columns)
    summary = (
        prepared.groupby(["food_id", "participant_id"], sort=True)["weight_g"]
        .agg(
            donor_event_count="count",
            participant_food_weight_min_g="min",
            participant_food_weight_median_g="median",
            participant_food_weight_max_g="max",
        )
        .reset_index()
    )
    return summary.loc[:, output_columns]


def add_other_participant_median_imputations(
    missing: pd.DataFrame,
    participant_food_donors: pd.DataFrame,
) -> pd.DataFrame:
    """按其他参与者的同 food_id 个人中位数之中位数生成并批准填补值。"""
    validate_columns(
        missing.columns.tolist(),
        ["participant_id", "food_id"],
        "缺重量事件表",
    )
    required_donor_columns = [
        "food_id",
        "participant_id",
        "donor_event_count",
        "participant_food_weight_median_g",
    ]
    validate_columns(
        participant_food_donors.columns.tolist(),
        required_donor_columns,
        "参与者-food_id 供体表",
    )
    donor_groups = {
        food_id: group.copy()
        for food_id, group in participant_food_donors.groupby(
            "food_id", sort=False
        )
    }
    rows: List[Dict[str, object]] = []
    for _, event in missing.iterrows():
        food_id = event["food_id"]
        participant_id = event["participant_id"]
        pool = donor_groups.get(food_id)
        if pool is None:
            pool = pd.DataFrame(columns=required_donor_columns)
        else:
            pool = pool.loc[pool["participant_id"].ne(participant_id)].copy()
        values = pd.to_numeric(
            pool["participant_food_weight_median_g"], errors="coerce"
        )
        values = values.loc[values.gt(0)].dropna()
        donor_count = len(values)
        event_count = int(
            pd.to_numeric(
                pool.loc[values.index, "donor_event_count"], errors="coerce"
            )
            .fillna(0)
            .sum()
        )
        if donor_count:
            minimum = float(values.min())
            p25 = float(values.quantile(0.25))
            median = float(values.median())
            p75 = float(values.quantile(0.75))
            maximum = float(values.max())
            relative_iqr = (p75 - p25) / median if median > 0 else pd.NA
            method = AUTOMATIC_IMPUTATION_METHOD
            status = "use_resolved_weight"
            resolved_weight = median
            notes = "auto-approved user-selected other-participant median rule"
        else:
            minimum = pd.NA
            p25 = pd.NA
            median = pd.NA
            p75 = pd.NA
            maximum = pd.NA
            relative_iqr = pd.NA
            method = "no_other_participant_donor"
            status = "unresolved"
            resolved_weight = pd.NA
            notes = "no positive same-food_id weight from another participant"
        rows.append(
            {
                "imputation_method": method,
                "other_participant_donor_count": donor_count,
                "other_event_donor_count": event_count,
                "other_participant_weight_min_g": minimum,
                "other_participant_weight_p25_g": p25,
                "other_participant_weight_median_g": median,
                "other_participant_weight_p75_g": p75,
                "other_participant_weight_max_g": maximum,
                "other_participant_weight_relative_iqr": relative_iqr,
                "suggested_weight_g": median,
                "resolution_status": status,
                "resolved_weight_g": resolved_weight,
                "review_notes": notes,
            }
        )
    details = pd.DataFrame(rows, index=missing.index)
    return pd.concat([missing.copy(), details], axis=1)


def preserve_existing_resolutions(
    review: pd.DataFrame,
    existing_review_csv: Path,
) -> pd.DataFrame:
    """只保留既有人工批准值；旧 pending/unresolved 让位于当前规则。"""
    if not existing_review_csv.is_file():
        return review
    header = pd.read_csv(
        existing_review_csv, encoding="utf-8-sig", nrows=0
    )
    validate_columns(
        header.columns.tolist(), RESOLUTION_COLUMNS, "既有 hPDI 缺重量复核表"
    )
    existing = pd.read_csv(
        existing_review_csv,
        encoding="utf-8-sig",
        usecols=RESOLUTION_COLUMNS,
        dtype={
            column: "string"
            for column in EVENT_KEY_COLUMNS
            if column != "source_event_row"
        },
        low_memory=False,
    )
    existing["source_event_row"] = pd.to_numeric(
        existing["source_event_row"], errors="coerce"
    ).astype("Int64")
    for column in EVENT_KEY_COLUMNS:
        if column != "source_event_row":
            existing[column] = normalize_text(existing[column])
    existing["resolution_status"] = normalize_text(
        existing["resolution_status"]
    ).str.casefold()
    invalid = ~existing["resolution_status"].isin(VALID_RESOLUTION_STATUSES)
    if invalid.any():
        statuses = sorted(
            existing.loc[invalid, "resolution_status"]
            .fillna("<missing>")
            .astype(str)
            .unique()
        )
        raise ValueError(
            "既有 hPDI 缺重量复核表含无效 resolution_status："
            + ", ".join(statuses)
        )
    if existing.duplicated(EVENT_KEY_COLUMNS).any():
        raise ValueError("既有 hPDI 缺重量复核表存在重复事件键。")
    existing_resolved = pd.to_numeric(
        existing["resolved_weight_g"], errors="coerce"
    )
    approved = existing["resolution_status"].eq("use_resolved_weight")
    invalid_approved = approved & (
        existing_resolved.isna() | existing_resolved.le(0)
    )
    if invalid_approved.any():
        raise ValueError("既有人工批准重量存在缺失或非正数。")
    existing = existing.loc[approved].copy()
    if existing.empty:
        return review
    existing["resolved_weight_g"] = existing_resolved.loc[approved]
    existing = existing.rename(
        columns={
            "resolution_status": "existing_resolution_status",
            "resolved_weight_g": "existing_resolved_weight_g",
            "review_notes": "existing_review_notes",
        }
    )
    merged = review.merge(
        existing,
        on=EVENT_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    has_existing = merged["existing_resolution_status"].notna()
    merged.loc[has_existing, "resolution_status"] = "use_resolved_weight"
    merged.loc[has_existing, "resolved_weight_g"] = merged.loc[
        has_existing, "existing_resolved_weight_g"
    ]
    merged.loc[has_existing, "review_notes"] = merged.loc[
        has_existing, "existing_review_notes"
    ].fillna("")
    if "imputation_method" in merged.columns:
        merged.loc[
            has_existing, "imputation_method"
        ] = "preserved_existing_approved_resolution"
    return merged.drop(
        columns=[
            "existing_resolution_status",
            "existing_resolved_weight_g",
            "existing_review_notes",
        ]
    )


def build_qc(
    review: pd.DataFrame,
    scan_metrics: Dict[str, int],
    cohort: str,
    research_stage: str,
) -> pd.DataFrame:
    approved = review["resolution_status"].eq("use_resolved_weight")
    unresolved = ~approved
    participant_complete = review.groupby("participant_id")[
        "resolution_status"
    ].apply(lambda item: item.eq("use_resolved_weight").all())
    donor_counts = pd.to_numeric(
        review.loc[approved, "other_participant_donor_count"], errors="coerce"
    )
    metrics: Dict[str, object] = {
        **scan_metrics,
        "script_version": SCRIPT_VERSION,
        "selected_cohort": cohort,
        "selected_research_stage": research_stage,
        "missing_weight_event_count": len(review),
        "affected_participant_count": review["participant_id"].nunique(),
        "affected_food_id_count": review["food_id"].nunique(),
        "imputed_event_count": int(approved.sum()),
        "unresolved_event_count": int(unresolved.sum()),
        "participants_with_all_missing_events_imputed": int(
            participant_complete.sum()
        ),
        "participants_with_any_unresolved_missing_weight": int(
            (~participant_complete).sum()
        ),
        "imputed_other_participant_donor_count_min": (
            donor_counts.min() if not donor_counts.empty else pd.NA
        ),
        "imputed_other_participant_donor_count_median": (
            donor_counts.median() if not donor_counts.empty else pd.NA
        ),
        "imputed_other_participant_donor_count_max": (
            donor_counts.max() if not donor_counts.empty else pd.NA
        ),
        "automatic_imputation_rule": AUTOMATIC_IMPUTATION_METHOD,
        "target_participant_excluded_from_donors": True,
        "donor_participant_weighting": "one within-person median per participant",
        "approved_resolution_status": "use_resolved_weight",
    }
    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )


def prepare_hpdi_weight_review(
    input_csv: Path,
    mapping_csv: Path,
    output_dir: Path,
    cohort: str,
    research_stage: str,
    chunk_size: int,
) -> Tuple[Path, Path, Path]:
    if not input_csv.is_file():
        raise FileNotFoundError(f"找不到饮食事件表：{input_csv}")
    header = pd.read_csv(input_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(
        header.columns.tolist(), REQUIRED_EVENT_COLUMNS, "饮食事件表"
    )
    mapping = load_mapping(mapping_csv)
    missing, scan_metrics = scan_missing_weight_events(
        input_csv, mapping, cohort, research_stage, chunk_size
    )
    target_food_ids = set(missing["food_id"].dropna().astype(str))
    donors = collect_food_id_donors(
        input_csv, target_food_ids, cohort, research_stage, chunk_size
    )
    participant_food_donors = build_participant_food_donors(donors)
    review = add_other_participant_median_imputations(
        missing, participant_food_donors
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "hpdi_missing_weight_review.csv"
    donor_path = output_dir / "hpdi_missing_weight_food_id_donors.csv"
    qc_path = output_dir / "hpdi_missing_weight_qc.csv"
    review = preserve_existing_resolutions(review, review_path)
    qc = build_qc(review, scan_metrics, cohort, research_stage)
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    participant_food_donors.to_csv(
        donor_path, index=False, encoding="utf-8-sig"
    )
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    approved = review["resolution_status"].eq("use_resolved_weight")
    print("-" * 72)
    print(f"脚本版本：{SCRIPT_VERSION}")
    print(f"缺重量事件：{len(review):,} 条")
    print(f"受影响参与者：{review['participant_id'].nunique():,} 人")
    print(f"按其他参与者同 food_id 中位数填补：{int(approved.sum()):,} 条")
    print(f"仍无法填补：{int((~approved).sum()):,} 条")
    print(f"重量复核及 override 表：{review_path}")
    print(f"参与者-food_id 供体统计：{donor_path}")
    print(f"QC：{qc_path}")
    return review_path, donor_path, qc_path


def parse_args():
    parser = ArgumentParser(
        description="用其他参与者相同 food_id 的等权中位数填补 hPDI 重量"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--research-stage", default=DEFAULT_RESEARCH_STAGE)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_hpdi_weight_review(
        args.input,
        args.mapping,
        args.output_dir,
        args.cohort,
        args.research_stage,
        args.chunk_size,
    )
