"""Aggregate HPP diet events into daily and participant mAHEI-7 inputs."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-18-ahei-intakes-v1"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR.parent
DEFAULT_INPUT_CSV = (
    DATA_DIR / "Transfer" / "diet_logging" / "diet_logging_events.csv"
)
DEFAULT_MAPPING_CSV = (
    PROJECT_DIR
    / "outputs"
    / "03_score_mapping"
    / "ahei"
    / "ahei_food_id_mapping.csv"
)
DEFAULT_AMED_SHARED_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "amed"
    / "amed_participant_component_intakes.csv"
)
DEFAULT_WEIGHT_OVERRIDES_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "hpdi"
    / "hpdi_missing_weight_review.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "04_score_intakes" / "ahei"
DEFAULT_COHORT = "10k"
DEFAULT_RESEARCH_STAGE = "00_00_visit"

GROUP_COLUMNS = [
    "participant_id",
    "cohort",
    "research_stage",
    "logging_day",
    "collection_date",
]
EVENT_KEY_COLUMNS = [
    "source_event_row",
    "participant_id",
    "cohort",
    "research_stage",
    "collection_timestamp",
    "food_id",
]
INPUT_EVENT_COLUMNS = [
    *GROUP_COLUMNS,
    "collection_timestamp",
    "food_id",
    "weight_g",
    "calories_kcal",
    "carbohydrate_g",
    "lipid_g",
    "protein_g",
]
REQUIRED_EVENT_COLUMNS = ["source_event_row", *INPUT_EVENT_COLUMNS]
TEXT_EVENT_COLUMNS = [
    "participant_id",
    "cohort",
    "research_stage",
    "collection_timestamp",
    "food_id",
]
REQUIRED_MAPPING_COLUMNS = ["food_id", "ahei_component", "mapping_status"]
REQUIRED_AMED_COLUMNS = [
    "participant_id",
    "cohort",
    "research_stage",
    "alcohol_complete",
    "mean_daily_alcohol_g",
    "total_alcohol_unresolved_events",
    "energy_complete",
    "mean_daily_energy_kcal",
]
REQUIRED_WEIGHT_OVERRIDE_COLUMNS = [
    *EVENT_KEY_COLUMNS,
    "resolution_status",
    "resolved_weight_g",
]

ENERGY_COMPONENTS = [
    "vegetables",
    "fruit",
    "ssb_plus_fruit_juice",
    "red_plus_processed_meat",
]
WEIGHT_COMPONENTS = ["whole_grains", "nuts_plus_legumes"]
FOOD_COMPONENTS = ENERGY_COMPONENTS + WEIGHT_COMPONENTS
KCAL_PER_SERVING = {
    "vegetables": 30.0,
    "fruit": 70.0,
    "ssb_plus_fruit_juice": 70.0,
    "red_plus_processed_meat": 220.0,
}
NUTS_LEGUMES_G_PER_SERVING = 28.35
STANDARD_DRINK_G = 14.0
VALID_MAPPING_STATUSES = {
    "mapped",
    "not_applicable",
    "unmapped_missing_labels",
}
EVENT_STATUSES = [
    "mapped",
    "not_applicable",
    "unmapped_missing_labels",
    "unmatched_food_id",
    "missing_food_id",
]


def normalize_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def parse_boolean(series: pd.Series, column_name: str) -> pd.Series:
    normalized = normalize_text(series).str.casefold()
    parsed = normalized.map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    )
    invalid = normalized.notna() & parsed.isna()
    if invalid.any():
        values = sorted(normalized.loc[invalid].astype(str).unique())
        raise ValueError(
            "{}含无效布尔值：{}".format(column_name, ", ".join(values[:10]))
        )
    return parsed.astype("boolean")


def validate_columns(columns: List[str], required: List[str], source_name: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(
            "{}缺少字段：{}".format(source_name, ", ".join(missing))
        )


def load_mapping(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError("找不到AHEI映射表：{}".format(path))
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    validate_columns(header.columns.tolist(), REQUIRED_MAPPING_COLUMNS, "AHEI映射表")
    mapping = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=REQUIRED_MAPPING_COLUMNS,
        dtype={"food_id": "string"},
        low_memory=False,
    )
    for column in REQUIRED_MAPPING_COLUMNS:
        mapping[column] = normalize_text(mapping[column])
    if mapping["food_id"].isna().any() or mapping["food_id"].duplicated().any():
        raise ValueError("AHEI映射表food_id必须非缺失且唯一。")
    invalid_status = ~mapping["mapping_status"].isin(VALID_MAPPING_STATUSES)
    if invalid_status.any():
        values = sorted(mapping.loc[invalid_status, "mapping_status"].astype(str).unique())
        raise ValueError("AHEI映射尚未冻结：" + ", ".join(values[:10]))
    mapped = mapping["mapping_status"].eq("mapped")
    invalid_component = mapped & ~mapping["ahei_component"].isin(FOOD_COMPONENTS)
    if invalid_component.any():
        values = sorted(mapping.loc[invalid_component, "ahei_component"].astype(str).unique())
        raise ValueError("AHEI映射含未知组件：" + ", ".join(values[:10]))
    if mapping.loc[~mapped, "ahei_component"].notna().any():
        raise ValueError("非mapped食物不应带有AHEI组件。")
    return mapping


def load_weight_overrides(path: Path) -> pd.DataFrame:
    """Read only approved hPDI event-level weights; missing file means none."""
    if not Path(path).is_file():
        return pd.DataFrame(columns=[*EVENT_KEY_COLUMNS, "resolved_weight_g"])
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    validate_columns(
        header.columns.tolist(),
        REQUIRED_WEIGHT_OVERRIDE_COLUMNS,
        "hPDI缺重量复核表",
    )
    data = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=REQUIRED_WEIGHT_OVERRIDE_COLUMNS,
        dtype={column: "string" for column in EVENT_KEY_COLUMNS if column != "source_event_row"},
        low_memory=False,
    )
    data["source_event_row"] = pd.to_numeric(
        data["source_event_row"], errors="coerce"
    ).astype("Int64")
    for column in EVENT_KEY_COLUMNS:
        if column != "source_event_row":
            data[column] = normalize_text(data[column])
    data["resolution_status"] = normalize_text(data["resolution_status"]).str.casefold()
    valid_statuses = {"pending", "unresolved", "use_resolved_weight"}
    if (~data["resolution_status"].isin(valid_statuses)).any():
        raise ValueError("hPDI缺重量复核表含无效resolution_status。")
    approved = data["resolution_status"].eq("use_resolved_weight")
    data["resolved_weight_g"] = pd.to_numeric(data["resolved_weight_g"], errors="coerce")
    invalid = approved & (
        data["resolved_weight_g"].isna() | data["resolved_weight_g"].le(0)
    )
    if invalid.any():
        raise ValueError("已批准的重量复核行缺少正resolved_weight_g。")
    result = data.loc[approved, [*EVENT_KEY_COLUMNS, "resolved_weight_g"]].copy()
    if result[EVENT_KEY_COLUMNS].isna().any(axis=1).any():
        raise ValueError("已批准重量复核行缺少事件键。")
    if result.duplicated(EVENT_KEY_COLUMNS).any():
        raise ValueError("已批准重量复核行存在重复事件键。")
    return result.reset_index(drop=True)


def _prepare_direct_overrides(overrides: Optional[pd.DataFrame]) -> pd.DataFrame:
    if overrides is None or overrides.empty:
        return pd.DataFrame(columns=[*EVENT_KEY_COLUMNS, "resolved_weight_g"])
    required = [*EVENT_KEY_COLUMNS, "resolved_weight_g"]
    validate_columns(overrides.columns.tolist(), required, "重量override")
    result = overrides.loc[:, required].copy()
    result["source_event_row"] = pd.to_numeric(
        result["source_event_row"], errors="coerce"
    ).astype("Int64")
    for column in EVENT_KEY_COLUMNS:
        if column != "source_event_row":
            result[column] = normalize_text(result[column])
    result["resolved_weight_g"] = pd.to_numeric(
        result["resolved_weight_g"], errors="coerce"
    )
    invalid = result["resolved_weight_g"].isna() | result["resolved_weight_g"].le(0)
    if invalid.any() or result[EVENT_KEY_COLUMNS].isna().any(axis=1).any():
        raise ValueError("重量override事件键必须完整且resolved_weight_g必须为正。")
    if result.duplicated(EVENT_KEY_COLUMNS).any():
        raise ValueError("重量override存在重复事件键。")
    return result


def aggregate_chunk(
    chunk: pd.DataFrame,
    mapping: pd.DataFrame,
    cohort: str,
    research_stage: str,
    weight_overrides: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Aggregate one CSV chunk into participant-day mAHEI proxy totals."""
    chunk = chunk.copy()
    if "source_event_row" not in chunk.columns:
        chunk["source_event_row"] = chunk.index.to_series().astype("int64") + 2
    validate_columns(chunk.columns.tolist(), REQUIRED_EVENT_COLUMNS, "饮食事件表")
    chunk = chunk.loc[:, REQUIRED_EVENT_COLUMNS].copy()
    input_rows = len(chunk)
    for column in TEXT_EVENT_COLUMNS:
        chunk[column] = normalize_text(chunk[column])
    chunk["source_event_row"] = pd.to_numeric(
        chunk["source_event_row"], errors="coerce"
    ).astype("Int64")
    selected = chunk["cohort"].eq(cohort) & chunk["research_stage"].eq(research_stage)
    chunk = chunk.loc[selected].copy()
    metrics = {
        "input_rows": input_rows,
        "selected_rows": len(chunk),
        "excluded_other_cohort_or_stage_rows": input_rows - len(chunk),
    }
    if chunk.empty:
        return pd.DataFrame(), metrics
    chunk["logging_day"] = pd.to_numeric(chunk["logging_day"], errors="coerce").astype("Int64")
    dates = pd.to_datetime(chunk["collection_date"], errors="coerce")
    chunk["collection_date"] = dates.dt.strftime("%Y-%m-%d")
    if chunk[GROUP_COLUMNS].isna().any(axis=1).any():
        raise ValueError("目标事件缺少参与者-日汇总键。")

    numeric_columns = [
        "weight_g",
        "calories_kcal",
        "carbohydrate_g",
        "lipid_g",
        "protein_g",
    ]
    for column in numeric_columns:
        original = chunk[column]
        numeric = pd.to_numeric(original, errors="coerce")
        metrics["invalid_numeric_{}".format(column)] = int(
            (original.notna() & numeric.isna()).sum()
        )
        chunk[column] = numeric
    negative_weight = chunk["weight_g"].lt(0).fillna(False)
    chunk.loc[negative_weight, "weight_g"] = pd.NA
    metrics["negative_weight_g_rows"] = int(negative_weight.sum())
    chunk["__missing_weight_before_override"] = chunk["weight_g"].isna()
    missing_weight_before_count = int(
        chunk["__missing_weight_before_override"].sum()
    )
    overrides = _prepare_direct_overrides(weight_overrides)
    if not overrides.empty:
        chunk = chunk.merge(
            overrides, on=EVENT_KEY_COLUMNS, how="left", validate="many_to_one"
        )
        missing_weight_before = chunk["__missing_weight_before_override"]
        override_available = chunk["resolved_weight_g"].notna()
        override_applied = missing_weight_before & override_available
        chunk.loc[override_applied, "weight_g"] = chunk.loc[
            override_applied, "resolved_weight_g"
        ]
        override_not_needed = ~missing_weight_before & override_available
        chunk = chunk.drop(columns=["resolved_weight_g"])
    else:
        missing_weight_before = chunk["__missing_weight_before_override"]
        override_applied = pd.Series(False, index=chunk.index)
        override_not_needed = pd.Series(False, index=chunk.index)
    chunk = chunk.drop(columns=["__missing_weight_before_override"])
    metrics["missing_weight_before_override_rows"] = missing_weight_before_count
    metrics["weight_override_applied_rows"] = int(override_applied.sum())
    metrics["weight_override_not_needed_rows"] = int(override_not_needed.sum())
    metrics["missing_weight_after_override_rows"] = int(chunk["weight_g"].isna().sum())

    mapping_for_merge = mapping.loc[:, REQUIRED_MAPPING_COLUMNS].copy()
    for column in REQUIRED_MAPPING_COLUMNS:
        mapping_for_merge[column] = normalize_text(mapping_for_merge[column])
    chunk = chunk.merge(mapping_for_merge, on="food_id", how="left", validate="many_to_one")
    chunk["event_mapping_status"] = chunk["mapping_status"]
    missing_food_id = chunk["food_id"].isna()
    unmatched_food_id = chunk["mapping_status"].isna() & ~missing_food_id
    chunk.loc[missing_food_id, "event_mapping_status"] = "missing_food_id"
    chunk.loc[unmatched_food_id, "event_mapping_status"] = "unmatched_food_id"
    excluded_missing_labels = chunk["event_mapping_status"].eq("unmapped_missing_labels")
    retained = ~excluded_missing_labels

    valid_observed_energy = (
        retained & chunk["calories_kcal"].notna() & chunk["calories_kcal"].ge(0)
    )
    macro = pd.DataFrame(index=chunk.index)
    for column in ["carbohydrate_g", "lipid_g", "protein_g"]:
        macro[column] = chunk[column].where(chunk[column].ge(0))
    macros_complete = macro.notna().all(axis=1)
    formula_energy = (
        4 * macro["carbohydrate_g"].fillna(0)
        + 9 * macro["lipid_g"].fillna(0)
        + 4 * macro["protein_g"].fillna(0)
    )
    missing_energy = retained & ~valid_observed_energy
    complete_formula = missing_energy & macros_complete
    partial_formula = missing_energy & ~macros_complete
    chunk["__resolved_energy_kcal"] = chunk["calories_kcal"].where(valid_observed_energy)
    chunk.loc[missing_energy, "__resolved_energy_kcal"] = formula_energy.loc[missing_energy]
    chunk.loc[~retained, "__resolved_energy_kcal"] = pd.NA

    chunk["__retained_event"] = retained.astype("int64")
    chunk["__excluded_missing_labels_event"] = excluded_missing_labels.astype("int64")
    chunk["__weight_override_applied"] = (retained & override_applied).astype("int64")
    chunk["__energy_observed_event"] = valid_observed_energy.astype("int64")
    chunk["__energy_complete_formula_event"] = complete_formula.astype("int64")
    chunk["__energy_partial_formula_event"] = partial_formula.astype("int64")
    for status in EVENT_STATUSES:
        chunk["__{}_event".format(status)] = (
            retained & chunk["event_mapping_status"].eq(status)
        ).astype("int64")

    for component in FOOD_COMPONENTS:
        mask = (
            retained
            & chunk["event_mapping_status"].eq("mapped")
            & chunk["ahei_component"].eq(component)
        )
        chunk["__{}_event".format(component)] = mask.astype("int64")
        if component in ENERGY_COMPONENTS:
            chunk["__{}_valid_event".format(component)] = (
                mask & chunk["__resolved_energy_kcal"].notna()
            ).astype("int64")
            chunk["__{}_total".format(component)] = chunk[
                "__resolved_energy_kcal"
            ].where(mask)
        else:
            chunk["__{}_valid_event".format(component)] = (
                mask & chunk["weight_g"].notna()
            ).astype("int64")
            chunk["__{}_total".format(component)] = chunk["weight_g"].where(mask)

    grouped = chunk.groupby(GROUP_COLUMNS, sort=False, dropna=False)
    daily = grouped["__retained_event"].sum().rename("event_count").to_frame()
    daily["excluded_missing_labels_event_count"] = grouped[
        "__excluded_missing_labels_event"
    ].sum()
    daily["weight_override_applied_event_count"] = grouped[
        "__weight_override_applied"
    ].sum()
    daily["resolved_energy_kcal_total"] = grouped["__resolved_energy_kcal"].sum(min_count=1)
    daily["energy_observed_event_count"] = grouped["__energy_observed_event"].sum()
    daily["energy_complete_formula_event_count"] = grouped[
        "__energy_complete_formula_event"
    ].sum()
    daily["energy_partial_formula_event_count"] = grouped[
        "__energy_partial_formula_event"
    ].sum()
    for status in EVENT_STATUSES:
        daily["{}_event_count".format(status)] = grouped[
            "__{}_event".format(status)
        ].sum()
    for component in FOOD_COMPONENTS:
        daily["{}_event_count".format(component)] = grouped[
            "__{}_event".format(component)
        ].sum()
        daily["{}_valid_event_count".format(component)] = grouped[
            "__{}_valid_event".format(component)
        ].sum()
        suffix = "kcal_total" if component in ENERGY_COMPONENTS else "g_total"
        daily["{}_{}".format(component, suffix)] = grouped[
            "__{}_total".format(component)
        ].sum(min_count=1)

    metrics["missing_food_id_rows"] = int(missing_food_id.sum())
    metrics["unmatched_food_id_rows"] = int(unmatched_food_id.sum())
    metrics["excluded_missing_labels_rows"] = int(excluded_missing_labels.sum())
    metrics["retained_selected_rows"] = int(retained.sum())
    return daily.reset_index(), metrics


def combine_daily_partials(partials: List[pd.DataFrame]) -> pd.DataFrame:
    if not partials:
        raise ValueError("没有可合并的AHEI日级局部结果。")
    combined = pd.concat(partials, ignore_index=True)
    grouped = combined.groupby(GROUP_COLUMNS, sort=True, dropna=False)
    count_columns = [
        "event_count",
        "excluded_missing_labels_event_count",
        "weight_override_applied_event_count",
        "energy_observed_event_count",
        "energy_complete_formula_event_count",
        "energy_partial_formula_event_count",
        *["{}_event_count".format(status) for status in EVENT_STATUSES],
        *["{}_event_count".format(component) for component in FOOD_COMPONENTS],
        *["{}_valid_event_count".format(component) for component in FOOD_COMPONENTS],
    ]
    total_columns = [
        "resolved_energy_kcal_total",
        *[
            "{}_{}".format(
                component,
                "kcal_total" if component in ENERGY_COMPONENTS else "g_total",
            )
            for component in FOOD_COMPONENTS
        ],
    ]
    daily = grouped[count_columns].sum().join(
        grouped[total_columns].sum(min_count=1)
    ).reset_index()
    daily.loc[daily["event_count"].eq(0), "resolved_energy_kcal_total"] = 0.0
    for component in FOOD_COMPONENTS:
        count = daily["{}_event_count".format(component)]
        valid = daily["{}_valid_event_count".format(component)]
        total_column = "{}_{}".format(
            component,
            "kcal_total" if component in ENERGY_COMPONENTS else "g_total",
        )
        daily.loc[count.eq(0), total_column] = 0.0
        daily["{}_complete".format(component)] = valid.eq(count)
    return daily.sort_values(GROUP_COLUMNS).reset_index(drop=True)


def _prepare_amed_shared(data: pd.DataFrame) -> pd.DataFrame:
    validate_columns(data.columns.tolist(), REQUIRED_AMED_COLUMNS, "AMED共享摄入表")
    shared = data.loc[:, REQUIRED_AMED_COLUMNS].copy()
    for column in ["participant_id", "cohort", "research_stage"]:
        shared[column] = normalize_text(shared[column])
    if shared.duplicated(["participant_id", "cohort", "research_stage"]).any():
        raise ValueError("AMED共享摄入表存在重复参与者键。")
    shared["alcohol_complete"] = parse_boolean(shared["alcohol_complete"], "alcohol_complete")
    shared["energy_complete"] = parse_boolean(shared["energy_complete"], "energy_complete")
    for column in [
        "mean_daily_alcohol_g",
        "total_alcohol_unresolved_events",
        "mean_daily_energy_kcal",
    ]:
        shared[column] = pd.to_numeric(shared[column], errors="coerce")
    shared["alcohol_complete"] = (
        shared["alcohol_complete"].eq(True)
        & shared["mean_daily_alcohol_g"].ge(0)
        & shared["total_alcohol_unresolved_events"].eq(0)
    )
    shared["energy_complete"] = (
        shared["energy_complete"].eq(True)
        & shared["mean_daily_energy_kcal"].gt(0)
    )
    return shared


def build_participant_summary(
    daily: pd.DataFrame, amed_shared: pd.DataFrame
) -> pd.DataFrame:
    """Average daily proxies over every original logging day and merge AMED inputs."""
    group_keys = ["participant_id", "cohort", "research_stage"]
    grouped = daily.groupby(group_keys, sort=True, dropna=False)
    participants = grouped.size().rename("observed_diet_days").to_frame()
    participants["unique_logging_days"] = grouped["logging_day"].nunique()
    participants["first_collection_date"] = grouped["collection_date"].min()
    participants["last_collection_date"] = grouped["collection_date"].max()
    for column in [
        "event_count",
        "excluded_missing_labels_event_count",
        "weight_override_applied_event_count",
        "energy_observed_event_count",
        "energy_complete_formula_event_count",
        "energy_partial_formula_event_count",
        *["{}_event_count".format(component) for component in FOOD_COMPONENTS],
    ]:
        participants["total_{}".format(column)] = grouped[column].sum()
    for component in ENERGY_COMPONENTS:
        total_column = "{}_kcal_total".format(component)
        mean_kcal_column = "mean_daily_{}_kcal_proxy".format(component)
        participants[mean_kcal_column] = grouped[total_column].mean()
        participants["mean_daily_{}_servings".format(component)] = (
            participants[mean_kcal_column] / KCAL_PER_SERVING[component]
        )
        participants["{}_complete".format(component)] = grouped[
            "{}_complete".format(component)
        ].all()
    participants["mean_daily_whole_grain_proxy_g"] = grouped[
        "whole_grains_g_total"
    ].mean()
    participants["whole_grains_complete"] = grouped["whole_grains_complete"].all()
    participants["mean_daily_nuts_plus_legumes_g_proxy"] = grouped[
        "nuts_plus_legumes_g_total"
    ].mean()
    participants["mean_daily_nuts_plus_legumes_servings"] = (
        participants["mean_daily_nuts_plus_legumes_g_proxy"]
        / NUTS_LEGUMES_G_PER_SERVING
    )
    participants["nuts_plus_legumes_complete"] = grouped[
        "nuts_plus_legumes_complete"
    ].all()
    participants = participants.reset_index()

    shared = _prepare_amed_shared(amed_shared)
    participants = participants.merge(
        shared, on=group_keys, how="left", validate="one_to_one"
    )
    participants["mean_daily_alcohol_servings"] = (
        participants["mean_daily_alcohol_g"] / STANDARD_DRINK_G
    )
    component_complete_columns = [
        "{}_complete".format(component) for component in FOOD_COMPONENTS
    ]
    participants["mahei7_food_components_complete"] = participants[
        component_complete_columns
    ].all(axis=1)
    participants["mahei7_shared_inputs_complete"] = (
        participants["alcohol_complete"].eq(True)
        & participants["energy_complete"].eq(True)
    )
    participants["mahei7_intake_complete"] = (
        participants["mahei7_food_components_complete"]
        & participants["mahei7_shared_inputs_complete"]
    )
    return participants.sort_values(group_keys).reset_index(drop=True)


def build_qc(
    daily: pd.DataFrame,
    participants: pd.DataFrame,
    metrics: Dict[str, int],
    cohort: str,
    research_stage: str,
) -> pd.DataFrame:
    rows = [
        ("protocol", "script_version", SCRIPT_VERSION),
        ("protocol", "cohort", cohort),
        ("protocol", "research_stage", research_stage),
        ("protocol", "standard_drink_g", STANDARD_DRINK_G),
        ("protocol", "nuts_legumes_g_per_serving", NUTS_LEGUMES_G_PER_SERVING),
        ("overall", "participant_count", len(participants)),
        ("overall", "participant_day_count", len(daily)),
        (
            "overall",
            "participants_with_complete_mahei7_intakes",
            int(participants["mahei7_intake_complete"].sum()),
        ),
        (
            "overall",
            "participants_with_any_weight_override",
            int(participants["total_weight_override_applied_event_count"].gt(0).sum()),
        ),
    ]
    for key, value in sorted(metrics.items()):
        rows.append(("events", key, value))
    for component in FOOD_COMPONENTS:
        rows.append(
            (
                "component",
                "{}_complete_participant_count".format(component),
                int(participants["{}_complete".format(component)].sum()),
            )
        )
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def prepare_ahei_score_intake(
    input_csv: Path = DEFAULT_INPUT_CSV,
    mapping_csv: Path = DEFAULT_MAPPING_CSV,
    amed_shared_csv: Path = DEFAULT_AMED_SHARED_CSV,
    weight_overrides_csv: Path = DEFAULT_WEIGHT_OVERRIDES_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    chunk_size: int = 200000,
    cohort: str = DEFAULT_COHORT,
    research_stage: str = DEFAULT_RESEARCH_STAGE,
) -> Tuple[Path, Path, Path]:
    """Run event aggregation and write daily, participant, and QC outputs."""
    if not Path(input_csv).is_file():
        raise FileNotFoundError("找不到饮食事件表：{}".format(input_csv))
    if not Path(amed_shared_csv).is_file():
        raise FileNotFoundError("找不到AMED共享摄入表：{}".format(amed_shared_csv))
    mapping = load_mapping(mapping_csv)
    overrides = load_weight_overrides(weight_overrides_csv)
    header = pd.read_csv(input_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(header.columns.tolist(), INPUT_EVENT_COLUMNS, "饮食事件表")
    partials = []
    totals: Dict[str, int] = {}
    for chunk in pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        usecols=INPUT_EVENT_COLUMNS,
        dtype={column: "string" for column in TEXT_EVENT_COLUMNS},
        chunksize=chunk_size,
        low_memory=False,
    ):
        chunk["source_event_row"] = chunk.index.to_series().astype("int64") + 2
        partial, metrics = aggregate_chunk(
            chunk,
            mapping,
            cohort,
            research_stage,
            weight_overrides=overrides,
        )
        if not partial.empty:
            partials.append(partial)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0) + int(value)
    daily = combine_daily_partials(partials)
    amed_shared = pd.read_csv(
        amed_shared_csv,
        encoding="utf-8-sig",
        dtype={"participant_id": "string", "cohort": "string", "research_stage": "string"},
        low_memory=False,
    )
    participants = build_participant_summary(daily, amed_shared)
    qc = build_qc(daily, participants, totals, cohort, research_stage)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "ahei_daily_component_intakes.csv"
    participant_path = output_dir / "ahei_participant_component_intakes.csv"
    qc_path = output_dir / "ahei_intake_qc.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    participants.to_csv(participant_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    return daily_path, participant_path, qc_path


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--amed-shared-csv", type=Path, default=DEFAULT_AMED_SHARED_CSV)
    parser.add_argument("--weight-overrides-csv", type=Path, default=DEFAULT_WEIGHT_OVERRIDES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--research-stage", default=DEFAULT_RESEARCH_STAGE)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for output_path in prepare_ahei_score_intake(
        input_csv=arguments.input_csv,
        mapping_csv=arguments.mapping_csv,
        amed_shared_csv=arguments.amed_shared_csv,
        weight_overrides_csv=arguments.weight_overrides_csv,
        output_dir=arguments.output_dir,
        chunk_size=arguments.chunk_size,
        cohort=arguments.cohort,
        research_stage=arguments.research_stage,
    ):
        print(output_path)
