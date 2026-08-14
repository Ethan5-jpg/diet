"""审计 HPP AHEI 所需输入，不计算或输出 AHEI 分数。

当前正式 AHEI 仅面向酒精资料完整者。脚本检查饮食事件字段、人口学性别、
AMED 已解析的酒精摄入，以及 Data/Transfer 下可能存在的食物营养 lookup。
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-09-ahei-input-audit-v1"

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
DEFAULT_TRANSFER_DIR = DATA_DIR / "Transfer"
DEFAULT_ALCOHOL_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "amed"
    / "amed_participant_component_intakes.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "03_score_mapping" / "ahei"
)

DEFAULT_COHORT = "10k"
DEFAULT_RESEARCH_STAGE = "00_00_visit"

ID_COLUMNS = ["participant_id", "cohort", "research_stage"]
AUDIT_EVENT_FIELDS = ["weight_g", "calories_kcal", "sodium_mg"]
EVENT_TEXT_COLUMNS = ["participant_id", "cohort", "research_stage"]
ALCOHOL_REQUIRED_COLUMNS = [
    "participant_id",
    "cohort",
    "research_stage",
    "alcohol_complete",
    "mean_daily_alcohol_g",
]

COLUMN_KEYWORDS = {
    "serving_or_portion": ["serv", "portion"],
    "trans_fat": ["trans_fat", "transfat", "trans fatty"],
    "pufa": ["pufa", "polyunsatur", "poly_unsatur"],
    "saturated_fat": ["saturated_fat", "saturated fat", "satur_fat"],
    "whole_grain": ["whole_grain", "whole grain", "wholegrain"],
    "nutrient": ["nutrient", "nutrition", "composition"],
}

CANDIDATE_FILE_KEYWORDS = [
    "food",
    "nutrient",
    "nutrition",
    "composition",
    "serving",
    "portion",
    "lookup",
    "menu",
    "item",
]

SUPPORTED_TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".xlsx", ".xls"}


def normalize_text(series: pd.Series) -> pd.Series:
    """去除文本首尾空白，并把空字符串视为缺失。"""
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def normalize_column_name(value: object) -> str:
    """把列名统一成适合关键词搜索的形式。"""
    text = str(value).strip().casefold()
    for character in ["-", "/", "\\", ".", "(", ")", "%"]:
        text = text.replace(character, "_")
    return "_".join(text.split())


def validate_columns(
    columns: Iterable[str], required: Iterable[str], source_name: str
) -> None:
    """确认输入包含所有必需字段。"""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(
            "{}缺少字段：{}".format(source_name, ", ".join(missing))
        )


def find_keyword_hits(columns: Iterable[object]) -> List[Dict[str, str]]:
    """返回字段名中的 AHEI 相关关键词命中。"""
    hits = []
    for original_column in columns:
        normalized = normalize_column_name(original_column)
        for keyword_group, patterns in COLUMN_KEYWORDS.items():
            if keyword_group == "saturated_fat" and (
                "unsatur" in normalized or "pufa" in normalized
            ):
                continue
            matched_patterns = []
            for pattern in patterns:
                normalized_pattern = normalize_column_name(pattern)
                if normalized_pattern in normalized:
                    matched_patterns.append(pattern)
            if matched_patterns:
                hits.append(
                    {
                        "column": str(original_column),
                        "normalized_column": normalized,
                        "keyword_group": keyword_group,
                        "matched_patterns": "|".join(matched_patterns),
                    }
                )
    return hits


def parse_boolean(series: pd.Series, source_name: str) -> pd.Series:
    """把常见布尔表达解析为 pandas BooleanDtype。"""
    normalized = normalize_text(series).str.casefold()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }
    parsed = normalized.map(mapping).astype("boolean")
    invalid = normalized.notna() & parsed.isna()
    if invalid.any():
        values = sorted(normalized.loc[invalid].astype(str).unique())
        raise ValueError(
            "{}有无法识别的布尔值：{}".format(
                source_name, ", ".join(values[:10])
            )
        )
    return parsed


def read_csv_header(path: Path) -> List[str]:
    """读取 CSV 表头。"""
    if not path.is_file():
        raise FileNotFoundError("找不到文件：{}".format(path))
    return pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()


def valid_numeric_mask(series: pd.Series, field: str) -> Tuple[pd.Series, int]:
    """按 AHEI 审计用途定义事件数值字段的有效性。"""
    original_nonmissing = series.notna()
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_numeric = int((original_nonmissing & numeric.isna()).sum())
    if field in {"weight_g", "calories_kcal"}:
        valid = numeric.notna() & numeric.gt(0)
    else:
        valid = numeric.notna() & numeric.ge(0)
    return valid, invalid_numeric


def aggregate_event_chunk(
    chunk: pd.DataFrame,
    cohort: str,
    research_stage: str,
    audit_fields: List[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """筛选一个事件数据块，并生成参与者级字段覆盖局部汇总。"""
    columns = [*ID_COLUMNS, *audit_fields]
    chunk = chunk.loc[:, columns].copy()
    input_rows = len(chunk)
    for column in EVENT_TEXT_COLUMNS:
        chunk[column] = normalize_text(chunk[column])

    selected = (
        chunk["cohort"].eq(cohort).fillna(False)
        & chunk["research_stage"].eq(research_stage).fillna(False)
    )
    chunk = chunk.loc[selected].copy()
    metrics = {
        "input_rows": input_rows,
        "selected_rows": len(chunk),
    }
    if chunk.empty:
        return pd.DataFrame(), metrics

    missing_participant = int(chunk["participant_id"].isna().sum())
    if missing_participant:
        raise ValueError(
            "目标事件中有 {:,} 行缺少 participant_id。".format(
                missing_participant
            )
        )

    partial = pd.DataFrame({"participant_id": chunk["participant_id"]})
    partial["event_count"] = 1
    for field in audit_fields:
        valid, invalid_numeric = valid_numeric_mask(chunk[field], field)
        partial["{}_valid_event_count".format(field)] = valid.astype("int64")
        metrics["invalid_numeric_{}".format(field)] = invalid_numeric

    grouped = partial.groupby("participant_id", sort=False, dropna=False)
    summary = grouped.sum().reset_index()
    return summary, metrics


def combine_event_partials(
    partials: List[pd.DataFrame], audit_fields: List[str]
) -> pd.DataFrame:
    """合并跨 CSV 数据块的参与者级覆盖局部汇总。"""
    if not partials:
        raise ValueError("没有找到目标 cohort/research_stage 的饮食事件。")
    combined = pd.concat(partials, ignore_index=True)
    value_columns = [
        "event_count",
        *["{}_valid_event_count".format(field) for field in audit_fields],
    ]
    summary = (
        combined.groupby("participant_id", sort=True, dropna=False)[value_columns]
        .sum()
        .reset_index()
    )
    return summary


def build_event_coverage(
    participant_summary: pd.DataFrame, audit_fields: List[str]
) -> pd.DataFrame:
    """计算事件覆盖和参与者严格完整人数。"""
    rows = []
    event_count = int(participant_summary["event_count"].sum())
    participant_count = len(participant_summary)
    for field in audit_fields:
        count_column = "{}_valid_event_count".format(field)
        valid_count = int(participant_summary[count_column].sum())
        strict_complete = participant_summary[count_column].eq(
            participant_summary["event_count"]
        )
        any_valid = participant_summary[count_column].gt(0)
        rows.append(
            {
                "field": field,
                "validity_rule": (
                    "> 0" if field in {"weight_g", "calories_kcal"} else ">= 0"
                ),
                "target_event_count": event_count,
                "valid_event_count": valid_count,
                "missing_or_invalid_event_count": event_count - valid_count,
                "valid_event_share": (
                    valid_count / event_count if event_count else float("nan")
                ),
                "target_participant_count": participant_count,
                "participants_with_any_valid_event": int(any_valid.sum()),
                "strictly_complete_participant_count": int(
                    strict_complete.sum()
                ),
                "not_strictly_complete_participant_count": int(
                    (~strict_complete).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def audit_events(
    events_csv: Path,
    chunk_size: int,
    cohort: str,
    research_stage: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], List[str]]:
    """分块审计主饮食事件表。"""
    header = read_csv_header(events_csv)
    validate_columns(header, ID_COLUMNS, "饮食事件表")
    present_audit_fields = [
        field for field in AUDIT_EVENT_FIELDS if field in header
    ]
    if not present_audit_fields:
        raise ValueError("饮食事件表不含 weight、energy 或 sodium 审计字段。")

    usecols = [*ID_COLUMNS, *present_audit_fields]
    reader = pd.read_csv(
        events_csv,
        encoding="utf-8-sig",
        usecols=usecols,
        dtype={column: "string" for column in EVENT_TEXT_COLUMNS},
        chunksize=chunk_size,
        low_memory=False,
    )
    partials = []
    accumulated_metrics: Dict[str, int] = {}
    for chunk_number, chunk in enumerate(reader, start=1):
        partial, metrics = aggregate_event_chunk(
            chunk, cohort, research_stage, present_audit_fields
        )
        if not partial.empty:
            partials.append(partial)
        for key, value in metrics.items():
            accumulated_metrics[key] = accumulated_metrics.get(key, 0) + value
        print(
            "处理事件块 {}：输入 {:,}，目标 {:,}".format(
                chunk_number,
                accumulated_metrics.get("input_rows", 0),
                accumulated_metrics.get("selected_rows", 0),
            )
        )

    participant_summary = combine_event_partials(
        partials, present_audit_fields
    )
    coverage = build_event_coverage(participant_summary, present_audit_fields)
    return participant_summary, coverage, accumulated_metrics, header


def audit_population(
    population_csv: Path, target_participants: Set[str], cohort: str
) -> Dict[str, int]:
    """核对基线饮食参与者能否一对一匹配有效性别。"""
    if not population_csv.is_file():
        raise FileNotFoundError("找不到 population 表：{}".format(population_csv))
    header = pd.read_csv(population_csv, encoding="utf-8-sig", nrows=0)
    required = ["participant_id", "cohort", "sex"]
    validate_columns(header.columns.tolist(), required, "population 表")
    population = pd.read_csv(
        population_csv,
        encoding="utf-8-sig",
        usecols=required,
        dtype={"participant_id": "string", "cohort": "string"},
        low_memory=False,
    )
    population["participant_id"] = normalize_text(population["participant_id"])
    population["cohort"] = normalize_text(population["cohort"])
    population = population.loc[
        population["cohort"].eq(cohort).fillna(False)
    ].copy()

    duplicated = population["participant_id"].duplicated(keep=False)
    if duplicated.any():
        preview = sorted(
            population.loc[duplicated, "participant_id"].dropna().astype(str).unique()
        )
        raise ValueError(
            "population 表目标队列存在重复 participant_id：{}".format(
                ", ".join(preview[:10])
            )
        )

    original_nonmissing = population["sex"].notna()
    sex = pd.to_numeric(population["sex"], errors="coerce")
    invalid_numeric = original_nonmissing & sex.isna()
    unknown_code = sex.notna() & ~sex.isin([0, 1])
    population["valid_sex"] = sex.isin([0, 1])
    population_ids = set(
        population["participant_id"].dropna().astype(str).tolist()
    )
    indexed = population.set_index("participant_id")
    matched_ids = target_participants & population_ids
    valid_sex_ids = set(
        indexed.index[indexed["valid_sex"]].dropna().astype(str).tolist()
    )
    return {
        "target_participant_count": len(target_participants),
        "population_matched_count": len(matched_ids),
        "population_missing_count": len(target_participants - population_ids),
        "valid_sex_matched_count": len(target_participants & valid_sex_ids),
        "invalid_or_missing_sex_count": len(
            target_participants - valid_sex_ids
        ),
        "population_invalid_numeric_sex_rows": int(invalid_numeric.sum()),
        "population_unknown_sex_code_rows": int(unknown_code.sum()),
    }


def audit_alcohol(
    alcohol_csv: Path,
    target_participants: Set[str],
    cohort: str,
    research_stage: str,
) -> Dict[str, int]:
    """核对 AMED 参与者摄入表中的酒精完整性和当前评分资格。"""
    if not alcohol_csv.is_file():
        raise FileNotFoundError("找不到 AMED 酒精输入表：{}".format(alcohol_csv))
    header = pd.read_csv(alcohol_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(
        header.columns.tolist(), ALCOHOL_REQUIRED_COLUMNS, "AMED 酒精输入表"
    )
    data = pd.read_csv(
        alcohol_csv,
        encoding="utf-8-sig",
        usecols=ALCOHOL_REQUIRED_COLUMNS,
        dtype={column: "string" for column in ID_COLUMNS},
        low_memory=False,
    )
    for column in ID_COLUMNS:
        data[column] = normalize_text(data[column])
    selected = (
        data["cohort"].eq(cohort).fillna(False)
        & data["research_stage"].eq(research_stage).fillna(False)
    )
    data = data.loc[selected].copy()

    duplicated = data["participant_id"].duplicated(keep=False)
    if duplicated.any():
        preview = sorted(
            data.loc[duplicated, "participant_id"].dropna().astype(str).unique()
        )
        raise ValueError(
            "AMED 酒精输入表存在重复 participant_id：{}".format(
                ", ".join(preview[:10])
            )
        )

    data["alcohol_complete_parsed"] = parse_boolean(
        data["alcohol_complete"], "alcohol_complete"
    )
    original_nonmissing = data["mean_daily_alcohol_g"].notna()
    alcohol_g = pd.to_numeric(data["mean_daily_alcohol_g"], errors="coerce")
    invalid_numeric = original_nonmissing & alcohol_g.isna()
    negative = alcohol_g.notna() & alcohol_g.lt(0)
    if invalid_numeric.any() or negative.any():
        raise ValueError(
            "mean_daily_alcohol_g 有 {:,} 个非数值或负值。".format(
                int((invalid_numeric | negative).sum())
            )
        )
    data["valid_alcohol_g"] = alcohol_g.notna() & alcohol_g.ge(0)

    data_ids = set(data["participant_id"].dropna().astype(str).tolist())
    data["in_target"] = data["participant_id"].isin(target_participants)
    target_data = data.loc[data["in_target"]].copy()
    complete = target_data["alcohol_complete_parsed"].eq(True).fillna(False)
    incomplete = target_data["alcohol_complete_parsed"].eq(False).fillna(False)
    missing_flag = target_data["alcohol_complete_parsed"].isna()
    eligible = complete & target_data["valid_alcohol_g"]
    complete_missing_intake = complete & ~target_data["valid_alcohol_g"]

    return {
        "target_participant_count": len(target_participants),
        "alcohol_table_matched_count": len(target_participants & data_ids),
        "alcohol_table_missing_count": len(target_participants - data_ids),
        "alcohol_complete_participant_count": int(complete.sum()),
        "alcohol_incomplete_participant_count": int(incomplete.sum()),
        "alcohol_completeness_flag_missing_count": int(missing_flag.sum()),
        "complete_but_alcohol_intake_missing_count": int(
            complete_missing_intake.sum()
        ),
        "current_official_ahei_eligible_count": int(eligible.sum()),
        "current_official_ahei_ineligible_count": (
            len(target_participants) - int(eligible.sum())
        ),
    }


def read_candidate_header(path: Path) -> Tuple[List[str], str, str]:
    """尽量只读表头；返回列名、状态和错误信息。"""
    suffix = path.suffix.casefold()
    try:
        if suffix == ".csv":
            columns = pd.read_csv(
                path, encoding="utf-8-sig", nrows=0
            ).columns.tolist()
        elif suffix == ".tsv":
            columns = pd.read_csv(
                path, encoding="utf-8-sig", sep="\t", nrows=0
            ).columns.tolist()
        elif suffix == ".txt":
            columns = pd.read_csv(
                path,
                encoding="utf-8-sig",
                sep=None,
                engine="python",
                nrows=0,
            ).columns.tolist()
        elif suffix in {".xlsx", ".xls"}:
            columns = pd.read_excel(path, nrows=0).columns.tolist()
        elif suffix == ".parquet":
            try:
                import pyarrow.parquet as parquet  # type: ignore

                columns = parquet.ParquetFile(str(path)).schema.names
            except ImportError:
                return [], "unsupported", "pyarrow 未安装，未读取 parquet 表头"
        else:
            return [], "unsupported", "不支持的文件类型"
        return [str(column) for column in columns], "read", ""
    except Exception as error:  # 审计要记录单个候选文件问题，而不是整体中止
        return [], "error", "{}: {}".format(type(error).__name__, error)


def has_food_id_column(columns: Iterable[object]) -> bool:
    """判断候选表是否包含可识别的 food_id 键。"""
    normalized = {normalize_column_name(column) for column in columns}
    return bool(normalized & {"food_id", "foodid", "id_food"})


def scan_candidate_sources(
    transfer_dir: Path,
    excluded_paths: Set[Path],
    file_limit: int,
    progress_every: int = 0,
) -> Tuple[pd.DataFrame, int, bool]:
    """扫描 Data/Transfer 中可能按 food_id 连接的营养或份量表。"""
    if not transfer_dir.is_dir():
        raise FileNotFoundError("找不到 Data/Transfer：{}".format(transfer_dir))
    excluded_resolved = {path.resolve() for path in excluded_paths}
    table_paths = []
    limit_reached = False
    for path in sorted(transfer_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_TABLE_SUFFIXES:
            continue
        if path.resolve() in excluded_resolved:
            continue
        if file_limit > 0 and len(table_paths) >= file_limit:
            limit_reached = True
            break
        table_paths.append(path)

    rows = []
    for file_number, path in enumerate(table_paths, start=1):
        columns, header_status, scan_error = read_candidate_header(path)
        hits = find_keyword_hits(columns)
        keyword_groups = sorted({hit["keyword_group"] for hit in hits})
        matched_columns = sorted({hit["column"] for hit in hits})
        has_food_id = has_food_id_column(columns)
        normalized_name = normalize_column_name(path.stem)
        filename_matches = sorted(
            keyword for keyword in CANDIDATE_FILE_KEYWORDS if keyword in normalized_name
        )
        likely_lookup = has_food_id and bool(
            set(keyword_groups)
            & {
                "serving_or_portion",
                "trans_fat",
                "pufa",
                "whole_grain",
                "nutrient",
            }
        )
        if has_food_id or keyword_groups or filename_matches or header_status != "read":
            rows.append(
                {
                    "path": str(path),
                    "suffix": path.suffix.casefold(),
                    "header_status": header_status,
                    "column_count": len(columns),
                    "has_food_id": has_food_id,
                    "matched_keyword_groups": "|".join(keyword_groups),
                    "matched_columns": "|".join(matched_columns),
                    "filename_keyword_matches": "|".join(filename_matches),
                    "likely_ahei_lookup": likely_lookup,
                    "scan_error": scan_error,
                }
            )
        if progress_every > 0 and file_number % progress_every == 0:
            print(
                "候选表头扫描进度：已检查 {:,}/{:,} 个表".format(
                    file_number, len(table_paths)
                )
            )
    columns = [
        "path",
        "suffix",
        "header_status",
        "column_count",
        "has_food_id",
        "matched_keyword_groups",
        "matched_columns",
        "filename_keyword_matches",
        "likely_ahei_lookup",
        "scan_error",
    ]
    return pd.DataFrame(rows, columns=columns), len(table_paths), limit_reached


def hit_columns(hits: List[Dict[str, str]], group: str) -> List[str]:
    """从关键词命中列表提取某一组的字段名。"""
    return sorted({hit["column"] for hit in hits if hit["keyword_group"] == group})


def candidate_count_for_group(candidates: pd.DataFrame, group: str) -> int:
    """统计含 food_id 且命中指定字段组的候选 lookup 数。"""
    if candidates.empty:
        return 0
    groups = candidates["matched_keyword_groups"].fillna("").astype(str)
    keyed = candidates["has_food_id"].eq(True)
    return int((keyed & groups.str.split("|").apply(lambda values: group in values)).sum())


def choose_availability_status(
    requirement_group: str,
    main_hits: List[Dict[str, str]],
    raw_hits: List[Dict[str, str]],
    candidates: pd.DataFrame,
    has_weight: bool,
) -> Tuple[str, str]:
    """按直接字段、待验证来源和重量代理分层判断输入可用性。"""
    if hit_columns(main_hits, requirement_group):
        return "strictly_available", "主事件表存在所需字段"
    if hit_columns(raw_hits, requirement_group):
        return "proxy_only", "仅 raw 事件表发现候选字段，需验证并恢复到分析表"
    if candidate_count_for_group(candidates, requirement_group) > 0:
        return "proxy_only", "发现可按 food_id 连接的候选表，需核实定义和覆盖"
    if has_weight and requirement_group == "serving_or_portion":
        return "proxy_only", "只有 weight_g；必须建立有来源的 food-specific 份量换算"
    if has_weight and requirement_group == "whole_grain":
        return "proxy_only", "可映射食物重量，但混合食物的实际全谷物克数未知"
    return "unavailable", "未发现所需直接字段或可连接候选来源"


def build_component_availability(
    main_columns: List[str],
    raw_columns: List[str],
    candidates: pd.DataFrame,
    alcohol_metrics: Dict[str, int],
    population_metrics: Dict[str, int],
) -> pd.DataFrame:
    """逐项标记 HPP 10 个 AHEI 组件的严格、代理或不可用状态。"""
    main_hits = find_keyword_hits(main_columns)
    raw_hits = find_keyword_hits(raw_columns)
    has_weight = "weight_g" in main_columns
    serving_status, serving_reason = choose_availability_status(
        "serving_or_portion", main_hits, raw_hits, candidates, has_weight
    )
    whole_status, whole_reason = choose_availability_status(
        "whole_grain", main_hits, raw_hits, candidates, has_weight
    )
    trans_status, trans_reason = choose_availability_status(
        "trans_fat", main_hits, raw_hits, candidates, has_weight
    )
    pufa_status, pufa_reason = choose_availability_status(
        "pufa", main_hits, raw_hits, candidates, has_weight
    )

    rows = []

    def add_row(
        component: str,
        unit: str,
        status: str,
        reason: str,
        requirement_group: str,
        next_action: str,
        eligible_count: Optional[int] = None,
    ) -> None:
        rows.append(
            {
                "ahei_component": component,
                "paper_unit": unit,
                "availability_status": status,
                "direct_event_columns": "|".join(
                    hit_columns(main_hits, requirement_group)
                ),
                "raw_only_candidate_columns": "|".join(
                    hit_columns(raw_hits, requirement_group)
                ),
                "food_id_candidate_source_count": candidate_count_for_group(
                    candidates, requirement_group
                ),
                "current_eligible_participant_count": eligible_count,
                "reason": reason,
                "next_action": next_action,
            }
        )

    food_actions = "建立 AHEI 独立映射和有来源的 food-specific serving conversion"
    add_row("vegetables", "servings/day", serving_status, serving_reason, "serving_or_portion", food_actions)
    add_row("fruit", "servings/day", serving_status, serving_reason, "serving_or_portion", food_actions)
    add_row("whole_grains", "g/day", whole_status, whole_reason, "whole_grain", "核实 whole-grain grams，复合食物不得按整份重量计")
    add_row("ssb_plus_fruit_juice", "servings/day", serving_status, serving_reason, "serving_or_portion", "核实 8 oz/份及饮料密度/体积字段")
    add_row("nuts_plus_legumes", "servings/day", serving_status, serving_reason, "serving_or_portion", "按坚果、花生酱、豆类分别建立换算")
    add_row("red_plus_processed_meat", "servings/day", serving_status, serving_reason, "serving_or_portion", "未加工红肉与加工肉使用不同份量定义")
    add_row("trans_fat", "% energy", trans_status, trans_reason, "trans_fat", "取得可靠 trans_fat_g 后再计算；不得把缺失设为0")
    add_row("pufa", "% energy", pufa_status, pufa_reason, "pufa", "取得可靠 PUFA_g 后再计算；不得用总脂肪代替")

    sodium_available = "sodium_mg" in main_columns
    add_row(
        "sodium",
        "mg/day; sample deciles",
        "strictly_available" if sodium_available else "unavailable",
        "主事件表有 sodium_mg" if sodium_available else "主事件表没有 sodium_mg",
        "nutrient",
        "在当前正式评分样本内重新计算钠的十分位端点",
    )

    alcohol_eligible = alcohol_metrics.get("current_official_ahei_eligible_count", 0)
    valid_sex = population_metrics.get("valid_sex_matched_count", 0)
    target_count = alcohol_metrics.get("target_participant_count", 0)
    alcohol_strict = alcohol_eligible > 0 and valid_sex == target_count
    add_row(
        "alcohol",
        "drinks/day; sex-specific",
        "strictly_available" if alcohol_strict else "unavailable",
        (
            "AMED 酒精摄入完整且全部基线参与者可匹配有效性别"
            if alcohol_strict
            else "酒精完整性或性别匹配仍有缺口"
        ),
        "nutrient",
        "当前仅酒精完整者进入正式 AHEI；1 drink 换算和分段公式需参数化",
        alcohol_eligible,
    )
    return pd.DataFrame(rows)


def build_column_inventory(
    sources: List[Tuple[str, Path, List[str]]]
) -> pd.DataFrame:
    """生成主表和 raw 表的完整列清单及关键词命中。"""
    rows = []
    for source_kind, path, columns in sources:
        hits = find_keyword_hits(columns)
        groups_by_column: Dict[str, List[str]] = {}
        for hit in hits:
            groups_by_column.setdefault(hit["column"], []).append(
                hit["keyword_group"]
            )
        for position, column in enumerate(columns, start=1):
            rows.append(
                {
                    "source_kind": source_kind,
                    "source_path": str(path),
                    "column_position": position,
                    "column": column,
                    "ahei_keyword_groups": "|".join(
                        sorted(groups_by_column.get(column, []))
                    ),
                }
            )
    return pd.DataFrame(rows)


def metrics_to_table(
    event_metrics: Dict[str, int],
    population_metrics: Dict[str, int],
    alcohol_metrics: Dict[str, int],
    scanned_table_count: int,
    candidate_limit_reached: bool,
) -> pd.DataFrame:
    """把各模块 QC 指标转成长表。"""
    rows = [
        {"section": "script", "metric": "script_version", "value": SCRIPT_VERSION},
        {"section": "candidate_scan", "metric": "scanned_table_count", "value": scanned_table_count},
        {"section": "candidate_scan", "metric": "file_limit_reached", "value": candidate_limit_reached},
    ]
    for section, metrics in [
        ("events", event_metrics),
        ("population", population_metrics),
        ("alcohol", alcohol_metrics),
    ]:
        for metric, value in metrics.items():
            rows.append({"section": section, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def audit_ahei_inputs(
    events_csv: Path,
    raw_events_csv: Path,
    population_csv: Path,
    alcohol_csv: Path,
    transfer_dir: Path,
    output_dir: Path,
    chunk_size: int,
    candidate_file_limit: int,
    cohort: str,
    research_stage: str,
) -> Tuple[Path, Path, Path, Path, Path]:
    """执行 AHEI 输入审计并保存可复核输出。"""
    raw_columns = read_csv_header(raw_events_csv)
    participant_summary, coverage, event_metrics, main_columns = audit_events(
        events_csv, chunk_size, cohort, research_stage
    )
    target_participants = set(
        participant_summary["participant_id"].astype(str).tolist()
    )
    population_metrics = audit_population(
        population_csv, target_participants, cohort
    )
    alcohol_metrics = audit_alcohol(
        alcohol_csv, target_participants, cohort, research_stage
    )
    candidates, scanned_table_count, limit_reached = scan_candidate_sources(
        transfer_dir,
        {events_csv, raw_events_csv, population_csv},
        candidate_file_limit,
    )
    availability = build_component_availability(
        main_columns,
        raw_columns,
        candidates,
        alcohol_metrics,
        population_metrics,
    )
    inventory = build_column_inventory(
        [
            ("main_events", events_csv, main_columns),
            ("raw_events", raw_events_csv, raw_columns),
        ]
    )
    qc = metrics_to_table(
        event_metrics,
        population_metrics,
        alcohol_metrics,
        scanned_table_count,
        limit_reached,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    availability_path = output_dir / "ahei_input_availability.csv"
    inventory_path = output_dir / "ahei_source_column_inventory.csv"
    candidate_path = output_dir / "ahei_candidate_data_sources.csv"
    coverage_path = output_dir / "ahei_event_field_coverage.csv"
    qc_path = output_dir / "ahei_input_audit_qc.csv"
    availability.to_csv(availability_path, index=False, encoding="utf-8-sig")
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    print("=" * 76)
    print("AHEI 输入可用性审计")
    print("SCRIPT_VERSION={}".format(SCRIPT_VERSION))
    print("筛选：cohort={}；research_stage={}".format(cohort, research_stage))
    print("主事件表列（{}）：{}".format(len(main_columns), ", ".join(main_columns)))
    print("raw 事件表列（{}）：{}".format(len(raw_columns), ", ".join(raw_columns)))
    print("目标事件：{:,}".format(int(participant_summary["event_count"].sum())))
    print("基线参与者：{:,}".format(len(target_participants)))
    print("\n字段覆盖：")
    print(
        coverage[
            [
                "field",
                "valid_event_count",
                "missing_or_invalid_event_count",
                "valid_event_share",
                "strictly_complete_participant_count",
            ]
        ].to_string(index=False)
    )
    print("\n性别匹配：")
    print(
        "有效性别 {:,}；缺失/无效 {:,}".format(
            population_metrics["valid_sex_matched_count"],
            population_metrics["invalid_or_missing_sex_count"],
        )
    )
    print("\n酒精完整性与当前正式评分资格：")
    print(
        "完整 {:,}；不完整 {:,}；当前正式 AHEI 资格 {:,}".format(
            alcohol_metrics["alcohol_complete_participant_count"],
            alcohol_metrics["alcohol_incomplete_participant_count"],
            alcohol_metrics["current_official_ahei_eligible_count"],
        )
    )
    likely_count = int(candidates["likely_ahei_lookup"].eq(True).sum()) if not candidates.empty else 0
    print("\n候选表扫描：")
    print(
        "扫描 {:,} 个表；可按 food_id 连接且命中 AHEI 字段的候选表 {:,} 个{}".format(
            scanned_table_count,
            likely_count,
            "（已达到文件上限）" if limit_reached else "",
        )
    )
    if likely_count:
        print(
            candidates.loc[
                candidates["likely_ahei_lookup"].eq(True),
                ["path", "matched_keyword_groups", "matched_columns"],
            ].head(20).to_string(index=False)
        )
    print("\n10项输入状态：")
    print(
        availability[
            ["ahei_component", "availability_status", "reason"]
        ].to_string(index=False)
    )
    print("\n输出目录：{}".format(output_dir))
    print("核心可用性表：{}".format(availability_path))
    print("=" * 76)
    return availability_path, inventory_path, candidate_path, coverage_path, qc_path


def parse_args():
    parser = ArgumentParser(
        description="审计 HPP AHEI 的份量、脂肪酸、钠、性别和酒精输入"
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_CSV)
    parser.add_argument("--raw-events", type=Path, default=DEFAULT_RAW_EVENTS_CSV)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION_CSV)
    parser.add_argument("--alcohol", type=Path, default=DEFAULT_ALCOHOL_CSV)
    parser.add_argument("--transfer-dir", type=Path, default=DEFAULT_TRANSFER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--candidate-file-limit", type=int, default=5000)
    parser.add_argument("--cohort", default=DEFAULT_COHORT)
    parser.add_argument("--research-stage", default=DEFAULT_RESEARCH_STAGE)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size 必须为正整数")
    if args.candidate_file_limit <= 0:
        parser.error("--candidate-file-limit 必须为正整数")
    return args


def main() -> None:
    args = parse_args()
    audit_ahei_inputs(
        events_csv=args.events,
        raw_events_csv=args.raw_events,
        population_csv=args.population,
        alcohol_csv=args.alcohol,
        transfer_dir=args.transfer_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        candidate_file_limit=args.candidate_file_limit,
        cohort=args.cohort,
        research_stage=args.research_stage,
    )


if __name__ == "__main__":
    main()
