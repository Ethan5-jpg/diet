"""只读审计 HPP rDII 21项营养素及可连接的候选数据源。

脚本只读取主事件表、raw事件表和 Data/Transfer 下其他表的表头，不计算DII、
不把缺失营养素设为0，也不读取或导出服务器完整数据。
"""

from argparse import ArgumentParser
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-09-rdii-nutrient-audit-v2"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR.parent
DEFAULT_EVENTS_CSV = (
    DATA_DIR / "Transfer" / "diet_logging" / "diet_logging_events.csv"
)
DEFAULT_RAW_EVENTS_CSV = (
    DATA_DIR / "Transfer" / "diet_logging" / "raw_diet_logging_events.csv"
)
DEFAULT_TRANSFER_DIR = DATA_DIR / "Transfer"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "03_score_mapping" / "rdii"


def nutrient_definition(
    component: str,
    reported_unit: str,
    effect_score: float,
    global_mean: float,
    global_sd: float,
    aliases: List[str],
    exclusions: Optional[List[str]] = None,
    unit_review_note: str = "",
) -> Dict[str, object]:
    """构造一条Supplementary Table 13营养素定义。"""
    return {
        "component": component,
        "reported_unit": reported_unit,
        "effect_score": effect_score,
        "global_mean": global_mean,
        "global_sd": global_sd,
        "aliases": aliases,
        "exclusions": exclusions or [],
        "unit_review_note": unit_review_note,
    }


NUTRIENT_DEFINITIONS = [
    nutrient_definition(
        "alcohol", "g", -0.278, 13.98, 3.72, ["alcohol", "ethanol"]
    ),
    nutrient_definition(
        "vitamin_b12",
        "mg",
        0.106,
        5.15,
        2.7,
        ["vitamin_b12", "vit_b12", "b12", "cobalamin"],
        unit_review_note="表13写mg；均值5.15疑似应按µg核对原始DII来源",
    ),
    nutrient_definition(
        "vitamin_b6",
        "mg",
        -0.365,
        1.47,
        0.74,
        ["vitamin_b6", "vit_b6", "b6", "pyridoxine"],
    ),
    nutrient_definition(
        "carbohydrate",
        "g",
        0.097,
        272.2,
        40.0,
        ["carbohydrate", "carbohydrates", "total_carbohydrate", "carb", "carbs"],
    ),
    nutrient_definition(
        "cholesterol", "mg", 0.11, 279.4, 51.2, ["cholesterol"]
    ),
    nutrient_definition(
        "energy",
        "kcal",
        0.18,
        2056.0,
        338.0,
        ["calories", "calorie", "energy_kcal", "energy", "kcal"],
    ),
    nutrient_definition(
        "total_fat",
        "g",
        0.298,
        71.4,
        19.4,
        [
            "total_fat",
            "fat_total",
            "fat_g",
            "fat_grams",
            "fat_content",
            "lipid",
        ],
        exclusions=[
            "saturated",
            "unsaturated",
            "monounsaturated",
            "polyunsaturated",
            "trans_fat",
            "mufa",
            "pufa",
            "fatty_acid",
        ],
    ),
    nutrient_definition(
        "fibre",
        "g",
        -0.663,
        18.8,
        4.9,
        ["dietary_fiber", "dietary_fibre", "fiber", "fibre"],
    ),
    nutrient_definition(
        "iron", "mg", 0.032, 13.35, 3.71, ["iron", "fe"]
    ),
    nutrient_definition(
        "magnesium", "mg", -0.484, 310.1, 139.4, ["magnesium"]
    ),
    nutrient_definition(
        "mufa",
        "g",
        -0.009,
        27.0,
        6.1,
        ["mufa", "monounsaturated_fat", "monounsaturated_fatty_acid"],
    ),
    nutrient_definition(
        "niacin",
        "mg",
        -0.246,
        25.9,
        11.77,
        ["niacin", "vitamin_b3", "vit_b3"],
    ),
    nutrient_definition(
        "protein", "g", 0.021, 79.4, 13.9, ["protein"]
    ),
    nutrient_definition(
        "pufa",
        "g",
        -0.337,
        13.88,
        3.76,
        ["pufa", "polyunsaturated_fat", "polyunsaturated_fatty_acid"],
    ),
    nutrient_definition(
        "riboflavin",
        "mg",
        -0.068,
        1.7,
        0.79,
        ["riboflavin", "vitamin_b2", "vit_b2"],
    ),
    nutrient_definition(
        "saturated_fat",
        "g",
        0.373,
        28.6,
        8.0,
        ["saturated_fat", "saturated_fatty_acid", "sfa"],
        exclusions=["unsaturated", "monounsaturated", "polyunsaturated"],
    ),
    nutrient_definition(
        "thiamin",
        "mg",
        -0.098,
        1.7,
        0.66,
        ["thiamin", "thiamine", "vitamin_b1", "vit_b1", "b1"],
        exclusions=["b12"],
    ),
    nutrient_definition(
        "vitamin_a",
        "RE",
        -0.401,
        983.9,
        518.6,
        ["vitamin_a", "vit_a", "retinol_equivalent", "retinol_equivalents"],
        exclusions=["beta_carotene", "carotene"],
        unit_review_note="需核对lookup中的RE/RAE/µg单位及换算",
    ),
    nutrient_definition(
        "vitamin_c",
        "mg",
        -0.424,
        118.2,
        43.46,
        ["vitamin_c", "vit_c", "ascorbic_acid", "ascorbate"],
    ),
    nutrient_definition(
        "vitamin_e",
        "mg",
        -0.419,
        8.73,
        1.49,
        ["vitamin_e", "vit_e", "alpha_tocopherol", "tocopherol"],
    ),
    nutrient_definition(
        "zinc", "mg", -0.313, 9.84, 2.19, ["zinc", "zn"]
    ),
]

EXPECTED_COMPONENT_COUNT = 21

FOOD_KEY_ALIASES = {
    "food_id",
    "foodid",
    "id_food",
    "food_code",
    "foodcode",
    "item_id",
    "itemid",
    "dish_id",
    "dishid",
    "product_id",
    "productid",
}
PARTICIPANT_KEY_ALIASES = {
    "participant_id",
    "participantid",
    "research_id",
    "researchid",
    "user_id",
    "userid",
}
NUTRIENT_DESCRIPTOR_ALIASES = {
    "nutrient",
    "nutrient_name",
    "nutrient_code",
    "nutrient_id",
    "component",
    "component_name",
}
VALUE_COLUMN_ALIASES = {
    "value",
    "amount",
    "nutrient_value",
    "nutrient_amount",
    "quantity",
}
FILE_NAME_KEYWORDS = [
    "nutrient",
    "nutrition",
    "composition",
    "food",
    "diet",
    "usda",
    "ministry",
]


def normalize_column_name(value: object) -> str:
    """把字段名转成小写下划线格式，便于安全匹配别名。"""
    text = str(value).strip().casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def alias_in_column(normalized_column: str, alias: str) -> bool:
    """按完整下划线词组匹配，避免B1误命中B12等字段。"""
    normalized_alias = normalize_column_name(alias)
    pattern = r"(^|_){}($|_)".format(re.escape(normalized_alias))
    return re.search(pattern, normalized_column) is not None


def matches_definition(column: object, definition: Dict[str, object]) -> bool:
    """判断字段是否对应一项rDII营养素。"""
    normalized = normalize_column_name(column)
    exclusions = definition["exclusions"]
    for exclusion in exclusions:
        if normalize_column_name(exclusion) in normalized:
            return False
    aliases = definition["aliases"]
    return any(alias_in_column(normalized, alias) for alias in aliases)


def match_nutrient_columns(
    columns: Iterable[object],
) -> Dict[str, List[str]]:
    """返回21项营养素各自命中的字段。"""
    matches: Dict[str, List[str]] = {
        str(definition["component"]): []
        for definition in NUTRIENT_DEFINITIONS
    }
    for column in columns:
        for definition in NUTRIENT_DEFINITIONS:
            if matches_definition(column, definition):
                component = str(definition["component"])
                matches[component].append(str(column))
    return {key: sorted(set(values)) for key, values in matches.items()}


def find_key_columns(
    columns: Iterable[object], aliases: Set[str]
) -> List[str]:
    """查找可用于连接候选表的键字段。"""
    result = []
    for column in columns:
        if normalize_column_name(column) in aliases:
            result.append(str(column))
    return sorted(set(result))


def find_exact_alias_columns(
    columns: Iterable[object], aliases: Set[str]
) -> List[str]:
    """查找long-format营养表的描述或数值字段。"""
    result = []
    for column in columns:
        if normalize_column_name(column) in aliases:
            result.append(str(column))
    return sorted(set(result))


def detect_table_format(path: Path) -> Optional[str]:
    """识别支持只读表头扫描的文件格式。"""
    name = path.name.casefold()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return "csv"
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return "tsv"
    if name.endswith(".txt") or name.endswith(".txt.gz"):
        return "txt"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".xlsx"):
        return "xlsx"
    if name.endswith(".xls"):
        return "xls"
    return None


def read_table_header(path: Path) -> Tuple[List[str], str, str]:
    """尽量只读表头，并记录依赖缺失或文件错误。"""
    table_format = detect_table_format(path)
    try:
        if table_format == "csv":
            columns = pd.read_csv(
                path, encoding="utf-8-sig", nrows=0
            ).columns.tolist()
        elif table_format == "tsv":
            columns = pd.read_csv(
                path, encoding="utf-8-sig", sep="\t", nrows=0
            ).columns.tolist()
        elif table_format == "txt":
            columns = pd.read_csv(
                path,
                encoding="utf-8-sig",
                sep=None,
                engine="python",
                nrows=0,
            ).columns.tolist()
        elif table_format in {"xlsx", "xls"}:
            columns = pd.read_excel(path, nrows=0).columns.tolist()
        elif table_format == "parquet":
            try:
                import pyarrow.parquet as parquet  # type: ignore

                columns = parquet.ParquetFile(str(path)).schema.names
            except ImportError:
                return [], "unsupported", "pyarrow未安装，未读取parquet表头"
        else:
            return [], "unsupported", "不支持的文件类型"
        return [str(column) for column in columns], "read", ""
    except Exception as error:  # 单个文件失败不应阻断全目录审计
        return [], "error", "{}: {}".format(type(error).__name__, error)


def list_table_paths(
    transfer_dir: Path, excluded_paths: Set[Path], file_limit: int
) -> Tuple[List[Path], bool]:
    """列出待扫描的表文件；0表示不设上限。"""
    if not transfer_dir.is_dir():
        raise FileNotFoundError("找不到Data/Transfer：{}".format(transfer_dir))
    excluded = {path.resolve() for path in excluded_paths}
    paths = []
    limit_reached = False
    for path in sorted(transfer_dir.rglob("*")):
        if not path.is_file() or detect_table_format(path) is None:
            continue
        if path.resolve() in excluded:
            continue
        if file_limit > 0 and len(paths) >= file_limit:
            limit_reached = True
            break
        paths.append(path)
    return paths, limit_reached


def scan_transfer_headers(
    transfer_dir: Path,
    excluded_paths: Set[Path],
    file_limit: int,
    progress_every: int,
) -> Tuple[pd.DataFrame, int, bool]:
    """扫描候选宽表和long-format营养表的表头。"""
    paths, limit_reached = list_table_paths(
        transfer_dir, excluded_paths, file_limit
    )
    rows = []
    for file_number, path in enumerate(paths, start=1):
        columns, header_status, scan_error = read_table_header(path)
        matches = match_nutrient_columns(columns)
        matched_components = sorted(
            component for component, values in matches.items() if values
        )
        matched_pairs = []
        for component in matched_components:
            for column in matches[component]:
                matched_pairs.append("{}={}".format(component, column))

        food_key_columns = find_key_columns(columns, FOOD_KEY_ALIASES)
        participant_key_columns = find_key_columns(
            columns, PARTICIPANT_KEY_ALIASES
        )
        descriptor_columns = find_exact_alias_columns(
            columns, NUTRIENT_DESCRIPTOR_ALIASES
        )
        value_columns = find_exact_alias_columns(columns, VALUE_COLUMN_ALIASES)
        long_format_candidate = bool(
            (food_key_columns or participant_key_columns)
            and descriptor_columns
            and value_columns
        )
        normalized_stem = normalize_column_name(path.stem)
        filename_matches = sorted(
            keyword for keyword in FILE_NAME_KEYWORDS if keyword in normalized_stem
        )

        if (
            matched_components
            or long_format_candidate
            or filename_matches
            or header_status != "read"
        ):
            rows.append(
                {
                    "path": str(path),
                    "table_format": detect_table_format(path),
                    "header_status": header_status,
                    "column_count": len(columns),
                    "food_key_columns": "|".join(food_key_columns),
                    "participant_key_columns": "|".join(
                        participant_key_columns
                    ),
                    "matched_rdii_component_count": len(matched_components),
                    "matched_rdii_components": "|".join(matched_components),
                    "matched_component_columns": "|".join(matched_pairs),
                    "nutrient_descriptor_columns": "|".join(
                        descriptor_columns
                    ),
                    "value_columns": "|".join(value_columns),
                    "long_format_nutrient_candidate": long_format_candidate,
                    "filename_keyword_matches": "|".join(filename_matches),
                    "scan_error": scan_error,
                }
            )
        if progress_every > 0 and file_number % progress_every == 0:
            print(
                "表头扫描进度：已检查 {:,}/{:,} 个表".format(
                    file_number, len(paths)
                )
            )

    output_columns = [
        "path",
        "table_format",
        "header_status",
        "column_count",
        "food_key_columns",
        "participant_key_columns",
        "matched_rdii_component_count",
        "matched_rdii_components",
        "matched_component_columns",
        "nutrient_descriptor_columns",
        "value_columns",
        "long_format_nutrient_candidate",
        "filename_keyword_matches",
        "scan_error",
    ]
    return pd.DataFrame(rows, columns=output_columns), len(paths), limit_reached


def candidate_mask_for_component(
    candidates: pd.DataFrame, component: str
) -> pd.Series:
    """定位候选表中命中指定rDII组件的记录。"""
    if candidates.empty:
        return pd.Series(False, index=candidates.index, dtype="bool")
    values = candidates["matched_rdii_components"].fillna("").astype(str)
    return values.str.split("|").apply(lambda items: component in items)


def build_availability(
    main_columns: List[str],
    raw_columns: List[str],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """逐项汇总21项营养素在事件表和候选数据源中的状态。"""
    main_matches = match_nutrient_columns(main_columns)
    raw_matches = match_nutrient_columns(raw_columns)
    rows = []
    long_format_count = (
        int(candidates["long_format_nutrient_candidate"].eq(True).sum())
        if not candidates.empty
        else 0
    )

    for definition in NUTRIENT_DEFINITIONS:
        component = str(definition["component"])
        component_mask = candidate_mask_for_component(candidates, component)
        component_candidates = candidates.loc[component_mask].copy()
        if component_candidates.empty:
            food_count = 0
            participant_count = 0
            unkeyed_count = 0
            path_preview = ""
        else:
            food_keyed = component_candidates["food_key_columns"].fillna("").ne("")
            participant_keyed = component_candidates[
                "participant_key_columns"
            ].fillna("").ne("")
            food_count = int(food_keyed.sum())
            participant_count = int((~food_keyed & participant_keyed).sum())
            unkeyed_count = int((~food_keyed & ~participant_keyed).sum())
            path_preview = "|".join(
                component_candidates["path"].astype(str).head(10).tolist()
            )

        if main_matches[component]:
            status = "direct_event_column"
            reason = "主事件表存在直接字段"
        elif raw_matches[component]:
            status = "raw_event_candidate_requires_recovery"
            reason = "仅raw事件表发现字段，需验证并恢复"
        elif food_count:
            status = "food_lookup_candidate_requires_validation"
            reason = "发现含食物键的候选营养宽表"
        elif participant_count:
            status = "participant_table_candidate_requires_validation"
            reason = "发现含参与者键的候选营养表"
        elif unkeyed_count:
            status = "unkeyed_candidate_requires_review"
            reason = "发现字段但没有可识别连接键"
        elif long_format_count:
            status = "long_format_candidate_requires_value_scan"
            reason = "发现long-format营养表，需检查营养素名称取值"
        else:
            status = "unavailable"
            reason = "未在事件表或候选表头发现"

        rows.append(
            {
                "rdii_component": component,
                "reported_unit": definition["reported_unit"],
                "overall_inflammatory_effect_score": definition[
                    "effect_score"
                ],
                "global_daily_mean": definition["global_mean"],
                "global_sd": definition["global_sd"],
                "availability_status": status,
                "main_event_columns": "|".join(main_matches[component]),
                "raw_event_columns": "|".join(raw_matches[component]),
                "food_keyed_candidate_count": food_count,
                "participant_keyed_candidate_count": participant_count,
                "unkeyed_candidate_count": unkeyed_count,
                "long_format_candidate_count": long_format_count,
                "candidate_path_preview": path_preview,
                "unit_review_note": definition["unit_review_note"],
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_source_inventory(
    events_csv: Path,
    main_columns: List[str],
    raw_events_csv: Path,
    raw_columns: List[str],
) -> pd.DataFrame:
    """保存主事件表和raw表的列级营养素命中。"""
    rows = []
    for source_kind, path, columns in [
        ("main_events", events_csv, main_columns),
        ("raw_events", raw_events_csv, raw_columns),
    ]:
        matches = match_nutrient_columns(columns)
        components_by_column: Dict[str, List[str]] = {}
        for component, matched_columns in matches.items():
            for column in matched_columns:
                components_by_column.setdefault(column, []).append(component)
        for position, column in enumerate(columns, start=1):
            rows.append(
                {
                    "source_kind": source_kind,
                    "source_path": str(path),
                    "column_position": position,
                    "column": column,
                    "matched_rdii_components": "|".join(
                        sorted(components_by_column.get(column, []))
                    ),
                }
            )
    return pd.DataFrame(rows)


def audit_rdii_nutrients(
    events_csv: Path,
    raw_events_csv: Path,
    transfer_dir: Path,
    output_dir: Path,
    file_limit: int,
    progress_every: int,
) -> Tuple[Path, Path, Path, Path]:
    """执行rDII表头审计并保存可复核输出。"""
    if len(NUTRIENT_DEFINITIONS) != EXPECTED_COMPONENT_COUNT:
        raise ValueError("rDII营养素定义数量不是21项。")
    main_columns, main_status, main_error = read_table_header(events_csv)
    raw_columns, raw_status, raw_error = read_table_header(raw_events_csv)
    if main_status != "read":
        raise ValueError("无法读取主事件表表头：{}".format(main_error))
    if raw_status != "read":
        raise ValueError("无法读取raw事件表表头：{}".format(raw_error))

    candidates, scanned_count, limit_reached = scan_transfer_headers(
        transfer_dir,
        {events_csv, raw_events_csv},
        file_limit,
        progress_every,
    )
    availability = build_availability(main_columns, raw_columns, candidates)
    inventory = build_source_inventory(
        events_csv, main_columns, raw_events_csv, raw_columns
    )

    direct_count = int(
        availability["availability_status"].eq("direct_event_column").sum()
    )
    unavailable_count = int(
        availability["availability_status"].eq("unavailable").sum()
    )
    candidate_count = len(availability) - direct_count - unavailable_count
    header_error_count = (
        int(candidates["header_status"].eq("error").sum())
        if not candidates.empty
        else 0
    )
    long_format_count = (
        int(candidates["long_format_nutrient_candidate"].eq(True).sum())
        if not candidates.empty
        else 0
    )
    qc = pd.DataFrame(
        [
            ("script_version", SCRIPT_VERSION),
            ("expected_rdii_component_count", EXPECTED_COMPONENT_COUNT),
            ("main_event_column_count", len(main_columns)),
            ("raw_event_column_count", len(raw_columns)),
            ("scanned_table_count", scanned_count),
            ("file_limit_reached", limit_reached),
            ("complete_unlimited_scan", file_limit == 0 and not limit_reached),
            ("header_error_count", header_error_count),
            ("long_format_nutrient_candidate_count", long_format_count),
            ("direct_event_component_count", direct_count),
            ("candidate_or_recovery_component_count", candidate_count),
            ("unavailable_component_count", unavailable_count),
            (
                "all_21_direct_or_candidate",
                unavailable_count == 0 and len(availability) == 21,
            ),
        ],
        columns=["metric", "value"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    availability_path = output_dir / "rdii_nutrient_input_availability.csv"
    candidate_path = output_dir / "rdii_candidate_data_sources.csv"
    inventory_path = output_dir / "rdii_source_column_inventory.csv"
    qc_path = output_dir / "rdii_input_audit_qc.csv"
    availability.to_csv(
        availability_path, index=False, encoding="utf-8-sig"
    )
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    print("=" * 88)
    print("rDII 21项营养素输入审计")
    print("SCRIPT_VERSION={}".format(SCRIPT_VERSION))
    print("主事件表列（{}）：{}".format(len(main_columns), ", ".join(main_columns)))
    print("raw事件表列（{}）：{}".format(len(raw_columns), ", ".join(raw_columns)))
    print(
        "扫描表数：{:,}；完整无限扫描：{}；表头错误：{:,}".format(
            scanned_count, file_limit == 0 and not limit_reached, header_error_count
        )
    )
    print(
        "直接可用：{}/21；候选或待恢复：{}/21；仍不可用：{}/21；long-format候选：{}".format(
            direct_count, candidate_count, unavailable_count, long_format_count
        )
    )
    print("\n21项状态：")
    print(
        availability[
            [
                "rdii_component",
                "reported_unit",
                "availability_status",
                "main_event_columns",
                "food_keyed_candidate_count",
                "participant_keyed_candidate_count",
                "long_format_candidate_count",
            ]
        ].to_string(index=False)
    )

    unavailable = availability.loc[
        availability["availability_status"].eq("unavailable"),
        "rdii_component",
    ].tolist()
    if unavailable:
        print("\n完全未找到：{}".format(", ".join(unavailable)))

    unit_review = availability.loc[
        availability["unit_review_note"].fillna("").ne(""),
        ["rdii_component", "reported_unit", "unit_review_note"],
    ]
    if not unit_review.empty:
        print("\n必须人工核对的单位：")
        print(unit_review.to_string(index=False))

    long_candidates = candidates.loc[
        candidates["long_format_nutrient_candidate"].eq(True)
    ] if not candidates.empty else pd.DataFrame()
    if not long_candidates.empty:
        print("\nlong-format营养表候选：")
        print(
            long_candidates[
                [
                    "path",
                    "food_key_columns",
                    "participant_key_columns",
                    "nutrient_descriptor_columns",
                    "value_columns",
                ]
            ].head(30).to_string(index=False)
        )

    component_candidates = candidates.loc[
        candidates["matched_rdii_component_count"].gt(0)
    ] if not candidates.empty else pd.DataFrame()
    if not component_candidates.empty:
        print("\n命中rDII营养素的候选表（前50）：")
        print(
            component_candidates[
                [
                    "path",
                    "food_key_columns",
                    "participant_key_columns",
                    "matched_rdii_components",
                    "matched_component_columns",
                ]
            ].head(50).to_string(index=False)
        )

    print("\n输出目录：{}".format(output_dir))
    print("核心可用性表：{}".format(availability_path))
    print("候选数据源：{}".format(candidate_path))
    print("QC：{}".format(qc_path))
    print("=" * 88)
    return availability_path, candidate_path, inventory_path, qc_path


def parse_args():
    parser = ArgumentParser(
        description="只读审计HPP rDII 21项营养素和Data/Transfer候选lookup"
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_CSV)
    parser.add_argument(
        "--raw-events", type=Path, default=DEFAULT_RAW_EVENTS_CSV
    )
    parser.add_argument(
        "--transfer-dir", type=Path, default=DEFAULT_TRANSFER_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--file-limit",
        type=int,
        default=0,
        help="最多扫描表数；0表示不设上限（默认：0）",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="每扫描多少个表打印进度；0关闭（默认：1000）",
    )
    args = parser.parse_args()
    if args.file_limit < 0:
        parser.error("--file-limit不能为负数")
    if args.progress_every < 0:
        parser.error("--progress-every不能为负数")
    return args


def main() -> None:
    args = parse_args()
    audit_rdii_nutrients(
        events_csv=args.events,
        raw_events_csv=args.raw_events,
        transfer_dir=args.transfer_dir,
        output_dir=args.output_dir,
        file_limit=args.file_limit,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
