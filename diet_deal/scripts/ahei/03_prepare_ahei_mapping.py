"""Build the reviewed food_id mapping for the approved HPP mAHEI-7 score."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-18-ahei-mapping-v1"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_DICTIONARY_CSV = (
    PROJECT_DIR / "outputs" / "02_food_dictionary" / "food_id_dictionary.csv"
)
DEFAULT_HPDI_MAPPING_CSV = (
    PROJECT_DIR
    / "outputs"
    / "03_score_mapping"
    / "hpdi"
    / "hpdi_food_id_mapping.csv"
)
LOCAL_HPDI_MAPPING_FALLBACK = (
    PROJECT_DIR
    / "outputs"
    / "03_score_mapping"
    / "hpdi"
    / "hpdi_food_id_mapping"
    / "hpdi_food_id_mapping.csv"
)
DEFAULT_AMED_MAPPING_CSV = (
    PROJECT_DIR
    / "outputs"
    / "03_score_mapping"
    / "amed"
    / "amed_food_id_mapping.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "03_score_mapping" / "ahei"

DICTIONARY_COLUMNS = [
    "food_id",
    "canonical_short_food_name",
    "canonical_product_name",
    "canonical_food_category",
    "food_id_event_count",
    "food_id_weight_g_total",
]
HPDI_COLUMNS = ["food_id", "hpdi_component", "mapping_status"]
AMED_COLUMNS = ["food_id", "amed_component", "mapping_status"]

HPDI_TO_AHEI: Dict[str, str] = {
    "fruits": "fruit",
    "vegetables": "vegetables",
    "whole_grains": "whole_grains",
    "nuts": "nuts_plus_legumes",
    "legumes": "nuts_plus_legumes",
    "fruit_juice": "ssb_plus_fruit_juice",
    "sugar_sweetened_beverages": "ssb_plus_fruit_juice",
}
AMED_RED_MEAT_COMPONENT = "red_processed_meat"
AHEI_INCLUDED_COMPONENTS = [
    "vegetables",
    "fruit",
    "whole_grains",
    "ssb_plus_fruit_juice",
    "nuts_plus_legumes",
    "red_plus_processed_meat",
    "alcohol",
]
FOOD_MAPPING_COMPONENTS = AHEI_INCLUDED_COMPONENTS[:-1]


def normalize_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def validate_columns(columns: List[str], required: List[str], source_name: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(
            "{}缺少字段：{}".format(source_name, ", ".join(missing))
        )


def component_definitions() -> pd.DataFrame:
    """Return the seven included and three transparently excluded components."""
    rows = [
        (
            "vegetables", True, "servings/day", "0", ">=5",
            "30 kcal/serving proxy", "available_by_food_mapping",
        ),
        (
            "fruit", True, "servings/day", "0", ">=4",
            "70 kcal/serving proxy; excludes fruit juice",
            "available_by_food_mapping",
        ),
        (
            "whole_grains", True, "g/day", "0", "female 75; male 90",
            "mapped event weight_g proxy", "available_by_food_mapping",
        ),
        (
            "ssb_plus_fruit_juice", True, "servings/day", ">=1", "0",
            "70 kcal/serving proxy", "available_by_food_mapping",
        ),
        (
            "nuts_plus_legumes", True, "servings/day", "0", ">=1",
            "28.35 g/serving", "available_by_food_mapping",
        ),
        (
            "red_plus_processed_meat", True, "servings/day", ">=1.5", "0",
            "220 kcal/serving proxy", "available_by_food_mapping",
        ),
        (
            "alcohol", True, "servings/day", "sex-specific heavy", "sex-specific moderate",
            "AMED ethanol g/day / 14 g per standard drink",
            "available_from_amed_shared_input",
        ),
        (
            "trans_fat", False, "% energy", ">=4", "<=0.5",
            "not calculated", "excluded_by_user_protocol",
        ),
        (
            "pufa", False, "% energy", "<=2", ">=10",
            "not calculated", "excluded_by_user_protocol",
        ),
        (
            "sodium", False, "mg/day", "highest decile", "lowest decile",
            "coverage-only QC; no imputation or scoring",
            "excluded_by_user_protocol",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "ahei_component",
            "included_in_final_score",
            "unit",
            "criterion_score_0",
            "criterion_score_10",
            "hpp_input_protocol",
            "availability_status",
        ],
    )


def _prepare_source(
    data: pd.DataFrame, required: List[str], source_name: str, prefix: str
) -> pd.DataFrame:
    validate_columns(data.columns.tolist(), required, source_name)
    source = data.loc[:, required].copy()
    source["food_id"] = normalize_text(source["food_id"])
    if source["food_id"].isna().any():
        raise ValueError("{}存在缺失food_id。".format(source_name))
    if source["food_id"].duplicated().any():
        raise ValueError("{}存在重复food_id。".format(source_name))
    for column in required:
        if column != "food_id":
            source[column] = normalize_text(source[column])
    return source.rename(
        columns={
            column: "{}_{}".format(prefix, column)
            for column in required
            if column != "food_id"
        }
    )


def build_ahei_mapping(
    dictionary: pd.DataFrame,
    hpdi_mapping: pd.DataFrame,
    amed_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Translate reviewed hPDI/AMED mappings into one mAHEI food component."""
    validate_columns(
        dictionary.columns.tolist(), DICTIONARY_COLUMNS, "食物字典"
    )
    base = dictionary.loc[:, DICTIONARY_COLUMNS].copy()
    base["food_id"] = normalize_text(base["food_id"])
    if base["food_id"].isna().any() or base["food_id"].duplicated().any():
        raise ValueError("食物字典food_id必须非缺失且唯一。")
    for column in [
        "canonical_short_food_name",
        "canonical_product_name",
        "canonical_food_category",
    ]:
        base[column] = normalize_text(base[column])

    hpdi = _prepare_source(hpdi_mapping, HPDI_COLUMNS, "hPDI映射", "hpdi")
    amed = _prepare_source(amed_mapping, AMED_COLUMNS, "AMED映射", "amed")
    mapping = base.merge(hpdi, on="food_id", how="left", validate="one_to_one")
    mapping = mapping.merge(amed, on="food_id", how="left", validate="one_to_one")

    mapping["ahei_component"] = mapping["hpdi_hpdi_component"].map(
        HPDI_TO_AHEI
    )
    mapping["mapping_status"] = "not_applicable"
    mapping["mapping_basis"] = "outside_mahei7_food_components"
    hpdi_mapped = mapping["ahei_component"].notna()
    mapping.loc[hpdi_mapped, "mapping_status"] = "mapped"
    mapping.loc[hpdi_mapped, "mapping_basis"] = (
        "reviewed_hpdi_component:" + mapping.loc[hpdi_mapped, "hpdi_hpdi_component"].astype(str)
    )

    red_meat = (
        mapping["amed_mapping_status"].eq("mapped")
        & mapping["amed_amed_component"].eq(AMED_RED_MEAT_COMPONENT)
    )
    conflicts = red_meat & mapping["ahei_component"].notna()
    if conflicts.any():
        food_ids = mapping.loc[conflicts, "food_id"].astype(str).tolist()
        raise ValueError(
            "AHEI映射冲突：红/加工肉同时进入其他组件：{}".format(
                ", ".join(food_ids[:10])
            )
        )
    mapping.loc[red_meat, "ahei_component"] = "red_plus_processed_meat"
    mapping.loc[red_meat, "mapping_status"] = "mapped"
    mapping.loc[red_meat, "mapping_basis"] = "reviewed_amed_red_processed_meat"

    labels_missing = mapping[
        [
            "canonical_short_food_name",
            "canonical_product_name",
            "canonical_food_category",
        ]
    ].isna().all(axis=1)
    source_missing_labels = (
        mapping["hpdi_mapping_status"].eq("unmapped_missing_labels")
        | mapping["amed_mapping_status"].eq("unmapped_missing_labels")
    )
    missing_labels = labels_missing | source_missing_labels
    mapping.loc[missing_labels, "ahei_component"] = pd.NA
    mapping.loc[missing_labels, "mapping_status"] = "unmapped_missing_labels"
    mapping.loc[missing_labels, "mapping_basis"] = "all_three_food_labels_missing"

    missing_source_rows = (
        mapping["hpdi_mapping_status"].isna()
        | mapping["amed_mapping_status"].isna()
    ) & ~missing_labels
    mapping.loc[missing_source_rows, "ahei_component"] = pd.NA
    mapping.loc[missing_source_rows, "mapping_status"] = "review_required"
    mapping.loc[missing_source_rows, "mapping_basis"] = "missing_reviewed_source_mapping"

    unresolved_source = (
        mapping["hpdi_mapping_status"].eq("review_required")
        | mapping["amed_mapping_status"].eq("review_required")
    ) & mapping["mapping_status"].ne("mapped") & ~missing_labels
    mapping.loc[unresolved_source, "mapping_status"] = "review_required"
    mapping.loc[unresolved_source, "mapping_basis"] = "source_mapping_requires_review"

    invalid_component = (
        mapping["mapping_status"].eq("mapped")
        & ~mapping["ahei_component"].isin(FOOD_MAPPING_COMPONENTS)
    )
    if invalid_component.any():
        raise AssertionError("生成了未知mAHEI-7食物组件。")
    mapping["proxy_metric"] = pd.NA
    mapping.loc[
        mapping["ahei_component"].isin(
            [
                "vegetables",
                "fruit",
                "ssb_plus_fruit_juice",
                "red_plus_processed_meat",
            ]
        ),
        "proxy_metric",
    ] = "resolved_energy_kcal"
    mapping.loc[
        mapping["ahei_component"].isin(
            ["whole_grains", "nuts_plus_legumes"]
        ),
        "proxy_metric",
    ] = "weight_g"

    output_columns = [
        *DICTIONARY_COLUMNS,
        "hpdi_hpdi_component",
        "hpdi_mapping_status",
        "amed_amed_component",
        "amed_mapping_status",
        "ahei_component",
        "mapping_status",
        "mapping_basis",
        "proxy_metric",
    ]
    return mapping.loc[:, output_columns].sort_values(
        ["mapping_status", "ahei_component", "food_id_event_count"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def build_mapping_qc(mapping: pd.DataFrame) -> pd.DataFrame:
    status = (
        mapping.groupby("mapping_status", dropna=False)
        .agg(
            food_id_count=("food_id", "nunique"),
            event_count=("food_id_event_count", "sum"),
        )
        .reset_index()
    )
    status.insert(0, "summary_type", "mapping_status")
    status["ahei_component"] = ""
    components = (
        mapping.loc[mapping["mapping_status"].eq("mapped")]
        .groupby("ahei_component", dropna=False)
        .agg(
            food_id_count=("food_id", "nunique"),
            event_count=("food_id_event_count", "sum"),
        )
        .reset_index()
    )
    components.insert(0, "summary_type", "mapped_component")
    components["mapping_status"] = "mapped"
    return pd.concat([status, components], ignore_index=True).loc[
        :,
        [
            "summary_type",
            "mapping_status",
            "ahei_component",
            "food_id_count",
            "event_count",
        ],
    ]


def _resolve_hpdi_path(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if path == DEFAULT_HPDI_MAPPING_CSV and LOCAL_HPDI_MAPPING_FALLBACK.is_file():
        return LOCAL_HPDI_MAPPING_FALLBACK
    return path


def prepare_ahei_mapping(
    dictionary_csv: Path = DEFAULT_DICTIONARY_CSV,
    hpdi_mapping_csv: Path = DEFAULT_HPDI_MAPPING_CSV,
    amed_mapping_csv: Path = DEFAULT_AMED_MAPPING_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Tuple[Path, Path, Path]:
    """Read frozen source mappings and write mAHEI definitions/mapping/QC."""
    hpdi_mapping_csv = _resolve_hpdi_path(hpdi_mapping_csv)
    for path in [dictionary_csv, hpdi_mapping_csv, amed_mapping_csv]:
        if not Path(path).is_file():
            raise FileNotFoundError("找不到映射输入：{}".format(path))
    dictionary = pd.read_csv(
        dictionary_csv,
        encoding="utf-8-sig",
        dtype={"food_id": "string"},
        low_memory=False,
    )
    hpdi = pd.read_csv(
        hpdi_mapping_csv,
        encoding="utf-8-sig",
        dtype={"food_id": "string"},
        low_memory=False,
    )
    amed = pd.read_csv(
        amed_mapping_csv,
        encoding="utf-8-sig",
        dtype={"food_id": "string"},
        low_memory=False,
    )
    mapping = build_ahei_mapping(dictionary, hpdi, amed)
    unresolved = mapping["mapping_status"].eq("review_required")
    if unresolved.any():
        raise ValueError(
            "AHEI映射仍有 {:,} 个food_id需要审核。".format(int(unresolved.sum()))
        )
    definitions = component_definitions()
    qc = build_mapping_qc(mapping)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions_path = output_dir / "ahei_component_definitions.csv"
    mapping_path = output_dir / "ahei_food_id_mapping.csv"
    qc_path = output_dir / "ahei_mapping_qc.csv"
    definitions.to_csv(definitions_path, index=False, encoding="utf-8-sig")
    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    return definitions_path, mapping_path, qc_path


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary-csv", type=Path, default=DEFAULT_DICTIONARY_CSV)
    parser.add_argument("--hpdi-mapping-csv", type=Path, default=DEFAULT_HPDI_MAPPING_CSV)
    parser.add_argument("--amed-mapping-csv", type=Path, default=DEFAULT_AMED_MAPPING_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for output_path in prepare_ahei_mapping(
        dictionary_csv=arguments.dictionary_csv,
        hpdi_mapping_csv=arguments.hpdi_mapping_csv,
        amed_mapping_csv=arguments.amed_mapping_csv,
        output_dir=arguments.output_dir,
    ):
        print(output_path)
