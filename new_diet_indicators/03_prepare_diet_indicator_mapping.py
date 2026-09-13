"""Build food-ID mappings for modified EAT-Lancet-13 and NOVA4."""

from argparse import ArgumentParser
from pathlib import Path
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-09-12-new-diet-indicator-mapping-v4"
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
DEFAULT_SUPPLEMENTARY_CSV = PROJECT_DIR / "config" / "diet_adherence_foods.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "03_score_mapping" / "new_diet_indicators"
)
DEFAULT_OVERRIDES_CSV = DEFAULT_OUTPUT_DIR / "new_diet_indicator_overrides.csv"

DICTIONARY_COLUMNS = [
    "food_id",
    "canonical_short_food_name",
    "canonical_product_name",
    "canonical_food_category",
    "food_id_event_count",
    "food_id_weight_g_total",
]
HPDI_COLUMNS = ["food_id", "hpdi_component", "mapping_status"]
NAME_COLUMNS = [
    "canonical_short_food_name",
    "canonical_product_name",
    "canonical_food_category",
]
EAT_COMPONENTS = [
    "vegetables",
    "fruits",
    "unsaturated_oils",
    "legumes",
    "nuts",
    "whole_grains",
    "fish",
    "beef_and_lamb",
    "pork",
    "poultry",
    "eggs",
    "dairy",
    "tubers",
]
HPDI_TO_EAT: Dict[str, str] = {
    "vegetables": "vegetables",
    "fruits": "fruits",
    "vegetable_oils": "unsaturated_oils",
    "legumes": "legumes",
    "nuts": "nuts",
    "whole_grains": "whole_grains",
    "fish_seafood": "fish",
    "poultry": "poultry",
    "eggs": "eggs",
    "potatoes": "tubers",
}
DIRECT_CATEGORY_TO_EAT: Dict[str, str] = {
    "vegetables": "vegetables",
    "fruits": "fruits",
    "med oil and fats": "unsaturated_oils",
    "pulses and products": "legumes",
    "nuts seeds and products": "nuts",
    "pasta grains and side dishes wholewheat": "whole_grains",
    "bread wholewheat": "whole_grains",
    "cereals wholewheat": "whole_grains",
    "fish and seafood": "fish",
    "milk cream cheese and yogurts": "dairy",
    "beef veal lamb and other meat products": "beef_and_lamb",
    "non kosher foods": "pork",
    "poultry and its products": "poultry",
    "eggs and their products": "eggs",
}
VALID_HPDI_STATUSES = {
    "mapped",
    "not_applicable",
    "unmapped_missing_labels",
}
VALID_EAT_STATUSES = {
    "mapped",
    "not_applicable",
    "unmapped_missing_labels",
    "review_required",
}
VALID_NOVA_STATUSES = {
    "mapped",
    "unmapped_missing_labels",
    "review_required",
}
OVERRIDE_COLUMNS = [
    "food_id",
    "override_eat_lancet_component",
    "override_nova_group",
    "review_status",
    "review_basis",
]

# Evidence: HPP server dictionary screenshots supplied on 2026-09-12.
# Match both ID and product name: IDs alone are not portable across datasets.
# These rules correct EAT food identity only; they do not assign NOVA groups.
SERVER_IDENTITY_CHECKS = [
    (
        "1012933",
        "Dietary fiber BENEFIBER food supplement",
        "not_applicable",
        "not_applicable",
        "fiber_supplement_is_not_whole_grain",
    ),
    (
        "1014956",
        "Boiled potato with skin",
        "tubers",
        "mapped",
        "boiled_potato_is_tuber_not_refined_grain",
    ),
]

# User-approved study adaptation (2026-09-12): exclude exactly these eight
# products from EAT component grams because ingredient-equivalent weights are
# unavailable. Preserve their events, energy, carbohydrate and NOVA mapping.
# This is a protocol exclusion, not evidence of zero legume/nut intake.
APPROVED_PLANT_ALTERNATIVE_EXCLUSIONS = {
    "1009259": "Soy drink with a natural soy milk flavor",
    "1010775": "Sugar-free almond milk",
    "1009011": "Natural organic soy drink soy milk",
    "1011847": "Coconut milk drink",
    "1012723": "Soy cheese 5% fat in different flavors",
    "1009092": "Alpro almond coconut milk drink",
    "1013950": "Creamy vegan soy cheese, garlic and dill",
    "1007077": "Soy yogurt ice cream",
}
SERVER_IDENTITY_CHECKS.extend([
    (food_id, product_name, "not_applicable", "not_applicable",
     "user_approved_plant_alternative_exclusion_no_ingredient_equivalents")
    for food_id, product_name in APPROVED_PLANT_ALTERNATIVE_EXCLUSIONS.items()
])


def normalize_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def normalize_food_name(value: object) -> str:
    """Match the supplied short-name table without fuzzy inference."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def validate_columns(columns: List[str], required: List[str], source_name: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(
            "{}缺少字段：{}".format(source_name, ", ".join(missing))
        )


def resolve_hpdi_path(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if path == DEFAULT_HPDI_MAPPING_CSV and LOCAL_HPDI_MAPPING_FALLBACK.is_file():
        return LOCAL_HPDI_MAPPING_FALLBACK
    return path


def load_supplementary_reference(path: Path) -> pd.DataFrame:
    """Convert the wide Supplementary Table 2 layout to one food per row."""
    if not Path(path).is_file():
        raise FileNotFoundError("找不到饮食分类补充表：{}".format(path))
    raw = pd.read_csv(
        path,
        header=None,
        encoding="utf-8-sig",
        dtype="string",
        keep_default_na=False,
    )
    if raw.shape[0] < 2 or raw.shape[1] < 2:
        raise ValueError("饮食分类补充表的宽表结构无效。")
    labels = normalize_text(raw.iloc[:, 0])
    duplicate_labels = labels.dropna().duplicated(keep=False)
    if duplicate_labels.any():
        values = sorted(labels.dropna().loc[duplicate_labels].astype(str).unique())
        raise ValueError("补充表存在重复行标签：{}".format(", ".join(values)))
    label_to_row = {
        str(label).strip(): index
        for index, label in labels.items()
        if not pd.isna(label)
    }
    for required in ["NOVA", "Tubers", "Pork"]:
        if required not in label_to_row:
            raise ValueError("补充表缺少{}行。".format(required))

    food_names = normalize_text(raw.iloc[0, 1:]).reset_index(drop=True)
    reference = pd.DataFrame({"supplementary_food_name": food_names})
    reference["normalized_food_name"] = reference[
        "supplementary_food_name"
    ].map(normalize_food_name)
    if reference["normalized_food_name"].eq("").any():
        raise ValueError("补充表存在空食物名称。")
    duplicated = reference["normalized_food_name"].duplicated(keep=False)
    if duplicated.any():
        values = sorted(
            reference.loc[duplicated, "supplementary_food_name"].astype(str).unique()
        )
        raise ValueError(
            "补充表食物名标准化后重复：{}".format(", ".join(values[:10]))
        )

    nova_values = raw.iloc[label_to_row["NOVA"], 1:].reset_index(drop=True)
    nova_numeric = pd.to_numeric(normalize_text(nova_values), errors="coerce")
    invalid_nova = nova_numeric.notna() & ~nova_numeric.isin([1, 2, 3, 4])
    if invalid_nova.any():
        raise ValueError("补充表NOVA行只能包含1、2、3、4或空值。")
    reference["nova_group"] = nova_numeric.astype("Int64")

    for label, output_column in [("Tubers", "is_tuber"), ("Pork", "is_pork")]:
        values = pd.to_numeric(
            normalize_text(raw.iloc[label_to_row[label], 1:].reset_index(drop=True)),
            errors="coerce",
        )
        invalid = values.isna() | ~values.isin([0, 1])
        if invalid.any():
            raise ValueError("补充表{}行只能包含0或1。".format(label))
        reference[output_column] = values.astype("Int64")
    if (reference["is_tuber"].eq(1) & reference["is_pork"].eq(1)).any():
        raise ValueError("同一补充表食物不能同时标为Tubers和Pork。")
    return reference


def prepare_dictionary(dictionary: pd.DataFrame) -> pd.DataFrame:
    validate_columns(dictionary.columns.tolist(), DICTIONARY_COLUMNS, "食物字典")
    result = dictionary.loc[:, DICTIONARY_COLUMNS].copy()
    result["food_id"] = normalize_text(result["food_id"])
    if result["food_id"].isna().any() or result["food_id"].duplicated().any():
        raise ValueError("食物字典food_id必须非缺失且唯一。")
    for column in NAME_COLUMNS:
        result[column] = normalize_text(result[column])
    result["food_id_event_count"] = pd.to_numeric(
        result["food_id_event_count"], errors="coerce"
    )
    result["food_id_weight_g_total"] = pd.to_numeric(
        result["food_id_weight_g_total"], errors="coerce"
    )
    if result["food_id_event_count"].isna().any():
        raise ValueError("食物字典food_id_event_count必须为数值。")
    result["normalized_short_food_name"] = result[
        "canonical_short_food_name"
    ].map(normalize_food_name)
    return result


def prepare_hpdi_mapping(hpdi: pd.DataFrame) -> pd.DataFrame:
    validate_columns(hpdi.columns.tolist(), HPDI_COLUMNS, "hPDI映射")
    result = hpdi.loc[:, HPDI_COLUMNS].copy()
    for column in HPDI_COLUMNS:
        result[column] = normalize_text(result[column])
    if result["food_id"].isna().any() or result["food_id"].duplicated().any():
        raise ValueError("hPDI映射food_id必须非缺失且唯一。")
    invalid = ~result["mapping_status"].isin(VALID_HPDI_STATUSES)
    if invalid.any():
        values = sorted(result.loc[invalid, "mapping_status"].astype(str).unique())
        raise ValueError("hPDI映射尚未冻结：{}".format(", ".join(values[:10])))
    return result.rename(
        columns={
            "hpdi_component": "source_hpdi_component",
            "mapping_status": "source_hpdi_mapping_status",
        }
    )


def load_overrides(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)
    data = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype="string",
        keep_default_na=False,
    )
    validate_columns(data.columns.tolist(), OVERRIDE_COLUMNS, "新指标审核override")
    data = data.loc[:, OVERRIDE_COLUMNS].copy()
    for column in OVERRIDE_COLUMNS:
        data[column] = normalize_text(data[column])
    data = data.loc[data["review_status"].str.casefold().eq("approved").fillna(False)]
    if data.empty:
        return data
    if data["food_id"].isna().any() or data["food_id"].duplicated().any():
        raise ValueError("已批准override的food_id必须非缺失且唯一。")
    eat = data["override_eat_lancet_component"]
    invalid_eat = eat.notna() & ~eat.isin(EAT_COMPONENTS + ["not_applicable"])
    if invalid_eat.any():
        values = sorted(eat.loc[invalid_eat].astype(str).unique())
        raise ValueError("override包含无效EAT-Lancet组件：{}".format(", ".join(values)))
    nova = pd.to_numeric(data["override_nova_group"], errors="coerce")
    invalid_nova = data["override_nova_group"].notna() & ~nova.isin([1, 2, 3, 4])
    if invalid_nova.any():
        raise ValueError("override_nova_group只能为1、2、3、4或空值。")
    data["override_nova_group"] = nova.astype("Int64")
    no_decision = eat.isna() & data["override_nova_group"].isna()
    if no_decision.any():
        raise ValueError("已批准override至少需要填写一个EAT或NOVA决定。")
    return data


def apply_food_identity_checks(mapping: pd.DataFrame) -> pd.DataFrame:
    """Correct evidenced identities and flag conflicts before approved overrides."""
    mapping = mapping.copy()
    mapping["food_identity_check"] = pd.NA
    product_names = mapping["canonical_product_name"].map(normalize_food_name)
    for food_id, product_name, component, status, reason in SERVER_IDENTITY_CHECKS:
        same_id = mapping["food_id"].eq(food_id)
        verified = same_id & product_names.eq(normalize_food_name(product_name))
        mapping.loc[verified, "eat_lancet_component"] = (
            component if status == "mapped" else pd.NA
        )
        mapping.loc[verified, "eat_lancet_mapping_status"] = status
        mapping.loc[verified, "eat_lancet_mapping_basis"] = (
            "server_identity_check_2026_09_12:" + reason
        )
        mapping.loc[verified, "food_identity_check"] = reason

        changed_identity = same_id & ~verified
        mapping.loc[changed_identity, "eat_lancet_component"] = pd.NA
        mapping.loc[changed_identity, "eat_lancet_mapping_status"] = "review_required"
        mapping.loc[changed_identity, "eat_lancet_mapping_basis"] = (
            "server_identity_check_product_name_mismatch"
        )
        mapping.loc[changed_identity, "food_identity_check"] = (
            "known_food_id_product_name_changed"
        )

    # Flag similarly conflicting labels, without guessing a new food component.
    names = (
        mapping["canonical_short_food_name"].fillna("") + " "
        + mapping["canonical_product_name"].fillna("")
    ).map(normalize_food_name)
    supplement = names.str.contains(r"\b(?:supplement|supplements)\b", regex=True)
    dairy_substitute = names.str.contains(
        r"\b(?:soy|soya|almond|oat|rice|coconut) (?:milk|cheese|yogurt|yoghurt)\b",
        regex=True,
    )
    conflict = mapping["food_identity_check"].isna() & (
        (supplement & mapping["eat_lancet_mapping_status"].eq("mapped"))
        | (dairy_substitute & mapping["eat_lancet_component"].eq("dairy"))
    )
    mapping.loc[conflict, "eat_lancet_component"] = pd.NA
    mapping.loc[conflict, "eat_lancet_mapping_status"] = "review_required"
    mapping.loc[conflict, "eat_lancet_mapping_basis"] = "product_category_conflict"
    mapping.loc[conflict, "food_identity_check"] = "supplement_or_plant_dairy_conflict"
    return mapping


def supplement_nova_mapping(mapping: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Use exact reference products, then tightly scoped plain-food descriptions.

    Rule source: FAO 2019, Ultra-processed foods, diet quality, and health
    using the NOVA classification system, group 1. Name-based classifications
    are inferred, not verified ingredient lists. No broad category inference.
    """
    mapping = mapping.copy()
    products = mapping["canonical_product_name"].map(normalize_food_name)
    lookup = reference.set_index("normalized_food_name")
    product_nova = products.map(lookup["nova_group"])
    exact = mapping["nova_mapping_status"].ne("mapped") & product_nova.isin([1, 2, 3, 4])
    mapping.loc[exact, "nova_group"] = product_nova.loc[exact].astype("Int64")
    mapping.loc[exact, "nova_mapping_status"] = "mapped"
    mapping.loc[exact, "nova_mapping_basis"] = "supplementary2_exact_normalized_product_name"

    # Exact descriptions only. Mixed dishes, branded pastries and rendered-vs-raw
    # ambiguous fats remain unclassified; a category such as Snacks is not NOVA.
    plain_foods = {
        "boiled potato with skin": 1,
        "cooked turkey thigh meat": 1,
        "lamb chops": 1,
        "beef ribs with fat": 1,
        "squeezed celery juice": 1,
        "golden berry": 1,
    }
    inferred = products.map(plain_foods)
    use_plain = mapping["nova_mapping_status"].ne("mapped") & inferred.notna()
    mapping.loc[use_plain, "nova_group"] = inferred.loc[use_plain].astype("Int64")
    mapping.loc[use_plain, "nova_mapping_status"] = "mapped"
    mapping.loc[use_plain, "nova_mapping_basis"] = "FAO2019_group1_exact_plain_product_description_inferred"
    return mapping


def build_mapping(
    dictionary: pd.DataFrame,
    hpdi_mapping: pd.DataFrame,
    supplementary_reference: pd.DataFrame,
    overrides: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build separate EAT-Lancet and NOVA classifications for every food ID."""
    base = prepare_dictionary(dictionary)
    hpdi = prepare_hpdi_mapping(hpdi_mapping)
    reference = supplementary_reference.copy()
    required_reference = [
        "supplementary_food_name",
        "normalized_food_name",
        "nova_group",
        "is_tuber",
        "is_pork",
    ]
    validate_columns(reference.columns.tolist(), required_reference, "补充表长表")
    mapping = base.merge(hpdi, on="food_id", how="left", validate="one_to_one")
    mapping = mapping.merge(
        reference.loc[:, required_reference],
        left_on="normalized_short_food_name",
        right_on="normalized_food_name",
        how="left",
        validate="many_to_one",
    )
    mapping["supplementary_name_matched"] = mapping[
        "supplementary_food_name"
    ].notna()

    mapping["eat_lancet_component"] = mapping["source_hpdi_component"].map(
        HPDI_TO_EAT
    )
    mapping["eat_lancet_mapping_basis"] = pd.NA
    from_hpdi = mapping["eat_lancet_component"].notna()
    mapping.loc[from_hpdi, "eat_lancet_mapping_basis"] = (
        "source_hpdi_component:"
        + mapping.loc[from_hpdi, "source_hpdi_component"].astype(str)
    )

    normalized_category = mapping["canonical_food_category"].map(normalize_food_name)
    direct_component = normalized_category.map(DIRECT_CATEGORY_TO_EAT)
    direct = mapping["eat_lancet_component"].isna() & direct_component.notna()
    mapping.loc[direct, "eat_lancet_component"] = direct_component.loc[direct]
    mapping.loc[direct, "eat_lancet_mapping_basis"] = (
        "canonical_category:"
        + mapping.loc[direct, "canonical_food_category"].astype(str)
    )

    tuber = mapping["is_tuber"].eq(1)
    pork = mapping["is_pork"].eq(1)
    mapping.loc[tuber, "eat_lancet_component"] = "tubers"
    mapping.loc[tuber, "eat_lancet_mapping_basis"] = "supplementary2_tubers"
    mapping.loc[pork, "eat_lancet_component"] = "pork"
    mapping.loc[pork, "eat_lancet_mapping_basis"] = "supplementary2_pork"

    labels_missing = mapping[NAME_COLUMNS].isna().all(axis=1)
    mapping["eat_lancet_mapping_status"] = "not_applicable"
    mapping.loc[
        mapping["eat_lancet_component"].notna(), "eat_lancet_mapping_status"
    ] = "mapped"
    mapping.loc[labels_missing, "eat_lancet_component"] = pd.NA
    mapping.loc[labels_missing, "eat_lancet_mapping_status"] = (
        "unmapped_missing_labels"
    )
    mapping.loc[labels_missing, "eat_lancet_mapping_basis"] = (
        "all_three_food_labels_missing"
    )
    mapping["nova_mapping_status"] = "review_required"
    mapping["nova_mapping_basis"] = "short_food_name_not_in_supplementary2"
    nova_mapped = mapping["nova_group"].isin([1, 2, 3, 4])
    mapping.loc[nova_mapped, "nova_mapping_status"] = "mapped"
    mapping.loc[nova_mapped, "nova_mapping_basis"] = (
        "supplementary2_normalized_short_name"
    )
    matched_missing_nova = (
        mapping["supplementary_name_matched"] & ~nova_mapped
    )
    mapping.loc[matched_missing_nova, "nova_mapping_basis"] = (
        "supplementary2_nova_value_missing"
    )
    mapping.loc[labels_missing, "nova_mapping_status"] = (
        "unmapped_missing_labels"
    )
    mapping.loc[labels_missing, "nova_mapping_basis"] = (
        "all_three_food_labels_missing"
    )

    mapping = apply_food_identity_checks(mapping)
    mapping = supplement_nova_mapping(mapping, reference)

    approved = overrides if overrides is not None else pd.DataFrame(columns=OVERRIDE_COLUMNS)
    if not approved.empty:
        unknown_ids = sorted(set(approved["food_id"]) - set(mapping["food_id"]))
        if unknown_ids:
            raise ValueError(
                "override含食物字典外food_id：{}".format(", ".join(unknown_ids[:10]))
            )
        approved_for_merge = approved.rename(
            columns={"review_basis": "override_review_basis"}
        )
        mapping = mapping.merge(
            approved_for_merge,
            on="food_id",
            how="left",
            validate="one_to_one",
        )
        eat_override = mapping["override_eat_lancet_component"].notna()
        eat_not_applicable = eat_override & mapping[
            "override_eat_lancet_component"
        ].eq("not_applicable")
        eat_mapped = eat_override & ~eat_not_applicable
        mapping.loc[eat_mapped, "eat_lancet_component"] = mapping.loc[
            eat_mapped, "override_eat_lancet_component"
        ]
        mapping.loc[eat_mapped, "eat_lancet_mapping_status"] = "mapped"
        mapping.loc[eat_not_applicable, "eat_lancet_component"] = pd.NA
        mapping.loc[eat_not_applicable, "eat_lancet_mapping_status"] = (
            "not_applicable"
        )
        mapping.loc[eat_override, "eat_lancet_mapping_basis"] = (
            "approved_override:"
            + mapping.loc[eat_override, "override_review_basis"].fillna(
                "no_basis_given"
            )
        )
        nova_override = mapping["override_nova_group"].notna()
        mapping.loc[nova_override, "nova_group"] = mapping.loc[
            nova_override, "override_nova_group"
        ].astype("Int64")
        mapping.loc[nova_override, "nova_mapping_status"] = "mapped"
        mapping.loc[nova_override, "nova_mapping_basis"] = (
            "approved_override:"
            + mapping.loc[nova_override, "override_review_basis"].fillna(
                "no_basis_given"
            )
        )
    else:
        for column in [
            "override_eat_lancet_component",
            "override_nova_group",
            "review_status",
            "override_review_basis",
        ]:
            mapping[column] = pd.NA

    invalid_eat_status = ~mapping["eat_lancet_mapping_status"].isin(
        VALID_EAT_STATUSES
    )
    invalid_nova_status = ~mapping["nova_mapping_status"].isin(
        VALID_NOVA_STATUSES
    )
    invalid_eat_component = (
        mapping["eat_lancet_mapping_status"].eq("mapped")
        & ~mapping["eat_lancet_component"].isin(EAT_COMPONENTS)
    )
    invalid_nova_group = (
        mapping["nova_mapping_status"].eq("mapped")
        & ~mapping["nova_group"].isin([1, 2, 3, 4])
    )
    if invalid_eat_status.any() or invalid_nova_status.any():
        raise AssertionError("生成了无效映射状态。")
    if invalid_eat_component.any() or invalid_nova_group.any():
        raise AssertionError("mapped记录缺少有效组件或NOVA组。")

    mapping["nova_analysis_status"] = "excluded_insufficient_evidence"
    mapping.loc[mapping["nova_mapping_status"].eq("unmapped_missing_labels"),
                "nova_analysis_status"] = "excluded_missing_labels"
    mapping.loc[mapping["nova_mapping_status"].eq("mapped"),
                "nova_analysis_status"] = "included"
    output_columns = [
        *DICTIONARY_COLUMNS,
        "source_hpdi_component",
        "source_hpdi_mapping_status",
        "supplementary_food_name",
        "supplementary_name_matched",
        "eat_lancet_component",
        "eat_lancet_mapping_status",
        "eat_lancet_mapping_basis",
        "nova_group",
        "nova_mapping_status",
        "nova_mapping_basis",
        "nova_analysis_status",
        "food_identity_check",
    ]
    return mapping.loc[:, output_columns].sort_values(
        ["nova_mapping_status", "eat_lancet_mapping_status", "food_id_event_count"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def component_definitions() -> pd.DataFrame:
    rows = [
        ("vegetables", "beneficial", 100.0, 200.0, 300.0, "g/day"),
        ("fruits", "beneficial", 50.0, 100.0, 200.0, "g/day"),
        ("unsaturated_oils", "beneficial", 10.0, 20.0, 40.0, "g/day"),
        ("legumes", "beneficial", 18.75, 37.5, 75.0, "g/day"),
        ("nuts", "beneficial", 12.5, 25.0, 50.0, "g/day"),
        ("whole_grains", "beneficial", 58.0, 116.0, 232.0, "g/day"),
        ("fish", "beneficial", 7.0, 14.0, 28.0, "g/day"),
        ("beef_and_lamb", "adverse", 7.0, 14.0, 28.0, "g/day"),
        ("pork", "adverse", 7.0, 14.0, 28.0, "g/day"),
        ("poultry", "adverse", 29.0, 58.0, 116.0, "g/day"),
        ("eggs", "adverse", 13.0, 25.0, 50.0, "g/day"),
        ("dairy", "adverse", 250.0, 500.0, 1000.0, "g/day"),
        ("tubers", "adverse", 50.0, 100.0, 200.0, "g/day"),
        ("added_sugar", "excluded", 31.0, 62.0, 124.0, "g/day"),
    ]
    definitions = pd.DataFrame(
        rows,
        columns=[
            "component",
            "direction",
            "cutoff_1",
            "cutoff_2",
            "cutoff_3",
            "unit",
        ],
    )
    definitions["included_in_modified_score"] = definitions[
        "component"
    ].ne("added_sugar")
    definitions["source"] = "Supplementary Table 1: The eat lancet index"
    definitions["protocol_note"] = ""
    definitions.loc[definitions["included_in_modified_score"], "protocol_note"] = (
        "user-approved 2026-09-12: eight specified plant dairy alternatives "
        "excluded from EAT component grams; no ingredient-equivalent weights; "
        "events, energy, carbohydrate and NOVA classification retained"
    )
    definitions.loc[
        definitions["component"].eq("added_sugar"), "protocol_note"
    ] = "excluded: HPP has no reliable added_sugar_g"
    return definitions


def build_review_template(mapping: pd.DataFrame) -> pd.DataFrame:
    review = mapping.loc[
        mapping["eat_lancet_mapping_status"].eq("review_required")
        | mapping["nova_mapping_status"].eq("review_required")
    ].copy()
    review["override_eat_lancet_component"] = pd.NA
    review["override_nova_group"] = pd.NA
    review["review_status"] = "pending"
    review["review_basis"] = pd.NA
    columns = [
        "food_id",
        *NAME_COLUMNS,
        "food_id_event_count",
        "food_id_weight_g_total",
        "source_hpdi_component",
        "source_hpdi_mapping_status",
        "eat_lancet_component",
        "eat_lancet_mapping_basis",
        "food_identity_check",
        "eat_lancet_mapping_status",
        "nova_mapping_status",
        *OVERRIDE_COLUMNS[1:],
    ]
    return review.loc[:, columns].sort_values(
        "food_id_event_count", ascending=False
    ).reset_index(drop=True)


def build_mapping_qc(mapping: pd.DataFrame) -> pd.DataFrame:
    total_events = float(mapping["food_id_event_count"].sum())
    rows = []
    for indicator, status_column in [
        ("eat_lancet", "eat_lancet_mapping_status"),
        ("nova", "nova_mapping_status"),
    ]:
        grouped = mapping.groupby(status_column, dropna=False)
        for status, data in grouped:
            event_count = float(data["food_id_event_count"].sum())
            rows.append(
                (
                    indicator,
                    "mapping_status",
                    status,
                    "",
                    int(data["food_id"].nunique()),
                    event_count,
                    event_count / total_events if total_events > 0 else pd.NA,
                    float(data["food_id_weight_g_total"].sum(min_count=1)),
                )
            )
    eat_mapped = mapping.loc[mapping["eat_lancet_mapping_status"].eq("mapped")]
    for component, data in eat_mapped.groupby("eat_lancet_component"):
        event_count = float(data["food_id_event_count"].sum())
        rows.append(
            (
                "eat_lancet",
                "mapped_component",
                "mapped",
                component,
                int(data["food_id"].nunique()),
                event_count,
                event_count / total_events if total_events > 0 else pd.NA,
                float(data["food_id_weight_g_total"].sum(min_count=1)),
            )
        )
    for group, data in mapping.loc[
        mapping["nova_mapping_status"].eq("mapped")
    ].groupby("nova_group"):
        event_count = float(data["food_id_event_count"].sum())
        rows.append(
            (
                "nova",
                "mapped_group",
                "mapped",
                "NOVA{}".format(int(group)),
                int(data["food_id"].nunique()),
                event_count,
                event_count / total_events if total_events > 0 else pd.NA,
                float(data["food_id_weight_g_total"].sum(min_count=1)),
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "indicator",
            "summary_type",
            "mapping_status",
            "component_or_group",
            "food_id_count",
            "dictionary_event_count",
            "dictionary_event_share",
            "dictionary_weight_g_total",
        ],
    )


def prepare_mapping(
    dictionary_csv: Path = DEFAULT_DICTIONARY_CSV,
    hpdi_mapping_csv: Path = DEFAULT_HPDI_MAPPING_CSV,
    supplementary_csv: Path = DEFAULT_SUPPLEMENTARY_CSV,
    overrides_csv: Path = DEFAULT_OVERRIDES_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Tuple[Path, Path, Path, Path, Path]:
    hpdi_mapping_csv = resolve_hpdi_path(hpdi_mapping_csv)
    for path in [dictionary_csv, hpdi_mapping_csv, supplementary_csv]:
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
    reference = load_supplementary_reference(supplementary_csv)
    overrides = load_overrides(overrides_csv)
    mapping = build_mapping(dictionary, hpdi, reference, overrides)
    definitions = component_definitions()
    review = build_review_template(mapping)
    qc = build_mapping_qc(mapping)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions_path = output_dir / "eat_lancet13_component_definitions.csv"
    reference_path = output_dir / "supplementary_food_classification_long.csv"
    mapping_path = output_dir / "new_diet_indicator_food_id_mapping.csv"
    review_path = output_dir / "new_diet_indicator_review_template.csv"
    qc_path = output_dir / "new_diet_indicator_mapping_qc.csv"
    definitions.to_csv(definitions_path, index=False, encoding="utf-8-sig")
    reference.to_csv(reference_path, index=False, encoding="utf-8-sig")
    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    print("SCRIPT_VERSION={}".format(SCRIPT_VERSION))
    print("food_ids={:,}".format(len(mapping)))
    print(
        "eat_review_required={:,}".format(
            int(mapping["eat_lancet_mapping_status"].eq("review_required").sum())
        )
    )
    print(
        "nova_review_required={:,}".format(
            int(mapping["nova_mapping_status"].eq("review_required").sum())
        )
    )
    print("nova_unmapped_missing_labels={:,}".format(
        int(mapping["nova_mapping_status"].eq("unmapped_missing_labels").sum())
    ))
    print("nova_unclassified_total={:,}".format(
        int(mapping["nova_mapping_status"].ne("mapped").sum())
    ))
    print("NOVA_PROTOCOL=exclude_unclassified_from_NOVA_numerator_and_denominator_only")
    for basis, count in mapping["nova_mapping_basis"].value_counts().items():
        print("nova_basis={} | food_ids={:,}".format(basis, int(count)))
    checks = mapping.loc[mapping["food_identity_check"].notna()]
    print("food_identity_checks={:,}".format(len(checks)))
    print("NOTE: mapped表示规则已匹配，不代表人工审核或完整分类验证。")
    for row in checks.sort_values("food_id_event_count", ascending=False).head(15).itertuples():
        print("\nfood_id={} | product={}".format(row.food_id, row.canonical_product_name))
        print("  source_hpdi={} | EAT={} | status={}".format(
            row.source_hpdi_component, row.eat_lancet_component,
            row.eat_lancet_mapping_status,
        ))
        print("  check={}".format(row.food_identity_check))
    return definitions_path, reference_path, mapping_path, review_path, qc_path


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary-csv", type=Path, default=DEFAULT_DICTIONARY_CSV)
    parser.add_argument("--hpdi-mapping-csv", type=Path, default=DEFAULT_HPDI_MAPPING_CSV)
    parser.add_argument("--supplementary-csv", type=Path, default=DEFAULT_SUPPLEMENTARY_CSV)
    parser.add_argument("--overrides-csv", type=Path, default=DEFAULT_OVERRIDES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for output_path in prepare_mapping(
        dictionary_csv=arguments.dictionary_csv,
        hpdi_mapping_csv=arguments.hpdi_mapping_csv,
        supplementary_csv=arguments.supplementary_csv,
        overrides_csv=arguments.overrides_csv,
        output_dir=arguments.output_dir,
    ):
        print(output_path)
