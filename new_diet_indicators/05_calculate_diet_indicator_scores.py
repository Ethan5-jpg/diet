"""Score modified EAT-Lancet-13 and standardize the three new diet indicators."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-09-12-new-diet-indicator-scores-v2"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "new_diet_indicators"
    / "diet_indicator_participant_intakes.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "05_diet_scores" / "new_diet_indicators"
)

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
BENEFICIAL_CUTOFFS: Dict[str, Tuple[float, float, float]] = {
    "vegetables": (100.0, 200.0, 300.0),
    "fruits": (50.0, 100.0, 200.0),
    "unsaturated_oils": (10.0, 20.0, 40.0),
    "legumes": (18.75, 37.5, 75.0),
    "nuts": (12.5, 25.0, 50.0),
    "whole_grains": (58.0, 116.0, 232.0),
    "fish": (7.0, 14.0, 28.0),
}
ADVERSE_CUTOFFS: Dict[str, Tuple[float, float, float]] = {
    "beef_and_lamb": (7.0, 14.0, 28.0),
    "pork": (7.0, 14.0, 28.0),
    "poultry": (29.0, 58.0, 116.0),
    "eggs": (13.0, 25.0, 50.0),
    "dairy": (250.0, 500.0, 1000.0),
    "tubers": (50.0, 100.0, 200.0),
}
SCORE_COLUMNS = ["{}_score_0_3".format(component) for component in EAT_COMPONENTS]
REQUIRED_COLUMNS = [
    "participant_id",
    "cohort",
    "research_stage",
    "observed_diet_days",
    "eat_lancet13_intake_complete",
    "nova4_classified_energy_pct",
    "participant_nova_mapping_energy_coverage",
    "nova4_classified_valid_day_count",
    "carbohydrate_energy_pct",
    "carbohydrate_valid_day_count",
    "carbohydrate_pct_out_of_range_day_count",
    *["mean_daily_{}_g".format(component) for component in EAT_COMPONENTS],
    *["{}_complete".format(component) for component in EAT_COMPONENTS],
]


def normalize_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def validate_columns(columns: List[str], required: List[str], source_name: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(
            "{}缺少字段：{}".format(source_name, ", ".join(missing))
        )


def parse_boolean(series: pd.Series, column_name: str) -> pd.Series:
    normalized = normalize_text(series).str.casefold()
    parsed = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        }
    )
    invalid = normalized.notna() & parsed.isna()
    if invalid.any():
        values = sorted(normalized.loc[invalid].astype(str).unique())
        raise ValueError(
            "{}含无效布尔值：{}".format(column_name, ", ".join(values[:10]))
        )
    return parsed.astype("boolean")


def score_beneficial(values: pd.Series, cutoffs: Tuple[float, float, float]) -> pd.Series:
    """Score [0,c1), [c1,c2), [c2,c3), [c3,+inf) as 0,1,2,3."""
    numeric = pd.to_numeric(values, errors="coerce")
    c1, c2, c3 = cutoffs
    if not 0 < c1 < c2 < c3:
        raise ValueError("有益组件阈值必须严格递增且为正。")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = numeric.notna() & numeric.ge(0)
    result.loc[valid & numeric.lt(c1)] = 0
    result.loc[valid & numeric.ge(c1) & numeric.lt(c2)] = 1
    result.loc[valid & numeric.ge(c2) & numeric.lt(c3)] = 2
    result.loc[valid & numeric.ge(c3)] = 3
    return result


def score_adverse(values: pd.Series, cutoffs: Tuple[float, float, float]) -> pd.Series:
    """Score [0,c1), [c1,c2), [c2,c3), [c3,+inf) as 3,2,1,0."""
    numeric = pd.to_numeric(values, errors="coerce")
    c1, c2, c3 = cutoffs
    if not 0 < c1 < c2 < c3:
        raise ValueError("不利组件阈值必须严格递增且为正。")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = numeric.notna() & numeric.ge(0)
    result.loc[valid & numeric.lt(c1)] = 3
    result.loc[valid & numeric.ge(c1) & numeric.lt(c2)] = 2
    result.loc[valid & numeric.ge(c2) & numeric.lt(c3)] = 1
    result.loc[valid & numeric.ge(c3)] = 0
    return result


def add_eat_lancet_scores(intakes: pd.DataFrame) -> pd.DataFrame:
    data = intakes.copy()
    for component in EAT_COMPONENTS:
        intake_column = "mean_daily_{}_g".format(component)
        complete_column = "{}_complete".format(component)
        data[complete_column] = parse_boolean(data[complete_column], complete_column)
        if component in BENEFICIAL_CUTOFFS:
            score = score_beneficial(data[intake_column], BENEFICIAL_CUTOFFS[component])
        else:
            score = score_adverse(data[intake_column], ADVERSE_CUTOFFS[component])
        score.loc[~data[complete_column].eq(True)] = pd.NA
        data["{}_score_0_3".format(component)] = score
    data["modified_eat_lancet13_complete"] = data[SCORE_COLUMNS].notna().all(axis=1)
    raw = data[SCORE_COLUMNS].sum(axis=1, min_count=len(EAT_COMPONENTS))
    data["modified_eat_lancet13_raw_0_39"] = raw.astype("Float64")
    invalid = data["modified_eat_lancet13_raw_0_39"].notna() & ~data[
        "modified_eat_lancet13_raw_0_39"
    ].between(0, 39)
    if invalid.any():
        raise AssertionError("modified EAT-Lancet-13总分超出0--39。")
    return data


def standardize(
    values: pd.Series, eligible: pd.Series, output_name: str
) -> Tuple[pd.Series, Dict[str, object]]:
    numeric = pd.to_numeric(values, errors="coerce")
    use = eligible.fillna(False) & numeric.notna()
    count = int(use.sum())
    if count < 2:
        raise ValueError("{}可标准化参与者少于2人。".format(output_name))
    mean = float(numeric.loc[use].mean())
    sd = float(numeric.loc[use].std(ddof=1))
    if not sd > 0:
        raise ValueError("{}没有变异，无法计算Z分数。".format(output_name))
    result = pd.Series(pd.NA, index=values.index, dtype="Float64")
    result.loc[use] = (numeric.loc[use] - mean) / sd
    parameters: Dict[str, object] = {
        "variable": output_name,
        "eligible_participant_count": count,
        "mean": mean,
        "sample_standard_deviation": sd,
        "ddof": 1,
    }
    return result, parameters


def calculate_scores(
    intakes: pd.DataFrame,
    confirm_eat_weight_basis: bool,
    minimum_nova_energy_coverage: float = 0.0,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    validate_columns(intakes.columns.tolist(), REQUIRED_COLUMNS, "新饮食指标摄入表")
    if not confirm_eat_weight_basis:
        raise ValueError(
            "modified EAT-Lancet-13计分前必须确认weight_g可作为实际摄入g/day。"
            "确认后使用--confirm-eat-weight-basis。"
        )
    if minimum_nova_energy_coverage < 0 or minimum_nova_energy_coverage > 1:
        raise ValueError("minimum_nova_energy_coverage必须在0到1之间。")
    data = intakes.copy()
    for column in ["participant_id", "cohort", "research_stage"]:
        data[column] = normalize_text(data[column])
    if data.duplicated(["participant_id", "cohort", "research_stage"]).any():
        raise ValueError("新饮食指标摄入表存在重复参与者键。")
    data = add_eat_lancet_scores(data)

    parameters = []
    eat_z, eat_parameters = standardize(
        data["modified_eat_lancet13_raw_0_39"],
        data["modified_eat_lancet13_complete"],
        "modified_eat_lancet13_raw_0_39",
    )
    data["modified_eat_lancet13_z"] = eat_z
    parameters.append(eat_parameters)

    nova = pd.to_numeric(data["nova4_classified_energy_pct"], errors="coerce")
    nova_coverage = pd.to_numeric(
        data["participant_nova_mapping_energy_coverage"], errors="coerce"
    )
    nova_eligible = (
        nova.between(0, 100)
        & data["nova4_classified_valid_day_count"].gt(0)
        & nova_coverage.ge(minimum_nova_energy_coverage)
    )
    data["nova4_meets_energy_mapping_coverage"] = nova_coverage.ge(
        minimum_nova_energy_coverage
    )
    nova_z, nova_parameters = standardize(
        nova, nova_eligible, "nova4_classified_energy_pct"
    )
    data["nova4_classified_energy_pct_z"] = nova_z
    # Reject legacy standardized results carried in by custom input tables.
    data = data.drop(columns=["nova4_energy_pct_z"], errors="ignore")
    nova_parameters["denominator"] = "classified_food_energy_only"
    nova_parameters["minimum_energy_mapping_coverage"] = (
        minimum_nova_energy_coverage
    )
    parameters.append(nova_parameters)

    carbohydrate = pd.to_numeric(
        data["carbohydrate_energy_pct"], errors="coerce"
    )
    carbohydrate_eligible = (
        carbohydrate.between(0, 100)
        & data["carbohydrate_valid_day_count"].gt(0)
        & data["carbohydrate_pct_out_of_range_day_count"].eq(0)
    )
    carb_z, carb_parameters = standardize(
        carbohydrate, carbohydrate_eligible, "carbohydrate_energy_pct"
    )
    data["carbohydrate_energy_pct_z"] = carb_z
    data["very_low_carb_flag"] = pd.Series(
        pd.NA, index=data.index, dtype="boolean"
    )
    data["low_carb_flag"] = pd.Series(pd.NA, index=data.index, dtype="boolean")
    valid_carb = carbohydrate.between(0, 100)
    data.loc[valid_carb, "very_low_carb_flag"] = carbohydrate.loc[
        valid_carb
    ].lt(10)
    data.loc[valid_carb, "low_carb_flag"] = (
        carbohydrate.loc[valid_carb].ge(10)
        & carbohydrate.loc[valid_carb].le(26)
    )
    parameters.append(carb_parameters)
    return data, parameters


def build_qc(
    scores: pd.DataFrame,
    parameters: List[Dict[str, object]],
    minimum_nova_energy_coverage: float,
) -> pd.DataFrame:
    rows = [
        ("protocol", "script_version", SCRIPT_VERSION),
        ("protocol", "eat_lancet_name", "modified EAT-Lancet-13"),
        ("protocol", "eat_lancet_component_count", 13),
        ("protocol", "eat_lancet_raw_range", "0--39"),
        ("protocol", "excluded_component", "added_sugar"),
        ("protocol", "nova_analysis_variable", "nova4_classified_energy_pct"),
        ("protocol", "nova_denominator", "classified_food_energy_only; unclassified events excluded from NOVA only"),
        (
            "protocol",
            "standardization",
            "raw participant value; sample SD; no winsorization; no energy residual",
        ),
        (
            "protocol",
            "minimum_nova_energy_coverage",
            minimum_nova_energy_coverage,
        ),
        ("overall", "participant_count", len(scores)),
        (
            "eat_lancet",
            "complete_score_count",
            int(scores["modified_eat_lancet13_raw_0_39"].notna().sum()),
        ),
        (
            "nova4",
            "standardized_count",
            int(scores["nova4_classified_energy_pct_z"].notna().sum()),
        ),
        (
            "carbohydrate",
            "standardized_count",
            int(scores["carbohydrate_energy_pct_z"].notna().sum()),
        ),
    ]
    for component in EAT_COMPONENTS:
        column = "{}_score_0_3".format(component)
        rows.extend(
            [
                (
                    "eat_lancet_component",
                    "{}_nonmissing_count".format(component),
                    int(scores[column].notna().sum()),
                ),
                (
                    "eat_lancet_component",
                    "{}_minimum".format(component),
                    scores[column].min(),
                ),
                (
                    "eat_lancet_component",
                    "{}_maximum".format(component),
                    scores[column].max(),
                ),
            ]
        )
    for parameter in parameters:
        variable = parameter["variable"]
        for key, value in parameter.items():
            if key != "variable":
                rows.append(("standardization", "{}.{}".format(variable, key), value))
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def calculate_diet_indicator_scores(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    confirm_eat_weight_basis: bool = False,
    minimum_nova_energy_coverage: float = 0.0,
) -> Tuple[Path, Path, Path, Path]:
    if not Path(input_csv).is_file():
        raise FileNotFoundError("找不到新饮食指标摄入表：{}".format(input_csv))
    intakes = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        dtype={
            "participant_id": "string",
            "cohort": "string",
            "research_stage": "string",
        },
        low_memory=False,
    )
    scores, parameters = calculate_scores(
        intakes,
        confirm_eat_weight_basis=confirm_eat_weight_basis,
        minimum_nova_energy_coverage=minimum_nova_energy_coverage,
    )
    qc = build_qc(scores, parameters, minimum_nova_energy_coverage)
    parameter_table = pd.DataFrame(parameters)
    component_columns = [
        "participant_id",
        "cohort",
        "research_stage",
        *SCORE_COLUMNS,
        "modified_eat_lancet13_complete",
        "modified_eat_lancet13_raw_0_39",
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    component_path = output_dir / "eat_lancet13_participant_component_scores.csv"
    score_path = output_dir / "diet_indicator_participant_scores.csv"
    qc_path = output_dir / "diet_indicator_score_qc.csv"
    parameter_path = output_dir / "diet_indicator_standardization_parameters.csv"
    scores.loc[:, component_columns].to_csv(
        component_path, index=False, encoding="utf-8-sig"
    )
    scores.to_csv(score_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    parameter_table.to_csv(parameter_path, index=False, encoding="utf-8-sig")
    print("SCRIPT_VERSION={}".format(SCRIPT_VERSION))
    print("participants={:,}".format(len(scores)))
    print(
        "eat_lancet13_complete={:,}".format(
            int(scores["modified_eat_lancet13_raw_0_39"].notna().sum())
        )
    )
    print(
        "nova4_classified_standardized={:,}".format(
            int(scores["nova4_classified_energy_pct_z"].notna().sum())
        )
    )
    print(
        "carbohydrate_standardized={:,}".format(
            int(scores["carbohydrate_energy_pct_z"].notna().sum())
        )
    )
    return component_path, score_path, qc_path, parameter_path


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--confirm-eat-weight-basis", action="store_true")
    parser.add_argument("--minimum-nova-energy-coverage", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for output_path in calculate_diet_indicator_scores(
        input_csv=arguments.input_csv,
        output_dir=arguments.output_dir,
        confirm_eat_weight_basis=arguments.confirm_eat_weight_basis,
        minimum_nova_energy_coverage=arguments.minimum_nova_energy_coverage,
    ):
        print(output_path)
