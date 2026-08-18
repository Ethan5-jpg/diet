"""Calculate raw and analysis-ready HPP mAHEI-7 participant scores."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-18-ahei-scores-v2"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR.parent
DEFAULT_INPUT_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "ahei"
    / "ahei_participant_component_intakes.csv"
)
DEFAULT_POPULATION_CSV = (
    DATA_DIR / "Transfer" / "population" / "population.csv"
)
LEGACY_POPULATION_CSV = (
    DATA_DIR / "Transfer" / "diet_logging" / "population.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "05_diet_scores" / "ahei"

SCORE_PROTOCOL_NAME = "modified_AHEI_2010_7_component"
SCORE_COMPONENT_COUNT = 7
SCORE_RAW_MIN = 0.0
SCORE_RAW_MAX = 70.0
WINSOR_LOWER_QUANTILE = 0.005
WINSOR_UPPER_QUANTILE = 0.995

COMPONENTS = [
    "vegetables",
    "fruit",
    "whole_grains",
    "ssb_plus_fruit_juice",
    "nuts_plus_legumes",
    "red_plus_processed_meat",
    "alcohol",
]
COMPONENT_SCORE_COLUMNS = [
    "{}_score_0_10".format(component) for component in COMPONENTS
]
INTAKE_COLUMNS = {
    "vegetables": "mean_daily_vegetables_servings",
    "fruit": "mean_daily_fruit_servings",
    "whole_grains": "mean_daily_whole_grain_proxy_g",
    "ssb_plus_fruit_juice": "mean_daily_ssb_plus_fruit_juice_servings",
    "nuts_plus_legumes": "mean_daily_nuts_plus_legumes_servings",
    "red_plus_processed_meat": "mean_daily_red_plus_processed_meat_servings",
    "alcohol": "mean_daily_alcohol_servings",
}
COMPLETE_COLUMNS = {
    component: "{}_complete".format(component) for component in COMPONENTS
}
WHOLE_GRAIN_TARGET_G = {"female": 75.0, "male": 90.0}


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


def resolve_population_csv(path: Path) -> Path:
    """Prefer the server layout while supporting the local snapshot layout."""
    path = Path(path)
    if path.is_file():
        return path
    if path == DEFAULT_POPULATION_CSV and LEGACY_POPULATION_CSV.is_file():
        return LEGACY_POPULATION_CSV
    return path


def score_beneficial(intake: pd.Series, target: object) -> pd.Series:
    """Linearly map zero to 0 and the beneficial target to 10."""
    numeric = pd.to_numeric(intake, errors="coerce")
    if isinstance(target, pd.Series):
        target_numeric = pd.to_numeric(target, errors="coerce")
        score = numeric / target_numeric * 10.0
        score = score.mask(target_numeric.le(0))
    else:
        target_value = float(target)
        if target_value <= 0:
            raise ValueError("有益组件十分端点必须为正。")
        score = numeric / target_value * 10.0
    return score.clip(lower=0.0, upper=10.0).astype("Float64")


def score_adverse(intake: pd.Series, zero_score_endpoint: float) -> pd.Series:
    """Linearly map zero intake to 10 and the adverse endpoint to 0."""
    endpoint = float(zero_score_endpoint)
    if endpoint <= 0:
        raise ValueError("不利组件零分端点必须为正。")
    numeric = pd.to_numeric(intake, errors="coerce")
    score = (1.0 - numeric / endpoint) * 10.0
    return score.clip(lower=0.0, upper=10.0).astype("Float64")


def score_alcohol(servings: pd.Series, sex: pd.Series) -> pd.Series:
    """Apply the AHEI sex-specific moderate, heavy, and nondrinker rules."""
    values = pd.to_numeric(servings, errors="coerce")
    normalized_sex = normalize_text(sex).str.casefold()
    result = pd.Series(pd.NA, index=servings.index, dtype="Float64")
    for index in servings.index:
        value = values.loc[index]
        sex_value = normalized_sex.loc[index]
        if pd.isna(value) or value < 0 or sex_value not in {"female", "male"}:
            continue
        moderate_upper = 1.5 if sex_value == "female" else 2.0
        heavy_endpoint = 2.5 if sex_value == "female" else 3.5
        if value < 0.5:
            score = 2.5 + (value / 0.5) * 7.5
        elif value <= moderate_upper:
            score = 10.0
        elif value < heavy_endpoint:
            score = (
                (heavy_endpoint - value)
                / (heavy_endpoint - moderate_upper)
                * 10.0
            )
        else:
            score = 0.0
        result.loc[index] = max(0.0, min(10.0, float(score)))
    return result


def _canonical_sex(series: pd.Series) -> pd.Series:
    normalized = normalize_text(series).str.casefold()
    mapping = {
        "0": "female",
        "0.0": "female",
        "female": "female",
        "f": "female",
        "woman": "female",
        "1": "male",
        "1.0": "male",
        "male": "male",
        "m": "male",
        "man": "male",
    }
    return normalized.map(mapping).astype("string")


def prepare_population(population: pd.DataFrame) -> pd.DataFrame:
    required = ["participant_id", "cohort", "sex"]
    validate_columns(population.columns.tolist(), required, "population")
    result = population.loc[:, required].copy()
    result["participant_id"] = normalize_text(result["participant_id"])
    result["cohort"] = normalize_text(result["cohort"])
    result["sex"] = _canonical_sex(result["sex"])
    if result[["participant_id", "cohort"]].isna().any(axis=1).any():
        raise ValueError("population存在缺失参与者键。")
    duplicates = result.duplicated(["participant_id", "cohort"], keep=False)
    if duplicates.any():
        distinct = result.groupby(["participant_id", "cohort"])["sex"].nunique(dropna=False)
        if distinct.gt(1).any():
            raise ValueError("population同一参与者存在冲突sex。")
        result = result.drop_duplicates(["participant_id", "cohort"])
    return result


def calculate_component_scores(
    intakes: pd.DataFrame, population: pd.DataFrame
) -> pd.DataFrame:
    """Calculate seven 0--10 component scores and the unscaled 0--70 raw sum."""
    required = [
        "participant_id",
        "cohort",
        "research_stage",
        "mean_daily_energy_kcal",
        "energy_complete",
        *INTAKE_COLUMNS.values(),
        *COMPLETE_COLUMNS.values(),
    ]
    validate_columns(intakes.columns.tolist(), required, "AHEI参与者摄入表")
    data = intakes.copy()
    for column in ["participant_id", "cohort", "research_stage"]:
        data[column] = normalize_text(data[column])
    if data.duplicated(["participant_id", "cohort", "research_stage"]).any():
        raise ValueError("AHEI参与者摄入表存在重复参与者键。")
    for column in [*COMPLETE_COLUMNS.values(), "energy_complete"]:
        data[column] = parse_boolean(data[column], column)
    prepared_population = prepare_population(population)
    data = data.merge(
        prepared_population,
        on=["participant_id", "cohort"],
        how="left",
        validate="many_to_one",
    )

    data["vegetables_score_0_10"] = score_beneficial(
        data[INTAKE_COLUMNS["vegetables"]], 5.0
    )
    data["fruit_score_0_10"] = score_beneficial(
        data[INTAKE_COLUMNS["fruit"]], 4.0
    )
    whole_target = data["sex"].map(WHOLE_GRAIN_TARGET_G)
    data["whole_grains_score_0_10"] = score_beneficial(
        data[INTAKE_COLUMNS["whole_grains"]], whole_target
    )
    data["ssb_plus_fruit_juice_score_0_10"] = score_adverse(
        data[INTAKE_COLUMNS["ssb_plus_fruit_juice"]], 1.0
    )
    data["nuts_plus_legumes_score_0_10"] = score_beneficial(
        data[INTAKE_COLUMNS["nuts_plus_legumes"]], 1.0
    )
    data["red_plus_processed_meat_score_0_10"] = score_adverse(
        data[INTAKE_COLUMNS["red_plus_processed_meat"]], 1.5
    )
    data["alcohol_score_0_10"] = score_alcohol(
        data[INTAKE_COLUMNS["alcohol"]], data["sex"]
    )

    for component in COMPONENTS:
        score_column = "{}_score_0_10".format(component)
        complete_column = COMPLETE_COLUMNS[component]
        data.loc[~data[complete_column].eq(True), score_column] = pd.NA
    data.loc[data["sex"].isna(), ["whole_grains_score_0_10", "alcohol_score_0_10"]] = pd.NA
    data["mAHEI7_score_complete"] = data[COMPONENT_SCORE_COLUMNS].notna().all(axis=1)
    raw = data[COMPONENT_SCORE_COLUMNS].sum(axis=1, min_count=SCORE_COMPONENT_COUNT)
    data["mAHEI7_raw_0_70"] = raw.astype("Float64")
    out_of_range = data["mAHEI7_raw_0_70"].notna() & ~data[
        "mAHEI7_raw_0_70"
    ].between(SCORE_RAW_MIN, SCORE_RAW_MAX)
    if out_of_range.any():
        raise AssertionError("mAHEI-7原始分超出0--70。")
    return data


def assign_tie_safe_quintiles(values: pd.Series) -> pd.Series:
    """Use average ranks so identical adjusted scores remain in one quintile."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    count = int(valid.sum())
    if count == 0:
        return result
    average_rank = numeric.loc[valid].rank(method="average", ascending=True)
    quintiles = (((average_rank - 1) * 5) // count + 1).clip(1, 5)
    result.loc[valid] = quintiles.astype("Int64")
    return result


def add_energy_adjusted_scores(
    scores: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Winsorize raw scores, residual-adjust for energy, Z-score, and quintile."""
    required = [
        "mAHEI7_raw_0_70",
        "mAHEI7_score_complete",
        "energy_complete",
        "mean_daily_energy_kcal",
    ]
    validate_columns(scores.columns.tolist(), required, "AHEI评分表")
    data = scores.copy()
    score_complete = parse_boolean(data["mAHEI7_score_complete"], "mAHEI7_score_complete")
    energy_complete = parse_boolean(data["energy_complete"], "energy_complete")
    raw = pd.to_numeric(data["mAHEI7_raw_0_70"], errors="coerce")
    energy = pd.to_numeric(data["mean_daily_energy_kcal"], errors="coerce")
    eligible = (
        score_complete.eq(True)
        & energy_complete.eq(True)
        & raw.notna()
        & energy.gt(0)
    )
    count = int(eligible.sum())
    if count < 2:
        raise ValueError("可用于mAHEI-7能量校正的参与者少于2人。")
    eligible_raw = raw.loc[eligible].astype("float64")
    eligible_energy = energy.loc[eligible].astype("float64")
    lower = float(eligible_raw.quantile(WINSOR_LOWER_QUANTILE))
    upper = float(eligible_raw.quantile(WINSOR_UPPER_QUANTILE))
    winsorized = eligible_raw.clip(lower=lower, upper=upper)
    centered_energy = eligible_energy - eligible_energy.mean()
    denominator = float((centered_energy ** 2).sum())
    if denominator <= 0:
        raise ValueError("总能量没有变异，无法进行残差校正。")
    centered_score = winsorized - winsorized.mean()
    slope = float((centered_energy * centered_score).sum() / denominator)
    adjusted = winsorized - slope * centered_energy
    adjusted_mean = float(adjusted.mean())
    adjusted_sd = float(adjusted.std(ddof=1))
    if not adjusted_sd > 0:
        raise ValueError("能量校正后的mAHEI-7没有变异，无法计算Z分数。")
    standardized = (adjusted - adjusted_mean) / adjusted_sd

    data["mAHEI7_score_winsorized"] = pd.Series(
        pd.NA, index=data.index, dtype="Float64"
    )
    data["mAHEI7_score_energy_adjusted"] = pd.Series(
        pd.NA, index=data.index, dtype="Float64"
    )
    data["mAHEI7_score_energy_adjusted_z"] = pd.Series(
        pd.NA, index=data.index, dtype="Float64"
    )
    data.loc[eligible, "mAHEI7_score_winsorized"] = winsorized
    data.loc[eligible, "mAHEI7_score_energy_adjusted"] = adjusted
    data.loc[eligible, "mAHEI7_score_energy_adjusted_z"] = standardized
    data["mAHEI7_score_energy_adjusted_quintile"] = assign_tie_safe_quintiles(
        data["mAHEI7_score_energy_adjusted"]
    )
    parameters: Dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "score_protocol_name": SCORE_PROTOCOL_NAME,
        "method": (
            "winsorize raw mAHEI7 at P0.5/P99.5; residual method versus "
            "mean daily energy; add residual to mean winsorized score; "
            "standardize using sample SD (ddof=1)"
        ),
        "eligible_participant_count": count,
        "ineligible_participant_count": int(len(data) - count),
        "winsor_lower_quantile": WINSOR_LOWER_QUANTILE,
        "winsor_upper_quantile": WINSOR_UPPER_QUANTILE,
        "winsor_lower_value": lower,
        "winsor_upper_value": upper,
        "winsorized_low_count": int(eligible_raw.lt(lower).sum()),
        "winsorized_high_count": int(eligible_raw.gt(upper).sum()),
        "mean_energy_kcal": float(eligible_energy.mean()),
        "slope_score_per_kcal": slope,
        "adjusted_score_mean": adjusted_mean,
        "adjusted_score_standard_deviation": adjusted_sd,
        "adjusted_score_energy_correlation": float(adjusted.corr(eligible_energy)),
        "z_score_mean": float(standardized.mean()),
        "z_score_standard_deviation": float(standardized.std(ddof=1)),
        "z_score_standard_deviation_definition": "sample SD, ddof=1",
        "quintile_method": "average_rank_floor; identical values stay tied",
    }
    return data, parameters


def build_qc(scores: pd.DataFrame, parameters: Dict[str, object]) -> pd.DataFrame:
    rows = [
        ("protocol", "script_version", SCRIPT_VERSION),
        ("protocol", "score_protocol_name", SCORE_PROTOCOL_NAME),
        ("protocol", "score_component_count", SCORE_COMPONENT_COUNT),
        ("protocol", "score_raw_min", SCORE_RAW_MIN),
        ("protocol", "score_raw_max", SCORE_RAW_MAX),
        ("protocol", "excluded_components", "trans_fat|pufa|sodium"),
        ("overall", "participant_count", len(scores)),
        (
            "overall",
            "participants_with_complete_raw_score",
            int(scores["mAHEI7_raw_0_70"].notna().sum()),
        ),
        (
            "overall",
            "participants_with_energy_adjusted_z",
            int(scores["mAHEI7_score_energy_adjusted_z"].notna().sum()),
        ),
    ]
    for component in COMPONENTS:
        column = "{}_score_0_10".format(component)
        rows.extend(
            [
                ("component", "{}_nonmissing_count".format(component), int(scores[column].notna().sum())),
                ("component", "{}_minimum".format(component), scores[column].min()),
                ("component", "{}_maximum".format(component), scores[column].max()),
            ]
        )
    for key, value in parameters.items():
        rows.append(("energy_adjustment", key, value))
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def calculate_ahei_scores(
    input_csv: Path = DEFAULT_INPUT_CSV,
    population_csv: Path = DEFAULT_POPULATION_CSV,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Tuple[Path, Path, Path, Path]:
    """Read participant inputs and write component, final, QC, and parameter CSVs."""
    population_csv = resolve_population_csv(population_csv)
    for path in [input_csv, population_csv]:
        if not Path(path).is_file():
            raise FileNotFoundError("找不到评分输入：{}".format(path))
    intakes = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        dtype={"participant_id": "string", "cohort": "string", "research_stage": "string"},
        low_memory=False,
    )
    population = pd.read_csv(
        population_csv,
        encoding="utf-8-sig",
        dtype={"participant_id": "string", "cohort": "string"},
        low_memory=False,
    )
    component_scores = calculate_component_scores(intakes, population)
    scores, parameters = add_energy_adjusted_scores(component_scores)
    qc = build_qc(scores, parameters)
    parameter_table = pd.DataFrame(
        [(key, value) for key, value in parameters.items()],
        columns=["parameter", "value"],
    )
    component_output_columns = [
        "participant_id",
        "cohort",
        "research_stage",
        "sex",
        *COMPONENT_SCORE_COLUMNS,
        "mAHEI7_score_complete",
        "mAHEI7_raw_0_70",
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    component_path = output_dir / "ahei_participant_component_scores.csv"
    score_path = output_dir / "ahei_participant_scores.csv"
    qc_path = output_dir / "ahei_score_qc.csv"
    parameter_path = output_dir / "ahei_energy_adjustment_parameters.csv"
    scores.loc[:, component_output_columns].to_csv(
        component_path, index=False, encoding="utf-8-sig"
    )
    scores.to_csv(score_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    parameter_table.to_csv(parameter_path, index=False, encoding="utf-8-sig")
    return component_path, score_path, qc_path, parameter_path


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--population-csv", type=Path, default=DEFAULT_POPULATION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for output_path in calculate_ahei_scores(
        input_csv=arguments.input_csv,
        population_csv=arguments.population_csv,
        output_dir=arguments.output_dir,
    ):
        print(output_path)
