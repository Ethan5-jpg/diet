"""按 g/day 计算17组件 rEDIH；酒精不完整者评分保留为空。"""

import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-08-redih-scores-v2"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "redih"
    / "redih_participant_component_intakes.csv"
)
DEFAULT_ALCOHOL_COMPLETENESS_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "amed"
    / "amed_participant_component_intakes.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "05_diet_scores" / "redih"

ID_COLUMNS = ["participant_id", "cohort", "research_stage"]
ENERGY_COLUMN = "mean_daily_calories_kcal"
ALCOHOL_COMPLETENESS_COLUMN = "alcohol_complete"
WINE_INTAKE_COLUMN = "mean_daily_wine_g"
WINSOR_LOWER_QUANTILE = 0.005
WINSOR_UPPER_QUANTILE = 0.995

COMPONENT_DEFINITIONS = [
    ("red_meat", 0.250),
    ("low_energy_beverages", 0.053),
    ("cream_soups", 0.787),
    ("processed_meat", 0.199),
    ("poultry", 0.183),
    ("butter", 0.094),
    ("french_fries", 0.581),
    ("other_fish", 0.172),
    ("high_energy_drinks", 0.104),
    ("tomatoes", 0.095),
    ("low_fat_dairy", 0.025),
    ("eggs", 0.124),
    ("wine", -0.165),
    ("coffee", -0.035),
    ("whole_fruits", -0.029),
    ("high_fat_dairy", -0.046),
    ("green_leafy_vegetables", -0.055),
]
CURRENT_SCORE_COMPONENTS = [component for component, _ in COMPONENT_DEFINITIONS]
COMPONENT_WEIGHTS: Dict[str, float] = dict(COMPONENT_DEFINITIONS)

SCORE_COLUMNS = [
    "edih_score_raw_17",
    "redih_score_raw_17",
    "redih_score_winsorized_17",
    "redih_score_energy_adjusted_17",
    "redih_energy_adjusted_quintile_17",
]


def normalize_text(series: pd.Series) -> pd.Series:
    """清理文本并将空字符串转为缺失。"""
    normalized = series.astype("string").str.strip()
    return normalized.mask(normalized.eq(""), pd.NA)


def parse_boolean(series: pd.Series, column_name: str) -> pd.Series:
    """兼容 CSV 中常见的布尔值表示。"""
    normalized = normalize_text(series).str.lower()
    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n"}
    known_values = true_values.union(false_values)
    unknown = normalized.notna() & ~normalized.isin(known_values)
    if unknown.any():
        examples = sorted(normalized.loc[unknown].dropna().unique())[:10]
        raise ValueError(
            "{} 包含无法识别的布尔值：{}".format(
                column_name,
                ", ".join(examples),
            )
        )
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[normalized.isin(true_values)] = True
    result.loc[normalized.isin(false_values)] = False
    return result


def coefficient_definitions() -> pd.DataFrame:
    """输出17个 HPP 可映射组及其计分公式。"""
    rows: List[Dict[str, object]] = []
    for component, weight in COMPONENT_DEFINITIONS:
        rows.append(
            {
                "redih_component": component,
                "edih_weight_per_paper_table": weight,
                "intake_column": "mean_daily_{}_g".format(component),
                "current_intake_unit": "g/day",
                "included_in_current_score": True,
                "current_exclusion_reason": "",
                "edih_contribution_formula": "g/day * weight",
                "redih_direction_transform": "rEDIH = -1 * EDIH",
            }
        )
    return pd.DataFrame(rows)


def validate_input(data: pd.DataFrame) -> pd.DataFrame:
    """验证索引、17组 g/day、能量与酒精完整性标记。"""
    intake_columns = [
        "mean_daily_{}_g".format(component)
        for component in CURRENT_SCORE_COMPONENTS
    ]
    required = [
        *ID_COLUMNS,
        *intake_columns,
        ENERGY_COLUMN,
        ALCOHOL_COMPLETENESS_COLUMN,
    ]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError("参与者摄入表缺少字段：" + ", ".join(missing))

    clean = data.copy()
    for column in ID_COLUMNS:
        clean[column] = normalize_text(clean[column])
    missing_ids = int(clean[ID_COLUMNS].isna().any(axis=1).sum())
    if missing_ids:
        raise ValueError("参与者摄入表有 {:,} 行缺少索引。".format(missing_ids))
    duplicated = clean.duplicated(ID_COLUMNS, keep=False)
    if duplicated.any():
        preview = clean.loc[duplicated, ID_COLUMNS].head(10)
        raise ValueError("参与者摄入表索引不唯一：\n" + preview.to_string(index=False))

    clean[ALCOHOL_COMPLETENESS_COLUMN] = parse_boolean(
        clean[ALCOHOL_COMPLETENESS_COLUMN],
        ALCOHOL_COMPLETENESS_COLUMN,
    )
    if clean[ALCOHOL_COMPLETENESS_COLUMN].isna().any():
        raise ValueError("alcohol_complete 存在缺失，无法确定 rEDIH 计分资格。")
    eligible = clean[ALCOHOL_COMPLETENESS_COLUMN].eq(True).fillna(False)
    if int(eligible.sum()) < 5:
        raise ValueError("酒精完整的参与者至少需要5人才能生成五分位。")

    numeric_columns = [*intake_columns, ENERGY_COLUMN]
    for column in numeric_columns:
        original_nonmissing = clean[column].notna()
        numeric = pd.to_numeric(clean[column], errors="coerce")
        invalid = original_nonmissing & numeric.isna()
        if invalid.any():
            raise ValueError(
                "{}有 {:,} 个无效数值。".format(column, int(invalid.sum()))
            )
        clean[column] = numeric

    eligible_intakes = clean.loc[eligible, intake_columns]
    if eligible_intakes.isna().any().any():
        raise ValueError("酒精完整者的17个 rEDIH 组件 g/day 存在缺失。")
    if eligible_intakes.lt(0).any().any():
        raise ValueError("酒精完整者的17个 rEDIH 组件 g/day 存在负值。")
    eligible_energy = clean.loc[eligible, ENERGY_COLUMN]
    if eligible_energy.isna().any():
        raise ValueError("酒精完整者的平均每日能量存在缺失，无法进行残差校正。")
    nonpositive_energy = eligible_energy.le(0)
    if nonpositive_energy.any():
        raise ValueError(
            "酒精完整者的平均每日能量有 {:,} 个非正值，无法进行残差校正。".format(
                int(nonpositive_energy.sum())
            )
        )
    return clean


def stable_quintiles(values: pd.Series, tie_breaker: pd.Series) -> pd.Series:
    """按数值和参与者稳定键排序，生成尽量均衡的1至5五分位。"""
    if values.isna().any():
        raise ValueError("五分位输入含缺失。")
    if len(values) < 5:
        raise ValueError("至少需要5名参与者才能生成五分位。")
    order = pd.DataFrame(
        {
            "score": pd.to_numeric(values, errors="raise"),
            "tie_breaker": normalize_text(tie_breaker).fillna(""),
        },
        index=values.index,
    ).sort_values(["score", "tie_breaker"], kind="mergesort")
    position = pd.Series(range(len(order)), index=order.index, dtype="int64")
    assigned = ((position * 5) // len(order) + 1).clip(upper=5).astype("Int64")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    result.loc[assigned.index] = assigned
    return result


def distribution_row(score_name: str, values: pd.Series) -> Dict[str, object]:
    """生成一个评分变量在可计分样本中的固定分位数摘要。"""
    numeric = pd.to_numeric(values, errors="raise").dropna()
    return {
        "score_variable": score_name,
        "participant_count": int(len(numeric)),
        "minimum": float(numeric.min()),
        "p0_5": float(numeric.quantile(0.005)),
        "p1": float(numeric.quantile(0.01)),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "mean": float(numeric.mean()),
        "p75": float(numeric.quantile(0.75)),
        "p99": float(numeric.quantile(0.99)),
        "p99_5": float(numeric.quantile(0.995)),
        "maximum": float(numeric.max()),
        "standard_deviation": float(numeric.std(ddof=1)),
    }


def calculate_redih_scores(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """计算17组件 EDIH、rEDIH，并仅在酒精完整者中完成后处理。"""
    scores = validate_input(data)
    eligible = scores[ALCOHOL_COMPLETENESS_COLUMN].eq(True).fillna(False)
    scores["redih_score_complete"] = eligible.astype(bool)

    contribution_columns: List[str] = []
    for component, weight in COMPONENT_DEFINITIONS:
        intake_column = "mean_daily_{}_g".format(component)
        contribution_column = "edih_{}_g_weighted_contribution".format(component)
        scores[contribution_column] = (scores[intake_column] * weight).mask(~eligible)
        contribution_columns.append(contribution_column)

    scores["edih_score_raw_17"] = scores[contribution_columns].sum(
        axis=1,
        min_count=len(contribution_columns),
    )
    scores["redih_score_raw_17"] = -1.0 * scores["edih_score_raw_17"]
    reversal_error = (
        scores.loc[eligible, "redih_score_raw_17"]
        + scores.loc[eligible, "edih_score_raw_17"]
    ).abs()
    if float(reversal_error.max()) > 1e-10:
        raise ValueError("EDIH 到 rEDIH 的方向反转验证失败。")

    raw_redih = scores.loc[eligible, "redih_score_raw_17"]
    winsor_lower = float(raw_redih.quantile(WINSOR_LOWER_QUANTILE))
    winsor_upper = float(raw_redih.quantile(WINSOR_UPPER_QUANTILE))
    scores["redih_score_winsorized_17"] = pd.Series(
        float("nan"), index=scores.index, dtype="float64"
    )
    scores.loc[eligible, "redih_score_winsorized_17"] = raw_redih.clip(
        lower=winsor_lower,
        upper=winsor_upper,
    )

    energy = scores.loc[eligible, ENERGY_COLUMN]
    winsorized = scores.loc[eligible, "redih_score_winsorized_17"]
    energy_mean = float(energy.mean())
    score_mean = float(winsorized.mean())
    centered_energy = energy - energy_mean
    denominator = float((centered_energy ** 2).sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("酒精完整者的平均每日能量没有可用于残差校正的变异。")
    slope = float((centered_energy * (winsorized - score_mean)).sum() / denominator)
    intercept = score_mean - slope * energy_mean
    predicted = intercept + slope * energy
    residual = winsorized - predicted
    adjusted = residual + score_mean
    if not adjusted.map(lambda value: math.isfinite(float(value))).all():
        raise ValueError("能量校正 rEDIH 出现非有限值。")
    scores["redih_score_energy_adjusted_17"] = pd.Series(
        float("nan"), index=scores.index, dtype="float64"
    )
    scores.loc[eligible, "redih_score_energy_adjusted_17"] = adjusted

    tie_breaker = (
        scores.loc[eligible, "participant_id"].astype("string")
        + "|"
        + scores.loc[eligible, "cohort"].astype("string")
        + "|"
        + scores.loc[eligible, "research_stage"].astype("string")
    )
    scores["redih_energy_adjusted_quintile_17"] = pd.Series(
        pd.NA, index=scores.index, dtype="Int64"
    )
    scores.loc[eligible, "redih_energy_adjusted_quintile_17"] = stable_quintiles(
        scores.loc[eligible, "redih_score_energy_adjusted_17"],
        tie_breaker,
    )

    parameter_row = {
        "script_version": SCRIPT_VERSION,
        "participant_count": len(scores),
        "eligible_participant_count": int(eligible.sum()),
        "alcohol_incomplete_participant_count": int((~eligible).sum()),
        "current_intake_unit": "g/day",
        "current_scored_component_count": len(CURRENT_SCORE_COMPONENTS),
        "current_scored_components": "|".join(CURRENT_SCORE_COMPONENTS),
        "margarine_in_current_score": False,
        "wine_in_current_score": True,
        "raw_edih_formula": "sum(mean_daily_component_g * published_weight) over 17 components",
        "redih_transform": "-1 * raw EDIH",
        "eligibility_rule": "alcohol_complete == true",
        "ineligible_score_handling": "retain participant row; official score fields are missing",
        "winsor_lower_quantile": WINSOR_LOWER_QUANTILE,
        "winsor_upper_quantile": WINSOR_UPPER_QUANTILE,
        "winsor_lower_value": winsor_lower,
        "winsor_upper_value": winsor_upper,
        "energy_column": ENERGY_COLUMN,
        "energy_mean": energy_mean,
        "winsorized_score_mean": score_mean,
        "energy_regression_slope": slope,
        "energy_regression_intercept": intercept,
        "energy_adjustment": "OLS residual plus winsorized score mean among eligible participants",
        "quintile_method": "stable score order with participant ID tie-breaker among eligible participants",
    }
    parameters = pd.DataFrame([parameter_row])

    distribution_columns = SCORE_COLUMNS[:4]
    distributions = pd.DataFrame(
        [distribution_row(column, scores[column]) for column in distribution_columns]
    )

    qc_metrics: Dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "participant_rows": len(scores),
        "unique_participants": scores["participant_id"].nunique(),
        "alcohol_complete_participant_count": int(eligible.sum()),
        "alcohol_incomplete_participant_count": int((~eligible).sum()),
        "official_score_missing_participant_count": int(
            scores["redih_score_energy_adjusted_17"].isna().sum()
        ),
        "current_intake_unit": "g/day",
        "current_scored_component_count": len(CURRENT_SCORE_COMPONENTS),
        "margarine_in_current_score": False,
        "wine_in_current_score": True,
        "wine_intake_nonzero_eligible_participant_count": int(
            scores.loc[eligible, WINE_INTAKE_COLUMN].gt(0).sum()
        ),
        "component_contribution_column_count": len(contribution_columns),
        "raw_edih_available_count": int(scores["edih_score_raw_17"].notna().sum()),
        "energy_adjusted_redih_available_count": int(
            scores["redih_score_energy_adjusted_17"].notna().sum()
        ),
        "eligible_nonpositive_energy_count": int(
            scores.loc[eligible, ENERGY_COLUMN].le(0).sum()
        ),
        "max_absolute_direction_reversal_error": float(reversal_error.max()),
        "winsorized_low_count": int(raw_redih.lt(winsor_lower).sum()),
        "winsorized_high_count": int(raw_redih.gt(winsor_upper).sum()),
    }
    for quintile, count in (
        scores.loc[eligible, "redih_energy_adjusted_quintile_17"]
        .value_counts()
        .sort_index()
        .items()
    ):
        qc_metrics["energy_adjusted_quintile_{}_count".format(int(quintile))] = int(
            count
        )
    qc = pd.DataFrame(
        {"metric": list(qc_metrics.keys()), "value": list(qc_metrics.values())}
    )
    return scores, parameters, distributions, qc


def load_alcohol_completeness(path: Path) -> pd.DataFrame:
    """读取 AMED 步骤已确定的酒精完整性标记，作为当前统一资格门控。"""
    if not path.is_file():
        raise FileNotFoundError("找不到酒精完整性来源表：{}".format(path))
    data = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={column: "string" for column in ID_COLUMNS},
        low_memory=False,
    )
    required = [*ID_COLUMNS, ALCOHOL_COMPLETENESS_COLUMN]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError("酒精完整性来源表缺少字段：" + ", ".join(missing))
    optional_audit_columns = [
        "mean_daily_alcohol_g",
        "total_alcohol_unresolved_events",
    ]
    columns = required + [
        column for column in optional_audit_columns if column in data.columns
    ]
    result = data.loc[:, columns].copy()
    for column in ID_COLUMNS:
        result[column] = normalize_text(result[column])
    if result[ID_COLUMNS].isna().any(axis=1).any():
        raise ValueError("酒精完整性来源表存在缺失参与者索引。")
    duplicated = result.duplicated(ID_COLUMNS, keep=False)
    if duplicated.any():
        preview = result.loc[duplicated, ID_COLUMNS].head(10)
        raise ValueError("酒精完整性来源表索引不唯一：\n" + preview.to_string(index=False))
    return result


def run(
    input_csv: Path,
    alcohol_completeness_csv: Path,
    output_dir: Path,
) -> Dict[str, Path]:
    """合并酒精完整性标记，计算并保存17组件 rEDIH。"""
    if not input_csv.is_file():
        raise FileNotFoundError("找不到 rEDIH 参与者摄入表：{}".format(input_csv))
    data = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        dtype={column: "string" for column in ID_COLUMNS},
        low_memory=False,
    )
    if ALCOHOL_COMPLETENESS_COLUMN in data.columns:
        raise ValueError(
            "rEDIH 摄入表已包含 alcohol_complete；请避免与资格来源表重复。"
        )
    alcohol = load_alcohol_completeness(alcohol_completeness_csv)
    merged = data.merge(
        alcohol,
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        preview = merged.loc[unmatched, ID_COLUMNS].head(10)
        raise ValueError(
            "有 {:,} 名 rEDIH 参与者未匹配到酒精完整性标记：\n{}".format(
                int(unmatched.sum()),
                preview.to_string(index=False),
            )
        )
    merged = merged.drop(columns="_merge")
    scores, parameters, distributions, qc = calculate_redih_scores(merged)
    coefficients = coefficient_definitions()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "scores": output_dir / "redih_participant_scores_17.csv",
        "coefficients": output_dir / "redih_component_coefficients.csv",
        "parameters": output_dir / "redih_scoring_parameters_17.csv",
        "distributions": output_dir / "redih_score_distribution_17.csv",
        "qc": output_dir / "redih_score_qc_17.csv",
    }
    scores.to_csv(paths["scores"], index=False, encoding="utf-8-sig")
    coefficients.to_csv(paths["coefficients"], index=False, encoding="utf-8-sig")
    parameters.to_csv(paths["parameters"], index=False, encoding="utf-8-sig")
    distributions.to_csv(paths["distributions"], index=False, encoding="utf-8-sig")
    qc.to_csv(paths["qc"], index=False, encoding="utf-8-sig")

    eligible = scores["redih_score_complete"].eq(True)
    print("=" * 88)
    print("rEDIH：17组件，g/day，包含 wine")
    print("SCRIPT_VERSION={}".format(SCRIPT_VERSION))
    print("输入参与者：{:,}".format(len(scores)))
    print("酒精完整、已计算评分：{:,}".format(int(eligible.sum())))
    print("酒精不完整、评分留空：{:,}".format(int((~eligible).sum())))
    print("计算公式：EDIH=Σ(g/day×权重)；rEDIH=-EDIH")
    print("后处理仅在酒精完整者中：0.5/99.5百分位温莎化 -> 能量残差校正 -> 五分位")
    print("-" * 88)
    print(distributions.to_string(index=False))
    print("-" * 88)
    print("酒精完整者的能量校正五分位人数：")
    print(
        scores.loc[eligible, "redih_energy_adjusted_quintile_17"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print("输出目录：{}".format(output_dir))
    print("=" * 88)
    return paths


def parse_args() -> object:
    """解析命令行参数。"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument(
        "--alcohol-completeness-csv",
        type=Path,
        default=DEFAULT_ALCOHOL_COMPLETENESS_CSV,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    run(args.input_csv, args.alcohol_completeness_csv, args.output_dir)


if __name__ == "__main__":
    main()
