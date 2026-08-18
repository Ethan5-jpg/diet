"""计算17组件 rEDIH、能量校正分、Z 分数和五分位。"""

import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-14-redih-scores-v3"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "redih"
    / "redih_participant_component_intakes.csv"
)
DEFAULT_AMED_RECOVERY_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "amed"
    / "amed_participant_component_intakes.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "05_diet_scores" / "redih"

ID_COLUMNS = ["participant_id", "cohort", "research_stage"]
ENERGY_COLUMN = "mean_daily_energy_kcal"
ENERGY_COMPLETENESS_COLUMN = "energy_complete"
ALCOHOL_COMPLETENESS_COLUMN = "alcohol_complete"
COMPONENT_WEIGHT_COMPLETENESS_COLUMN = (
    "redih_all_current_score_component_weights_complete"
)
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
    "redih_score_energy_adjusted_z_17",
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
    """验证索引、17组 g/day、重量、酒精和能量完整性标记。"""
    intake_columns = [
        "mean_daily_{}_g".format(component)
        for component in CURRENT_SCORE_COMPONENTS
    ]
    required = [
        *ID_COLUMNS,
        *intake_columns,
        ENERGY_COLUMN,
        ALCOHOL_COMPLETENESS_COLUMN,
        ENERGY_COMPLETENESS_COLUMN,
        COMPONENT_WEIGHT_COMPLETENESS_COLUMN,
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

    completeness_columns = [
        ALCOHOL_COMPLETENESS_COLUMN,
        ENERGY_COMPLETENESS_COLUMN,
        COMPONENT_WEIGHT_COMPLETENESS_COLUMN,
    ]
    for column in completeness_columns:
        clean[column] = parse_boolean(clean[column], column)
        if clean[column].isna().any():
            raise ValueError("{} 存在缺失，无法确定 rEDIH 计分资格。".format(column))
    eligible = pd.Series(True, index=clean.index, dtype="boolean")
    for column in completeness_columns:
        eligible &= clean[column].eq(True)
    eligible = eligible.fillna(False)
    if int(eligible.sum()) < 5:
        raise ValueError("完整的 rEDIH 参与者至少需要5人才能完成后处理。")

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
        raise ValueError("合格者的17个 rEDIH 组件 g/day 存在缺失。")
    if eligible_intakes.lt(0).any().any():
        raise ValueError("合格者的17个 rEDIH 组件 g/day 存在负值。")
    eligible_energy = clean.loc[eligible, ENERGY_COLUMN]
    if eligible_energy.isna().any():
        raise ValueError("合格者的 AMED 恢复能量存在缺失，无法进行残差校正。")
    nonpositive_energy = eligible_energy.le(0)
    if nonpositive_energy.any():
        raise ValueError(
            "合格者的 AMED 恢复能量有 {:,} 个非正值，无法进行残差校正。".format(
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
    """计算17组件 EDIH、rEDIH，并在完整者中完成统一后处理。"""
    scores = validate_input(data)
    eligible = (
        scores[ALCOHOL_COMPLETENESS_COLUMN].eq(True)
        & scores[ENERGY_COMPLETENESS_COLUMN].eq(True)
        & scores[COMPONENT_WEIGHT_COMPLETENESS_COLUMN].eq(True)
    ).fillna(False)
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
        raise ValueError("合格者的平均每日能量没有可用于残差校正的变异。")
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

    adjusted_sd = float(adjusted.std(ddof=1))
    if not math.isfinite(adjusted_sd) or adjusted_sd <= 0:
        raise ValueError("能量校正后的 rEDIH 没有变异，无法进行 Z 标准化。")
    scores["redih_score_energy_adjusted_z_17"] = pd.Series(
        float("nan"), index=scores.index, dtype="float64"
    )
    scores.loc[eligible, "redih_score_energy_adjusted_z_17"] = (
        adjusted - float(adjusted.mean())
    ) / adjusted_sd

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
        "alcohol_incomplete_participant_count": int(
            (~scores[ALCOHOL_COMPLETENESS_COLUMN].eq(True)).sum()
        ),
        "energy_incomplete_participant_count": int(
            (~scores[ENERGY_COMPLETENESS_COLUMN].eq(True)).sum()
        ),
        "component_weight_incomplete_participant_count": int(
            (~scores[COMPONENT_WEIGHT_COMPLETENESS_COLUMN].eq(True)).sum()
        ),
        "current_intake_unit": "g/day",
        "current_scored_component_count": len(CURRENT_SCORE_COMPONENTS),
        "current_scored_components": "|".join(CURRENT_SCORE_COMPONENTS),
        "margarine_in_current_score": False,
        "wine_in_current_score": True,
        "wine_intake_definition": (
            "wine beverage weight_g/day; not pure alcohol_g/day"
        ),
        "alcohol_recovery_source": (
            "AMED participant intake table after alcohol recovery"
        ),
        "raw_edih_formula": "sum(mean_daily_component_g * published_weight) over 17 components",
        "redih_transform": "-1 * raw EDIH",
        "eligibility_rule": (
            "alcohol_complete and energy_complete and "
            "redih_all_current_score_component_weights_complete"
        ),
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
        "energy_adjusted_score_standard_deviation": adjusted_sd,
        "z_score_standard_deviation_definition": "sample SD, ddof=1",
        "standardized_energy_adjusted_score_mean": float(
            scores.loc[eligible, "redih_score_energy_adjusted_z_17"].mean()
        ),
        "standardized_energy_adjusted_score_sd": float(
            scores.loc[eligible, "redih_score_energy_adjusted_z_17"].std(ddof=1)
        ),
        "quintile_method": "stable score order with participant ID tie-breaker among eligible participants",
    }
    parameters = pd.DataFrame([parameter_row])

    distribution_columns = SCORE_COLUMNS[:5]
    distributions = pd.DataFrame(
        [distribution_row(column, scores[column]) for column in distribution_columns]
    )

    qc_metrics: Dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "participant_rows": len(scores),
        "unique_participants": scores["participant_id"].nunique(),
        "eligible_participant_count": int(eligible.sum()),
        "alcohol_complete_participant_count": int(
            scores[ALCOHOL_COMPLETENESS_COLUMN].eq(True).sum()
        ),
        "alcohol_incomplete_participant_count": int(
            (~scores[ALCOHOL_COMPLETENESS_COLUMN].eq(True)).sum()
        ),
        "energy_complete_participant_count": int(
            scores[ENERGY_COMPLETENESS_COLUMN].eq(True).sum()
        ),
        "energy_incomplete_participant_count": int(
            (~scores[ENERGY_COMPLETENESS_COLUMN].eq(True)).sum()
        ),
        "component_weight_complete_participant_count": int(
            scores[COMPONENT_WEIGHT_COMPLETENESS_COLUMN].eq(True).sum()
        ),
        "component_weight_incomplete_participant_count": int(
            (~scores[COMPONENT_WEIGHT_COMPLETENESS_COLUMN].eq(True)).sum()
        ),
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
        "energy_adjusted_redih_z_available_count": int(
            scores["redih_score_energy_adjusted_z_17"].notna().sum()
        ),
        "eligible_nonpositive_energy_count": int(
            scores.loc[eligible, ENERGY_COLUMN].le(0).sum()
        ),
        "max_absolute_direction_reversal_error": float(reversal_error.max()),
        "winsorized_low_count": int(raw_redih.lt(winsor_lower).sum()),
        "winsorized_high_count": int(raw_redih.gt(winsor_upper).sum()),
        "adjusted_score_energy_correlation": float(adjusted.corr(energy)),
        "z_score_mean": float(
            scores.loc[eligible, "redih_score_energy_adjusted_z_17"].mean()
        ),
        "z_score_standard_deviation": float(
            scores.loc[eligible, "redih_score_energy_adjusted_z_17"].std(ddof=1)
        ),
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


def load_amed_recovery(path: Path) -> pd.DataFrame:
    """读取 AMED 已恢复的酒精完整性、纯酒精和总能量。"""
    if not path.is_file():
        raise FileNotFoundError("找不到 AMED 恢复来源表：{}".format(path))
    data = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={column: "string" for column in ID_COLUMNS},
        low_memory=False,
    )
    required = [
        *ID_COLUMNS,
        ALCOHOL_COMPLETENESS_COLUMN,
        ENERGY_COMPLETENESS_COLUMN,
        ENERGY_COLUMN,
    ]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError("AMED 恢复来源表缺少字段：" + ", ".join(missing))
    optional_audit_columns = [
        "mean_daily_alcohol_g",
        "total_alcoholic_drink_events",
        "total_alcohol_observed_events",
        "total_alcohol_energy_formula_events",
        "total_alcohol_standard_abv_events",
        "total_alcohol_zero_fallback_events",
        "total_alcohol_unresolved_events",
    ]
    columns = required + [
        column for column in optional_audit_columns if column in data.columns
    ]
    result = data.loc[:, columns].copy()
    for column in ID_COLUMNS:
        result[column] = normalize_text(result[column])
    if result[ID_COLUMNS].isna().any(axis=1).any():
        raise ValueError("AMED 恢复来源表存在缺失参与者索引。")
    duplicated = result.duplicated(ID_COLUMNS, keep=False)
    if duplicated.any():
        preview = result.loc[duplicated, ID_COLUMNS].head(10)
        raise ValueError("AMED 恢复来源表索引不唯一：\n" + preview.to_string(index=False))
    return result


def run(
    input_csv: Path,
    amed_recovery_csv: Path,
    output_dir: Path,
) -> Dict[str, Path]:
    """合并 AMED 酒精和能量恢复结果，计算并保存17组件 rEDIH。"""
    if not input_csv.is_file():
        raise FileNotFoundError("找不到 rEDIH 参与者摄入表：{}".format(input_csv))
    data = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        dtype={column: "string" for column in ID_COLUMNS},
        low_memory=False,
    )
    duplicate_source_columns = [
        column
        for column in [
            ALCOHOL_COMPLETENESS_COLUMN,
            ENERGY_COMPLETENESS_COLUMN,
            ENERGY_COLUMN,
        ]
        if column in data.columns
    ]
    if duplicate_source_columns:
        raise ValueError(
            "rEDIH 摄入表已包含 AMED 恢复字段："
            + ", ".join(duplicate_source_columns)
        )
    amed_recovery = load_amed_recovery(amed_recovery_csv)
    merged = data.merge(
        amed_recovery,
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged["_merge"].ne("both")
    if unmatched.any():
        preview = merged.loc[unmatched, ID_COLUMNS].head(10)
        raise ValueError(
            "有 {:,} 名 rEDIH 参与者未匹配到 AMED 恢复结果：\n{}".format(
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
    print("重量、酒精和能量完整，已计算评分：{:,}".format(int(eligible.sum())))
    print("任一必要信息不完整，评分留空：{:,}".format(int((~eligible).sum())))
    print("计算公式：EDIH=Σ(g/day×权重)；rEDIH=-EDIH")
    print("wine 使用葡萄酒饮料 weight_g/day；AMED 的纯酒精 g/day 不替代 wine 重量。")
    print(
        "后处理：0.5/99.5百分位温莎化 -> AMED恢复能量残差校正 "
        "-> Z标准化 -> 五分位"
    )
    print("-" * 88)
    print(distributions.to_string(index=False))
    print("-" * 88)
    print("完整者的能量校正五分位人数：")
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
        "--amed-recovery-csv",
        type=Path,
        default=DEFAULT_AMED_RECOVERY_CSV,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    run(args.input_csv, args.amed_recovery_csv, args.output_dir)


if __name__ == "__main__":
    main()
