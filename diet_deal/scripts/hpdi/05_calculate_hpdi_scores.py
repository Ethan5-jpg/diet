"""计算18项 hPDI 原始总分，并按总能量摄入进行残差法校正。"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-08-hpdi-scores-v3"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "hpdi"
    / "hpdi_participant_component_intakes.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "05_diet_scores" / "hpdi"
ID_COLUMNS = ["participant_id", "cohort", "research_stage"]
COMPONENT_DEFINITIONS = [
    ("whole_grains", "positive"),
    ("fruits", "positive"),
    ("vegetables", "positive"),
    ("nuts", "positive"),
    ("legumes", "positive"),
    ("vegetable_oils", "positive"),
    ("tea_coffee", "positive"),
    ("fruit_juice", "reverse"),
    ("refined_grains", "reverse"),
    ("potatoes", "reverse"),
    ("sugar_sweetened_beverages", "reverse"),
    ("sweets_desserts", "reverse"),
    ("animal_fat", "reverse"),
    ("dairy", "reverse"),
    ("eggs", "reverse"),
    ("fish_seafood", "reverse"),
    ("meat", "reverse"),
    ("miscellaneous_animal_foods", "reverse"),
]
COMPONENTS = [component for component, _ in COMPONENT_DEFINITIONS]
ENERGY_COLUMN = "mean_daily_calories_kcal_proxy"
WINSOR_LOWER_QUANTILE = 0.005
WINSOR_UPPER_QUANTILE = 0.995


def validate_columns(columns: List[str], required: List[str], source_name: str) -> None:
    """确认输入表包含所需字段。"""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"{source_name}缺少字段：{', '.join(missing)}")


def intake_column(component: str) -> str:
    """返回组件的参与者平均每日克数代理字段。"""
    return f"mean_daily_{component}_g_proxy"


def weight_complete_column(component: str) -> str:
    """返回组件重量严格完整性字段。"""
    return f"{component}_weight_complete"


def parse_boolean(series: pd.Series, column: str) -> pd.Series:
    """将 CSV 中常见布尔表示解析为不含缺失的 bool。"""
    normalized = series.astype("string").str.strip().str.lower()
    parsed = normalized.map(
        {"true": True, "1": True, "false": False, "0": False}
    )
    invalid = parsed.isna()
    if invalid.any():
        values = sorted(
            normalized.loc[invalid].fillna("<missing>").astype(str).unique()
        )
        raise ValueError(
            f"{column} 含无效布尔值：" + ", ".join(values[:10])
        )
    return parsed.astype(bool)


def assign_tie_safe_quintiles(values: pd.Series) -> pd.Series:
    """用平均秩分组；相同摄入量始终获得相同五分位。"""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid_count = int(valid.sum())
    if valid_count == 0:
        return result

    average_rank = numeric.loc[valid].rank(method="average", ascending=True)
    quintile = (((average_rank - 1) * 5) // valid_count + 1).clip(1, 5)
    result.loc[valid] = quintile.astype("Int64")
    return result


def build_threshold_row(
    component: str,
    direction: str,
    values: pd.Series,
    quintiles: pd.Series,
) -> Dict[str, object]:
    """记录名义分位点、实际分组范围和并列值诊断。"""
    valid_values = values.dropna()
    row: Dict[str, object] = {
        "hpdi_component": component,
        "scoring_direction": direction,
        "intake_column": intake_column(component),
        "intake_basis": "mean daily grams proxy over observed diet days",
        "quintile_method": "average_rank_floor; identical values stay tied",
        "valid_participant_count": len(valid_values),
        "missing_participant_count": int(values.isna().sum()),
        "distinct_intake_values": int(valid_values.nunique()),
        "intake_min": valid_values.min() if not valid_values.empty else pd.NA,
        "intake_max": valid_values.max() if not valid_values.empty else pd.NA,
    }
    for probability, label in [
        (0.2, "p20"),
        (0.4, "p40"),
        (0.6, "p60"),
        (0.8, "p80"),
    ]:
        row[f"nominal_{label}"] = (
            valid_values.quantile(probability) if not valid_values.empty else pd.NA
        )
    for quintile in range(1, 6):
        mask = quintiles.eq(quintile)
        group_values = values.loc[mask]
        row[f"q{quintile}_participant_count"] = int(mask.sum())
        row[f"q{quintile}_intake_min"] = (
            group_values.min() if not group_values.empty else pd.NA
        )
        row[f"q{quintile}_intake_max"] = (
            group_values.max() if not group_values.empty else pd.NA
        )
    row["observed_quintile_count"] = int(quintiles.dropna().nunique())
    return row


def load_intakes(input_csv: Path) -> pd.DataFrame:
    """读取并验证参与者级 hPDI 组件摄入量。"""
    if not input_csv.is_file():
        raise FileNotFoundError(f"找不到 hPDI 参与者摄入量：{input_csv}")
    required = [
        *ID_COLUMNS,
        ENERGY_COLUMN,
        *[intake_column(item) for item in COMPONENTS],
        *[weight_complete_column(item) for item in COMPONENTS],
    ]
    header = pd.read_csv(input_csv, encoding="utf-8-sig", nrows=0)
    validate_columns(header.columns.tolist(), required, "hPDI 摄入量表")
    data = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        dtype={column: "string" for column in ID_COLUMNS},
        low_memory=False,
    )
    if data[ID_COLUMNS].isna().any(axis=1).any():
        raise ValueError("hPDI 摄入量表存在缺失参与者键。")
    if data.duplicated(ID_COLUMNS).any():
        raise ValueError("hPDI 摄入量表存在重复参与者键。")
    for component in COMPONENTS:
        column = intake_column(component)
        original = data[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"{column} 含 {int(invalid.sum()):,} 个非数值记录。")
        if numeric.lt(0).fillna(False).any():
            raise ValueError(f"{column} 含负摄入量。")
        data[column] = numeric
        complete_column = weight_complete_column(component)
        data[complete_column] = parse_boolean(
            data[complete_column], complete_column
        )
    original_energy = data[ENERGY_COLUMN]
    energy = pd.to_numeric(original_energy, errors="coerce")
    invalid_energy = original_energy.notna() & energy.isna()
    if invalid_energy.any():
        raise ValueError(
            f"{ENERGY_COLUMN} 含 {int(invalid_energy.sum()):,} 个非数值记录。"
        )
    nonpositive_energy = energy.notna() & energy.le(0)
    data[ENERGY_COLUMN] = energy.mask(nonpositive_energy)
    return data


def energy_adjust_score(
    raw_score: pd.Series,
    energy_kcal: pd.Series,
) -> Tuple[pd.Series, pd.Series, Dict[str, object]]:
    """先温莎化原始分，再用总能量残差法校正并保留原分量纲。"""
    raw_numeric = pd.to_numeric(raw_score, errors="coerce")
    energy_numeric = pd.to_numeric(energy_kcal, errors="coerce")
    eligible = raw_numeric.notna() & energy_numeric.gt(0)
    adjusted = pd.Series(pd.NA, index=raw_score.index, dtype="Float64")
    winsorized = pd.Series(pd.NA, index=raw_score.index, dtype="Float64")
    count = int(eligible.sum())
    if count < 2:
        raise ValueError("可用于 hPDI 能量校正的参与者少于2人。")

    eligible_score = raw_numeric.loc[eligible].astype("float64")
    eligible_energy = energy_numeric.loc[eligible].astype("float64")
    lower = float(eligible_score.quantile(WINSOR_LOWER_QUANTILE))
    upper = float(eligible_score.quantile(WINSOR_UPPER_QUANTILE))
    score_winsorized = eligible_score.clip(lower=lower, upper=upper)
    centered_energy = eligible_energy - eligible_energy.mean()
    denominator = float((centered_energy ** 2).sum())
    if denominator <= 0:
        raise ValueError("总能量摄入没有变异，无法进行残差法校正。")
    centered_score = score_winsorized - score_winsorized.mean()
    slope = float((centered_energy * centered_score).sum() / denominator)
    adjusted_values = score_winsorized - slope * centered_energy
    winsorized.loc[eligible] = score_winsorized
    adjusted.loc[eligible] = adjusted_values
    parameters: Dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "method": (
            "winsorize raw score at 0.5th/99.5th percentiles; "
            "residual method versus mean daily total energy; "
            "add residual to mean winsorized score"
        ),
        "eligible_participant_count": count,
        "missing_or_nonpositive_energy_count": int(
            raw_numeric.notna().sum() - count
        ),
        "winsor_lower_quantile": WINSOR_LOWER_QUANTILE,
        "winsor_upper_quantile": WINSOR_UPPER_QUANTILE,
        "winsor_lower_value": lower,
        "winsor_upper_value": upper,
        "mean_energy_kcal": float(eligible_energy.mean()),
        "mean_winsorized_raw_score": float(score_winsorized.mean()),
        "slope_score_per_kcal": slope,
        "adjusted_score_min": float(adjusted_values.min()),
        "adjusted_score_max": float(adjusted_values.max()),
        "adjusted_score_mean": float(adjusted_values.mean()),
    }
    return winsorized, adjusted, parameters


def build_qc(
    scores: pd.DataFrame,
    thresholds: pd.DataFrame,
    energy_parameters: Dict[str, object],
) -> pd.DataFrame:
    """生成整体、组件和总分分布 QC。"""
    rows = [
        ("overall", "", "script_version", SCRIPT_VERSION),
        ("overall", "", "participant_rows", len(scores)),
        (
            "overall",
            "",
            "participants_with_all_18_intake_values",
            int(scores["hpdi_complete_18_components"].sum()),
        ),
        (
            "overall",
            "",
            "participants_with_any_missing_intake_value",
            int((~scores["hpdi_complete_18_components"]).sum()),
        ),
        (
            "overall",
            "",
            "participants_with_all_18_component_weights_complete",
            int(scores["hpdi_all_18_component_weights_complete"].sum()),
        ),
        (
            "overall",
            "",
            "participants_with_incomplete_component_weights",
            int((~scores["hpdi_all_18_component_weights_complete"]).sum()),
        ),
        (
            "overall",
            "",
            "hpdi_raw_score_min",
            scores["hpdi_score_raw_18_90"].min(),
        ),
        (
            "overall",
            "",
            "hpdi_raw_score_max",
            scores["hpdi_score_raw_18_90"].max(),
        ),
        (
            "overall",
            "",
            "hpdi_raw_score_mean",
            scores["hpdi_score_raw_18_90"].mean(),
        ),
        (
            "overall",
            "",
            "hpdi_raw_score_standard_deviation",
            scores["hpdi_score_raw_18_90"].std(),
        ),
        (
            "overall",
            "",
            "participants_with_energy_adjusted_score",
            int(scores["hpdi_score_energy_adjusted"].notna().sum()),
        ),
        (
            "overall",
            "",
            "hpdi_energy_adjusted_score_min",
            scores["hpdi_score_energy_adjusted"].min(),
        ),
        (
            "overall",
            "",
            "hpdi_energy_adjusted_score_max",
            scores["hpdi_score_energy_adjusted"].max(),
        ),
        (
            "overall",
            "",
            "hpdi_energy_adjusted_score_mean",
            scores["hpdi_score_energy_adjusted"].mean(),
        ),
    ]
    for metric, value in energy_parameters.items():
        rows.append(("energy_adjustment", "", metric, value))
    for _, threshold in thresholds.iterrows():
        component = str(threshold["hpdi_component"])
        rows.extend(
            [
                (
                    "component",
                    component,
                    "valid_participant_count",
                    threshold["valid_participant_count"],
                ),
                (
                    "component",
                    component,
                    "missing_participant_count",
                    threshold["missing_participant_count"],
                ),
                (
                    "component",
                    component,
                    "distinct_intake_values",
                    threshold["distinct_intake_values"],
                ),
                (
                    "component",
                    component,
                    "observed_quintile_count",
                    threshold["observed_quintile_count"],
                ),
            ]
        )
        score_column = f"{component}_score"
        for score_value in range(1, 6):
            rows.append(
                (
                    "component_score_distribution",
                    component,
                    f"score_{score_value}_participant_count",
                    int(scores[score_column].eq(score_value).sum()),
                )
            )
    distribution = scores["hpdi_score_raw_18_90"].value_counts(
        dropna=False
    ).sort_index()
    for score_value, count in distribution.items():
        label = "missing" if pd.isna(score_value) else str(int(score_value))
        rows.append(("total_score_distribution", "", f"hpdi_score_{label}", int(count)))
    return pd.DataFrame(
        rows, columns=["summary_type", "hpdi_component", "metric", "value"]
    )


def calculate_hpdi_scores(
    input_csv: Path,
    output_dir: Path,
) -> Tuple[Path, Path, Path, Path]:
    """计算组件五分位分数、18项总分、阈值和 QC。"""
    intakes = load_intakes(input_csv)
    scores = intakes.loc[:, ID_COLUMNS].copy()
    scores[ENERGY_COLUMN] = intakes[ENERGY_COLUMN]
    threshold_rows = []
    score_columns = []
    for component, direction in COMPONENT_DEFINITIONS:
        column = intake_column(component)
        values = intakes[column]
        quintiles = assign_tie_safe_quintiles(values)
        component_scores = (
            quintiles if direction == "positive" else 6 - quintiles
        ).astype("Int64")
        scores[column] = values
        scores[weight_complete_column(component)] = intakes[
            weight_complete_column(component)
        ]
        scores[f"{component}_intake_quintile"] = quintiles
        scores[f"{component}_score"] = component_scores
        score_columns.append(f"{component}_score")
        threshold_rows.append(
            build_threshold_row(component, direction, values, quintiles)
        )

    scores["hpdi_nonmissing_component_count"] = (
        scores[score_columns].notna().sum(axis=1)
    )
    scores["hpdi_complete_18_components"] = scores[
        "hpdi_nonmissing_component_count"
    ].eq(18)
    weight_complete_columns = [
        weight_complete_column(component) for component in COMPONENTS
    ]
    scores["hpdi_component_weight_complete_count"] = scores[
        weight_complete_columns
    ].sum(axis=1)
    scores["hpdi_all_18_component_weights_complete"] = scores[
        "hpdi_component_weight_complete_count"
    ].eq(18)
    total = scores[score_columns].sum(axis=1, min_count=18)
    scores["hpdi_score_raw_18_90"] = total.astype("Int64")
    complete_scores = scores.loc[
        scores["hpdi_complete_18_components"], "hpdi_score_raw_18_90"
    ]
    if not complete_scores.between(18, 90).all():
        raise AssertionError("完整 hPDI 总分超出理论范围 18–90。")
    for score_column in score_columns:
        valid_scores = scores[score_column].dropna()
        if not valid_scores.between(1, 5).all():
            raise AssertionError(f"{score_column} 超出 1–5 分范围。")

    winsorized, adjusted, energy_parameters = energy_adjust_score(
        scores["hpdi_score_raw_18_90"],
        scores[ENERGY_COLUMN],
    )
    scores["hpdi_score_winsorized"] = winsorized
    scores["hpdi_score_energy_adjusted"] = adjusted
    scores["hpdi_energy_adjusted_quintile"] = assign_tie_safe_quintiles(
        adjusted
    )

    thresholds = pd.DataFrame(threshold_rows)
    qc = build_qc(scores, thresholds, energy_parameters)
    adjustment = pd.DataFrame([energy_parameters])
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = output_dir / "hpdi_quintile_thresholds.csv"
    score_path = output_dir / "hpdi_participant_scores.csv"
    qc_path = output_dir / "hpdi_score_qc.csv"
    adjustment_path = output_dir / "hpdi_energy_adjustment_parameters.csv"
    thresholds.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    scores.to_csv(score_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    adjustment.to_csv(adjustment_path, index=False, encoding="utf-8-sig")
    print("-" * 72)
    print(f"脚本版本：{SCRIPT_VERSION}")
    print(f"参与者：{len(scores):,} 人")
    print(
        "18项摄入值可用于主分析："
        f"{int(scores['hpdi_complete_18_components'].sum()):,} 人"
    )
    print(
        "18项重量严格完整（敏感性分析）："
        f"{int(scores['hpdi_all_18_component_weights_complete'].sum()):,} 人"
    )
    print(
        "原始 hPDI 范围："
        f"{scores['hpdi_score_raw_18_90'].min()}–"
        f"{scores['hpdi_score_raw_18_90'].max()}"
    )
    print(
        "能量校正分可用参与者："
        f"{int(scores['hpdi_score_energy_adjusted'].notna().sum()):,} 人"
    )
    print(
        "能量校正 hPDI 范围："
        f"{scores['hpdi_score_energy_adjusted'].min():.3f}–"
        f"{scores['hpdi_score_energy_adjusted'].max():.3f}"
    )
    print(f"五分位定义：{threshold_path}")
    print(f"参与者分数：{score_path}")
    print(f"QC 结果：{qc_path}")
    print(f"能量校正参数：{adjustment_path}")
    return threshold_path, score_path, qc_path, adjustment_path


def parse_args():
    parser = ArgumentParser(description="计算 hPDI 原始分和总能量残差校正分")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calculate_hpdi_scores(args.input, args.output_dir)
