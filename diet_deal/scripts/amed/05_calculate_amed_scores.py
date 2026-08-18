"""计算 AMED 原始分、能量校正分、Z 分数和五分位。"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-14-amed-scores-v2"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR.parent
DEFAULT_INPUT_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "amed"
    / "amed_participant_component_intakes.csv"
)
DEFAULT_POPULATION_CSV = (
    DATA_DIR / "Transfer" / "population" / "population.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "05_diet_scores" / "amed"
)

ID_COLUMNS = ["participant_id", "cohort", "research_stage"]
MERGE_KEYS = ["participant_id", "cohort"]
COMPONENT_DEFINITIONS = [
    ("fruit", "mean_daily_fruit_g_proxy", "beneficial"),
    ("vegetables", "mean_daily_vegetables_g_proxy", "beneficial"),
    ("whole_grains", "mean_daily_whole_grains_g_proxy", "beneficial"),
    ("nuts", "mean_daily_nuts_g_proxy", "beneficial"),
    ("legumes", "mean_daily_legumes_g_proxy", "beneficial"),
    ("fish", "mean_daily_fish_g_proxy", "beneficial"),
    (
        "red_processed_meat",
        "mean_daily_red_processed_meat_g_proxy",
        "adverse",
    ),
]
ALCOHOL_COLUMNS = [
    "alcohol_complete",
    "mean_daily_alcohol_g",
    "total_alcohol_unresolved_events",
]
ENERGY_COLUMNS = [
    "energy_complete",
    "mean_daily_energy_kcal",
    "total_energy_missing_or_invalid_events",
]

FEMALE_CODE = 0
MALE_CODE = 1
FEMALE_ALCOHOL_RANGE = (5.0, 15.0)
MALE_ALCOHOL_RANGE = (10.0, 25.0)
WINSOR_LOWER_QUANTILE = 0.005
WINSOR_UPPER_QUANTILE = 0.995


def normalize_text(series: pd.Series) -> pd.Series:
    """去除文本首尾空白，并将空字符串视为缺失。"""
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
            f"{column_name} 包含无法识别的布尔值："
            + ", ".join(examples)
        )

    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[normalized.isin(true_values)] = True
    result.loc[normalized.isin(false_values)] = False
    return result


def validate_input(data: pd.DataFrame) -> None:
    """验证参与者摄入表的字段和参与者索引。"""
    intake_columns = [column for _, column, _ in COMPONENT_DEFINITIONS]
    required = [
        *ID_COLUMNS,
        *intake_columns,
        *ALCOHOL_COLUMNS,
        *ENERGY_COLUMNS,
    ]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError("参与者摄入表缺少字段：" + ", ".join(missing))

    missing_keys = int(data[ID_COLUMNS].isna().any(axis=1).sum())
    if missing_keys:
        raise ValueError(f"参与者摄入表有 {missing_keys:,} 行缺少索引。")

    duplicated = data.duplicated(ID_COLUMNS, keep=False)
    if duplicated.any():
        preview = data.loc[duplicated, ID_COLUMNS].head(10)
        raise ValueError(
            "参与者摄入表的索引不唯一：\n"
            + preview.to_string(index=False)
        )


def prepare_population(population: pd.DataFrame) -> pd.DataFrame:
    """清理人口学表，并保留酒精评分所需的性别。"""
    required = [*MERGE_KEYS, "sex"]
    missing = sorted(set(required) - set(population.columns))
    if missing:
        raise ValueError("population 表缺少字段：" + ", ".join(missing))

    population = population.loc[:, required].copy()
    for column in MERGE_KEYS:
        population[column] = normalize_text(population[column])

    missing_keys = int(population[MERGE_KEYS].isna().any(axis=1).sum())
    if missing_keys:
        raise ValueError(f"population 表有 {missing_keys:,} 行缺少索引。")

    duplicated = population.duplicated(MERGE_KEYS, keep=False)
    if duplicated.any():
        preview = population.loc[duplicated, MERGE_KEYS].head(10)
        raise ValueError(
            "population 表的 participant_id-cohort 不唯一：\n"
            + preview.to_string(index=False)
        )

    original_nonmissing = population["sex"].notna()
    sex = pd.to_numeric(population["sex"], errors="coerce")
    invalid_numeric = original_nonmissing & sex.isna()
    if invalid_numeric.any():
        raise ValueError(
            "population.sex 有 "
            f"{int(invalid_numeric.sum()):,} 个无法转换的值。"
        )

    unknown_codes = sex.notna() & ~sex.isin([FEMALE_CODE, MALE_CODE])
    if unknown_codes.any():
        codes = sorted(sex.loc[unknown_codes].unique())
        raise ValueError(
            "population.sex 出现预期外编码："
            + ", ".join(str(code) for code in codes)
        )

    population["sex_code"] = sex.astype("Int64")
    population["sex_label"] = population["sex_code"].map(
        {FEMALE_CODE: "female", MALE_CODE: "male"}
    )
    return population.drop(columns=["sex"])


def score_component(
    intake: pd.Series, median: float, direction: str
) -> pd.Series:
    """按队列中位数规则计算一个二元食物分项。"""
    score = pd.Series(pd.NA, index=intake.index, dtype="Int64")
    observed = intake.notna()
    if direction == "beneficial":
        score.loc[observed] = intake.loc[observed].gt(median).astype("int64")
    elif direction == "adverse":
        score.loc[observed] = intake.loc[observed].le(median).astype("int64")
    else:
        raise ValueError(f"未知评分方向：{direction}")
    return score


def score_alcohol(
    sex_code: pd.Series,
    alcohol_complete: pd.Series,
    mean_daily_alcohol_g: pd.Series,
) -> pd.Series:
    """按性别特异范围计算 AMED 酒精二元分项。"""
    score = pd.Series(pd.NA, index=sex_code.index, dtype="Int64")
    eligible = (
        alcohol_complete.eq(True).fillna(False)
        & sex_code.isin([FEMALE_CODE, MALE_CODE])
        & mean_daily_alcohol_g.notna()
    )
    score.loc[eligible] = 0

    female = eligible & sex_code.eq(FEMALE_CODE)
    female_in_range = (
        mean_daily_alcohol_g.ge(FEMALE_ALCOHOL_RANGE[0])
        & mean_daily_alcohol_g.le(FEMALE_ALCOHOL_RANGE[1])
    )
    score.loc[female & female_in_range] = 1

    male = eligible & sex_code.eq(MALE_CODE)
    male_in_range = (
        mean_daily_alcohol_g.ge(MALE_ALCOHOL_RANGE[0])
        & mean_daily_alcohol_g.le(MALE_ALCOHOL_RANGE[1])
    )
    score.loc[male & male_in_range] = 1
    return score


def add_energy_adjusted_scores(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """先截尾 AMED 原始分，再做残差校正和样本 Z 标准化。"""
    adjusted = data.copy()
    adjusted["energy_complete"] = parse_boolean(
        adjusted["energy_complete"], "energy_complete"
    )

    original_energy_nonmissing = adjusted[
        "mean_daily_energy_kcal"
    ].notna()
    energy = pd.to_numeric(
        adjusted["mean_daily_energy_kcal"], errors="coerce"
    )
    invalid_energy_numeric = original_energy_nonmissing & energy.isna()
    if invalid_energy_numeric.any():
        raise ValueError(
            "mean_daily_energy_kcal 有 "
            f"{int(invalid_energy_numeric.sum()):,} 个无效数值。"
        )
    adjusted["mean_daily_energy_kcal"] = energy

    inconsistent_energy = adjusted["energy_complete"].eq(True).fillna(
        False
    ) & (energy.isna() | energy.le(0))
    if inconsistent_energy.any():
        raise ValueError(
            "energy_complete=True 但 mean_daily_energy_kcal 缺失或非正数："
            f"{int(inconsistent_energy.sum()):,} 人。"
        )

    raw_score = pd.to_numeric(
        adjusted["amed_total_score_0_8"], errors="coerce"
    )
    eligible = (
        adjusted["amed_score_complete"].eq(True).fillna(False)
        & adjusted["energy_complete"].eq(True).fillna(False)
        & raw_score.notna()
        & energy.gt(0)
    )
    eligible_count = int(eligible.sum())
    if eligible_count < 5:
        raise ValueError(
            "可用于 AMED 能量调整的参与者不足 5 人："
            f"{eligible_count:,} 人。"
        )

    x = energy.loc[eligible].astype("float64")
    y_raw = raw_score.loc[eligible].astype("float64")
    winsor_lower = float(y_raw.quantile(WINSOR_LOWER_QUANTILE))
    winsor_upper = float(y_raw.quantile(WINSOR_UPPER_QUANTILE))
    y_winsorized = y_raw.clip(lower=winsor_lower, upper=winsor_upper)

    winsorized = pd.Series(
        pd.NA, index=adjusted.index, dtype="Float64"
    )
    winsorized.loc[eligible] = y_winsorized.to_numpy()
    adjusted["amed_score_winsorized"] = winsorized

    mean_energy = float(x.mean())
    mean_raw_score = float(y_raw.mean())
    mean_winsorized_score = float(y_winsorized.mean())
    centered_energy = x - mean_energy
    denominator = float(centered_energy.pow(2).sum())
    if denominator <= 0:
        raise ValueError("总能量摄入没有变异，无法进行残差法调整。")

    slope = float(
        (
            centered_energy
            * (y_winsorized - mean_winsorized_score)
        ).sum()
        / denominator
    )
    intercept = mean_winsorized_score - slope * mean_energy

    energy_adjusted = pd.Series(
        pd.NA, index=adjusted.index, dtype="Float64"
    )
    energy_adjusted.loc[eligible] = (
        y_winsorized - slope * (x - mean_energy)
    ).to_numpy()
    adjusted["amed_energy_adjusted_score"] = energy_adjusted

    adjusted_mean = float(energy_adjusted.loc[eligible].mean())
    adjusted_sd = float(energy_adjusted.loc[eligible].std(ddof=1))
    if not adjusted_sd > 0:
        raise ValueError("能量调整后的 AMED 没有变异，无法进行 Z 标准化。")
    z_score = pd.Series(pd.NA, index=adjusted.index, dtype="Float64")
    z_score.loc[eligible] = (
        (energy_adjusted.loc[eligible] - adjusted_mean) / adjusted_sd
    ).to_numpy()
    adjusted["amed_energy_adjusted_score_z"] = z_score

    try:
        quintile_values = pd.qcut(
            energy_adjusted.loc[eligible],
            q=5,
            labels=[1, 2, 3, 4, 5],
        )
    except ValueError as error:
        raise ValueError(
            "AMED 能量调整分无法划分为五分位；请检查重复值。"
        ) from error
    quintile = pd.Series(pd.NA, index=adjusted.index, dtype="Int64")
    quintile.loc[eligible] = quintile_values.astype("int64").to_numpy()
    adjusted["amed_energy_adjusted_score_quintile"] = quintile

    metrics: Dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "energy_adjustment_method": (
            "winsorize raw AMED score at 0.5th/99.5th percentiles; "
            "residual method versus mean daily energy; residual plus "
            "mean winsorized score; standardize adjusted score to mean "
            "0 and sample SD 1"
        ),
        "energy_adjustment_method_source": (
            "analyst implementation based on HPP cited reference 71; "
            "exact HPP code unavailable"
        ),
        "energy_adjustment_eligible_count": eligible_count,
        "energy_adjustment_missing_count": int((~eligible).sum()),
        "winsor_lower_quantile": WINSOR_LOWER_QUANTILE,
        "winsor_upper_quantile": WINSOR_UPPER_QUANTILE,
        "winsor_lower_value": winsor_lower,
        "winsor_upper_value": winsor_upper,
        "winsorized_low_count": int(y_raw.lt(winsor_lower).sum()),
        "winsorized_high_count": int(y_raw.gt(winsor_upper).sum()),
        "energy_adjustment_mean_energy_kcal": mean_energy,
        "energy_adjustment_mean_raw_score": mean_raw_score,
        "energy_adjustment_mean_winsorized_score": (
            mean_winsorized_score
        ),
        "energy_adjustment_intercept": intercept,
        "energy_adjustment_slope_per_kcal": slope,
        "energy_adjusted_score_energy_correlation": float(
            energy_adjusted.loc[eligible].corr(x)
        ),
        "energy_adjusted_score_min": float(
            energy_adjusted.loc[eligible].min()
        ),
        "energy_adjusted_score_max": float(
            energy_adjusted.loc[eligible].max()
        ),
        "energy_adjusted_score_mean": adjusted_mean,
        "energy_adjusted_score_standard_deviation": adjusted_sd,
        "z_score_standard_deviation_definition": "sample SD, ddof=1",
        "standardized_energy_adjusted_score_mean": float(
            z_score.loc[eligible].mean()
        ),
        "standardized_energy_adjusted_score_sd": float(
            z_score.loc[eligible].std(ddof=1)
        ),
    }
    for value, count in (
        quintile.loc[eligible].value_counts().sort_index().items()
    ):
        metrics[f"energy_adjusted_score_quintile_{int(value)}_count"] = int(
            count
        )
    return adjusted, metrics


def calculate_amed_scores(
    data: pd.DataFrame,
    population: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """一次完成八项原始分、能量调整、Z 分数、五分位和 QC。"""
    data = data.copy()
    for column in ID_COLUMNS:
        data[column] = normalize_text(data[column])
    validate_input(data)

    score_columns = []
    threshold_rows: List[Dict[str, object]] = []
    qc_metrics: Dict[str, object] = {
        "participant_rows": len(data),
        "unique_participants": data["participant_id"].nunique(),
        "scoring_sample": "all rows in participant intake input",
        "food_intake_unit": "mean daily grams proxy",
        "food_component_count": len(COMPONENT_DEFINITIONS),
    }

    for component, intake_column, direction in COMPONENT_DEFINITIONS:
        original_nonmissing = data[intake_column].notna()
        intake = pd.to_numeric(data[intake_column], errors="coerce")
        invalid_numeric = original_nonmissing & intake.isna()
        if invalid_numeric.any():
            raise ValueError(
                f"{intake_column} 有 {int(invalid_numeric.sum()):,} 个无效数值。"
            )
        if intake.notna().sum() == 0:
            raise ValueError(f"{intake_column} 没有可用于计算中位数的值。")

        data[intake_column] = intake
        median = float(intake.median())
        score_column = f"amed_{component}_score"
        data[score_column] = score_component(intake, median, direction)
        score_columns.append(score_column)

        equal_to_median = int(intake.eq(median).sum())
        one_point_count = int(data[score_column].eq(1).sum())
        observed_count = int(intake.notna().sum())
        rule = (
            f"{intake_column} > median"
            if direction == "beneficial"
            else f"{intake_column} <= median"
        )
        threshold_rows.append(
            {
                "amed_component": component,
                "intake_column": intake_column,
                "intake_unit": "mean daily grams proxy",
                "direction": direction,
                "one_point_rule": rule,
                "population_median": median,
                "participant_count_used": observed_count,
                "missing_intake_count": int(intake.isna().sum()),
                "equal_to_median_count": equal_to_median,
                "one_point_count": one_point_count,
                "one_point_share": one_point_count / observed_count,
            }
        )
        qc_metrics[f"{component}_median"] = median
        qc_metrics[f"{component}_missing_intake_count"] = int(
            intake.isna().sum()
        )
        qc_metrics[f"{component}_equal_to_median_count"] = equal_to_median
        qc_metrics[f"{component}_one_point_count"] = one_point_count

    data["amed_food_components_complete"] = data[
        score_columns
    ].notna().all(axis=1)
    data["amed_food_score_0_7"] = data[score_columns].sum(
        axis=1, min_count=len(score_columns)
    ).astype("Int64")

    data["alcohol_complete"] = parse_boolean(
        data["alcohol_complete"], "alcohol_complete"
    )
    original_alcohol_nonmissing = data["mean_daily_alcohol_g"].notna()
    alcohol = pd.to_numeric(data["mean_daily_alcohol_g"], errors="coerce")
    invalid_alcohol = original_alcohol_nonmissing & alcohol.isna()
    if invalid_alcohol.any():
        raise ValueError(
            "mean_daily_alcohol_g 有 "
            f"{int(invalid_alcohol.sum()):,} 个无效数值。"
        )
    negative_alcohol = alcohol.notna() & alcohol.lt(0)
    if negative_alcohol.any():
        raise ValueError(
            "mean_daily_alcohol_g 有 "
            f"{int(negative_alcohol.sum()):,} 个负值。"
        )
    data["mean_daily_alcohol_g"] = alcohol

    population_clean = prepare_population(population)
    merged = data.merge(
        population_clean,
        on=MERGE_KEYS,
        how="left",
        validate="one_to_one",
    )
    merged["amed_alcohol_score"] = score_alcohol(
        merged["sex_code"],
        merged["alcohol_complete"],
        merged["mean_daily_alcohol_g"],
    )
    merged["amed_score_complete"] = (
        merged["amed_food_components_complete"].eq(True).fillna(False)
        & merged["amed_alcohol_score"].notna()
    )
    merged["amed_total_score_0_8"] = pd.Series(
        pd.NA, index=merged.index, dtype="Int64"
    )
    complete = merged["amed_score_complete"]
    merged.loc[complete, "amed_total_score_0_8"] = (
        merged.loc[complete, "amed_food_score_0_7"]
        + merged.loc[complete, "amed_alcohol_score"]
    ).astype("Int64")

    merged, energy_adjustment_metrics = add_energy_adjusted_scores(merged)

    qc_metrics.update(
        {
            "population_rows": len(population),
            "population_unique_participants": population[
                "participant_id"
            ].nunique(),
            "matched_sex_count": int(merged["sex_code"].notna().sum()),
            "missing_sex_count": int(merged["sex_code"].isna().sum()),
            "female_count": int(merged["sex_code"].eq(FEMALE_CODE).sum()),
            "male_count": int(merged["sex_code"].eq(MALE_CODE).sum()),
            "female_code": FEMALE_CODE,
            "male_code": MALE_CODE,
            "female_alcohol_one_point_range_g_day": "5-15 inclusive",
            "male_alcohol_one_point_range_g_day": "10-25 inclusive",
            "food_components_complete_count": int(
                merged["amed_food_components_complete"].sum()
            ),
            "alcohol_complete_count": int(
                merged["alcohol_complete"].eq(True).sum()
            ),
            "alcohol_incomplete_count": int(
                merged["alcohol_complete"].eq(False).sum()
            ),
            "alcohol_score_available_count": int(
                merged["amed_alcohol_score"].notna().sum()
            ),
            "alcohol_score_zero_count": int(
                merged["amed_alcohol_score"].eq(0).sum()
            ),
            "alcohol_score_one_count": int(
                merged["amed_alcohol_score"].eq(1).sum()
            ),
            "complete_amed_count": int(merged["amed_score_complete"].sum()),
            "incomplete_amed_count": int(
                (~merged["amed_score_complete"]).sum()
            ),
            **energy_adjustment_metrics,
        }
    )
    for score, count in (
        merged["amed_total_score_0_8"]
        .value_counts(dropna=False)
        .sort_index()
        .items()
    ):
        label = "missing" if pd.isna(score) else str(int(score))
        qc_metrics[f"amed_total_score_{label}_count"] = int(count)

    thresholds = pd.DataFrame(threshold_rows)
    qc = pd.DataFrame(
        {"metric": list(qc_metrics.keys()), "value": list(qc_metrics.values())}
    )
    leading_columns = [
        *MERGE_KEYS,
        "research_stage",
        "sex_code",
        "sex_label",
    ]
    remaining_columns = [
        column for column in merged.columns if column not in leading_columns
    ]
    scores = merged.loc[:, [*leading_columns, *remaining_columns]]
    return scores, thresholds, qc


def run(
    input_csv: Path,
    population_csv: Path,
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    """读取摄入量与人口学数据，计算并保存完整 AMED 评分。"""
    if not input_csv.is_file():
        raise FileNotFoundError(f"找不到参与者摄入量表：{input_csv}")
    if not population_csv.is_file():
        raise FileNotFoundError(f"找不到 population 表：{population_csv}")

    data = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        dtype={column: "string" for column in ID_COLUMNS},
        low_memory=False,
    )
    population = pd.read_csv(
        population_csv,
        encoding="utf-8-sig",
        dtype={column: "string" for column in MERGE_KEYS},
        low_memory=False,
    )
    scores, thresholds, qc = calculate_amed_scores(data, population)

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "amed_participant_scores.csv"
    threshold_path = output_dir / "amed_food_component_thresholds.csv"
    qc_path = output_dir / "amed_score_qc.csv"
    scores.to_csv(score_path, index=False, encoding="utf-8-sig")
    thresholds.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print("AMED 八项评分（七项食物 + 酒精）")
    print(f"脚本版本：{SCRIPT_VERSION}")
    print(thresholds.to_string(index=False))
    print("-" * 72)
    print(f"输入参与者：{len(scores):,} 人")
    print(f"成功匹配性别：{int(scores['sex_code'].notna().sum()):,} 人")
    print(
        "酒精分项可评分："
        f"{int(scores['amed_alcohol_score'].notna().sum()):,} 人"
    )
    print(f"八项完整：{int(scores['amed_score_complete'].sum()):,} 人")
    print("八项总分分布：")
    print(scores["amed_total_score_0_8"].value_counts().sort_index())
    print("能量调整：残差法（总分 ~ 参与者日均总能量）")
    print(
        "能量调整可用："
        f"{int(scores['amed_energy_adjusted_score'].notna().sum()):,} 人"
    )
    print(
        "最终 Z 分数（均值/样本SD）："
        f"{scores['amed_energy_adjusted_score_z'].mean():.6f}/"
        f"{scores['amed_energy_adjusted_score_z'].std(ddof=1):.6f}"
    )
    print("能量调整分五分位分布：")
    print(
        scores["amed_energy_adjusted_score_quintile"]
        .value_counts()
        .sort_index()
    )
    print("-" * 72)
    print("说明：酒精记录不完整者保留缺失，未按 0 分处理。")
    print(f"参与者评分：{score_path}")
    print(f"中位数阈值：{threshold_path}")
    print(f"QC 结果：{qc_path}")
    return score_path, threshold_path, qc_path


def parse_args():
    parser = ArgumentParser(description="计算完整 AMED 八项评分")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"参与者 AMED 摄入量表（默认：{DEFAULT_INPUT_CSV}）",
    )
    parser.add_argument(
        "--population",
        type=Path,
        default=DEFAULT_POPULATION_CSV,
        help=f"population CSV（默认：{DEFAULT_POPULATION_CSV}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.population, args.output_dir)
