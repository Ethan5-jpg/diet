"""计算 AMED 七项食物分、性别特异酒精分和八项总分。"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


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

FEMALE_CODE = 0
MALE_CODE = 1
FEMALE_ALCOHOL_RANGE = (5.0, 15.0)
MALE_ALCOHOL_RANGE = (10.0, 25.0)


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
    required = [*ID_COLUMNS, *intake_columns, *ALCOHOL_COLUMNS]
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


def calculate_amed_scores(
    data: pd.DataFrame,
    population: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """一次完成七项食物分、酒精分、八项总分和 QC。"""
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
