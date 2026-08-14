"""审计 hPDI 缺重量事件在正向与反向计分组件中的分布。"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-14-hpdi-missing-weight-direction-audit-v1"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_CSV = (
    PROJECT_DIR
    / "outputs"
    / "04_score_intakes"
    / "hpdi"
    / "hpdi_participant_component_intakes.csv"
)
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


def missing_weight_column(component: str) -> str:
    return "total_{}_missing_weight_events".format(component)


def component_event_column(component: str) -> str:
    return "total_{}_events".format(component)


def validate_columns(columns: List[str], required: List[str]) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError("hPDI 参与者组件表缺少字段：{}".format(", ".join(missing)))


def prepare_participant_table(participants: pd.DataFrame) -> pd.DataFrame:
    """校验并规范各组件事件数；每一行必须对应唯一参与者。"""
    components = [component for component, _ in COMPONENT_DEFINITIONS]
    count_columns = []  # type: List[str]
    for component in components:
        count_columns.extend(
            [missing_weight_column(component), component_event_column(component)]
        )
    validate_columns(participants.columns.tolist(), ID_COLUMNS + count_columns)

    prepared = participants.copy()
    if prepared[ID_COLUMNS].isna().any().any():
        raise ValueError("参与者标识字段存在缺失。")
    if prepared.duplicated(ID_COLUMNS).any():
        raise ValueError("参与者组件表存在重复参与者记录。")

    for column in count_columns:
        numeric = pd.to_numeric(prepared[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError("{} 含缺失或非数值。".format(column))
        if numeric.lt(0).any():
            raise ValueError("{} 含负数。".format(column))
        if (numeric % 1).ne(0).any():
            raise ValueError("{} 含非整数事件数。".format(column))
        prepared[column] = numeric.astype("int64")
    return prepared


def build_component_summary(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []  # type: List[Dict[str, object]]
    for component, direction in COMPONENT_DEFINITIONS:
        missing_column = missing_weight_column(component)
        event_column = component_event_column(component)
        missing_count = int(participants[missing_column].sum())
        event_count = int(participants[event_column].sum())
        rows.append(
            {
                "hpdi_component": component,
                "scoring_direction": direction,
                "missing_weight_event_count": missing_count,
                "affected_participant_count": int(
                    participants[missing_column].gt(0).sum()
                ),
                "component_event_count": event_count,
                "missing_weight_event_share": (
                    missing_count / event_count if event_count else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def direction_masks(participants: pd.DataFrame) -> Dict[str, pd.Series]:
    masks = {}  # type: Dict[str, pd.Series]
    for direction in ["positive", "reverse"]:
        columns = [
            missing_weight_column(component)
            for component, item_direction in COMPONENT_DEFINITIONS
            if item_direction == direction
        ]
        masks[direction] = participants[columns].sum(axis=1).gt(0)
    return masks


def build_direction_summary(
    participants: pd.DataFrame,
    component_summary: pd.DataFrame,
) -> pd.DataFrame:
    masks = direction_masks(participants)
    rows = []  # type: List[Dict[str, object]]
    for direction in ["positive", "reverse"]:
        selected = component_summary["scoring_direction"].eq(direction)
        rows.append(
            {
                "scoring_direction": direction,
                "component_count": int(selected.sum()),
                "missing_weight_event_count": int(
                    component_summary.loc[
                        selected, "missing_weight_event_count"
                    ].sum()
                ),
                "affected_participant_count": int(masks[direction].sum()),
                "participant_component_incidence_count": int(
                    component_summary.loc[
                        selected, "affected_participant_count"
                    ].sum()
                ),
                "component_event_count": int(
                    component_summary.loc[selected, "component_event_count"].sum()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["missing_weight_event_share"] = (
        summary["missing_weight_event_count"]
        / summary["component_event_count"].replace(0, pd.NA)
    ).fillna(0.0)
    return summary


def build_participant_pattern_summary(
    participants: pd.DataFrame,
) -> pd.DataFrame:
    masks = direction_masks(participants)
    positive = masks["positive"]
    reverse = masks["reverse"]
    rows = [
        ("positive_only", positive & ~reverse),
        ("reverse_only", ~positive & reverse),
        ("both_directions", positive & reverse),
        ("any_missing_weight", positive | reverse),
        ("no_missing_weight", ~positive & ~reverse),
    ]  # type: List[Tuple[str, pd.Series]]
    return pd.DataFrame(
        [
            {
                "participant_pattern": label,
                "participant_count": int(mask.sum()),
                "participant_share": float(mask.mean()) if len(mask) else 0.0,
            }
            for label, mask in rows
        ]
    )


def audit_missing_weight_direction(
    participants: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared = prepare_participant_table(participants)
    component_summary = build_component_summary(prepared)
    direction_summary = build_direction_summary(prepared, component_summary)
    participant_pattern_summary = build_participant_pattern_summary(prepared)
    return component_summary, direction_summary, participant_pattern_summary


def print_summary(
    input_csv: Path,
    participant_count: int,
    component_summary: pd.DataFrame,
    direction_summary: pd.DataFrame,
    participant_pattern_summary: pd.DataFrame,
) -> None:
    positive_events = int(
        direction_summary.loc[
            direction_summary["scoring_direction"].eq("positive"),
            "missing_weight_event_count",
        ].iloc[0]
    )
    reverse_events = int(
        direction_summary.loc[
            direction_summary["scoring_direction"].eq("reverse"),
            "missing_weight_event_count",
        ].iloc[0]
    )
    maximum = max(positive_events, reverse_events)
    relative_difference = (
        abs(positive_events - reverse_events) / maximum if maximum else 0.0
    )

    print("脚本版本：{}".format(SCRIPT_VERSION))
    print("输入：{}".format(input_csv))
    print("参与者：{:,} 人".format(participant_count))
    print("\n按计分方向汇总")
    print(direction_summary.to_string(index=False))
    print("\n按参与者正反向重叠汇总")
    print(participant_pattern_summary.to_string(index=False))
    print("\n按组件汇总")
    print(component_summary.to_string(index=False))
    print(
        "\n正反向缺重量事件数绝对差：{:,}".format(
            abs(positive_events - reverse_events)
        )
    )
    print("正反向相对差（相对于较大者）：{:.1%}".format(relative_difference))
    print(
        "注意：即使事件总数接近，也不能据此认定五分位计分偏差会互相抵消；"
        "还需结合参与者重叠、组件分布和可能重量大小。"
    )


def parse_args():
    parser = ArgumentParser(description="审计 hPDI 缺重量事件的正反向分布")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError("找不到 hPDI 参与者组件表：{}".format(args.input))
    input_table = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    component_result, direction_result, pattern_result = (
        audit_missing_weight_direction(input_table)
    )
    print_summary(
        args.input,
        len(input_table),
        component_result,
        direction_result,
        pattern_result,
    )
