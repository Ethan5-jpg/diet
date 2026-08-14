"""完整扫描 Data/Transfer 中可能支持 AHEI 的 food_id lookup。

本脚本不重读饮食事件，不计算分数。它复用第00步的表头检查函数，并允许
无限扫描，以解决第00步达到候选文件上限后仍不能排除遗漏 lookup 的问题。
"""

from argparse import ArgumentParser
import importlib.util
from pathlib import Path
import sys
from typing import Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-09-ahei-lookup-scan-v1"

SCRIPT_DIR = Path(__file__).resolve().parent
AUDITOR_PATH = SCRIPT_DIR / "00_audit_ahei_inputs.py"
SPEC = importlib.util.spec_from_file_location("ahei_input_auditor", AUDITOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载 AHEI 输入审计模块：{}".format(AUDITOR_PATH))
AUDITOR = importlib.util.module_from_spec(SPEC)
sys.modules["ahei_input_auditor"] = AUDITOR
SPEC.loader.exec_module(AUDITOR)

DEFAULT_TRANSFER_DIR = AUDITOR.DEFAULT_TRANSFER_DIR
DEFAULT_OUTPUT_DIR = AUDITOR.DEFAULT_OUTPUT_DIR
DEFAULT_EVENTS_CSV = AUDITOR.DEFAULT_EVENTS_CSV
DEFAULT_RAW_EVENTS_CSV = AUDITOR.DEFAULT_RAW_EVENTS_CSV
DEFAULT_POPULATION_CSV = AUDITOR.DEFAULT_POPULATION_CSV


def scan_lookup_sources(
    transfer_dir: Path,
    output_dir: Path,
    file_limit: int,
    progress_every: int,
) -> Tuple[Path, Path]:
    """扫描候选 lookup，并保存完整候选表和扫描 QC。"""
    candidates, scanned_table_count, limit_reached = AUDITOR.scan_candidate_sources(
        transfer_dir,
        {
            DEFAULT_EVENTS_CSV,
            DEFAULT_RAW_EVENTS_CSV,
            DEFAULT_POPULATION_CSV,
        },
        file_limit,
        progress_every=progress_every,
    )

    if candidates.empty:
        likely_count = 0
        keyed_count = 0
        keyword_count = 0
        header_error_count = 0
    else:
        likely_count = int(candidates["likely_ahei_lookup"].eq(True).sum())
        keyed_count = int(candidates["has_food_id"].eq(True).sum())
        keyword_count = int(
            candidates["matched_keyword_groups"].fillna("").ne("").sum()
        )
        header_error_count = int(candidates["header_status"].eq("error").sum())

    qc = pd.DataFrame(
        [
            ("script_version", SCRIPT_VERSION),
            ("transfer_dir", str(transfer_dir)),
            ("requested_file_limit", file_limit),
            ("unlimited_scan_requested", file_limit == 0),
            ("scanned_table_count", scanned_table_count),
            ("file_limit_reached", limit_reached),
            ("candidate_rows_written", len(candidates)),
            ("tables_with_food_id", keyed_count),
            ("tables_with_ahei_keyword_columns", keyword_count),
            ("likely_ahei_lookup_count", likely_count),
            ("header_error_count", header_error_count),
            (
                "scan_complete_without_limit",
                file_limit == 0 and not limit_reached,
            ),
        ],
        columns=["metric", "value"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "ahei_candidate_data_sources.csv"
    qc_path = output_dir / "ahei_lookup_scan_qc.csv"
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")

    print("=" * 76)
    print("AHEI food_id lookup 完整扫描")
    print("SCRIPT_VERSION={}".format(SCRIPT_VERSION))
    print("Data/Transfer：{}".format(transfer_dir))
    print(
        "扫描表数：{:,}；达到上限：{}；表头错误：{:,}".format(
            scanned_table_count, limit_reached, header_error_count
        )
    )
    print(
        "含 food_id：{:,}；命中 AHEI 字段：{:,}；严格候选 lookup：{:,}".format(
            keyed_count, keyword_count, likely_count
        )
    )
    print(
        "完整无限扫描：{}".format(file_limit == 0 and not limit_reached)
    )

    if likely_count:
        print("\n可按 food_id 连接且命中 AHEI 字段的候选表：")
        print(
            candidates.loc[
                candidates["likely_ahei_lookup"].eq(True),
                ["path", "matched_keyword_groups", "matched_columns"],
            ].to_string(index=False)
        )
    else:
        print("\n未发现同时含 food_id 和 AHEI 所需字段的候选 lookup。")

    if keyword_count:
        keyword_only = candidates.loc[
            candidates["matched_keyword_groups"].fillna("").ne("")
            & ~candidates["likely_ahei_lookup"].eq(True),
            [
                "path",
                "has_food_id",
                "matched_keyword_groups",
                "matched_columns",
            ],
        ]
        if not keyword_only.empty:
            print("\n命中字段但尚不能按 food_id 连接的表（前30）：")
            print(keyword_only.head(30).to_string(index=False))

    if header_error_count:
        print("\n无法读取表头的文件（前20）：")
        print(
            candidates.loc[
                candidates["header_status"].eq("error"),
                ["path", "scan_error"],
            ].head(20).to_string(index=False)
        )

    print("\n候选表：{}".format(candidate_path))
    print("扫描 QC：{}".format(qc_path))
    print("=" * 76)
    return candidate_path, qc_path


def parse_args():
    parser = ArgumentParser(
        description="完整扫描 Data/Transfer 中可能支持 AHEI 的 food_id lookup"
    )
    parser.add_argument(
        "--transfer-dir", type=Path, default=DEFAULT_TRANSFER_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--file-limit",
        type=int,
        default=0,
        help="最多扫描的表数；0表示不设上限（默认：0）",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="每扫描多少个表打印一次进度；0表示关闭（默认：1000）",
    )
    args = parser.parse_args()
    if args.file_limit < 0:
        parser.error("--file-limit 不能为负数")
    if args.progress_every < 0:
        parser.error("--progress-every 不能为负数")
    return args


def main() -> None:
    args = parse_args()
    scan_lookup_sources(
        transfer_dir=args.transfer_dir,
        output_dir=args.output_dir,
        file_limit=args.file_limit,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
