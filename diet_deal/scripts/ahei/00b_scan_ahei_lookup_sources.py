"""Perform an unlimited, header-only scan for optional AHEI lookup sources."""

import importlib.util
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import pandas as pd


SCRIPT_VERSION = "2026-08-18-ahei-lookup-scan-v3"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR.parent
DEFAULT_TRANSFER_DIR = DATA_DIR / "Transfer"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "00_input_audit" / "ahei_lookup_scan"

AUDITOR_PATH = SCRIPT_DIR / "00_audit_ahei_inputs.py"
SPEC = importlib.util.spec_from_file_location("ahei_input_auditor_for_lookup", AUDITOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载AHEI输入审计模块：{}".format(AUDITOR_PATH))
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def _requirement_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (
            "serving_or_portion",
            AUDITOR.candidate_count_for_group(candidates, "serving_or_portion"),
            "not_required_hpp_proxy_conversion_frozen",
            "HPP kcal/g serving proxies are frozen; candidates are sensitivity-only.",
        ),
        (
            "whole_grain",
            AUDITOR.candidate_count_for_group(candidates, "whole_grain"),
            "optional_strict_source_proxy_protocol_approved",
            "Mapped food weight is the approved proxy; a strict source would support sensitivity analysis.",
        ),
        (
            "trans_fat",
            AUDITOR.candidate_count_for_group(candidates, "trans_fat"),
            "not_required_excluded_by_user_protocol",
            "Excluded from mAHEI-7; discovered sources remain informational only.",
        ),
        (
            "pufa",
            AUDITOR.candidate_count_for_group(candidates, "pufa"),
            "not_required_excluded_by_user_protocol",
            "Excluded from mAHEI-7; discovered sources remain informational only.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "requirement",
            "candidate_table_count",
            "formal_ahei_status",
            "interpretation",
        ],
    )


def scan_lookup_sources(
    transfer_dir: Path = DEFAULT_TRANSFER_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    file_limit: int = 0,
    progress_every: int = 500,
) -> Tuple[Path, Path, Path]:
    """Scan table headers and write candidates, scan QC, and requirement status."""
    del progress_every
    candidates, scanned, limit_reached = AUDITOR.scan_candidate_sources(
        Path(transfer_dir), set(), file_limit
    )
    summary = _requirement_summary(candidates)
    errors = 0
    if not candidates.empty:
        errors = int(candidates["header_error"].fillna("").ne("").sum())
    qc = pd.DataFrame(
        [
            ("script_version", SCRIPT_VERSION),
            ("transfer_dir", str(transfer_dir)),
            ("requested_file_limit", file_limit),
            ("scanned_file_count", scanned),
            ("file_limit_reached", limit_reached),
            ("scan_complete_without_limit", file_limit == 0 and not limit_reached),
            ("header_error_count", errors),
            (
                "likely_candidate_count",
                int(candidates["likely_ahei_lookup"].fillna(False).astype(bool).sum())
                if not candidates.empty
                else 0,
            ),
        ],
        columns=["metric", "value"],
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "ahei_lookup_candidates.csv"
    qc_path = output_dir / "ahei_lookup_scan_qc.csv"
    summary_path = output_dir / "ahei_lookup_requirement_summary.csv"
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return candidate_path, qc_path, summary_path


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-dir", type=Path, default=DEFAULT_TRANSFER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--file-limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    for output_path in scan_lookup_sources(
        transfer_dir=arguments.transfer_dir,
        output_dir=arguments.output_dir,
        file_limit=arguments.file_limit,
        progress_every=arguments.progress_every,
    ):
        print(output_path)
