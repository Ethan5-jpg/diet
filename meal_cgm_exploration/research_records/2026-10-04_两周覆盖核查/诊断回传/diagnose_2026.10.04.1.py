#!/usr/bin/env python3
"""Read-only diagnosis of the 2026-10-04 window QC run. No refitting/filter changes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

VERSION = "2026.10.04.1"
KEYS = ["participant_id", "connection_id"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize(series):
    return series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def flag(series):
    return series.astype(str).str.lower().eq("true")


def table(title, frame):
    print("\n" + title)
    print(frame.to_string(index=False, float_format=lambda x: "{:.5g}".format(x)))


def modes(series, label, n=6):
    return series.dropna().value_counts().head(n).rename_axis(label).reset_index(name="connections")


def missing_partition(quality):
    d = quality.loc[quality.status.eq("AUDITED")].copy()
    fields = ["leading_missing_slots", "trailing_missing_slots", "internal_missing_slots", "whole_window_missing_slots"]
    require(d[fields].notna().all().all(), "Missing partition fields unavailable")
    require(d[fields].sum(axis=1).eq(d.missing_slots).all(), "Missing partition identity failed")
    total = d.missing_slots.sum()
    partition = pd.DataFrame([
        dict(location=col, missing_slots=int(d[col].sum()),
             share_of_all_missing_slots=d[col].sum()/total if total else float("nan"))
        for col in fields])
    return d, partition


def day_profile(days):
    d = days.copy()
    dates = pd.to_datetime(d.local_date, format="%Y-%m-%d", errors="raise")
    first = dates.groupby([d.participant_id, d.connection_id]).transform("min")
    d["day_index"] = (dates-first).dt.days + 1
    require(not d.duplicated(KEYS + ["day_index"]).any(), "Duplicate diet connection-date")
    counts = d.groupby(KEYS).day_index.agg(["min", "max", "size"])
    require((counts["min"].eq(1) & counts["max"].eq(14) & counts["size"].eq(14)).all(), "Expected exactly 14 dates per audited connection")
    d["recorded_bool"] = flag(d.recorded)
    d["metadata_positive"] = pd.to_numeric(d.metadata_items, errors="coerce").gt(0)
    d["events_positive"] = pd.to_numeric(d.event_count, errors="coerce").gt(0)
    d["timezone_problem_bool"] = flag(d.timezone_problem)
    out = d.groupby("day_index").agg(connections=("recorded_bool", "size"),
        recorded=("recorded_bool", "sum"), metadata_positive=("metadata_positive", "sum"),
        events_positive=("events_positive", "sum"), timezone_problem=("timezone_problem_bool", "sum")).reset_index()
    out["recorded_fraction"] = out.recorded/out.connections
    return out


def metadata_comparison(q, meta, config):
    scope = ["participant_id", "cohort", "research_stage", "connection_id"]
    require(set(scope).issubset(meta), "CGM metadata identity fields unavailable")
    m = meta.copy()
    for col in scope:
        m[col] = normalize(m[col])
    m = m.loc[m.cohort.eq(config["cohort"]) & m.research_stage.eq(config["research_stage"])]
    require(not m.duplicated(KEYS).any(), "Duplicate scoped CGM metadata keys")
    d = q.merge(m, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_metadata"), indicator=True)
    rows = [dict(metric="audited_connections", value=len(q)),
            dict(metric="matched_metadata_connections", value=int(d._merge.eq("both").sum()))]
    field = "number_of_cgm_datapoints_available_for_the_connection"
    if field in d:
        available = pd.to_numeric(d[field], errors="coerce")
        known = available.notna()
        rows.extend([dict(metric="metadata_available_points_known", value=int(known.sum())),
            dict(metric="cached_raw_rows_equal_metadata_points", value=int((known & d.raw_rows.eq(available)).sum())),
            dict(metric="cached_raw_rows_less_than_metadata_points", value=int((known & d.raw_rows.lt(available)).sum())),
            dict(metric="cached_raw_rows_more_than_metadata_points", value=int((known & d.raw_rows.gt(available)).sum()))])
    return pd.DataFrame(rows), d


def diary_offsets(q, diary, config):
    d = diary.copy()
    for col in ["participant_id", "cohort", "research_stage"]:
        d[col] = normalize(d[col])
    d = d.loc[d.cohort.eq(config["cohort"]) & d.research_stage.eq(config["research_stage"])]
    d["date"] = pd.to_datetime(d.collection_date.astype(str).str[:10], format="%Y-%m-%d", errors="raise")
    require(not d.duplicated(["participant_id", "date"]).any(), "Duplicate scoped diary dates")
    d["positive"] = pd.to_numeric(d.food_items, errors="coerce").gt(0)
    # Multiple connections deliberately get separate comparisons; denominator is connections.
    joined = q[KEYS + ["diet_first_date"]].merge(d[["participant_id", "date", "positive"]], on="participant_id", how="left")
    joined["offset"] = (joined.date-pd.to_datetime(joined.diet_first_date)).dt.days
    counts = joined.loc[joined.positive.eq(True)].groupby("offset").size()
    out = pd.DataFrame({"day_offset": list(range(-3, 17))})
    out["connections_with_positive_diary"] = out.day_offset.map(counts).fillna(0).astype(int)
    out["audited_connections"] = len(q)
    out["fraction"] = out.connections_with_positive_diary/len(q) if len(q) else float("nan")
    totals = d.groupby("participant_id").positive.sum()
    return out, pd.DataFrame([dict(scoped_diary_people=len(totals),
        people_with_at_least_14_positive_dates_anywhere_in_visit=int(totals.ge(14).sum()))])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data-root", default="..")
    args = parser.parse_args(argv)
    root, run = Path(args.data_root).expanduser().resolve(), Path(args.run).expanduser().resolve()
    manifest = json.loads((run/"manifest.json").read_text(encoding="utf-8"))
    state = manifest["status"]["window_qc_followup"]
    require(state["status"] in ["PARTIAL", "COMPLETE_WITH_LIMITATIONS", "AUDIT_COMPLETE"], "Run has not finished coverage analysis")
    require(manifest["data_kind"] in ["REAL", "SYNTHETIC"], "Unverified data kind")
    hashes = state["result_sha256"]
    checked = {}

    def result(relative, csv=True):
        path = run/relative
        require(relative in hashes and path.is_file() and sha256(path) == hashes[relative], "Missing/changed result: " + relative)
        checked[path] = hashes[relative]
        if not csv:
            return json.loads(path.read_text(encoding="utf-8"))
        return pd.read_csv(path, dtype={"participant_id": str, "connection_id": str})

    spec = result("analysis_spec.json", csv=False)
    config = manifest["config"]
    old_root = Path(manifest["root"])

    def relocate(value):
        path = Path(value)
        try:
            return root/path.relative_to(old_root)
        except ValueError:
            return path

    pinned = {str(relocate(x["path"]).resolve()): x["sha256"] for x in manifest["inputs"]}

    def source(label):
        value = spec["sources"].get(label)
        if value is None:
            return None
        path = relocate(value).resolve()
        require(str(path) in pinned and path.is_file() and sha256(path) == pinned[str(path)], "Missing/changed source: " + label)
        checked[path] = pinned[str(path)]
        return pd.read_csv(path, dtype=str)

    print("WINDOW_QC_DIAGNOSIS=" + VERSION)
    print("RUN={} DATA={} STATUS={}".format(run.name, manifest["data_kind"], state["status"]))
    print("READ_ONLY; fixed 14-day thresholds and original models are unchanged.")
    quality = result("private/connection_quality.csv")
    q, partition = missing_partition(quality)
    table("A | 缺口位置（连接加权；不是受试者漏测原因）", partition)
    for field in ["valid_slots", "raw_rows", "trailing_missing_slots", "internal_missing_slots"]:
        table("A | 最常见 " + field, modes(q[field], field))
    table("A | 校验指标", pd.DataFrame([dict(
        audited_connections=len(q), valid_slots_1248=int(q.valid_slots.eq(1248).sum()),
        tail_96_and_no_other_missing=int((q.trailing_missing_slots.eq(96) & q.missing_slots.eq(96)).sum()),
        raw_span_near_13_days=int(q.observed_span_days.between(12.95,13.05).sum()),
        off_grid_rows=int(q.off_grid_or_unassignable_rows.sum()),
        conflict_slots=int(q.conflicting_slots.sum()),
        rows_after_336h=int(q.rows_after_window.sum()))]))
    table("B | 饮食第1至14日位置诊断", day_profile(result("private/diet_window_days.csv")))
    meta = source("cgm_metadata")
    if meta is not None:
        comparison, matched = metadata_comparison(q, meta, config)
        table("C | 原缓存行数与上游元数据（字段语义差异仍需核实）", comparison)
        for field in ["number_of_cgm_days_available_for_the_connection", "number_of_cgm_datapoints_available_for_the_connection", "number_of_cgm_datapoints_lost_in_qc"]:
            if field in matched:
                table("C | 最常见上游 " + field, modes(pd.to_numeric(matched[field], errors="coerce"), field))
    else:
        print("C | CGM metadata unavailable in source run")
    diary = source("diary")
    require(diary is not None, "Diary source unavailable")
    offsets, totals = diary_offsets(q, diary, config)
    table("D | 全访视饮食日表（仅metadata，不能替代事件核对）", totals)
    table("D | 相对默认饮食起日偏移；0至13为原窗口", offsets)
    for name in ["diet_models.csv", "gut_models.csv"]:
        if "tables/"+name in hashes:
            models = result("tables/"+name)
            table("E | 模型完成情况 " + name, models.groupby(["setting", "status"]).size().reset_index(name="models"))
    for path, digest in checked.items():
        require(sha256(path) == digest, "A source changed while diagnosing")
    print("\nDIAGNOSIS_COMPLETE; all read sources unchanged; aggregate output only.")
    print("这些诊断不能单独证明上游删去第一天；需结合提取代码或数据提供方说明。")
    print("不据此自动改成13天，不按显著性选阈值，不从首末观测缩短主分母。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("DIAGNOSIS_FAILED: " + str(error), file=sys.stderr)
        sys.exit(2)
