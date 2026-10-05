#!/usr/bin/env python3
"""Read-only diagnosis of the 2026-10-04 window QC run. No refitting/filter changes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

VERSION = "2026.10.04.2"
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


def boundary_tables(q, meta, diary, config):
    """Test calendar truncation/index alignment without choosing a new window."""
    timezone = config["time"]["local_timezone"]
    proxy = q.loc[q.anchor_source.eq("FIRST_RECORDED_POINT_PROXY")].copy()
    # Only first-record anchors allow reconstruction of the raw last timestamp
    # from start + raw observed span. Never do this for declared plan anchors.
    first = pd.to_datetime(proxy.window_start, unit="s", utc=True).dt.tz_convert(timezone)
    last = pd.to_datetime(proxy.window_start + proxy.observed_span_days*86400,
                         unit="s", utc=True).dt.round("s").dt.tz_convert(timezone)
    end = last + pd.Timedelta(minutes=15)
    first_clock = first.dt.hour*60 + first.dt.minute + first.dt.second/60
    end_clock = end.dt.hour*60 + end.dt.minute + end.dt.second/60
    # Ignore sub-microsecond floating-point round-off in span reconstruction.
    proxy["first_date"] = first.dt.tz_localize(None).dt.normalize()
    proxy["last_date"] = last.dt.tz_localize(None).dt.normalize()
    out = [("F | CGM首点及末点后一采样间隔：当地零点检验", pd.DataFrame([dict(
        audited_connections=len(q), first_record_proxy_connections=len(proxy),
        start_in_first_15_minutes=int(first_clock.lt(15).sum()),
        last_plus_15min_in_first_15_minutes=int(end_clock.lt(15).sum()))]))]
    if meta is not None:
        fields = ["collection_timestamp", "connection_timestamp", "cgm_connection_first_date",
            "cgm_connection_last_date", "number_of_cgm_datapoints_available_for_the_connection",
            "number_of_cgm_days_available_for_the_connection", "number_of_cgm_datapoints_lost_in_qc",
            "percentage_of_cgm_datapoints_lost_in_qc"]
        out.append(("F | CGM上游字段是否实际存在（未显示点数比较不代表比较通过）",
                    pd.DataFrame([dict(field=c, present=c in meta) for c in fields])))
        _, matched = metadata_comparison(proxy, meta, config)
        stamp = "collection_timestamp" if "collection_timestamp" in matched else "connection_timestamp"
        if stamp in matched:
            aware = matched[stamp].fillna("").str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, case=False)
            options = {"format": "mixed"} if int(pd.__version__.split(".")[0]) >= 2 else {}
            parsed = pd.to_datetime(matched[stamp].where(aware), utc=True, errors="coerce", **options)
            raw_start = pd.to_datetime(matched.window_start, unit="s", utc=True)
            diff = (raw_start-parsed).dt.total_seconds()/60
            out.append(("F | 缓存首点减元数据时间戳：分钟差", modes(diff.round(5), "minutes", 8)))
            out.append(("F | 时间戳匹配分母", pd.DataFrame([dict(
                connections=len(matched), metadata_timestamp_parseable=int(parsed.notna().sum()),
                within_one_second=int(diff.abs().le(1/60).sum()))])))
        loss = "percentage_of_cgm_datapoints_lost_in_qc"
        if loss in matched:
            values = pd.to_numeric(matched[loss], errors="coerce")
            out.append(("F | 平台QC丢点字段原值分位数：不是1344分母缺失率",
                values.quantile([0,.25,.5,.75,1]).rename_axis("quantile").reset_index(name=loss)))

    required = ["participant_id", "cohort", "research_stage", "collection_date", "food_items", "logging_day"]
    out.append(("G | 饮食日期编号字段", pd.DataFrame([dict(field=c, present=c in diary) for c in required])))
    if not set(required).issubset(diary):
        return out
    d = diary.copy()
    for col in ["participant_id", "cohort", "research_stage"]:
        d[col] = normalize(d[col])
    d = d.loc[d.cohort.eq(config["cohort"]) & d.research_stage.eq(config["research_stage"])]
    d["date"] = pd.to_datetime(d.collection_date.astype(str).str[:10], format="%Y-%m-%d", errors="raise")
    require(not d.duplicated(["participant_id", "date"]).any(), "Duplicate scoped diary dates")
    d["day_number"] = pd.to_numeric(d.logging_day, errors="coerce")
    good = d.day_number.notna() & d.day_number.mod(1).eq(0) & d.day_number.between(-10000,10000)
    d["number_valid"] = good
    d["day_zero"] = d.date - pd.to_timedelta(d.day_number.where(good), unit="D")
    g = d.groupby("participant_id")
    anchors = g.agg(day_zero=("day_zero", "first"), origin_count=("day_zero", "nunique"),
        all_number_valid=("number_valid", "all"), rows=("date", "size"))
    anchors["consistent"] = anchors.origin_count.eq(1) & anchors.all_number_valid
    out.append(("G | 日期减logging_day是否给出固定起日（不是已验证佩戴起日）",
        pd.DataFrame([dict(scoped_people=len(anchors),
            consistent_people=int(anchors.consistent.sum()),
            inconsistent_or_invalid_people=int((~anchors.consistent).sum()),
            consistent_people_with_only_one_row=int((anchors.consistent & anchors.rows.eq(1)).sum()))])))
    # Inconsistent ranks/calendar indices are explicitly excluded from offset
    # summaries. No minimum-date fallback and no best-coverage window search.
    valid = anchors.loc[anchors.consistent]
    joined = proxy.merge(valid[["day_zero"]], left_on="participant_id", right_index=True,
                         how="inner", validate="many_to_one")
    out.append(("G | 编号起日与CGM边界匹配分母", pd.DataFrame([dict(
        first_record_proxy_connections=len(proxy), matched_consistent_connections=len(joined))])))
    for col in ["first_date", "last_date"]:
        offset = (joined[col]-joined.day_zero).dt.days
        out.append(("G | CGM {} 减饮食编号0起日：天数".format(col), modes(offset, "days", 8)))
    positive = d.loc[d.participant_id.isin(valid.index) & pd.to_numeric(d.food_items,errors="coerce").gt(0)]
    in14 = positive.loc[positive.day_number.between(0,13)]
    n14 = in14.groupby("participant_id").date.nunique().reindex(valid.index,fill_value=0)
    out.append(("G | 编号0至13范围内日表覆盖（不是新的纳入人数）", pd.DataFrame([dict(
        consistent_people=len(valid), day_metadata_complete_14=int(n14.eq(14).sum()),
        at_least_13=int(n14.ge(13).sum()), at_least_12=int(n14.ge(12).sum()))])))
    byday = positive.groupby("day_number").participant_id.nunique().reindex(range(-1,15),fill_value=0)
    out.append(("G | 正值饮食日表按logging_day分布", byday.rename_axis("logging_day").reset_index(name="people")))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data-root", default="..")
    parser.add_argument("--boundaries-only", action="store_true", help="Inspect timestamps and logging_day without repeating A-E")
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
    if args.boundaries_only:
        meta, diary = source("cgm_metadata"), source("diary")
        require(diary is not None, "Diary source unavailable")
        for title, frame in boundary_tables(q, meta, diary, config):
            table(title, frame)
        for path, digest in checked.items():
            require(sha256(path) == digest, "A source changed while diagnosing")
        print("\nBOUNDARY_DIAGNOSIS_COMPLETE; all read sources unchanged; aggregate output only.")
        print("Only field consistency is assessed; no activation date, revised denominator, or new model is asserted.")
        return 0
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
