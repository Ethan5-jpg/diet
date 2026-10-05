import json
from pathlib import Path

import pandas as pd
import pytest

from window_qc_diagnosis.diagnose import (day_profile, diary_offsets, main,
    metadata_comparison, missing_partition, sha256, boundary_tables)


def fixture(tmp_path):
    run = tmp_path/"outputs"/"qc_synthetic"
    (run/"private").mkdir(parents=True)
    (run/"tables").mkdir()
    scope = dict(cohort="10k", research_stage="00_00_visit")
    config = dict(scope, time=dict(local_timezone="UTC"))
    rows = []
    for i in range(2):
        rows.append(dict(participant_id="person_private_"+str(i), connection_id=str(i), status="AUDITED",
            raw_rows=1248-i, valid_slots=1248-i, missing_slots=96+i,
            leading_missing_slots=0, trailing_missing_slots=96, internal_missing_slots=i,
            whole_window_missing_slots=0, observed_span_days=1247/96,
            off_grid_or_unassignable_rows=0, conflicting_slots=0, rows_after_window=0,
            diet_first_date="2024-01-02", anchor_source="FIRST_RECORDED_POINT_PROXY",
            window_start=pd.Timestamp("2024-01-02",tz="UTC").timestamp()))
    quality = pd.DataFrame(rows)
    quality.to_csv(run/"private/connection_quality.csv", index=False)
    days, diary = [], []
    for i in range(2):
        for offset in range(14):
            date = str((pd.Timestamp("2024-01-02")+pd.Timedelta(days=offset)).date())
            good = offset < 13 and not (i == 1 and offset == 6)
            days.append(dict(participant_id="person_private_"+str(i), connection_id=str(i), local_date=date,
                recorded=good, metadata_items=int(good), event_count=int(good), timezone_problem=False))
        for offset in range(-1, 13):
            date = str((pd.Timestamp("2024-01-02")+pd.Timedelta(days=offset)).date())
            diary.append(dict(participant_id="person_private_"+str(i), collection_date=date, food_items=1,
                              logging_day=offset+1, **scope))
    days, diary = pd.DataFrame(days), pd.DataFrame(diary)
    days.to_csv(run/"private/diet_window_days.csv", index=False)
    diary.to_csv(tmp_path/"diary.csv", index=False)
    meta = quality[["participant_id", "connection_id"]].assign(**scope,
        collection_timestamp="2024-01-02T00:00:00+00:00",
        number_of_cgm_datapoints_available_for_the_connection=[1248,1248],
        number_of_cgm_days_available_for_the_connection=13, number_of_cgm_datapoints_lost_in_qc=96)
    meta.to_csv(tmp_path/"meta.csv", index=False)
    sources = dict(diary=str(tmp_path/"diary.csv"), cgm_metadata=str(tmp_path/"meta.csv"))
    (run/"analysis_spec.json").write_text(json.dumps(dict(sources=sources)))
    pd.DataFrame([dict(setting="qc_primary", status="NOT_ESTIMATED"),
        dict(setting="qc10", status="ESTIMATED")]).to_csv(run/"tables/diet_models.csv", index=False)
    hashes = {str(path.relative_to(run)):sha256(path) for path in run.rglob("*") if path.is_file()}
    body = dict(root=str(tmp_path), config=config, data_kind="SYNTHETIC",
        inputs=[dict(path=value,sha256=sha256(value)) for value in sources.values()],
        status=dict(window_qc_followup=dict(status="PARTIAL",result_sha256=hashes)))
    (run/"manifest.json").write_text(json.dumps(body))
    return run, quality, days, diary, meta, config


def test_partition_detects_tail_dominance_and_keeps_internal(tmp_path):
    _, q, _, _, _, _ = fixture(tmp_path)
    audited, out = missing_partition(q)
    assert len(audited) == 2
    rows = out.set_index("location")
    assert rows.loc["trailing_missing_slots", "missing_slots"] == 192
    assert rows.loc["internal_missing_slots", "missing_slots"] == 1
    q.loc[0,"missing_slots"] += 1
    with pytest.raises(ValueError, match="identity"):
        missing_partition(q)


def test_day_positions_distinguish_last_day_and_internal(tmp_path):
    _, _, days, _, _, _ = fixture(tmp_path)
    out = day_profile(days).set_index("day_index")
    assert out.loc[14,"recorded"] == 0
    assert out.loc[7,"recorded"] == 1
    assert out.loc[1,"recorded_fraction"] == 1
    with pytest.raises(ValueError, match="14 dates"):
        day_profile(days.iloc[1:])


def test_metadata_counts_and_duplicate_fail(tmp_path):
    _, q, _, _, meta, config = fixture(tmp_path)
    counts, _ = metadata_comparison(q,meta,config)
    counts = counts.set_index("metric").value
    assert counts.cached_raw_rows_equal_metadata_points == 1
    assert counts.cached_raw_rows_less_than_metadata_points == 1
    with pytest.raises(ValueError, match="Duplicate"):
        metadata_comparison(q,pd.concat([meta,meta]),config)


def test_diary_offsets_reveal_reference_shift(tmp_path):
    _, q, _, diary, _, config = fixture(tmp_path)
    offsets, totals = diary_offsets(q,diary,config)
    assert totals.iloc[0].people_with_at_least_14_positive_dates_anywhere_in_visit == 2
    offsets = offsets.set_index("day_offset")
    assert offsets.loc[-1,"fraction"] == 1
    assert offsets.loc[13,"fraction"] == 0


def test_end_to_end_read_only_and_no_identifiers(tmp_path,capsys):
    run, *_ = fixture(tmp_path)
    before = {str(p):sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert main(["--run",str(run),"--data-root",str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "DIAGNOSIS_COMPLETE" in out and "DATA=SYNTHETIC" in out
    assert "person_private_" not in out
    after = {str(p):sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("relative",["outputs/qc_synthetic/private/connection_quality.csv","meta.csv"])
def test_changed_source_rejected(tmp_path,relative):
    run, *_ = fixture(tmp_path)
    file = tmp_path/relative
    file.write_text(file.read_text()+"\n")
    with pytest.raises(ValueError,match="changed"):
        main(["--run",str(run),"--data-root",str(tmp_path)])


def test_boundaries_detects_calendar_day_shift(tmp_path):
    _, q, _, diary, meta, config = fixture(tmp_path)
    tables = dict(boundary_tables(q,meta,diary,config))
    row = next(v for k,v in tables.items() if "零点检验" in k).iloc[0]
    assert row.start_in_first_15_minutes == row.last_plus_15min_in_first_15_minutes == 2
    first = next(v for k,v in tables.items() if "first_date 减" in k)
    assert first.iloc[0].days == 1 and first.iloc[0].connections == 2
    complete = next(v for k,v in tables.items() if "编号0至13" in k).iloc[0]
    assert complete.day_metadata_complete_14 == 2


def test_inconsistent_logging_index_excluded_without_fallback(tmp_path):
    _, q, _, diary, meta, config = fixture(tmp_path)
    diary.loc[0,"logging_day"] = 9
    tables = dict(boundary_tables(q,meta,diary,config))
    summary = next(v for k,v in tables.items() if "固定起日" in k).iloc[0]
    assert summary.consistent_people == summary.inconsistent_or_invalid_people == 1
    joined = next(v for k,v in tables.items() if "边界匹配分母" in k).iloc[0]
    assert joined.matched_consistent_connections == 1


def test_declared_anchor_not_used_to_reconstruct_raw_end(tmp_path):
    _, q, _, diary, meta, config = fixture(tmp_path)
    q.loc[0,"anchor_source"] = "DECLARED_PLAN"
    tables = dict(boundary_tables(q,meta,diary,config))
    row = next(v for k,v in tables.items() if "零点检验" in k).iloc[0]
    assert row.audited_connections == 2 and row.first_record_proxy_connections == 1


def test_calendar_boundaries_dst_and_naive_metadata(tmp_path):
    _, q, _, diary, meta, config = fixture(tmp_path)
    config["time"]["local_timezone"] = "Asia/Jerusalem"
    start = pd.Timestamp("2024-10-20T00:03:00",tz="Asia/Jerusalem")
    end = pd.Timestamp("2024-11-02T00:03:00",tz="Asia/Jerusalem")
    q["window_start"] = start.timestamp()
    q["observed_span_days"] = (end.timestamp()-start.timestamp()-900)/86400 - 1e-12
    meta["collection_timestamp"] = "2024-10-20 00:03:00"  # Explicitly naive: do not assume UTC.
    tables = dict(boundary_tables(q,meta,diary,config))
    row = next(v for k,v in tables.items() if "零点检验" in k).iloc[0]
    assert row.start_in_first_15_minutes == row.last_plus_15min_in_first_15_minutes == 2
    parse = next(v for k,v in tables.items() if "时间戳匹配分母" in k).iloc[0]
    assert parse.metadata_timestamp_parseable == 0


def test_boundaries_cli_no_identifiers_and_missing_optional_fields(tmp_path,capsys):
    run, q, _, diary, _, config = fixture(tmp_path)
    assert main(["--run",str(run),"--data-root",str(tmp_path),"--boundaries-only"]) == 0
    out=capsys.readouterr().out
    assert "BOUNDARY_DIAGNOSIS_COMPLETE" in out and "person_private_" not in out
    assert "A |" not in out
    tables=boundary_tables(q,None,diary.drop(columns="logging_day"),config)
    assert not any("固定起日" in title for title,_ in tables)
