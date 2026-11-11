"""Select documented measurements before the first analyzed meal."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RAW = "/home/ec2-user/studies/hpp_datasets"
SPECS = {
    "hba1c": dict(file="blood_tests/blood_tests.parquet", field="bt__hba1c_float_value",
                  keys=("participant_id", "cohort"), min_value=3., max_value=18., max_lag=365,
                  reference=5.5, step=.5),
    "bmi": dict(file="anthropometrics/anthropometrics.parquet", field="bmi",
                keys=("participant_id", "cohort", "research_stage"), min_value=10., max_value=80.,
                max_lag=90, reference=25., step=5.),
}


def select_antecedent(anchors, values, spec):
    keys = list(spec["keys"])
    field = spec["field"]
    if anchors.duplicated(keys).any() or anchors.input_row.duplicated().any():
        raise ValueError("Clinical link must resolve to one discovery identity")
    linked = values.merge(anchors[keys + ["input_row", "first_day"]], on=keys, validate="many_to_one")
    linked["collection_date"] = pd.to_datetime(linked.collection_date, errors="coerce").dt.normalize()
    linked["lag"] = (linked.first_day - linked.collection_date).dt.days
    linked[field] = pd.to_numeric(linked[field], errors="coerce")
    eligible = linked.loc[
        linked.lag.between(1, spec["max_lag"]) & linked[field].between(spec["min_value"], spec["max_value"])
    ].copy()
    latest = eligible.loc[eligible.lag.eq(eligible.groupby("input_row").lag.transform("min"))]
    chosen = latest.groupby("input_row").agg(value=(field, "first"), distinct=(field, "nunique"),
                                             lag=("lag", "first"))
    result = chosen.loc[chosen.distinct.eq(1), ["value", "lag"]]
    return result, dict(matched=int(linked.input_row.nunique()), eligible=len(chosen),
                        conflicting=int(chosen.distinct.gt(1).sum()), selected=len(result))


def read_measurements(raw, meal, index, names, digest):
    dates = pd.to_datetime(meal.groupby("input_row").date.min(), errors="raise")
    anchors = index.loc[index.input_row.isin(dates.index)].copy()
    if len(anchors) != len(dates) or not anchors.final_role.eq("discovery").all():
        raise ValueError("Clinical linkage escaped the archived discovery rows")
    anchors["first_day"] = anchors.input_row.map(dates)
    result, audit, sources = {}, {}, {}
    for name in names:
        spec = SPECS[name]
        path = raw / spec["file"]
        fingerprint = digest(path)
        cols = [*spec["keys"], "collection_date", spec["field"]]
        rows = pq.read_table(path, columns=cols, filters=[("participant_id", "in", anchors.participant_id.tolist())],
                             use_threads=False, pre_buffer=False).to_pandas(ignore_metadata=True)
        if digest(path) != fingerprint:
            raise ValueError("Clinical source changed during read")
        sources[str(path)] = fingerprint
        selected, metrics = select_antecedent(anchors, rows, spec)
        result[name] = selected.value
        metrics["median_lag_days"] = float(selected.lag.median()) if len(selected) else None
        audit[name] = metrics
    frame = pd.DataFrame(result)
    return frame, audit, sources


def scale(values, names):
    offsets = np.array([SPECS[k]["reference"] for k in names])
    steps = np.array([SPECS[k]["step"] for k in names])
    return (values - offsets) / steps
