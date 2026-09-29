import copy
import json
import math

import numpy as np
import pandas as pd
import pytest

from hppmeal.common import parse_times, validate_config
from hppmeal.cgm import areas, match_meal, prepare_connections, response
from hppmeal.meals import build_meals
from hppmeal.models import bh_planned, within_fit
from hppmeal.report import repeat_pairs


def food_data(times):
    return pd.DataFrame([dict(participant_id="0001", cohort="10k", research_stage="00_00_visit",
                              t=1609459200 + t * 60, food_id=str(i), tz="Asia/Jerusalem", nova=4,
                              duplicate_item=0, weight_g=100.0, calories_kcal=100.0,
                              carbohydrate_g=20.0, lipid_g=2.0, protein_g=3.0, dietary_fiber_g=1.0)
                         for i, t in enumerate(times)])


def test_timezone_equivalence_and_naive_rejection():
    out = parse_times(pd.Series(["2022-09-16T08:00:00+03:00", "2022-09-16T07:00:00+02:00",
                                 "2022-09-16 08:00:00", "bad"]))
    assert out[0] == out[1]
    assert out[2:].isna().all()
    local = parse_times(pd.Series(["2022-09-16 08:00:00"]), "Asia/Jerusalem")
    assert local[0] == out[0]


def test_dst_ambiguous_and_nonexistent_not_guessed():
    result = parse_times(pd.Series(["2021-11-07 01:30:00", "2021-03-14 02:30:00"]), "America/New_York")
    assert result.isna().all()


def test_config_string_false_rejected(config):
    config["time"]["consumption_time_verified"] = "false"
    with pytest.raises(ValueError):
        validate_config(config)


def test_episode_span_no_chaining_and_no_identity_loss(config):
    m = build_meals(food_data([0, 20, 40, 60, 180]), config)
    assert list(m["items"]) == [2, 2, 1]
    assert m.participant_id.eq("0001").all()
    assert m.iloc[1].previous_gap_minutes == 20


def test_nutrients_and_mapping_unknown_not_zero(config):
    f = food_data([0, 5, 200])
    f.loc[0, "carbohydrate_g"] = np.nan
    f.loc[1, "nova"] = np.nan
    m = build_meals(f, config)
    assert math.isnan(m.iloc[0].carbohydrate_g)
    assert m.iloc[0].nova_coverage == 0.5
    assert m.iloc[0].nova4_pct == 50
    assert not m.iloc[0].nova_eligible
    assert not m.iloc[0].meal_eligible


def test_future_snack_excludes_window_and_boundary_unknown(config):
    f = food_data([0, 180, 270, 600])
    f.loc[2, ["calories_kcal", "carbohydrate_g"]] = 0
    m = build_meals(f, config)
    assert not m.iloc[0].isolated_window
    assert not m.iloc[1].isolated_window  # Even a zero-energy logged event is visible.
    assert not m.iloc[-1].isolated_window


def test_triangle_area_and_baseline_crossing():
    assert areas([0, 60, 120], [0, 40, 0]) == (2400.0, 2400.0)
    net, positive = areas([0, 10], [-2, 2])
    assert net == 0 and positive == 5  # Clipping endpoints then trapezoids would wrongly give 10.


def known_curve():
    t = np.arange(-30, 151, 15) * 60 + 1_000_000
    r = (t - 1_000_000) / 60
    g = 100 + np.maximum(0, 1 - np.abs(r - 60) / 60) * 40
    return t, g


def test_known_postprandial_response(config):
    t, g = known_curve()
    row, curve = response(t, g, 1_000_000, config["cgm"])
    assert row["cgm_valid"]
    assert row["iauc_net_0_120"] == 2400
    assert row["peak_rise_0_120"] == 40
    assert row["peak_time_minutes"] == 60
    assert row["baseline_glucose"] == 100


def test_long_gap_not_interpolated(config):
    t, g = known_curve()
    keep = t != 1_000_000 + 60 * 60
    row, _ = response(t[keep], g[keep], 1_000_000, config["cgm"])
    assert not row["cgm_valid"] and row["cgm_reason"] == "post_gap"


def test_no_endpoint_extrapolation(config):
    t, g = known_curve()
    keep = t < 1_000_000 + 120 * 60
    row, _ = response(t[keep], g[keep], 1_000_000, config["cgm"])
    assert not row["cgm_valid"]


def test_multiple_connections_rejected(config):
    t, g = known_curve()
    row, _ = match_meal({"a": (t, g), "b": (t, g)}, 1_000_000, config["cgm"])
    assert row["cgm_reason"] == "multiple_connections_in_window"


def test_duplicate_cgm_collapse_but_conflict_invalidates():
    raw = pd.DataFrame(dict(connection_id=["a"] * 4, t=[0, 0, 15, 15], glucose=[100, 100, 105, 110], valid=[1] * 4))
    connections, stats = prepare_connections(raw)
    t, g = connections["a"]
    assert len(t) == 2 and g[0] == 100 and math.isnan(g[1])
    assert stats[0]["conflicting_timepoints"] == 1
    assert stats[0]["duplicate_excess_rows"] == 2


def test_within_fit_matches_explicit_dummy_regression():
    rng = np.random.default_rng(177)
    n_people, m = 30, 12
    n = n_people * m
    pid = np.repeat(np.arange(n_people), m)
    x, z = rng.normal(size=n), rng.normal(size=n)
    y = np.repeat(rng.normal(size=n_people) * 50, m) + 2.5 * x - 0.8 * z + rng.normal(size=n)
    d = pd.DataFrame(dict(participant_id=pid.astype(str), cohort="10k", research_stage="v",
                         meal_id=np.arange(n).astype(str), nova_per_10pp=x, control=z, y=y))
    result = within_fit(d, "y", ["nova_per_10pp", "control"])
    D = np.eye(n_people)[pid]
    X = np.column_stack([x, z, D])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X.dot(b)
    scores = np.zeros((n_people, X.shape[1]))
    np.add.at(scores, pid, X * residual[:, None])
    bread = np.linalg.inv(X.T.dot(X))
    V = bread.dot(scores.T.dot(scores)).dot(bread) * n_people / (n_people - 1) * (n - 1) / (n - X.shape[1])
    assert result["coefficients"]["nova_per_10pp"]["estimate"] == pytest.approx(b[0], abs=1e-10)
    assert result["coefficients"]["nova_per_10pp"]["se"] == pytest.approx(np.sqrt(V[0, 0]), abs=1e-10)
    assert result["residual_df"] == n - n_people - 2


def test_missing_planned_test_kept_in_fdr_denominator():
    original = np.array([0.01, np.nan])
    out = bh_planned(original)
    assert out[0] == 0.02 and math.isnan(out[1])
    assert math.isnan(original[1])


def test_actual_peak_excludes_interpolated_endpoints(config):
    # Meal occurs 5 minutes after a sensor reading; window endpoints are interpolated.
    t = 1_000_000 + np.arange(-35, 146, 15) * 60
    g = np.full(len(t), 100.0)
    g[(t - 1_000_000) / 60 == 55] = 130
    row, _ = response(t, g, 1_000_000, config["cgm"])
    assert row["cgm_valid"]
    assert row["peak_time_minutes"] == 55
    assert row["peak_rise_0_120"] == 30


def test_constant_personal_exposure_cannot_be_estimated():
    n = 100
    ids = np.repeat(np.arange(10), 10).astype(str)
    df = pd.DataFrame(dict(participant_id=ids, cohort="c", research_stage="v", meal_id=np.arange(n).astype(str),
                           nova_per_10pp=np.repeat(np.arange(10), 10), y=np.arange(n)))
    with pytest.raises(ValueError, match="No within-person NOVA"):
        within_fit(df, "y", ["nova_per_10pp"])


def test_microbe_interaction_uses_individual_not_meal_standardization():
    rng = np.random.default_rng(200)
    people, repeats = 25, 16
    n = people * repeats
    ids = np.repeat(np.arange(people), repeats)
    micro = np.repeat(rng.normal(size=people), repeats)
    x = rng.uniform(0, 10, n)
    y = ids * 10 + 2 * x + 4 * micro * x + rng.normal(0, 0.05, n)
    df = pd.DataFrame(dict(participant_id=ids.astype(str), cohort="c", research_stage="v",
                           meal_id=np.arange(n).astype(str), nova_per_10pp=x, microbiome_x_nova=micro * x, y=y))
    result = within_fit(df, "y", ["nova_per_10pp", "microbiome_x_nova"])
    assert result["coefficients"]["microbiome_x_nova"]["estimate"] == pytest.approx(4, abs=0.01)


def test_repeat_pairs_disjoint_and_different_days():
    df = pd.DataFrame(dict(participant_id=["1"] * 4, cohort="c", research_stage="v", signature="s",
                           analysis_ready=True, start=[1, 2, 3, 4], local_hour=[8, 8, 8, 8],
                           local_date=["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
                           iauc_net_0_120=[10, 12, 14, 16], meal_id=["a", "b", "c", "d"]))
    pairs = repeat_pairs(df)
    assert len(pairs) == 2
    assert pairs.delta.tolist() == [2, 2]
