import numpy as np
import pandas as pd
import pytest

from daily.build import cgm_day, cgm_person, day_bounds, diet_day
from daily.common import date_only, epoch, parse_times, validate_config


@pytest.mark.parametrize("date,hours", [("2024-03-29", 23), ("2024-10-27", 25), ("2024-01-10", 24)])
def test_actual_calendar_day_and_sample_sd(config, date, hours):
    left, right = day_bounds(date, "Asia/Jerusalem")
    assert (right - left) / 3600 == hours
    t = np.arange(left, right, 900)
    y = 100 + 10 * np.sin(np.arange(len(t)))
    row = cgm_day(t, y, date, config)
    assert row["expected_bins"] == hours * 4
    assert row["cgm_coverage"] == 1
    assert row["cgm_strict"]
    assert row["daily_mean"] == pytest.approx(y.mean())
    assert row["daily_sd"] == pytest.approx(y.std(ddof=1))
    assert row["daily_cv"] == pytest.approx(100 * y.std(ddof=1) / y.mean())


def test_gaps_and_partial_day_do_not_pass_by_span_denominator(config):
    left, right = day_bounds("2024-01-10", "Asia/Jerusalem")
    t = np.arange(left, right, 900)
    y = np.linspace(90, 110, len(t))
    assert not cgm_day(t[20:], y[20:], "2024-01-10", config)["cgm_pass"]
    y[20:24] = np.nan
    row = cgm_day(t, y, "2024-01-10", config)
    assert row["cgm_coverage"] > .9
    assert row["max_gap_minutes"] == 75
    assert not row["cgm_pass"]


def test_conflicting_duplicate_and_invalid_glucose_never_averaged(config):
    left, right = day_bounds("2024-01-10", "Asia/Jerusalem")
    t = np.arange(left, right, 900)
    raw = pd.DataFrame(dict(t=t, glucose=np.ones(len(t)) * 100, valid=1))
    raw = pd.concat([raw, pd.DataFrame([dict(t=t[2], glucose=100, valid=1), dict(t=t[3], glucose=200, valid=1), dict(t=t[4], glucose=np.nan, valid=0)])], ignore_index=True)
    rows, stats = cgm_person(raw, config)
    assert stats["duplicate_excess"] == 3
    assert stats["conflicting_times"] == 1
    assert stats["invalid_times"] == 2
    assert rows[0]["observed_bins"] == 94
    assert rows[0]["daily_mean"] == 100
    assert not cgm_person(raw, config, True)[0][0]["cgm_pass"]


def food_day():
    return pd.DataFrame(dict(energy=[600., 600., 800.], nova=[4., 1., np.nan],
        t=np.array([7., 12., 18.]) * 3600, rowhash=["a", "b", "c"], bad_day=[0, 0, 0], formula_energy=[0, 0, 0],
        dietary_fiber_g=[5., 10., 5.], carbohydrate_g=[60., 70., 80.], protein_g=[20., 25., 30.], alcohol_g=[0., 0., 0.]))


def test_classified_denominator_and_fiber_missing(config):
    d = food_day()
    r = diet_day(d, config)
    assert r["diet_pass"]
    assert r["nova4_classified_pct"] == 50
    assert r["nova4_total_pct"] == 30
    assert r["nova_coverage"] == .6
    assert r["fiber_density"] == 10
    d.loc[1, "dietary_fiber_g"] = np.nan
    assert np.isnan(diet_day(d, config)["fiber_density"])
    assert diet_day(d, config)["fiber_g_energy_coverage"] == .7


def test_duplicate_food_and_unknown_energy_reject_day(config):
    d = food_day()
    d.loc[1, "rowhash"] = "a"
    assert "SUSPECT_DUPLICATE_FOOD" in diet_day(d, config)["diet_reasons"]
    d.loc[0, "energy"] = np.nan
    r = diet_day(d, config)
    assert not r["diet_pass"]
    assert np.isnan(r["nova4_classified_pct"])
    assert "INCOMPLETE_ENERGY" in r["diet_reasons"]


def test_naive_time_is_not_silently_utc():
    s = pd.Series(["2024-01-01 23:30:00+02:00", "2024-01-01 23:30:00"])
    x = parse_times(s)
    assert x.iloc[0].hour == 21
    assert pd.isna(x.iloc[1])
    assert parse_times(s, "Asia/Jerusalem").iloc[0] == parse_times(s, "Asia/Jerusalem").iloc[1]


def test_invalid_configuration(config):
    config["cgm"]["minimum_coverage"] = 1.1
    with pytest.raises(ValueError, match="coverage"):
        validate_config(config)


def test_partial_or_ambiguous_dates_are_not_imputed():
    s = pd.Series(["2024", "2024-01", "01/02/2024", "NaT", "2024-02-30",
                   "2024-01-01", "2024-01-01T23:30:00Z"])
    out = date_only(s, "Asia/Jerusalem")
    assert out.iloc[:5].isna().all()
    assert out.iloc[5:].tolist() == ["2024-01-01", "2024-01-02"]


@pytest.mark.parametrize("values", [[], ["invalid", ""],
                                   ["2024-01-01T12:00:00+02:00", "invalid", "2024-01-01T10:15:00Z"]])
def test_epoch_empty_missing_and_valid_are_always_float(values):
    source = pd.Series(values, index=pd.Index(range(17, 17 + len(values)), name="row"),
                       dtype=str, name="collection_timestamp")
    parsed = parse_times(source)
    actual = epoch(parsed)
    assert actual.dtype == np.dtype("float64")
    pd.testing.assert_index_equal(actual.index, source.index)
    assert actual.name == source.name
    expected = np.array([x.timestamp() if pd.notna(x) else np.nan for x in parsed], dtype=float)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
