import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(HERE))
from daily.models import make_design as old_design, within_fit, choose_sample as old_sample
from run_analysis import load_saved_panel
from revision_models import SPECS, choose_sample, make_design, fit_all, bh
from source_audit import attach_alcohol, checkpoints, verify_paired_samples

FIXTURE = PARENT.parent / "tests/_artifacts/daily_within_person/fix_empty_cgm_20261011_v3"


@pytest.fixture
def fixture():
    p = load_saved_panel(FIXTURE / "private/daily_panel.csv")
    c = json.loads((FIXTURE / "analysis_config.json").read_text())
    return p, c


def amed_for(p, c):
    a = p[["participant_id", "date", "events"]].rename(columns={"date": "collection_date", "events": "event_count"}).copy()
    a["cohort"], a["research_stage"] = c["cohort"], c["research_stage"]
    a["alcohol_g_total"] = p.alcohol_g.fillna(0).to_numpy()  # fictional fixture only
    drinks = a.alcohol_g_total.gt(0).astype(int)
    a["alcoholic_drink_event_count"] = drinks
    a["alcohol_observed_event_count"] = drinks
    for field in ["excluded_missing_labels_event_count", "alcohol_energy_formula_event_count",
                  "alcohol_standard_abv_event_count", "alcohol_zero_fallback_event_count",
                  "alcohol_unresolved_event_count", "alcohol_invalid_observed_event_count"]:
        a[field] = 0
    return a


def attach(tmp_path, p, c, a):
    path = tmp_path / "amed.csv"
    a.to_csv(path, index=False)
    return attach_alcohol(p, path, c)


def test_observed_case_matches_previous_design_and_fit(fixture, tmp_path):
    p, c = fixture
    new, _ = attach(tmp_path, p, c, amed_for(p, c))
    for spec in SPECS[:7]:
        d, _ = choose_sample(new, c, "NOVA4", "T068", "SD", spec)
        previous, _ = old_sample(p, c, "NOVA4", "T068", "SD", spec)
        assert d.day_id.tolist() == previous.day_id.tolist()
        anchors = dict(mean=d.drop_duplicates("participant_id").T068.mean())
        x, cols = make_design(d, "NOVA4", "T068", spec, anchors)
        old, old_cols = old_design(previous, "NOVA4", "T068", spec, anchors)
        assert cols == old_cols
        np.testing.assert_allclose(x[cols], old[cols], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("failure", ["duplicate", "nan_flag", "negative_flag", "fraction_flag", "fallback", "excluded", "invalid", "event_mismatch", "closure", "negative_alcohol", "invalid_date"])
def test_bad_source_days_never_enter_adjusted_models(fixture, tmp_path, failure):
    p, c = fixture
    a = amed_for(p, c)
    if failure == "duplicate":
        a = pd.concat([a, a], ignore_index=True)
    elif failure == "nan_flag": a["alcohol_zero_fallback_event_count"] = np.nan
    elif failure == "negative_flag": a["alcohol_zero_fallback_event_count"] = -1
    elif failure == "fraction_flag": a["alcohol_zero_fallback_event_count"] = .5
    elif failure == "fallback":
        a["alcohol_observed_event_count"] = 0
        a["alcoholic_drink_event_count"] = 1
        a["alcohol_zero_fallback_event_count"] = 1
    elif failure == "excluded":
        a["event_count"] -= 1
        a["excluded_missing_labels_event_count"] = 1
    elif failure == "invalid": a["alcohol_invalid_observed_event_count"] = 1
    elif failure == "event_mismatch": a["event_count"] += 1
    elif failure == "closure": a["alcoholic_drink_event_count"] += 1
    elif failure == "negative_alcohol": a["alcohol_g_total"] = -1
    elif failure == "invalid_date": a["collection_date"] = "2020"
    revised, _ = attach(tmp_path, p, c, a)
    assert not revised.alcohol_observed_ok.any()
    assert not revised.alcohol_recovered_ok.any()
    assert choose_sample(revised, c, "NOVA4", "T068", "SD", "primary")[0].empty
    assert not choose_sample(revised, c, "NOVA4", "T068", "SD", "all_days_no_alcohol_control")[0].empty


def test_source_policy_and_same_sample_controls(fixture, tmp_path):
    p, c = fixture
    a = amed_for(p, c)
    ix = a.index % 4 == 0
    a.loc[ix, "alcohol_energy_formula_event_count"] = a.loc[ix, "alcohol_observed_event_count"]
    a.loc[ix, "alcohol_observed_event_count"] = 0
    revised, _ = attach(tmp_path, p, c, a)
    assert revised.alcohol_recovered_ok.sum() > revised.alcohol_observed_ok.sum()
    assert verify_paired_samples(revised, c).identical.all()
    assert len(checkpoints(revised, c)) == 12
    d, _ = choose_sample(revised, c, "NOVA4", "T068", "SD", "primary_no_alcohol_control")
    anchors = dict(mean=d.drop_duplicates("participant_id").T068.mean())
    x, cols = make_design(d, "NOVA4", "T068", "primary_no_alcohol_control", anchors)
    assert not any("alcohol" in col for col in cols)
    d["alcohol_amed_g"] = np.nan
    y, cols2 = make_design(d, "NOVA4", "T068", "primary_no_alcohol_control", anchors)
    assert cols == cols2
    np.testing.assert_allclose(x[cols], y[cols])


def test_leading_zero_and_literal_na_ids(fixture, tmp_path):
    p, c = fixture
    ids = p.participant_id.unique()
    p["participant_id"] = p.participant_id.replace({ids[0]: "001", ids[1]: "NA"})
    revised, audit = attach(tmp_path, p, c, amed_for(p, c))
    assert {"001", "NA"}.issubset(revised.participant_id)
    assert audit["unmatched_days"] == 0
    assert revised.alcohol_observed_ok.all()


def test_missing_source_column_refused(fixture, tmp_path):
    p, c = fixture
    a = amed_for(p, c).drop(columns=["alcohol_zero_fallback_event_count"])
    with pytest.raises(ValueError, match="Missing columns"):
        attach(tmp_path, p, c, a)


def test_primary_fixed_effect_fit_matches_explicit_dummies(fixture, tmp_path):
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster
    p, c = fixture
    revised, _ = attach(tmp_path, p, c, amed_for(p, c))
    d, _ = choose_sample(revised, c, "NOVA4", "T068", "SD", "primary")
    anchors = dict(mean=d.drop_duplicates("participant_id").T068.mean())
    design, cols = make_design(d, "NOVA4", "T068", "primary", anchors)
    fit = within_fit(design, "daily_sd", cols)
    active = fit["names"]
    x = pd.concat([pd.get_dummies(design.participant_id, dtype=float), design[active]], axis=1)
    explicit = sm.OLS(design.daily_sd, x).fit()
    cov = cov_cluster(explicit, design.participant_id, use_correction=True)[-len(active):, -len(active):]
    np.testing.assert_allclose([fit["coefficients"][k]["beta"] for k in active], explicit.params.iloc[-len(active):], rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(fit["covariance"], cov, rtol=1e-7, atol=1e-9)


def test_failed_primary_stays_in_all_families(fixture, tmp_path):
    p, c = fixture
    revised, _ = attach(tmp_path, p, c, amed_for(p, c))
    revised["T096"] = np.nan
    results, _, _, _, _ = fit_all(revised, c)
    assert len(results) == 132
    for spec, d in results.groupby("spec"):
        assert len(d) == 12
        assert len(d.loc[d.outcome.eq("SD")]) == 4
        assert d.loc[d.taxon.eq("T096"), ["beta", "p", "q_primary4", "q_all12"]].isna().all().all()
        np.testing.assert_allclose(d.q_all12, bh(d.p), equal_nan=True)
        sd = d.loc[d.outcome.eq("SD")]
        np.testing.assert_allclose(sd.q_primary4, bh(sd.p), equal_nan=True)
