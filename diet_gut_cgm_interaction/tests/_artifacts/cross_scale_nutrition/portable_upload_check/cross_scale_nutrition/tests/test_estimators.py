import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

from crossscale.estimators import fit_linear, contrast, paired_contrast
from crossscale.common import bh


@pytest.mark.parametrize("equal_person", [True, False])
def test_absorbed_fe_matches_explicit_weighted_dummies(panel, equal_person):
    d = panel.copy()
    d["x"] = d.nova4_total_pct / 10
    d["xm"] = d.x * d.T096
    columns = ["x", "xm", "energy_kcal", "alcohol_amed_g"]
    fit = fit_linear(d, "daily_sd", columns, ["x", "xm"], True, equal_person, 30)
    x = pd.concat([pd.get_dummies(d.participant_id, dtype=float), d[columns]], axis=1)
    weight = 1 / d.groupby("participant_id").participant_id.transform("size") if equal_person else np.ones(len(d))
    explicit = sm.WLS(d.daily_sd, x, weights=weight).fit()
    covariance = cov_cluster(explicit, d.participant_id, use_correction=True)[-len(columns):, -len(columns):]
    np.testing.assert_allclose([fit["coefficients"][c]["beta"] for c in columns], explicit.params.iloc[-len(columns):], rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(fit["covariance"], covariance, rtol=2e-8, atol=1e-10)
    np.testing.assert_allclose(fit["influence"].to_numpy().T.dot(fit["influence"]), covariance, rtol=2e-8, atol=1e-10)
    coef = fit["coefficients"]["xm"]
    assert np.isclose(coef["p"], 2 * stats.t.sf(abs(coef["beta"] / coef["se"]), d.participant_id.nunique() - 1))


def test_between_matches_hc1_ols(panel):
    d = panel.groupby("participant_id", sort=True)[["daily_sd", "nova4_total_pct", "T096", "bmi"]].mean().reset_index()
    d["intercept"], d["x"] = 1., d.nova4_total_pct / 10
    d["xm"] = d.x * d.T096
    columns = ["intercept", "x", "T096", "xm", "bmi"]
    fit = fit_linear(d, "daily_sd", columns, ["x", "xm"], False)
    model = sm.OLS(d.daily_sd, d[columns]).fit(cov_type="HC1")
    np.testing.assert_allclose([fit["coefficients"][c]["beta"] for c in columns], model.params, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(fit["covariance"], model.cov_params(), rtol=1e-9, atol=1e-9)


def test_paired_contrast_keeps_covariance_and_identity_alignment(panel):
    d = panel.copy()
    d["x"], d["xm"] = d.nova4_total_pct / 10, d.nova4_total_pct * d.T096 / 10
    columns = ["x", "xm", "energy_kcal"]
    left = fit_linear(d, "daily_sd", columns, ["x", "xm"], True)
    right = fit_linear(d.sample(frac=1, random_state=42), "daily_sd", columns + ["alcohol_amed_g"], ["x", "xm"], True)
    delta = paired_contrast(right, left, {"xm": 1}, {"xm": 1})
    a = right["influence"]["xm"].sort_index().to_numpy()
    b = left["influence"]["xm"].sort_index().to_numpy()
    covariance = np.column_stack([a, b]).T.dot(np.column_stack([a, b]))
    expected_var = np.array([1., -1.]).dot(covariance).dot(np.array([1., -1.]))
    assert np.isclose(delta["se"] ** 2, expected_var)
    independent_var = right["coefficients"]["xm"]["se"] ** 2 + left["coefficients"]["xm"]["se"] ** 2
    assert abs(independent_var - expected_var) > .25 * independent_var
    identical = paired_contrast(left, left, {"xm": 1}, {"xm": 1})
    assert identical["beta"] == 0 and identical["se"] == 0 and identical["p"] == 1


def test_overlapping_samples_and_partial_overlap(panel):
    d = panel.copy()
    d["x"], d["xm"] = d.nova4_total_pct / 10, d.nova4_total_pct * d.T096 / 10
    a = fit_linear(d, "daily_sd", ["x", "xm"], ["x", "xm"], True)
    sub = d.loc[d.participant_id.isin(d.participant_id.unique()[20:])]
    b = fit_linear(sub, "daily_sd", ["x", "xm"], ["x", "xm"], True)
    delta = paired_contrast(b, a, {"xm": 1}, {"xm": 1})
    ids = a["influence"].index.union(b["influence"].index)
    z = b["influence"]["xm"].reindex(ids, fill_value=0) - a["influence"]["xm"].reindex(ids, fill_value=0)
    assert np.isclose(delta["se"], np.linalg.norm(z))
    assert delta["overlap_people"] == 80


def test_collinear_focal_rejected_and_failed_tests_stay_in_bh(panel):
    d = panel.assign(x=panel.nova4_total_pct, xm=panel.nova4_total_pct)
    with pytest.raises(ValueError, match="Collinear"):
        fit_linear(d, "daily_sd", ["x", "xm"], ["x", "xm"], True)
    np.testing.assert_allclose(bh([.01, np.nan, .03, .9]), [.04, np.nan, .06, 1.], equal_nan=True)


def test_linear_contrast_uses_off_diagonal_covariance(panel):
    d = panel.copy()
    d["x"], d["xm"] = d.nova4_total_pct / 10, d.nova4_total_pct * d.T096 / 10
    fit = fit_linear(d, "daily_sd", ["x", "xm"], ["x", "xm"], True)
    row = contrast(fit, {"x": 1., "xm": .7})
    v = np.array([1., .7])
    assert np.isclose(row["se"] ** 2, v.dot(fit["covariance"]).dot(v))
