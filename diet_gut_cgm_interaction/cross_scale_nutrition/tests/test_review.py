"""Independent stacked-regression checks added during the pre-upload review."""
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

from crossscale.analysis import VARIANTS, anchors_for, design_for, select_samples
from crossscale.estimators import fit_linear, paired_contrast


@pytest.mark.parametrize("comparison", ["scale", "nutrition", "selection"])
def test_dependent_difference_matches_independent_stacked_wls(panel, protocol, comparison):
    # Force loss of participants AND dates, not merely a subset of covariates.
    ids = panel.participant_id.unique()
    panel.loc[panel.participant_id.isin(ids[:12]), "fiber_density"] = np.nan
    samples, _ = select_samples(panel, protocol["reference_config"], protocol)
    anchors = anchors_for(samples["F"])
    variants = {v["name"]: v for v in VARIANTS}
    pairs = {
        "scale": [("F_joint", "W"), ("F_joint", "B")],
        "nutrition": [("N1_joint", "W"), ("N0_joint", "W")],
        "selection": [("N0_joint", "W"), ("F_joint", "W")],
    }
    fits, matrices, weights, outcomes, groups, terms, corrections = [], [], [], [], [], [], []
    for name, scale in pairs[comparison]:
        v = variants[name]
        d, columns, focal = design_for(samples[v["sample"]], v, scale, anchors)
        fit = fit_linear(d, "daily_sd", columns, focal, scale == "W", min_df=8)
        fits.append(fit)
        active = fit["names"]
        X = d[active].copy()
        if scale == "W":
            # Independent implementation: explicit person intercepts and raw
            # regressors; no use of production demeaning, scores, or bread.
            X = pd.concat([pd.get_dummies(d.participant_id, dtype=float), X], axis=1)
            w = 1 / d.groupby("participant_id").participant_id.transform("size")
        else:
            w = np.ones(len(d))
        n, g, k = len(d), d.participant_id.nunique(), X.shape[1]
        corrections.append(g / (g - 1) * (n - 1) / (n - k))
        matrices.append(X.to_numpy())
        weights.append(w)
        outcomes.append(d.daily_sd.to_numpy())
        groups.extend(d.participant_id.tolist())
        terms.append(X.columns.tolist())
    joint = sm.WLS(np.concatenate(outcomes), block_diag(*matrices),
                   weights=np.concatenate(weights)).fit()
    # This covariance is computed from one independent stacked regression;
    # the cluster identities link rows belonging to different model blocks.
    uncorrected = cov_cluster(joint, pd.factorize(np.asarray(groups))[0], use_correction=False)
    for taxon in ["T068", "T096"]:
        name = "xm_" + taxon
        a = terms[0].index(name)
        b = len(terms[0]) + terms[1].index(name)
        contrast_weights = np.zeros(len(joint.params))
        contrast_weights[a] = np.sqrt(corrections[0])
        contrast_weights[b] = -np.sqrt(corrections[1])
        expected_variance = contrast_weights.dot(uncorrected).dot(contrast_weights)
        result = paired_contrast(fits[0], fits[1], {name: 1}, {name: 1})
        np.testing.assert_allclose(result["beta"], joint.params[a] - joint.params[b], atol=1e-9, rtol=1e-8)
        np.testing.assert_allclose(result["se"] ** 2, expected_variance, atol=1e-11, rtol=1e-8)


def test_constant_control_blocks_and_absorbed_person_offsets(panel, protocol):
    panel["energy_kcal"] = 2000.
    panel["alcohol_amed_g"] = 0.
    panel["vitamin_use"] = 1.
    samples, _ = select_samples(panel, protocol["reference_config"], protocol)
    d = samples["F"]
    anchors = anchors_for(d)
    between, bcols, bfocal = design_for(d, VARIANTS[0], "B", anchors)
    assert not {"energy", "alcohol", "T068_x_energy", "T096_x_alcohol"}.intersection(bcols)
    bfit = fit_linear(between, "daily_sd", bcols, bfocal, False, min_df=8)
    assert "vitamin_use" not in bfit["names"]
    within, cols, focal = design_for(d, VARIANTS[0], "W", anchors)
    wfit = fit_linear(within, "daily_sd", cols, focal, True, min_df=8)
    offsets = dict(zip(sorted(within.participant_id.unique()), np.arange(d.participant_id.nunique()) * 20.))
    within["daily_sd"] += within.participant_id.map(offsets)
    shifted = fit_linear(within, "daily_sd", cols, focal, True, min_df=8)
    np.testing.assert_allclose([wfit["coefficients"][c]["beta"] for c in focal],
                               [shifted["coefficients"][c]["beta"] for c in focal], atol=1e-10)
    np.testing.assert_allclose(wfit["covariance"], shifted["covariance"], atol=1e-10)
