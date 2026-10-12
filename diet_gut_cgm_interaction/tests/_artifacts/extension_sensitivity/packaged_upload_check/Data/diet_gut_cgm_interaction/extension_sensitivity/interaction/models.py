"""OLS/HC3 interaction inference and covariance-aware linear contrasts."""
import numpy as np
import pandas as pd
from scipy import stats

from .common import sample_hash
from .inputs import CATEGORICAL


class ModelUnavailable(ValueError):
    pass


class RobustFit:
    def cov_params(self):
        return self.covariance


def ols_hc3(y, x):
    """OLS via SVD; HC3 sandwich matches statsmodels, with residual-df t inference."""
    fit = RobustFit()
    rank = np.linalg.matrix_rank(x)
    if rank != x.shape[1] or len(y) <= rank:
        raise ModelUnavailable("Rank deficient design")
    inverse = np.linalg.pinv(x)
    fit.params = np.dot(inverse, y)
    residual = y - np.dot(x, fit.params)
    hat = np.sum(x * inverse.T, axis=1)
    if np.max(hat) >= 1 - 1e-8:
        raise ModelUnavailable("Near-unit leverage; HC3 undefined")
    adjusted = inverse * (residual / (1.0 - hat))[None, :]
    fit.covariance = np.dot(adjusted, adjusted.T)
    fit.df_resid = len(y) - rank
    fit.ssr = float(np.dot(residual, residual))
    total = float(np.sum((y - np.mean(y)) ** 2))
    fit.rsquared = 1 - fit.ssr / total
    return fit, hat


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    sorted_q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.clip(sorted_q, 0, 1)
    return result


def make_design(frame, exposure, taxon, covs, reference, variant):
    d = frame[exposure].to_numpy(float) - reference["d_center"]
    m = frame[taxon].to_numpy(float) - reference["m_center"]
    columns = {"Intercept": np.ones(len(frame)), "D": d, "M": m, "D:M": d * m}
    omitted, nuisance = [], {}
    for name in covs:
        if name in CATEGORICAL:
            values = frame[name].astype(str)
            levels = sorted(values.unique())
            if len(levels) == 1:
                omitted.append(name + ":constant")
            for level in levels[1:]:
                term = "C[{}]={}".format(name, level)
                columns[term] = values.eq(level).to_numpy(float)
                nuisance[term] = name
        else:
            values = frame[name].to_numpy(float)
            sd = np.std(values, ddof=1)
            if sd <= 1e-12:
                omitted.append(name + ":constant")
                continue
            columns[name] = (values - np.mean(values)) / sd
            nuisance[name] = name
    if variant == "quadratic":
        columns["D2"], columns["M2"] = d ** 2, m ** 2
    if variant == "covariate_interactions":
        # Control plausible covariate-driven heterogeneity on both margins.
        for term, original in nuisance.items():
            if original in ["age_years", "sex", "bmi"]:
                columns["D*" + term] = d * columns[term]
                columns["M*" + term] = m * columns[term]
    matrix = pd.DataFrame(columns, index=frame.index, dtype=float)
    return matrix, omitted


def contrast(fit, names, weights):
    vector = np.array([weights.get(name, 0.0) for name in names])
    effect = float(np.dot(vector, fit.params))
    var = float(np.dot(vector, np.dot(np.asarray(fit.cov_params()), vector)))
    if not np.isfinite(var) or var < -1e-10:
        raise ModelUnavailable("Invalid contrast variance")
    se = float(np.sqrt(max(var, 0)))
    if se <= 0:
        raise ModelUnavailable("Zero contrast SE")
    critical = stats.t.ppf(0.975, fit.df_resid)
    return dict(beta=effect, se=se, ci_low=effect - critical * se, ci_high=effect + critical * se,
                p=float(2 * stats.t.sf(abs(effect / se), fit.df_resid)))


def build_reference(frame, exposure, taxon):
    ref = dict(d_center=float(frame[exposure].mean()), m_center=float(frame[taxon].mean()))
    for prefix, col in [("d", exposure), ("m", taxon)]:
        for q in [10, 25, 50, 75, 90]:
            ref[prefix + str(q)] = float(frame[col].quantile(q / 100.0))
    return ref


def support_counts(frame, exposure, taxon, ref):
    result = {}
    distinct_tails = ref["d75"] > ref["d25"] and ref["m75"] > ref["m25"]
    for dlevel, dmask in [("low", frame[exposure].le(ref["d25"])), ("high", frame[exposure].ge(ref["d75"]))]:
        for mlevel, mmask in [("low", frame[taxon].le(ref["m25"])), ("high", frame[taxon].ge(ref["m75"]))]:
            # Coincident quartiles produce overlapping 'low'/'high' groups,
            # which must not be mistaken for adequate independent tail support.
            result["tail_{}_{}_n".format(dlevel, mlevel)] = int((dmask & mmask).sum()) if distinct_tails else np.nan
    result["diet_microbe_r"] = float(frame[exposure].corr(frame[taxon]))
    return result


def fit_interaction(frame, exposure, taxon, outcome, covs, reference, variant, config, y_scale):
    n = len(frame)
    if n < config["minimum_model_n"]:
        raise ModelUnavailable("N below minimum ({})".format(n))
    if frame[[exposure, taxon, outcome]].std().le(1e-10).any():
        raise ModelUnavailable("Constant exposure, taxon or outcome")
    matrix, omitted = make_design(frame, exposure, taxon, covs, reference, variant)
    a = matrix.to_numpy()
    rank = int(np.linalg.matrix_rank(a))
    if rank != a.shape[1]:
        raise ModelUnavailable("Rank deficient design: {} / {}".format(rank, a.shape[1]))
    if n - rank < config["minimum_residual_df"]:
        raise ModelUnavailable("Residual degrees of freedom below minimum")
    fit, hat = ols_hc3(frame[outcome].to_numpy(float), a)
    if not np.isfinite(fit.params).all() or not np.isfinite(fit.cov_params()).all():
        raise ModelUnavailable("Nonfinite fit/covariance")
    names = list(matrix.columns)
    interaction = contrast(fit, names, {"D:M": 1.0})
    coef = {"interaction_" + key: val for key, val in interaction.items()}
    for key in ["beta", "se", "ci_low", "ci_high"]:
        coef["interaction_" + key + "_native"] = interaction[key] * y_scale
    delta_d, delta_m = reference["d75"] - reference["d25"], reference["m75"] - reference["m25"]
    degenerate_iqr = delta_d <= 0 or delta_m <= 0
    # The main coefficient may be identifiable despite a tied middle 50%.
    # An uninformative optional IQR contrast must not discard that hypothesis.
    difference = {key: np.nan for key in ["beta", "se", "ci_low", "ci_high"]}
    if not degenerate_iqr:
        difference = contrast(fit, names, {"D:M": delta_d * delta_m})
    coef["double_iqr_status"] = "NOT_INFORMATIVE_ZERO_IQR" if degenerate_iqr else "OK"
    for key in ["beta", "se", "ci_low", "ci_high"]:
        coef["double_iqr_" + key + "_native"] = difference[key] * y_scale
    # Incremental in-sample variance is descriptive, never a validation metric.
    j = names.index("D:M")
    reduced_x = np.delete(a, j, axis=1)
    reduced_y = frame[outcome].to_numpy(float)
    reduced_residual = reduced_y - np.dot(reduced_x, np.linalg.lstsq(reduced_x, reduced_y, rcond=None)[0])
    reduced_r2 = 1 - float(np.sum(reduced_residual ** 2)) / float(np.sum((reduced_y - reduced_y.mean()) ** 2))
    support = support_counts(frame, exposure, taxon, reference)
    coef.update(n=n, parameters=rank, residual_df=float(fit.df_resid),
                sample_sha256=sample_hash(frame.index), covariates=";".join(covs),
                constant_covariates=";".join(omitted), condition_number=float(np.linalg.cond(a)),
                max_leverage=float(np.max(hat)), high_leverage_n=int(np.sum(hat > 2 * rank / n)),
                r2=float(fit.rsquared), interaction_delta_r2=float(fit.rsquared - reduced_r2),
                support_min_tail_n=min(support[k] for k in ["tail_low_low_n", "tail_low_high_n", "tail_high_low_n", "tail_high_high_n"]))
    coef.update(support)
    coef["limited_joint_support"] = degenerate_iqr or coef["support_min_tail_n"] < config["minimum_tail_cell_n"]
    coef.update(reference)
    slopes = []
    # Quadratic model: derivative at primary Diet median. Additional interactions:
    # average derivative over this model's observed covariate distribution.
    for q in [25, 50, 75]:
        weights = {"D": 1, "D:M": reference["m" + str(q)] - reference["m_center"]}
        if "D2" in names:
            weights["D2"] = 2 * (reference["d50"] - reference["d_center"])
        for name in names:
            if name.startswith("D*"):
                weights[name] = float(matrix[name[2:]].mean())
        estimate = contrast(fit, names, weights)
        slopes.append(dict(microbial_quantile=q, microbial_original_z=reference["m" + str(q)],
                           beta_native=estimate["beta"] * y_scale, ci_low_native=estimate["ci_low"] * y_scale,
                           ci_high_native=estimate["ci_high"] * y_scale, p_pointwise=estimate["p"],
                           slope_note="Per original Diet Z; quadratic at Diet median; covariate interactions averaged"))
    return coef, slopes


def apply_fdr(results):
    """Frozen families include non-estimable tests as p=1; their q stays missing."""
    results = results.copy()
    results["q_focus"] = np.nan
    results["q_all"] = np.nan
    audit = []
    for model in results.model.unique():
        for family, mask in [("focus", results.model.eq(model) & results.family.eq("focus")),
                             ("all", results.model.eq(model))]:
            subset = results.loc[mask]
            if subset.empty:
                continue
            good = subset.status.eq("OK") & subset.interaction_p.notna()
            p = subset.interaction_p.where(good, 1.0).to_numpy(float)
            q = bh_adjust(p)
            results.loc[subset.index[good], "q_" + family] = q[good.to_numpy()]
            audit.append(dict(model=model, family=family, planned_tests=len(subset), estimable_tests=int(good.sum()),
                              nonestimable_in_denominator=int((~good).sum()), correction="BH; failed tests p=1, q reported missing"))
    return results, pd.DataFrame(audit)
