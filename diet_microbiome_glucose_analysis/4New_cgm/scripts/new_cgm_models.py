#!/usr/bin/env python3
"""Reuse legacy OLS estimates with new outcome families and isolated reports."""
import numpy as np
import pandas as pd

import legacy_diet_models as diet_legacy
import legacy_microbiome_models as micro_legacy
from new_cgm_data import (OUTCOMES, EXPOSURES, BASE_COVARIATES, boolean, sample_hash)


def specs(config):
    items = [("primary", 0, 0), ("primary", 2, 2), ("primary", 3, 3)]
    if config["run_strict_sensitivity"]:
        items += [("strict", 2, 2), ("strict", 3, 3)]
    if config["run_same_cohort_comparisons"]:
        items += [("same_m2", 0, 2), ("same_m3", 2, 3)]
    return items


def complete_mask(frame, columns):
    mask = pd.Series(True, index=frame.index)
    for column in columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            mask &= values.notna() & np.isfinite(values)
        else:
            mask &= values.notna() & ~values.astype(str).str.strip().str.lower().isin(["", "nan", "na", "none", "null", "<na>"])
    return mask


def select_diet(frame, score, outcome, cohort_mode, selection_model):
    score_col, family = EXPOSURES[score]
    required = [score_col, OUTCOMES[outcome]]
    eligible = frame["strict_eligible"] if cohort_mode == "strict" else frame["primary_eligible"]
    mask = eligible.copy()
    if selection_model >= 2:
        required += BASE_COVARIATES
        if selection_model == 3:
            required += ["bmi"]
        if score == "hPDI":
            required += ["alcohol_intake_g_day"]
        if family != "original_four":
            required += ["mean_daily_energy_kcal"]
        prefix = "hpdi" if score == "hPDI" else "amed"
        flag = prefix + "_cgm_model%d_covariates_complete" % selection_model
        mask &= boolean(frame[flag], flag)
    return frame.loc[mask & complete_mask(frame, required)].copy()


def diet_design(selected, score, model):
    original = diet_legacy.BASE_CONTINUOUS
    try:
        if EXPOSURES[score][1] != "original_four":
            diet_legacy.BASE_CONTINUOUS = list(original) + ["mean_daily_energy_kcal"]
        return diet_legacy.build_design(selected, score, EXPOSURES[score][0], model)
    finally:
        diet_legacy.BASE_CONTINUOUS = original


def correct_fdr(frame, branch):
    """Keep every planned test. Failed fits get p=1 only inside BH, not in output."""
    result = frame.copy()
    result["FDR_family"] = np.nan
    result["FDR_global"] = np.nan
    family_column = "family" if branch == "diet" else "outcome"
    for _, indexes in result.groupby(["analysis_set", "model"], sort=False).groups.items():
        part = result.loc[indexes]
        valid = part["status"].eq("computed") & part["p_value"].notna()
        p = part["p_value"].where(valid, 1.0).to_numpy(float)
        q = diet_legacy.bh_fdr(p)
        result.loc[indexes, "FDR_global"] = np.where(valid, q, np.nan)
        result.loc[indexes, "global_tests_planned"] = len(part)
        for _, family_indexes in part.groupby(family_column, sort=False).groups.items():
            group = result.loc[family_indexes]
            good = group["status"].eq("computed") & group["p_value"].notna()
            q = diet_legacy.bh_fdr(group["p_value"].where(good, 1.0).to_numpy(float))
            result.loc[family_indexes, "FDR_family"] = np.where(good, q, np.nan)
            result.loc[family_indexes, "family_tests_planned"] = len(group)
    result["significant_family_05"] = result["FDR_family"].lt(0.05)
    result["significant_global_05"] = result["FDR_global"].lt(0.05)
    return result


def base_result(analysis_set, model, selection_model, outcome, selected):
    return {"analysis_set": analysis_set, "model": model, "selection_model": selection_model,
            "outcome": outcome, "outcome_column": OUTCOMES[outcome],
            "outcome_role": "auxiliary" if outcome == "TIR70_180" else "core",
            "N": len(selected), "sample_sha256": sample_hash(selected["participant_id"]),
            "unknown_exclusion_n": int(selected["exclusion_status"].isna().sum()),
            "beta": np.nan, "SE": np.nan, "CI95_lower": np.nan, "CI95_upper": np.nan,
            "p_value": np.nan, "status": "not_fitted", "error": ""}


def fit_diet(frame, config):
    rows, coefficients, design_rows, membership = [], [], [], {"participant_id": frame["participant_id"].tolist()}
    for analysis_set, model, selection_model in specs(config):
        for score in EXPOSURES:
            for outcome in OUTCOMES:
                selected = select_diet(frame, score, outcome, analysis_set, selection_model)
                row = base_result(analysis_set, model, selection_model, outcome, selected)
                row.update(exposure=score, exposure_column=EXPOSURES[score][0], family=EXPOSURES[score][1])
                membership["%s_M%d_%s_%s" % (analysis_set, model, score, outcome)] = frame["participant_id"].isin(selected["participant_id"]).astype(int).tolist()
                try:
                    if len(selected) < config["minimum_model_n"]:
                        raise ValueError("Insufficient N: %d < %d" % (len(selected), config["minimum_model_n"]))
                    y = selected[OUTCOMES[outcome]].to_numpy(float)
                    if np.ptp(y) == 0:
                        raise ValueError("Constant outcome in selected sample")
                    X, names, parameters, condition = diet_design(selected, score, model)
                    fit = diet_legacy.fit_ols(X, y)
                    if not np.isfinite(fit["p"][1]) or not np.isfinite(fit["se"][1]) or fit["se"][1] <= 0:
                        raise ValueError("Nonfinite/degenerate model inference")
                    row.update(beta=float(fit["coef"][1]), SE=float(fit["se"][1]),
                               CI95_lower=float(fit["ci_lower"][1]), CI95_upper=float(fit["ci_upper"][1]),
                               p_value=float(fit["p"][1]), R2=fit["r2"], df_resid=fit["df_resid"],
                               design_condition=condition, status="computed")
                    meta = {"analysis_set": analysis_set, "model": model, "exposure": score,
                            "outcome": outcome, "N": len(selected), "sample_sha256": row["sample_sha256"]}
                    for index, name in enumerate(names):
                        record = dict(meta, term=name, beta=fit["coef"][index], SE=fit["se"][index], p_value=fit["p"][index])
                        coefficients.append(record)
                    for parameter in parameters.to_dict("records"):
                        design_rows.append(dict(parameter, **{k: v for k, v in meta.items() if k not in parameter}))
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    row.update(status="failed", error=str(error))
                rows.append(row)
                print("DIET %-7s M%d %-16s %-10s N=%d beta=% .5g p=%.4g %s" %
                      (analysis_set, model, score, outcome, row["N"], row["beta"], row["p_value"], row["status"]), flush=True)
    return correct_fdr(pd.DataFrame(rows), "diet"), pd.DataFrame(coefficients), pd.DataFrame(design_rows), pd.DataFrame(membership)


def select_micro(frame, outcome, cohort_mode, selection_model):
    required = [OUTCOMES[outcome]]
    if selection_model >= 2:
        required += BASE_COVARIATES
    if selection_model == 3:
        required += ["bmi"]
    eligible = frame["strict_eligible"] if cohort_mode == "strict" else frame["primary_eligible"]
    return frame.loc[eligible & complete_mask(frame, required)].copy()


def fit_microbiome(frame, species, config):
    rows, design_rows, membership = [], [], {"participant_id": frame["participant_id"].tolist()}
    for analysis_set, model, selection_model in specs(config):
        for outcome in OUTCOMES:
            selected = select_micro(frame, outcome, analysis_set, selection_model)
            base = base_result(analysis_set, model, selection_model, outcome, selected)
            membership["%s_M%d_%s" % (analysis_set, model, outcome)] = frame["participant_id"].isin(selected["participant_id"]).astype(int).tolist()
            group = [dict(base, species=species_name) for species_name in species]
            try:
                if len(selected) < config["minimum_model_n"]:
                    raise ValueError("Insufficient N: %d < %d" % (len(selected), config["minimum_model_n"]))
                y = selected[OUTCOMES[outcome]].to_numpy(float)
                if np.ptp(y) == 0:
                    raise ValueError("Constant outcome in selected sample")
                if model == 0:
                    C, names, rank, condition, reference = np.ones((len(selected), 1)), ["intercept"], 1, 1.0, "none"
                else:
                    C, names, rank, condition, reference = micro_legacy.build_covariate_design(selected, include_bmi=(model == 3))
                for index, name in enumerate(names):
                    design_rows.append({"analysis_set": analysis_set, "model": model, "outcome": outcome,
                                        "term": name, "N": len(selected), "rank": rank,
                                        "condition": condition, "device_reference": reference,
                                        "encoded_mean": C[:, index].mean(), "encoded_sd_ddof0": C[:, index].std(),
                                        "sample_sha256": base["sample_sha256"]})
                X = selected[species].to_numpy(float)
                residual = micro_legacy.residualize_against_covariates(C, X)
                estimable = np.sum(residual ** 2, axis=0) > 1e-12
                for i in np.flatnonzero(~estimable):
                    group[i].update(status="failed", error="Species has no residual variance after adjustment")
                if estimable.any():
                    fit = micro_legacy.fit_all_species_adjusted(X[:, estimable], y, C)
                    for j, i in enumerate(np.flatnonzero(estimable)):
                        if not np.isfinite(fit["p_value"][j]) or not np.isfinite(fit["SE"][j]) or fit["SE"][j] <= 0:
                            group[i].update(status="failed", error="Nonfinite/degenerate model inference")
                            continue
                        group[i].update(beta=fit["beta"][j], SE=fit["SE"][j], CI95_lower=fit["CI95_lower"][j],
                                        CI95_upper=fit["CI95_upper"][j], p_value=fit["p_value"][j],
                                        partial_R2=fit["partial_R2"][j], df_resid=fit["df_resid"], status="computed")
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                for row in group:
                    row.update(status="failed", error=str(error))
            rows += group
            print("MICRO %-7s M%d %-10s N=%d species=%d computed=%d" %
                  (analysis_set, model, outcome, len(selected), len(species), sum(r["status"] == "computed" for r in group)), flush=True)
    return correct_fdr(pd.DataFrame(rows), "microbiome"), pd.DataFrame(design_rows), pd.DataFrame(membership)


def comparisons(results, branch):
    feature = "exposure" if branch == "diet" else "species"
    keys = [feature, "outcome"]
    columns = keys + ["N", "sample_sha256", "beta", "p_value", "FDR_family", "status"]
    output = []
    for label, original_model, same_set, lower_model, higher_model in [
        ("M0_to_M2", 0, "same_m2", 0, 2),
        ("M2_to_M3", 2, "same_m3", 2, 3),
    ]:
        original = results.loc[results.analysis_set.eq("primary") & results.model.eq(original_model), columns]
        same = results.loc[results.analysis_set.eq(same_set) & results.model.eq(lower_model), columns]
        higher = results.loc[results.analysis_set.eq("primary") & results.model.eq(higher_model), columns]
        if same.empty:
            continue
        a = original.rename(columns={c: c + "_original" for c in columns if c not in keys})
        b = same.rename(columns={c: c + "_same_sample" for c in columns if c not in keys})
        c = higher.rename(columns={c: c + "_adjusted" for c in columns if c not in keys})
        merged = a.merge(b, on=keys, validate="one_to_one").merge(c, on=keys, validate="one_to_one")
        if not (merged.sample_sha256_same_sample == merged.sample_sha256_adjusted).all():
            raise ValueError("Same-cohort comparison does not have identical participant IDs")
        merged["comparison"] = label
        merged["sample_selection_beta_change"] = merged.beta_same_sample - merged.beta_original
        merged["adjustment_beta_change_same_sample"] = merged.beta_adjusted - merged.beta_same_sample
        output.append(merged)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame(columns=keys + ["comparison"])
