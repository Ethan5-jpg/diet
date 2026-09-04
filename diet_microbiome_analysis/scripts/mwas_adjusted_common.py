from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as t_dist

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
ANALYSIS = ROOT / "diet_microbiome_analysis"
DATA_DIR = ANALYSIS / "data"
REPORT_DIR = ANALYSIS / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DIET_FILE = DATA_DIR / "00_aligned_four_scores.csv"
MICRO_FILE = DATA_DIR / "00_aligned_species_clr_zscore.csv"
COV_FILE = ROOT / "co-variant" / "outputs" / "data" / "02_covariate_master.csv"

SCORES = {
    "AHEI": "AHEI_z",
    "AMED": "AMED_z",
    "hPDI": "hPDI_z",
    "rEDIH": "rEDIH_z",
}
META_COLS = ["participant_id", "cohort", "research_stage", "array_index"]
BASE_CONT = ["age_years", "sleep_duration_hours_day", "physical_activity_met_h_week"]
BINARY = ["vitamin_use", "hormone_use"]
CATEGORICAL = ["sex", "education_level", "smoking_status"]


def normalize_id(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all():
        raise RuntimeError("Non-finite p value supplied to BH-FDR.")
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def to_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    x = s.astype(str).str.strip().str.lower()
    m = {"true": True, "1": True, "1.0": True,
         "false": False, "0": False, "0.0": False,
         "nan": False, "none": False, "": False}
    bad = sorted(set(x.unique()) - set(m))
    if bad:
        raise RuntimeError(f"Unexpected completeness values: {bad[:20]}")
    return x.map(m).astype(bool)


def z_numeric(s, name):
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    if not np.isfinite(x).all():
        raise RuntimeError(f"{name} contains missing/non-finite values.")
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if sd <= 0 or not np.isfinite(sd):
        raise RuntimeError(f"{name} has zero/non-finite SD.")
    return (x - mu) / sd, mu, sd


def binary_numeric(s, name):
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    if not np.isfinite(x).all():
        raise RuntimeError(f"{name} contains missing/non-finite values.")
    vals = set(np.unique(x).tolist())
    if not vals.issubset({0.0, 1.0}):
        raise RuntimeError(f"{name} expected 0/1, got {sorted(vals)[:20]}")
    return x


def category_text(s):
    return s.astype(str).str.strip().str.lower()


def build_design(df, score_name, score_col, model):
    names = ["intercept", score_col]
    diet_x = pd.to_numeric(df[score_col], errors="coerce").to_numpy(float)
    if not np.isfinite(diet_x).all():
        raise RuntimeError(f"{score_name}: invalid dietary-score values.")
    cols = [np.ones(len(df)), diet_x]
    params = []

    continuous = BASE_CONT.copy()
    if model == 3:
        continuous.append("bmi")
    if score_name == "hPDI":
        continuous.append("alcohol_intake_g_day")

    for c in continuous:
        z, mu, sd = z_numeric(df[c], c)
        cols.append(z)
        names.append(f"{c}_z")
        params.append({
            "score": score_name, "model": model, "variable": c,
            "encoding": "z_continuous", "mean": mu, "sd": sd, "reference": ""
        })

    for c in BINARY:
        x = binary_numeric(df[c], c)
        cols.append(x)
        names.append(c)
        params.append({
            "score": score_name, "model": model, "variable": c,
            "encoding": "binary_0_1", "mean": float(x.mean()),
            "sd": float(x.std(ddof=0)), "reference": "0"
        })

    sex = category_text(df["sex"]).replace({"f": "female", "m": "male", "0": "female", "1": "male"})
    bad = ~sex.isin(["female", "male"])
    if bad.any():
        raise RuntimeError(f"{score_name}: unexpected sex values {df.loc[bad,'sex'].value_counts().head().to_dict()}")
    male = sex.eq("male").to_numpy(float)
    cols.append(male); names.append("sex_male")
    params.append({"score": score_name, "model": model, "variable": "sex",
                   "encoding": "dummy", "mean": float(male.mean()), "sd": float(male.std()),
                   "reference": "female"})

    edu = category_text(df["education_level"]).str.replace("education_", "", regex=False)
    bad = ~edu.isin(["high", "low"])
    if bad.any():
        raise RuntimeError(f"{score_name}: unexpected education values {df.loc[bad,'education_level'].value_counts().head().to_dict()}")
    low = edu.eq("low").to_numpy(float)
    cols.append(low); names.append("education_low")
    params.append({"score": score_name, "model": model, "variable": "education_level",
                   "encoding": "dummy", "mean": float(low.mean()), "sd": float(low.std()),
                   "reference": "high"})

    smoking = category_text(df["smoking_status"])
    bad = ~smoking.isin(["never", "former", "current"])
    if bad.any():
        raise RuntimeError(f"{score_name}: unexpected smoking values {df.loc[bad,'smoking_status'].value_counts().head().to_dict()}")
    former = smoking.eq("former").to_numpy(float)
    current = smoking.eq("current").to_numpy(float)
    cols += [former, current]
    names += ["smoking_former", "smoking_current"]
    params += [
        {"score": score_name, "model": model, "variable": "smoking_status",
         "encoding": "dummy_former", "mean": float(former.mean()), "sd": float(former.std()),
         "reference": "never"},
        {"score": score_name, "model": model, "variable": "smoking_status",
         "encoding": "dummy_current", "mean": float(current.mean()), "sd": float(current.std()),
         "reference": "never"},
    ]

    X = np.column_stack(cols).astype(float)
    rank = np.linalg.matrix_rank(X)
    if rank != X.shape[1]:
        raise RuntimeError(f"{score_name}: rank-deficient design {rank}/{X.shape[1]}: {names}")
    return X, names, pd.DataFrame(params), float(np.linalg.cond(X))


def fit_all_species(X, Y):
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ Y
    resid = Y - X @ coef
    df_resid = n - k
    sse = np.sum(resid ** 2, axis=0)
    mse = sse / df_resid

    beta = coef[1]
    se = np.sqrt(mse * xtx_inv[1, 1])
    t = beta / se
    p = 2 * t_dist.sf(np.abs(t), df=df_resid)
    tcrit = t_dist.ppf(0.975, df=df_resid)

    y0 = Y - Y.mean(axis=0)
    r2 = 1 - sse / np.sum(y0 ** 2, axis=0)
    return beta, se, t, p, r2, beta - tcrit * se, beta + tcrit * se, df_resid


def load_inputs(model):
    diet = pd.read_csv(DIET_FILE, low_memory=False)
    micro = pd.read_csv(MICRO_FILE, low_memory=False)

    flag_cols = [
        f"amed_microbiome_model{model}_covariates_complete",
        f"hpdi_microbiome_model{model}_covariates_complete",
    ]
    needed = set(
        ["participant_id", "age_years", "sleep_duration_hours_day",
         "physical_activity_met_h_week", "vitamin_use", "hormone_use",
         "sex", "education_level", "smoking_status", "alcohol_intake_g_day", "bmi"]
        + flag_cols
    )
    master = pd.read_csv(COV_FILE, usecols=lambda c: c in needed, low_memory=False)

    for label, df in [("diet", diet), ("microbiome", micro), ("master", master)]:
        df["participant_id"] = normalize_id(df["participant_id"])
        if df["participant_id"].duplicated().any():
            raise RuntimeError(f"Duplicate participant IDs in {label}.")

    if diet["participant_id"].tolist() != micro["participant_id"].tolist():
        raise RuntimeError("Diet and microbiome participant order differs.")

    missing = sorted(needed - set(master.columns))
    if missing:
        raise RuntimeError("Missing master columns: " + ", ".join(missing))

    merged = diet.merge(master, on="participant_id", how="left", validate="one_to_one")
    species = [c for c in micro.columns if c not in META_COLS]
    Y = micro[species].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(Y).all():
        raise RuntimeError("Species matrix contains NaN/Inf.")
    return merged, species, Y


def compare_to_previous(current, score_name, previous_file):
    if not previous_file.exists():
        return None
    old = pd.read_csv(previous_file)[["species", "beta", "FDR", "significant_FDR05"]]
    new = current[["species", "beta", "FDR", "significant_FDR05"]]
    z = old.merge(new, on="species", suffixes=("_old", "_new"), validate="one_to_one")
    old_sig = z["significant_FDR05_old"].astype(bool)
    new_sig = z["significant_FDR05_new"].astype(bool)
    rho, rho_p = spearmanr(z["beta_old"], z["beta_new"])
    both = old_sig & new_sig
    same = int((np.sign(z.loc[both, "beta_old"]) == np.sign(z.loc[both, "beta_new"])).sum())
    return {
        "score": score_name,
        "previous_significant_FDR05": int(old_sig.sum()),
        "current_significant_FDR05": int(new_sig.sum()),
        "retained_significant": int((old_sig & new_sig).sum()),
        "lost_after_adjustment": int((old_sig & ~new_sig).sum()),
        "gained_after_adjustment": int((~old_sig & new_sig).sum()),
        "same_direction_among_both_significant": same,
        "beta_spearman_rho_all_species": float(rho),
        "beta_spearman_p_all_species": float(rho_p),
    }


def run_model(model):
    if model not in (2, 3):
        raise ValueError("model must be 2 or 3")

    prefix = "04" if model == 2 else "05"
    previous_prefix = "03" if model == 2 else "04"
    previous_model = "model0" if model == 2 else "model2"

    merged, species, Y = load_inputs(model)

    print("=" * 95)
    print(f"MICROBIOME-WIDE ASSOCIATION — MODEL {model}")
    print("=" * 95)
    print("Aligned participants:", len(merged))
    print("Species:", len(species))
    print("hPDI additionally adjusts for alcohol; AHEI/AMED/rEDIH do not.")
    if model == 3:
        print("Model 3 = Model 2 + BMI.")

    results, summaries, params, comparisons = [], [], [], []

    for score_name, score_col in SCORES.items():
        flag_prefix = "hpdi" if score_name == "hPDI" else "amed"
        flag_col = f"{flag_prefix}_microbiome_model{model}_covariates_complete"

        score = pd.to_numeric(merged[score_col], errors="coerce")
        valid = to_bool(merged[flag_col]) & score.notna() & np.isfinite(score.to_numpy(float))
        selected = merged.loc[valid].copy()
        idx = np.flatnonzero(valid.to_numpy())
        Y_use = Y[idx]

        X, design_names, ptab, cond = build_design(selected, score_name, score_col, model)
        ptab["N"] = len(selected)
        ptab["design_condition_number"] = cond
        params.append(ptab)

        beta, se, t, p, r2, lo, hi, df_resid = fit_all_species(X, Y_use)
        fdr = bh_fdr(p)

        res = pd.DataFrame({
            "score": score_name, "species": species, "N": len(selected), "model": model,
            "beta": beta, "SE": se, "t": t, "p": p, "FDR": fdr,
            "R2_full_model": r2, "CI_lower": lo, "CI_upper": hi,
            "df_resid": df_resid, "n_parameters": X.shape[1],
            "design_condition_number": cond,
        })
        res["significant_FDR05"] = res["FDR"] < 0.05
        res["direction"] = np.where(res["beta"] > 0, "positive", "negative")

        sig = res[res["significant_FDR05"]]
        print(f"\n{score_name}: N={len(selected)}, parameters={X.shape[1]}, condition={cond:.2f}")
        print(f"FDR<0.05={len(sig)}; positive={(sig.beta>0).sum()}; negative={(sig.beta<0).sum()}")
        print("TOP POSITIVE")
        print(res.nlargest(10, "beta")[["species","beta","CI_lower","CI_upper","p","FDR"]].to_string(index=False))
        print("TOP NEGATIVE")
        print(res.nsmallest(10, "beta")[["species","beta","CI_lower","CI_upper","p","FDR"]].to_string(index=False))

        res.to_csv(REPORT_DIR / f"{prefix}_MWAS_{score_name}_model{model}.csv", index=False)
        results.append(res)
        summaries.append({
            "score": score_name, "N": len(selected), "model": model,
            "species_tested": len(species), "n_parameters": X.shape[1], "df_resid": df_resid,
            "design_condition_number": cond, "significant_FDR05": len(sig),
            "positive_FDR05": int((sig.beta > 0).sum()),
            "negative_FDR05": int((sig.beta < 0).sum()),
            "min_p": float(res.p.min()), "min_FDR": float(res.FDR.min()),
            "max_positive_beta": float(res.beta.max()), "min_negative_beta": float(res.beta.min()),
        })

        cmp = compare_to_previous(
            res, score_name,
            REPORT_DIR / f"{previous_prefix}_MWAS_{score_name}_{previous_model}.csv"
        )
        if cmp is not None:
            comparisons.append(cmp)

    combined = pd.concat(results, ignore_index=True)
    summary = pd.DataFrame(summaries)

    wide_fdr = combined.pivot(index="species", columns="score", values="FDR")
    wide_beta = combined.pivot(index="species", columns="score", values="beta")
    required = list(SCORES)
    sig4 = wide_fdr[required].lt(0.05).all(axis=1)
    pos4 = wide_beta[required].gt(0).all(axis=1)
    neg4 = wide_beta[required].lt(0).all(axis=1)
    same4 = pos4 | neg4

    shared = pd.DataFrame({
        "all4_FDR05": sig4, "all4_same_direction": same4,
        "all4_positive": pos4, "all4_negative": neg4,
    })
    for s in required:
        shared[f"{s}_beta"] = wide_beta[s]
        shared[f"{s}_FDR"] = wide_fdr[s]
    shared["shared_4of4_final"] = shared["all4_FDR05"] & shared["all4_same_direction"]
    shared_final = shared[shared["shared_4of4_final"]].reset_index()

    combined.to_csv(REPORT_DIR / f"{prefix}_MWAS_all_scores_model{model}.csv", index=False)
    summary.to_csv(REPORT_DIR / f"{prefix}_MWAS_summary_model{model}.csv", index=False)
    shared.reset_index().to_csv(REPORT_DIR / f"{prefix}_shared_species_all379_model{model}.csv", index=False)
    shared_final.to_csv(REPORT_DIR / f"{prefix}_shared_4of4_species_model{model}.csv", index=False)
    pd.concat(params, ignore_index=True).to_csv(
        REPORT_DIR / f"{prefix}_design_parameters_model{model}.csv", index=False
    )

    if comparisons:
        pd.DataFrame(comparisons).to_csv(
            REPORT_DIR / f"{prefix}_{previous_model}_vs_model{model}_summary.csv", index=False
        )

    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(summary.to_string(index=False))
    print("\nSHARED 4/4")
    print("FDR<0.05 in all 4:", int(sig4.sum()))
    print("FDR<0.05 + same direction:", len(shared_final))
    print("All positive:", int((shared["shared_4of4_final"] & shared["all4_positive"]).sum()))
    print("All negative:", int((shared["shared_4of4_final"] & shared["all4_negative"]).sum()))
    print("NOTE: the paper's 138 shared species were a 5/5 intersection including rDII;")
    print("our 4/4 count is therefore not directly comparable.")

    if comparisons:
        print("\nPREVIOUS MODEL COMPARISON")
        print(pd.DataFrame(comparisons).to_string(index=False))

    if model == 2:
        print("\nPAPER SANITY REFERENCE: Model 2 reported 197–271 significant species")
        print("per score across five dietary scores; rEDIH fewest and hPDI most.")
    else:
        print("\nMODEL 3 is the BMI-adjusted sensitivity analysis.")

    print("\nSAVED TO:", REPORT_DIR)
