#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
CGM_PACKAGE = ROOT / "cgm_deal" / "cgm论文新增17指标"

PAPER20 = {
    "Mean glucose": "cgm_mean",
    "Median glucose": "cgm_median",
    "IQR": "cgm_iqr",
    "GMI": "cgm_gmi",
    "CV": "cgm_cv",
    "MAD": "cgm_mad",
    "MAG": "cgm_mag",
    "MODD": "cgm_modd",
    "SD of ROC": "cgm_sd_roc",
    "Time-of-Day SD (SDhhmm)": "cgm_sdhhmm",
    "SDw (paper label: Interday SD)": "cgm_sdw",
    "Within-Hour SD (SDwsh)": "cgm_sdwsh",
    "TAR140": "cgm_above_140",
    "TAR180": "cgm_above_180",
    "HBGI": "cgm_hbgi",
    "LBGI": "cgm_lbgi",
    "ADRR": "cgm_adrr",
    "COGI": "cgm_cogi",
    "GRADE": "cgm_grade",
    "GRADE eugly": "cgm_grade_eugly",
}
PROJECT5 = {
    "MAGE": "cgm_mage",
    "TBR70": "cgm_below_70",
    "TIR70_180": "cgm_in_range_70_180",
    "SDb": "cgm_sdb",
    "SDdm": "cgm_sddm",
}
BOUNDED_0_100 = {
    "cgm_above_140", "cgm_above_180", "cgm_below_70",
    "cgm_in_range_70_180", "cgm_cogi", "cgm_grade_eugly",
}

def latest_final():
    xs=[]
    for p in (CGM_PACKAGE/"outputs").glob("run_*/data/07_cgm_paper_extended_phenotypes.csv"):
        if p.is_file():
            xs.append((p.stat().st_mtime,p))
    if not xs:
        raise FileNotFoundError("No finalized CGM table found")
    xs.sort(reverse=True)
    return xs[0][1]

def num(df,col):
    if col not in df.columns:
        return pd.Series(np.nan,index=df.index,dtype=float)
    x=pd.to_numeric(df[col],errors="coerce").astype(float)
    x[~np.isfinite(x)] = np.nan
    return x

def modeled(df,field):
    for s in ("_clean","_raw",""):
        c=field+s
        if c in df.columns:
            return c,num(df,c)
    return None,pd.Series(np.nan,index=df.index,dtype=float)

def q(x,p):
    y=x.dropna()
    return float(y.quantile(p)) if len(y) else np.nan

def action_n(df,field,action):
    c=field+"_clean_action"
    if c not in df.columns:
        return np.nan
    return int(df[c].astype(str).str.strip().eq(action).sum())

def z_audit(df,field):
    c=field+"_z"
    if c not in df.columns:
        return None,0,np.nan,np.nan,False
    z=num(df,c); n=int(z.notna().sum())
    m=float(z.mean()) if n else np.nan
    sd=float(z.std(ddof=1)) if n>=2 else np.nan
    ok=bool(n>=2 and np.isfinite(m) and np.isfinite(sd)
            and abs(m)<=1e-6 and abs(sd-1)<=1e-6)
    return c,n,m,sd,ok

def make_row(df,family,label,field):
    c,x=modeled(df,field)
    raw=num(df,field+"_raw") if field+"_raw" in df.columns else x
    n=len(df); nv=int(x.notna().sum()); miss=n-nv
    rv=raw.dropna(); z0=int(rv.eq(0).sum())
    zf=z0/len(rv) if len(rv) else np.nan
    uniq=int(x.dropna().nunique())
    skew=float(x.skew()) if nv>=3 else np.nan
    ex=action_n(df,field,"set_missing_beyond_8sd")
    wi=action_n(df,field,"winsorized_at_5sd")
    vals=[v for v in (ex,wi) if not (isinstance(v,float) and math.isnan(v))]
    proc=sum(int(v) for v in vals) if vals else np.nan
    proc_frac=(proc/n) if np.isfinite(proc) and n else np.nan
    bounded_violation=np.nan
    if field in BOUNDED_0_100 and len(rv):
        bounded_violation=int((~rv.between(0,100)).sum())
    zcol,zn,zm,zsd,zok=z_audit(df,field)
    flags=[]
    if not zok: flags.append("z_check")
    if n and miss/n>=0.05: flags.append("missing_ge5pct")
    if np.isfinite(zf) and zf>=0.20: flags.append("zero_ge20pct")
    if np.isfinite(zf) and zf>=0.50: flags.append("zero_ge50pct")
    if np.isfinite(skew) and abs(skew)>=2: flags.append("abs_skew_ge2")
    if np.isfinite(proc_frac) and proc_frac>=0.02: flags.append("cleaning_ge2pct")
    if uniq<20: flags.append("unique_lt20")
    if np.isfinite(bounded_violation) and bounded_violation>0: flags.append("bounded_domain_violation")
    sensitive=any(f in flags for f in ("zero_ge20pct","abs_skew_ge2","cleaning_ge2pct","unique_lt20"))
    return {
        "family":family,"label":label,"field":field,"modeled_value_column":c,
        "N_total":n,"N_valid":nv,"missing_n":miss,"missing_pct":100*miss/n,
        "unique_values":uniq,"raw_zero_n":z0,"raw_zero_fraction":zf,
        "min":float(x.min()) if nv else np.nan,"p01":q(x,.01),"p05":q(x,.05),
        "p25":q(x,.25),"median":q(x,.5),"mean":float(x.mean()) if nv else np.nan,
        "sd":float(x.std(ddof=1)) if nv>=2 else np.nan,"p75":q(x,.75),
        "p95":q(x,.95),"p99":q(x,.99),"max":float(x.max()) if nv else np.nan,
        "skewness":skew,"extreme_set_missing_n":ex,"winsorized_n":wi,
        "cleaning_fraction":proc_frac,"bounded_0_100":field in BOUNDED_0_100,
        "bounded_violation_n":bounded_violation,"z_column":zcol,
        "z_valid_n":zn,"z_mean":zm,"z_sd":zsd,"z_standardized_ok":zok,
        "distribution_sensitive_review":sensitive,
        "qc_flags":";".join(flags) if flags else "none",
    }

def corr_input(df,fields):
    d={}
    for f in fields:
        z=f+"_z"
        if z in df.columns: d[f]=num(df,z)
        else: d[f]=modeled(df,f)[1]
    return pd.DataFrame(d,index=df.index)

def high_pairs(corr,thr=.8):
    rows=[]; cols=list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            rho=corr.iloc[i,j]
            if pd.notna(rho) and abs(float(rho))>=thr:
                rows.append({"outcome_1":cols[i],"outcome_2":cols[j],
                             "spearman_rho":float(rho),"abs_rho":abs(float(rho))})
    if not rows:
        return pd.DataFrame(columns=["outcome_1","outcome_2","spearman_rho","abs_rho"])
    return pd.DataFrame(rows).sort_values("abs_rho",ascending=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cgm-csv",default=None)
    ap.add_argument("--out-dir",default=None)
    a=ap.parse_args()
    src=Path(a.cgm_csv).expanduser().resolve() if a.cgm_csv else latest_final()
    if not src.is_file(): raise FileNotFoundError(src)
    df=pd.read_csv(src,low_memory=False)
    if len(df)!=7493: raise RuntimeError(f"Expected N=7493, got {len(df)}")
    if "participant_id" not in df.columns: raise RuntimeError("participant_id missing")
    if df["participant_id"].astype(str).duplicated().any(): raise RuntimeError("duplicate participant_id")
    defs=[("paper20_primary",l,f) for l,f in PAPER20.items()] +          [("project_supplementary",l,f) for l,f in PROJECT5.items()]
    missing=[f+"_z" for _,_,f in defs if f+"_z" not in df.columns]
    if missing: raise RuntimeError("Missing required Z columns: "+", ".join(missing))
    qc=pd.DataFrame([make_row(df,*x) for x in defs])
    pf=list(PAPER20.values()); af=pf+list(PROJECT5.values())
    pc=corr_input(df,pf).corr(method="spearman",min_periods=100)
    ac=corr_input(df,af).corr(method="spearman",min_periods=100)
    pp=high_pairs(pc); apairs=high_pairs(ac)
    review=qc.loc[qc["distribution_sensitive_review"] | qc["qc_flags"].ne("none")].copy()
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=Path(a.out_dir).expanduser().resolve() if a.out_dir else CGM_PACKAGE/"outputs"/f"paper20_plus5_qc_{stamp}"
    out.mkdir(parents=True,exist_ok=False)
    qc.to_csv(out/"01_outcome_qc_summary.csv",index=False)
    pc.to_csv(out/"02_paper20_spearman.csv")
    ac.to_csv(out/"03_all25_spearman.csv")
    pp.to_csv(out/"04_paper20_high_correlation_pairs.csv",index=False)
    apairs.to_csv(out/"05_all25_high_correlation_pairs.csv",index=False)
    review.to_csv(out/"06_model_review_flags.csv",index=False)
    paper=qc[qc.family.eq("paper20_primary")]
    supp=qc[qc.family.eq("project_supplementary")]
    lines=[
        "=== CGM PAPER-20 + PROJECT-5 QC ===",
        f"INPUT={src}",f"N={len(df)}",
        f"PAPER20_PRESENT={len(paper)}/20",f"PROJECT5_PRESENT={len(supp)}/5",
        f"ALL25_Z_STANDARDIZED={bool(qc.z_standardized_ok.all())}",
        f"PAPER20_DISTRIBUTION_REVIEW_N={int(paper.distribution_sensitive_review.sum())}",
        f"PROJECT5_DISTRIBUTION_REVIEW_N={int(supp.distribution_sensitive_review.sum())}",
        f"PAPER20_ABS_RHO_GE_0_80_PAIRS={len(pp)}",
        f"ALL25_ABS_RHO_GE_0_80_PAIRS={len(apairs)}","",
        "--- OUTCOME QC ---",
        qc[["family","label","field","N_valid","missing_pct","raw_zero_fraction",
            "skewness","extreme_set_missing_n","winsorized_n","z_standardized_ok",
            "distribution_sensitive_review","qc_flags"]].to_string(index=False),
        "","--- PAPER20 HIGH CORRELATION PAIRS |rho|>=0.80 ---",
        "NONE" if pp.empty else pp.to_string(index=False),
        "","--- REVIEW FLAGS ---",
        "NONE" if review.empty else review[["family","label","field","raw_zero_fraction",
            "skewness","cleaning_fraction","qc_flags"]].to_string(index=False),
        "","INTERPRETATION:",
        "- QC flags are review triggers, not automatic exclusions.",
        "- Paper-20 is the primary phenotype family.",
        "- MAGE/TBR70/TIR70_180/SDb/SDdm remain supplementary.",
        "- High phenotype correlation is descriptive; do not collapse outcomes post hoc.",
        "- Zero-inflated/heavily skewed outcomes need model-sensitivity review before treating ordinary OLS as sufficient.",
    ]
    (out/"07_qc_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps({
        "status":"completed","input":str(src),"N":len(df),
        "paper20_present":len(paper),"project5_present":len(supp),
        "all25_z_standardized":bool(qc.z_standardized_ok.all()),
        "paper20_review_n":int(paper.distribution_sensitive_review.sum()),
        "project5_review_n":int(supp.distribution_sensitive_review.sum()),
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("\n".join(lines))
    print(f"\nOUTPUT_DIR={out}")

if __name__ == "__main__":
    main()
