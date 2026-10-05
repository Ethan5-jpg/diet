#!/usr/bin/env python3
"""Export existing same-visit gut associations. No model fitting or new filtering."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026.10.04.1"
PACKAGE = Path(__file__).resolve().parents[1]
TAXA = {"T023": "GGB6012_SGB8640", "T046": "Clostridium phoceensis",
        "T068": "Eubacterium rectale", "T096": "Ruthenibacterium lactatiformans"}
MEAL = ["mean_increment", "baseline_glucose", "peak_rise_0_120"]
REFERENCE = ["cgm_mean_z", "cgm_cv_z", "cgm_sdw_z"]
SETTINGS = ["primary_common", "strict_common"]
TITLES = {"mean_increment": "Mean postprandial increment", "baseline_glucose": "Pre-meal glucose",
          "peak_rise_0_120": "Peak glucose rise", "cgm_mean_z": "Whole-record CGM mean",
          "cgm_cv_z": "Whole-record CGM CV", "cgm_sdw_z": "Whole-record CGM SDw"}
EXPORT = ["setting", "taxon", "species", "outcome", "status", "people", "meals", "estimate",
          "se", "ci_low", "ci_high", "p", "q_family", "q_meal12", "family",
          "planned_family_size", "unit", "adjustment", "reason", "design_condition",
          "design_columns", "absorbed_effects", "residual_df", "inference_df", "se_method", "dropped_constants"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def bh(values):
    p = np.asarray(values, float)
    finite = np.isfinite(p)
    work = np.where(finite, p, 1.)
    order = np.argsort(work, kind="stable")
    n = len(work)
    ranked = np.minimum.accumulate((work[order]*n/np.arange(1,n+1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked,1.)
    q[~finite] = np.nan
    return q


def family(outcome):
    return "net_primary" if outcome == "mean_increment" else (
        "baseline_peak_secondary" if outcome in MEAL else "whole_cgm_reference")


def validate_models(d, sample):
    keys = ["setting", "taxon", "outcome"]
    expected = {(s,t,o) for s in SETTINGS for t in TAXA for o in MEAL+REFERENCE}
    require(len(d)==48 and not d.duplicated(keys).any() and
            set(d[keys].itertuples(index=False,name=None))==expected, "Expected exactly 48 original specifications")
    require(set(d.status).issubset({"ESTIMATED", "NOT_ESTIMATED"}), "Unexpected model status")
    require(not sample.setting.duplicated().any() and set(sample.setting)==set(SETTINGS), "Invalid sample table")
    for col in ["estimate","se","ci_low","ci_high","p","q_family","q_meal12","people","meals"]:
        require(col in d, "Missing model field: " + col)
    for col in ["people","meals"]:
        require(np.isfinite(d[col]).all() and d[col].ge(0).all() and d[col].mod(1).eq(0).all(), "Invalid sample counts")
    estimated = d.loc[d.status.eq("ESTIMATED")]
    require(np.isfinite(estimated[["estimate","se","ci_low","ci_high","p","q_family"]]).all().all(), "Nonfinite estimated model")
    require((estimated.se.ge(0) & estimated.p.between(0,1) & estimated.q_family.between(0,1) &
             estimated.ci_low.le(estimated.estimate) & estimated.ci_high.ge(estimated.estimate)).all(), "Invalid estimated statistics")
    require(d.loc[d.status.ne("ESTIMATED"),["estimate","p","q_family"]].isna().all().all(), "NE model contains effect/p/q")
    require(d.family.eq(d.outcome.map(family)).all(), "Family assignment differs from original contract")
    for s,g in d.groupby("setting"):
        r = sample.set_index("setting").loc[s]
        for col in ["people","meals","person_sha256","meal_sha256"]:
            require(g[col].nunique(dropna=False)==1 and g.iloc[0][col]==r[col], "Common sample differs: "+s+"/"+col)
        for fam,part in g.groupby("family"):
            size = {"net_primary":4,"baseline_peak_secondary":8,"whole_cgm_reference":12}[fam]
            require(len(part)==size and part.planned_family_size.eq(size).all(), "Incorrect family size")
            require(np.allclose(bh(part.p),part.q_family,rtol=1e-7,atol=1e-12,equal_nan=True), "Original family q mismatch")
        part = g.loc[g.outcome.isin(MEAL)]
        require(np.allclose(bh(part.p),part.q_meal12,rtol=1e-7,atol=1e-12,equal_nan=True), "Original meal q12 mismatch")
    counts = sample.set_index("setting")
    require(counts.loc["strict_common","people"]<=counts.loc["primary_common","people"], "Strict count exceeds primary")


def load_full(source):
    source = Path(source).resolve()
    manifest_file = source/"manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    state = manifest["status"]["gut_meal_bridge_v2"]
    require(state["status"] in ["COMPLETE","COMPLETE_WITH_LIMITATIONS"], "Source run is not complete")
    require(manifest["data_kind"] in ["REAL","SYNTHETIC"], "Unverified data kind")
    require(state.get("version")=="2026.09.29.2", "Requires original gut bridge version 2026.09.29.2")
    pins = {manifest_file:digest(manifest_file)}
    frames = {}
    for name in ["models","sample","02_sample_flow","timing","descriptive"]:
        relative = "tables/"+name+".csv"
        p = source/relative
        expected = state["result_sha256"].get(relative)
        require(expected and p.is_file() and digest(p)==expected, "Missing/changed source result: "+relative)
        pins[p]=expected
        frames[name]=pd.read_csv(p,float_precision="round_trip")
    d,sample = frames["models"],frames["sample"]
    validate_models(d,sample)
    require(int(d.status.eq("ESTIMATED").sum())==state["estimated"], "Estimated count differs from source manifest")
    return frames, pins, manifest["data_kind"], False


def load_preview(path):
    path = Path(path).resolve()
    body = json.loads(path.read_text(encoding="utf-8"))
    require(body.get("data_label_reported") in ["REAL","SYNTHETIC"], "Unverified screenshot data label")
    mapping = dict(net=MEAL[0],pre=MEAL[1],peak=MEAL[2],Mean=REFERENCE[0],CV=REFERENCE[1],SDw=REFERENCE[2])
    rows=[]
    for raw in body["models"]:
        row=dict(raw)
        row.update(outcome=mapping[row["outcome"]], people=row.pop("N"),status="ESTIMATED")
        row["family"]=family(row["outcome"])
        row["planned_family_size"]=row.pop("family_size")
        rows.append(row)
    d=pd.DataFrame(rows)
    keys=["setting","taxon","outcome"]
    require(not d.duplicated(keys).any(), "Duplicate screenshot result")
    expected={(s,t,o) for s in SETTINGS for t in TAXA for o in MEAL+REFERENCE}
    require(set(d[keys].itertuples(index=False,name=None)).issubset(expected), "Unexpected screenshot specification")
    meal=d.loc[d.outcome.isin(MEAL)]
    require(len(meal)==24 and meal[["estimate","ci_low","ci_high","q_family"]].notna().all().all(), "Need all 24 meal effect/CI/q records for preview")
    require(np.isfinite(meal[["estimate","ci_low","ci_high","q_family"]]).all().all() and
            (meal.ci_low.le(meal.estimate)&meal.ci_high.ge(meal.estimate)&meal.q_family.between(0,1)).all(), "Invalid screenshot meal statistics")
    samples=pd.DataFrame(body["samples"])
    require(not samples.setting.duplicated().any() and set(samples.setting)==set(SETTINGS), "Invalid screenshot samples")
    for s,g in d.groupby("setting"):
        require(g.people.eq(samples.set_index("setting").loc[s,"people"]).all(), "Screenshot sample count mismatch")
    frames=dict(models=d,sample=samples)
    if "known_date" in samples:
        frames["timing"]=samples.copy()
    return frames,{path:digest(path)},body["data_label_reported"]+"_SCREENSHOT_ROUNDED",True


def num(x):
    return "NA" if pd.isna(x) else "{:.4g}".format(float(x))


def qnum(x):
    return "NA" if pd.isna(x) else ("0*" if float(x)==0 else num(x))


def cell(row):
    if row.status != "ESTIMATED":
        return "NE"
    return "{} [{}, {}]; q={}".format(num(row.estimate),num(row.ci_low),num(row.ci_high),qnum(row.q_family))


def forest(d,sample,out,preview,reference=False,data_kind="REAL"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"ps.fonttype":42,
                         "svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    outcomes=REFERENCE if reference else MEAL
    fig,axes=plt.subplots(3,1,figsize=(10.5,10.2))
    colors={"primary_common":"#146C94","strict_common":"#C16B22"}
    for k,(ax,outcome) in enumerate(zip(axes,outcomes)):
        selected=d.loc[d.outcome.eq(outcome)]
        valid=selected.loc[selected.status.eq("ESTIMATED")]
        require(valid[["ci_low","ci_high"]].notna().all().all(), "Missing plotted CI; do not fabricate from q")
        extent=max(abs(valid.ci_low.min()),abs(valid.ci_high.max()),.15) if len(valid) else 1.
        ax.set_xlim(-extent*1.15,extent*1.15)
        ax.axvline(0,color="#9AA7B0",lw=.8,ls="--",zorder=1)
        for i,t in enumerate(TAXA):
            if i%2==0:
                ax.axhspan(i-.47,i+.47,color="#F2F5F7",zorder=0)
            for s,shift,marker in [(SETTINGS[0],-.13,"o"),(SETTINGS[1],.13,"s")]:
                row=selected.loc[selected.taxon.eq(t)&selected.setting.eq(s)].iloc[0]
                if row.status=="ESTIMATED":
                    ax.errorbar(row.estimate,i+shift,xerr=[[row.estimate-row.ci_low],[row.ci_high-row.estimate]],
                                fmt=marker,color=colors[s],ms=5.5,lw=1.3,capsize=2.5,zorder=3)
                else:
                    ax.text(0,i+shift,"NE",fontsize=8,color=colors[s],ha="center",va="center")
                ax.text(1.04 if s==SETTINGS[0] else 1.22,i,qnum(row.q_family),
                        transform=ax.get_yaxis_transform(),ha="left",va="center",fontsize=9,color=colors[s])
        ax.set_yticks(range(4))
        ax.set_yticklabels([t+"  "+name for t,name in TAXA.items()],fontsize=9)
        ax.set_ylim(3.6,-.7)
        ax.tick_params(axis="y",length=0,pad=10)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#9AA7B0")
        ax.grid(axis="x",alpha=.15)
        ax.set_title(chr(65+k)+"  "+TITLES[outcome],loc="left",fontsize=12,fontweight="bold",pad=14)
        ax.set_xlabel("Difference in original CGM Z per original CLR-Z" if reference else "Difference in mg/dL per original CLR-Z",fontsize=9)
        ax.text(1.04,1.01,"q primary",transform=ax.transAxes,fontsize=8,color=colors[SETTINGS[0]])
        ax.text(1.22,1.01,"q strict",transform=ax.transAxes,fontsize=8,color=colors[SETTINGS[1]])
    counts=sample.set_index("setting")
    legend=[Line2D([0],[0],color=colors[s],marker=m,linestyle="none",
        label="{} (n={:,})".format(label,int(counts.loc[s,"people"]))) for s,m,label in [(SETTINGS[0],"o","Primary"),(SETTINGS[1],"s","Strict")]]
    fig.suptitle("Same-visit gut microbiome and "+("whole-record CGM" if reference else "meal glucose phenotypes"),x=.035,y=.982,ha="left",fontsize=15,fontweight="bold")
    fig.legend(handles=legend,loc="upper left",bbox_to_anchor=(.035,.958),frameon=False,ncol=2,fontsize=10)
    qualifier=("PREVIEW: "+data_kind+"; no new model fits.") if preview else (
        "SYNTHETIC TEST DATA; not HPP findings." if data_kind=="SYNTHETIC" else "Source: verified full-precision original model table; no new model fits.")
    text=qualifier+"\nPoints: adjusted coefficients; bars: pointwise 95% CIs. Strict is an overlapping subset.\n"
    text+=("q: 12 whole-CGM reference tests per sample." if reference else "q: 4 net-increment tests; 8 pre-meal/peak tests per sample. Available all-meal q12 values are in tables.")
    text+="\n* Numerical underflow, if shown; not exact zero."
    fig.text(.035,.021,text,ha="left",va="bottom",fontsize=8,color="#45545D",linespacing=1.6)
    fig.subplots_adjust(left=.33,right=.79,top=.885,bottom=.14,hspace=.70)
    name="Figure2_whole_CGM_reference" if reference else "Figure1_meal_associations"
    for ext in ["png","pdf","svg"]:
        fig.savefig(out/(name+"."+ext),dpi=220,facecolor="white")
    plt.close(fig)


def markdown_table(d,outcomes):
    lines=["| 结局 | 菌 | 主样本 β [95%CI]; q | strict β [95%CI]; q |",
           "|---|---|---|---|"]
    for o in outcomes:
        for t in TAXA:
            cells=[]
            for s in SETTINGS:
                r=d.loc[d.setting.eq(s)&d.taxon.eq(t)&d.outcome.eq(o)]
                cells.append(cell(r.iloc[0]) if len(r) else "未在本地截图显示")
            lines.append("| {} | {} | {} | {} |".format(TITLES[o],t,*cells))
    return "\n".join(lines)


def reports(d,sample,out,preview,data_kind):
    counts=sample.set_index("setting")
    a,b=counts.loc[SETTINGS[0]],counts.loc[SETTINGS[1]]
    provenance=("截图精度预览；未重新估计模型。未显示值不补造，正式写作待服务器全精度导出。" if preview else
        "原服务器模型表已通过SHA256、规格、共同样本及FDR核对；本次只导出，不重新拟合。")
    stats=[]
    for s in SETTINGS:
        for o in MEAL+REFERENCE:
            g=d.loc[d.setting.eq(s)&d.outcome.eq(o)]
            stats.append(dict(setting=s,outcome=o,available_rows=len(g),estimated=int(g.status.eq("ESTIMATED").sum()),
                              q_family_below_005=int((g.status.eq("ESTIMATED")&g.q_family.lt(.05)).sum())))
    pd.DataFrame(stats).to_csv(out/"tables/significance_counts.csv",index=False)
    def hits(s,o):
        g=d.loc[d.setting.eq(s)&d.outcome.eq(o)&d.status.eq("ESTIMATED")&d.q_family.lt(.05)]
        return ", ".join(g.taxon) or "none"
    def effects(s,o):
        g=d.loc[d.setting.eq(s)&d.outcome.eq(o)&d.status.eq("ESTIMATED")&d.q_family.lt(.05)]
        return "; ".join("{}: beta {} mg/dL, 95% CI {} to {}, q {}".format(r.taxon,num(r.estimate),num(r.ci_low),num(r.ci_high),qnum(r.q_family)) for r in g.itertuples()) or "none met the threshold"
    result_en=("In the primary sample, {} participants contributed {} eligible meals; the overlapping strict subset included {} participants and {} meals. "
        "The four candidates were selected in earlier analyses of the same cohort. Using the original multiple-testing families, associations with pre-meal glucose met q < 0.05 for {}. "
        "For the primary outcome, mean postprandial increment, candidates meeting the threshold were {}; for peak rise, they were {}. "
        "In the strict subset, candidates meeting the threshold for mean increment were {}. "
        "These comparisons describe associations conditional on the specified covariates. Different significance patterns across outcomes or overlapping samples do not establish differences between effects, independent replication, or causal pathways.").format(
            int(a.people),int(a.meals),int(b.people),int(b.meals),hits(SETTINGS[0],MEAL[1]),hits(SETTINGS[0],MEAL[0]),hits(SETTINGS[0],MEAL[2]),hits(SETTINGS[1],MEAL[0]))
    result_en+="\n\nPrimary-sample pre-meal estimates per original CLR-Z unit were {}. Strict-subset mean-increment estimates meeting the original family threshold were {}.".format(effects(SETTINGS[0],MEAL[1]),effects(SETTINGS[1],MEAL[0]))
    if d.status.ne("ESTIMATED").any():
        result_en+=" Unestimable specifications are marked NE in the tables and do not support either presence or absence of an association."
    report="# 同访视四候选菌关联结果\n\nDATA={}\n\n{}\n\n".format(data_kind,provenance)
    report+="本次按已确认口径：保留每餐CGM质控、每人至少5餐/3日及必要变量完整性；不新增全程缺失<5%门槛，不按菌群采样先后或日期未知排除。\n\n"
    report+="主样本：{:,}人、{:,}餐；strict：{:,}人、{:,}餐。个人等权，餐不是独立的回归样本。\n\n".format(int(a.people),int(a.meals),int(b.people),int(b.meals))
    report+="按原定家族q<0.05，主样本平均净增量通过的候选：{}；餐前血糖：{}；峰值升幅：{}。strict中平均净增量通过的候选：{}。none表示没有候选通过，不等于证明无关联；NE或本地未显示的模型不能作为阴性结果。\n\n".format(hits(SETTINGS[0],MEAL[0]),hits(SETTINGS[0],MEAL[1]),hits(SETTINGS[0],MEAL[2]),hits(SETTINGS[1],MEAL[0]))
    report+="## 餐次关联\n\n"+markdown_table(d,MEAL)+"\n\n"
    report+="单位：mg/dL/原菌群CLR-Z。餐前水平不是空腹血糖；平均净增量为净iAUC/120，可为负；峰值升幅取实际观测的峰值减餐前基线。\n\n"
    report+="## 同连接整体CGM参照\n\n"+markdown_table(d,REFERENCE)+"\n\n"
    report+="参照单位为原CGM Z/原CLR-Z，不能与餐次系数直接比较大小；监测时长/覆盖影响仍适用于整体CGM。整体和餐次可能共享读数，不是独立验证。\n\n"
    report+="## 结果解读边界\n\n主结局仍为平均餐后净增量，不因餐前更显著而改变主次。候选由同队列选择；FDR没有消除选择后推断问题。strict与主样本重叠。缺失只在不涉及某餐所需读数时不直接改变该餐计算，仍可能影响餐次代表性。饮食日志也可能漏记进食。菌样本身份/访视仍须正确匹配；同访视允许关联分析，不单独建立时间因果顺序。\n\n"
    report+="## 英文Results草稿\n\n"+result_en+"\n\n所有模型、阴性结果和缺失字段均保留。p/q显示为0只表示数值下溢，不代表真实概率为0。\n"
    (out/"RESULTS_zh.md").write_text(report,encoding="utf-8")
    (out/"RESULTS_en.md").write_text("# Same-visit associations\n\nDATA="+data_kind+"\n\n"+provenance+"\n\n"+result_en+"\n",encoding="utf-8")
    methods="""# Methods draft: selected-candidate meal associations

Four previously selected microbial candidates (GGB6012_SGB8640, Clostridium phoceensis, Eubacterium rectale and Ruthenibacterium lactatiformans) were evaluated using their original centred log-ratio standardized values (CLR-Z). Microbiome measurements were treated as measurements from the same study visit as diet and CGM. Sample timing was retained for descriptive auditing and was not an eligibility criterion. The analysis was observational and did not assume prospective or causal ordering.

Meals were retained using the existing eligibility, logged adjacent-intake separation and CGM quality rules. The 30-min pre-meal window required at least two valid readings, with the nearest reading no more than 15 min before meal onset; their median defined the pre-meal baseline. The 120-min post-meal window required at least seven actual readings; gaps exceeding 20 min, invalid readings and endpoints lacking valid support were rejected. Short bracketing interpolation was used only where allowed by the original endpoint rules; peak rise was derived from actual observations. Participants contributed at least five eligible meals across at least three dates. No additional fixed 14-day completeness threshold was imposed for this meal association analysis. Missingness may still affect the number and representativeness of observed meals.

All four microbial exposures, outcomes and required covariates were available in a common analysis sample. Eligible meals were summarized within each participant and participants received equal weights in ordinary least-squares regression. HC3 standard errors and residual-degrees-of-freedom t inference produced pointwise 95% confidence intervals. Exposures and whole-CGM reference outcomes retained their original standardization. A separate model was fitted for each taxon; the four taxa were not mutually adjusted.

All models adjusted for age, sex, education, smoking, usual sleep duration, usual physical activity, vitamin use, hormone use, BMI, CGM device, corrected hPDI, participant NOVA4, mean daily energy and alcohol intake. Meal models additionally adjusted for mean meal carbohydrate content, the mean of squared meal carbohydrate content, meal energy and meal timing/context variables (sine/cosine of hour, previous-intake gap, observation day and meal duration). Net increment and peak rise additionally adjusted for mean pre-meal glucose and pre-meal slope. The pre-meal glucose model did not adjust for its own outcome or slope. Mean increment was net incremental area under the 0–120-min curve divided by 120; negative values were retained. Pre-meal glucose was not equated with fasting glucose.

The primary sample excluded participants with known diabetes/A10-positive status under the existing eligibility definition while permitting unknown status when required covariates were complete. The strict sensitivity sample retained only participants explicitly classified as negative under that definition. These samples overlap and do not constitute independent replication.

Benjamini–Hochberg correction was performed separately within each sample for four primary net-increment tests, eight secondary pre-meal/peak tests and twelve whole-CGM reference tests. A supplementary q12 covered all twelve meal tests. Unestimable specifications remained in the planned multiplicity denominators. Whole-CGM reference outcomes were original same-connection mean, CV and SDw values with background/diet adjustment; they may share glucose observations with the meal outcomes and are not newly calculated fixed-14-day phenotypes. The four taxa were selected in earlier analyses of this cohort; results therefore remain exploratory selected-candidate associations.

This export reused the completed original models without refitting. Formal submission requires review of the full-precision server tables, eligibility definitions, original score definitions and cohort-level ethics/data-access statements.
"""
    methods=methods.replace("# Methods draft: selected-candidate meal associations\n", "# Methods draft: selected-candidate meal associations\n\nDATA="+data_kind+"\n\n"+provenance+"\n",1)
    (out/"Methods_en.md").write_text(methods,encoding="utf-8")
    caption=("# Figure legends\n\nDATA="+data_kind+"\n\nFigure 1. Same-visit associations of four previously selected microbial candidates with meal glucose phenotypes. "
        "Panels show mean postprandial increment (A; original primary outcome), pre-meal glucose (B) and peak rise (C). "
        "Circles and squares show the primary and overlapping strict samples. Points are adjusted coefficients per original CLR-Z unit; error bars are pointwise 95% confidence intervals. "
        "Family q values are shown at right; the original families comprise four net-increment tests and eight pre-meal/peak tests per sample. "
        "Supplementary q12 values for the twelve meal tests are retained in the full server tables; unshown preview values remain missing. See Methods for covariates and meal eligibility. "
        "No additional 14-day missingness or microbiome date-order filter is applied.\n\n"
        "Figure 2 (full server export only). Same-participant, same-connection whole-CGM reference associations for mean, CV and SDw. "
        "Units are original CGM Z per original CLR-Z; q uses twelve reference tests per sample. "
        "These models use background/diet covariates and may share readings with meal analyses.\n\n"+provenance+"\n")
    (out/"Figure_legends.md").write_text(caption,encoding="utf-8")
    (out/"TERMS.md").write_text("# 统一术语\n\n| 中文 | English | 定义/边界 |\n|---|---|---|\n| 同访视关联 | same-visit association | 不声称前瞻预测 |\n| 餐前血糖 | pre-meal glucose | 餐前30分钟基线，不等于空腹 |\n| 餐后平均净增量 | mean postprandial increment | 净iAUC/120，原主结局，可为负 |\n| 峰值升幅 | peak glucose rise | 0–120分钟实际观测峰值减基线 |\n| 原菌群CLR-Z | original microbial CLR-Z | 非绝对菌量，筛样本后不重新标准化 |\n| 严格样本 | strict subset | 与主样本重叠，不是独立复制 |\n",encoding="utf-8")


def create(args):
    source=Path(args.preview_json or args.source).expanduser().resolve()
    root=Path(args.output_root).expanduser().resolve()
    require(root != source and source not in root.parents, "Output must be outside the source")
    for p in [root]+list(root.parents):
        require(not (p/"manifest.json").exists(), "Output cannot be inside an existing run")
    frames,pins,data_kind,preview=load_preview(source) if args.preview_json else load_full(source)
    root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out=root/(("gut_same_visit_PREVIEW_" if preview else "gut_same_visit_complete_")+stamp)
    out.mkdir(exist_ok=False)
    state=dict(version=VERSION,status="RUNNING",data_kind=data_kind,source=str(source),
        preview=preview,refit=False,date_order_filter=False,additional_14day_filter=False,
        source_sha256={str(p):h for p,h in pins.items()},code_sha256=digest(Path(__file__)))
    write_json(out/"manifest.json",state)
    try:
        (out/"tables").mkdir(); (out/"figures").mkdir()
        d=frames["models"].copy()
        d["species"]=d.taxon.map(TAXA)
        for col in EXPORT:
            if col not in d:
                d[col]=np.nan
        d=d.sort_values(["setting","outcome","taxon"])
        public=d[EXPORT]
        public.to_csv(out/"tables/all_available_models.csv",index=False,float_format="%.17g")
        public.loc[public.outcome.isin(MEAL)].to_csv(out/"tables/meal_models_24.csv",index=False,float_format="%.17g")
        public.loc[public.outcome.isin(REFERENCE)].to_csv(out/"tables/whole_CGM_reference.csv",index=False,float_format="%.17g")
        allowed={"sample":["setting","people","meals","meals_p50","dates_p50"],
            "02_sample_flow":["stage","people","meals"],
            "timing":["setting","people","known_date","unknown_date","ambiguous","before","within","after","first_gap_p50","distant_over_30d"],
            "descriptive":["setting","field","people","mean","sd","q25","median","q75"]}
        for name,cols in allowed.items():
            if name in frames:
                frames[name][[c for c in cols if c in frames[name]]].to_csv(out/"tables"/(name+".csv"),index=False,float_format="%.17g")
        available=set(d[["setting","taxon","outcome"]].itertuples(index=False,name=None))
        missing=[dict(setting=s,taxon=t,outcome=o,status="NOT_IN_LOCAL_SCREENSHOTS") for s in SETTINGS for t in TAXA for o in MEAL+REFERENCE if (s,t,o) not in available]
        pd.DataFrame(missing,columns=["setting","taxon","outcome","status"]).to_csv(out/"tables/missing_local_specs.csv",index=False)
        reports(d,frames["sample"],out,preview,data_kind)
        forest(d,frames["sample"],out/"figures",preview,data_kind=data_kind)
        if not preview:
            forest(d,frames["sample"],out/"figures",False,reference=True,data_kind=data_kind)
        for p,h in pins.items():
            require(digest(p)==h,"Source changed during export")
        state.update(status="PREVIEW_ONLY" if preview else "COMPLETE_EXPORT",available_models=len(d),
            missing_local_specs=len(missing),estimated_models=int(d.status.eq("ESTIMATED").sum()),
            result_sha256={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="manifest.json"})
        write_json(out/"manifest.json",state)
        print("RUN_DIR="+str(out))
        print("STATUS={} DATA={} MODELS={}/48 REFIT=NO".format(state["status"],data_kind,len(d)))
        print("FULL_FDR_CHECK="+("NOT_APPLICABLE_TO_ROUNDED_SCREENSHOTS" if preview else "PASS"))
        print("SOURCES_UNCHANGED=PASS; individual files/identifiers are not exported.")
        print(pd.read_csv(out/"tables/sample.csv").to_string(index=False))
        print(pd.read_csv(out/"tables/significance_counts.csv").to_string(index=False))
        print("Share only the new aggregate export folder; do not include source private/.")
        return out
    except Exception as error:
        state.update(status="FAILED",error=str(error))
        write_json(out/"manifest.json",state)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--source",default=str(PACKAGE/"outputs/gut_meal_bridge_v2_20260929T044951_469958Z"))
    group.add_argument("--preview-json")
    parser.add_argument("--output-root",default=str(PACKAGE/"outputs"))
    args=parser.parse_args(argv)
    try:
        create(args)
        return 0
    except Exception as error:
        print("GUT_COMPLETION_FAILED: "+str(error),file=sys.stderr)
        return 2


if __name__=="__main__":
    sys.exit(main())
