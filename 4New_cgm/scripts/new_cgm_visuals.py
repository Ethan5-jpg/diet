#!/usr/bin/env python3
"""Descriptive tables and static figures from saved association results only."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from new_cgm_data import EXPOSURES, OUTCOMES, fingerprint

COLORS = {0: "#65748B", 2: "#237B83", 3: "#CA7B36"}
POSITIVE, NEGATIVE, NEUTRAL = "#B74446", "#3679A8", "#BFC5CE"
OUTCOME_LABELS = {"MAGE": "MAGE", "TAR180": "TAR180", "TBR70": "TBR70", "TIR70_180": "TIR70–180"}
DIET_LABELS = {name: name for name in EXPOSURES}
DIET_LABELS["Carbohydrate_pct"] = "Carbohydrate %"
REQUIRED = ["analysis_set", "model", "outcome", "N", "beta", "CI95_lower", "CI95_upper", "p_value", "FDR_family", "FDR_global", "status"]


def prepare_results(frame, branch):
    feature = "exposure" if branch == "diet" else "species"
    missing = [c for c in REQUIRED + [feature] if c not in frame]
    if missing:
        raise ValueError("%s visualization input missing columns: %s" % (branch, missing))
    primary = frame.loc[frame.analysis_set.eq("primary")].copy()
    if primary.empty:
        raise ValueError("No primary models in " + branch)
    for c in ["model", "N", "beta", "CI95_lower", "CI95_upper", "p_value", "FDR_family", "FDR_global"]:
        primary[c] = pd.to_numeric(primary[c], errors="raise")
    if primary.duplicated(["model", "outcome", feature]).any():
        raise ValueError("Duplicate primary associations: " + branch)
    if not primary.outcome.isin(OUTCOMES).all() or not primary.model.isin([0, 2, 3]).all():
        raise ValueError("Unknown model/outcome in visualization input")
    if branch == "diet" and not primary.exposure.isin(EXPOSURES).all():
        raise ValueError("Unknown diet exposure")
    good = primary.status.eq("computed")
    if not np.isfinite(primary.loc[good, ["N", "beta", "CI95_lower", "CI95_upper", "p_value", "FDR_family", "FDR_global"]].to_numpy()).all():
        raise ValueError("Computed models contain nonfinite statistics")
    for c in ["p_value", "FDR_family", "FDR_global"]:
        if not primary.loc[good, c].between(0, 1).all():
            raise ValueError("Invalid p/q values in " + c)
    primary["family_sig"] = good & primary.FDR_family.lt(0.05)
    primary["global_sig"] = good & primary.FDR_global.lt(0.05)
    return primary


def intuitive_counts(diet, micro):
    rows = []
    for branch, frame in [("diet", diet), ("microbiome", micro)]:
        for model in [0, 2, 3]:
            for outcome in OUTCOMES:
                group = frame.loc[frame.model.eq(model) & frame.outcome.eq(outcome)]
                good = group.status.eq("computed")
                rows.append({"branch": branch, "model": model, "outcome": outcome,
                             "planned_tests": len(group), "computed_tests": int(good.sum()),
                             "failed_tests": int((~good).sum()),
                             "N_min": group.N.min(), "N_max": group.N.max(),
                             "family_significant_positive": int((group.family_sig & group.beta.gt(0)).sum()),
                             "family_significant_negative": int((group.family_sig & group.beta.lt(0)).sum()),
                             "family_significant_total": int(group.family_sig.sum()),
                             "global_significant_total": int(group.global_sig.sum())})
    return pd.DataFrame(rows)


def heatmap_arrays(frame, feature, order):
    beta = np.full((len(order), len(OUTCOMES)), np.nan)
    family = np.zeros(beta.shape, dtype=bool)
    global_sig = np.zeros(beta.shape, dtype=bool)
    for i, name in enumerate(order):
        for j, outcome in enumerate(OUTCOMES):
            match = frame.loc[frame[feature].eq(name) & frame.outcome.eq(outcome)]
            if not match.empty and match.iloc[0].status == "computed":
                beta[i, j] = match.iloc[0].beta
                family[i, j] = match.iloc[0].family_sig
                global_sig[i, j] = match.iloc[0].global_sig
    return beta, family, global_sig


def draw_heatmap(ax, arrays, row_labels, vmax, title):
    beta, family, global_sig = arrays
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E8EBEF")
    image = ax.imshow(np.ma.masked_invalid(beta), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(OUTCOMES)))
    ax.set_xticklabels([OUTCOME_LABELS[x] for x in OUTCOMES], rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.tick_params(length=0)
    for i in range(beta.shape[0]):
        for j in range(beta.shape[1]):
            value = beta[i, j]
            marker = ("*" if family[i, j] else "") + ("+" if global_sig[i, j] else "")
            label = "NA" if not np.isfinite(value) else "%.2f%s" % (value, marker)
            color = "white" if np.isfinite(value) and abs(value) > 0.65 * vmax else "#172B3A"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def save_figure(fig, folder, stem, synthetic_label=None):
    if synthetic_label:
        fig.text(0.995, 0.002, synthetic_label, ha="right", va="bottom", fontsize=9, color="#AA3333")
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = folder / (stem + "." + suffix)
        fig.savefig(str(path), dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def diet_heatmap(diet):
    arrays = [heatmap_arrays(diet.loc[diet.model.eq(m)], "exposure", list(EXPOSURES)) for m in (0, 2, 3)]
    finite = np.concatenate([a[0].ravel() for a in arrays])
    finite = finite[np.isfinite(finite)]
    vmax = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 0.01)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.subplots_adjust(left=0.11, right=0.91, bottom=0.22, top=0.82, wspace=0.48)
    for ax, values, model in zip(axes, arrays, [0, 2, 3]):
        im = draw_heatmap(ax, values, [DIET_LABELS[x] for x in EXPOSURES], vmax, "Model %d" % model)
    cax = fig.add_axes([0.94, 0.25, 0.014, 0.53])
    fig.colorbar(im, cax=cax, label="CGM Z change per diet Z unit")
    fig.suptitle("Diet associations with four CGM outcomes", x=0.11, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.11, 0.065, "Primary cohort | * family q < 0.05; + global q < 0.05 | Grey / NA: model unavailable", fontsize=10)
    fig.text(0.11, 0.025, "Shared colour scale. Red = positive beta; blue = negative beta. TIR is an auxiliary outcome.", fontsize=9)
    return fig


def diet_forest(diet):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.88, bottom=0.12, hspace=0.5, wspace=0.65)
    for ax, outcome in zip(axes.flat, OUTCOMES):
        ax.axvline(0, color="#8E99A6", lw=1)
        for index, score in enumerate(EXPOSURES):
            for model, offset in [(2, -0.14), (3, 0.14)]:
                row = diet.loc[diet.model.eq(model) & diet.outcome.eq(outcome) & diet.exposure.eq(score)]
                if row.empty or row.iloc[0].status != "computed":
                    continue
                row = row.iloc[0]
                y = index + offset
                ax.hlines(y, row.CI95_lower, row.CI95_upper, color=COLORS[model], lw=1.4)
                ax.scatter([row.beta], [y], s=34, edgecolors=COLORS[model],
                           facecolors=COLORS[model] if row.family_sig else "white", zorder=3)
        ax.set_yticks(range(len(EXPOSURES)))
        ax.set_yticklabels([DIET_LABELS[x] for x in EXPOSURES])
        ax.set_ylim(len(EXPOSURES) - 0.4, -0.6)
        ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontweight="bold")
        ax.set_xlabel("Beta (CGM Z per diet Z unit), 95% CI")
        ax.grid(axis="x", color="#E8EBEF", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [Line2D([0], [0], marker="o", color=COLORS[m], label="Model %d" % m) for m in [2, 3]]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.97), ncol=2, frameon=False)
    fig.suptitle("Diet effects before and after BMI adjustment", x=0.08, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.08, 0.045, "Filled point: family q < 0.05; hollow: not significant. Each model uses its own eligible sample.", fontsize=10)
    fig.text(0.08, 0.018, "Differences combine sample selection and BMI adjustment; use the same-cohort comparison tables to separate them.", fontsize=9)
    return fig


def count_figure(counts):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, hspace=0.65, wspace=0.35)
    for i, branch in enumerate(["diet", "microbiome"]):
        for j, model in enumerate([0, 2, 3]):
            ax = axes[i, j]
            rows = counts.loc[counts.branch.eq(branch) & counts.model.eq(model)].set_index("outcome").reindex(OUTCOMES)
            positive = rows.family_significant_positive.to_numpy(float)
            negative = -rows.family_significant_negative.to_numpy(float)
            x = np.arange(4)
            ax.bar(x, positive, color=POSITIVE, width=0.58)
            ax.bar(x, negative, color=NEGATIVE, width=0.58)
            ax.axhline(0, color="#768494", lw=0.8)
            for k, (p, n) in enumerate(zip(positive, negative)):
                if p:
                    ax.annotate(str(int(p)), (k, p), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
                if n:
                    ax.annotate(str(int(n)), (k, n), xytext=(0, -3), textcoords="offset points", ha="center", va="top", fontsize=9)
            if not np.any(positive) and not np.any(negative):
                ax.set_ylim(-1, 1)
                ax.text(0.5, 0.75, "No significant associations", transform=ax.transAxes, ha="center", fontsize=9)
            else:
                extent = max(np.max(positive), abs(np.min(negative)), 1)
                ax.set_ylim(min(np.min(negative), 0) - 0.2 * extent, max(np.max(positive), 0) + 0.25 * extent)
            ax.set_xticks(x)
            ax.set_xticklabels([OUTCOME_LABELS[o] for o in OUTCOMES], rotation=25, ha="right")
            ax.set_title(("Diet" if branch == "diet" else "Microbiome") + " | Model %d" % model, loc="left", fontweight="bold")
            ax.set_ylabel("Significant associations")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle("How many associations meet family FDR < 0.05?", x=0.08, ha="left", fontsize=17, fontweight="bold")
    fig.legend(handles=[Line2D([0], [0], color=POSITIVE, lw=7, label="Positive beta"),
                        Line2D([0], [0], color=NEGATIVE, lw=7, label="Negative beta")], loc="upper right", bbox_to_anchor=(0.98, 0.96), ncol=2, frameon=False)
    fig.text(0.08, 0.04, "Counts are exposure/species–outcome pairs, not people. Negative bars indicate negative beta, not negative counts.", fontsize=9)
    fig.text(0.08, 0.015, "Diet: each exposure's prespecified family q. Microbiome: q within each CGM outcome. Failed tests are excluded from significance counts.", fontsize=8.5)
    return fig


def volcano(micro, model):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.12, hspace=0.43, wspace=0.30)
    for ax, outcome in zip(axes.flat, OUTCOMES):
        all_rows = micro.loc[micro.model.eq(model) & micro.outcome.eq(outcome)]
        rows = all_rows.loc[all_rows.status.eq("computed")].copy()
        if rows.empty:
            ax.text(0.5, 0.5, "No estimable models", transform=ax.transAxes, ha="center")
        else:
            y = -np.log10(rows.FDR_family.clip(lower=1e-300))
            clipped = y.gt(50)
            y = y.clip(upper=50)
            colors = np.where(rows.family_sig, np.where(rows.beta.gt(0), POSITIVE, NEGATIVE), NEUTRAL)
            ax.scatter(rows.beta, y, c=colors, s=18, alpha=0.8, edgecolors="none")
            if clipped.any():
                ax.scatter(rows.loc[clipped, "beta"], y.loc[clipped], marker="^", c=np.asarray(colors)[clipped], s=32)
            positives = int((rows.family_sig & rows.beta.gt(0)).sum())
            negatives = int((rows.family_sig & rows.beta.lt(0)).sum())
            ax.text(1.0, 1.015, "q < 0.05: +%d / −%d\nN: %d–%d | failed: %d" %
                    (positives, negatives, rows.N.min(), rows.N.max(), len(all_rows) - len(rows)),
                    transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
                    clip_on=False)
        ax.axvline(0, color="#9DA7B2", lw=0.7)
        ax.axhline(-np.log10(0.05), color="#6D7887", lw=0.8, ls="--")
        ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontweight="bold")
        ax.set_xlabel("Beta (CGM Z per species CLR-Z unit)")
        ax.set_ylabel("−log10(family q)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Microbiome associations | Model %d" % model, x=0.08, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.08, 0.045, "Primary cohort. Dashed line: family q = 0.05. Red/blue: significant positive/negative beta; grey: not significant.", fontsize=9)
    fig.text(0.08, 0.018, "Each dot is a species. Values above −log10(q) = 50 are capped and marked by triangles; tables retain the original q.", fontsize=8.5)
    return fig


def choose_species(micro, model, limit=20):
    rows = micro.loc[micro.model.eq(model) & micro.status.eq("computed")]
    ranked = rows.groupby("species", as_index=False).FDR_family.min().rename(columns={"FDR_family": "minimum_family_q"})
    ranked = ranked.sort_values(["minimum_family_q", "species"], kind="mergesort")
    significant = ranked.loc[ranked.minimum_family_q.lt(0.05)]
    mode = "significant_in_at_least_one_outcome" if len(significant) else "exploratory_no_significant_species"
    chosen = (significant if len(significant) else ranked).head(limit).copy()
    chosen["selection_rule"] = mode
    chosen["row_number"] = range(1, len(chosen) + 1)
    return chosen, mode


def micro_heatmap(micro, model):
    chosen, mode = choose_species(micro, model)
    height = max(4.5, 2.8 + 0.36 * len(chosen))
    fig, ax = plt.subplots(figsize=(10, height))
    fig.subplots_adjust(left=0.43, right=0.87, top=0.87, bottom=0.17)
    if chosen.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No estimable species", ha="center", transform=ax.transAxes)
    else:
        labels = []
        for row in chosen.itertuples():
            short = row.species.rsplit("s__", 1)[-1].replace("_", " ")
            if len(short) > 46:
                short = short[:43] + "…"
            labels.append("%02d  %s" % (row.row_number, short))
        values = heatmap_arrays(micro.loc[micro.model.eq(model)], "species", chosen.species.tolist())
        finite = values[0][np.isfinite(values[0])]
        vmax = max(float(np.max(np.abs(finite))) if len(finite) else 1, 0.01)
        image = draw_heatmap(ax, values, labels, vmax, "")
        fig.colorbar(image, cax=fig.add_axes([0.90, 0.26, 0.02, 0.48]), label="CGM Z per species CLR-Z unit")
    fig.suptitle("Representative species | Model %d" % model, x=0.05, ha="left", fontsize=16, fontweight="bold")
    text = "At most 20 species significant in >=1 outcome, ordered by minimum family q." if mode.startswith("significant") else "Exploratory ranking only: no species met family q < 0.05."
    fig.text(0.05, 0.075, text, fontsize=9)
    fig.text(0.05, 0.035, "* family q < 0.05; + global q < 0.05. Selection table contains full species names. Grey / NA: unavailable.", fontsize=8.5)
    return fig, chosen


def build_html(counts, diet, micro, figures, run_id, synthetic_label=None):
    main_diet = diet.loc[diet.model.eq(2)]
    main_micro = micro.loc[micro.model.eq(2)]
    distinct_sig = main_micro.loc[main_micro.family_sig, "species"].nunique()
    cards = [("Model 2 饮食显著组合", int(main_diet.family_sig.sum())),
             ("Model 2 菌群显著组合", int(main_micro.family_sig.sum())),
             ("至少一个结局显著的菌种", distinct_sig),
             ("主分析失败检验数（M0/2/3）", int(diet.status.ne("computed").sum() + micro.status.ne("computed").sum()))]
    sections = []
    for stem, title, caption in figures:
        sections.append('<section><h2>%s</h2><p>%s</p><a href="figures/%s.png"><img src="figures/%s.png" alt="%s"></a><p><a href="figures/%s.pdf">PDF</a> · <a href="figures/%s.svg">SVG</a></p></section>' % tuple(html.escape(s) for s in (title, caption, stem, stem, title, stem, stem)))
    labels = {"branch": "分支", "model": "模型", "outcome": "CGM", "planned_tests": "计划检验",
              "computed_tests": "完成", "failed_tests": "失败", "N_min": "最小N", "N_max": "最大N",
              "family_significant_positive": "家族FDR正关联", "family_significant_negative": "家族FDR负关联",
              "family_significant_total": "家族FDR显著合计", "global_significant_total": "全局FDR显著合计"}
    count_html = counts.rename(columns=labels).to_html(index=False, escape=True, border=0, float_format=lambda x: "%.0f" % x)
    diet_html = main_diet[["exposure", "outcome", "N", "beta", "CI95_lower", "CI95_upper", "p_value", "FDR_family", "FDR_global", "status"]].to_html(index=False, escape=True, border=0, float_format=lambda x: "%.4g" % x)
    banner = '<p class="warning">%s</p>' % html.escape(synthetic_label) if synthetic_label else ""
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>新四项 CGM 关联结果总览</title><style>
body{margin:0;background:#f4f6f8;color:#203044;font:15px/1.65 system-ui,sans-serif}main{max-width:1280px;margin:36px auto;padding:0 24px}h1{font-size:30px;margin-bottom:8px}h2{font-size:22px}section{background:white;border:1px solid #dbe1e8;border-radius:10px;padding:24px;margin:24px 0}img{display:block;width:100%%;height:auto}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:#203d54;color:white;padding:20px;border-radius:8px}.card strong{display:block;font-size:32px}table{border-collapse:collapse;font-size:13px;width:100%%;white-space:nowrap}td,th{padding:9px;border-bottom:1px solid #dde4eb;text-align:right}th{background:#edf2f6}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}a{color:#176a87}.warning{background:#fff0d4;padding:12px;border-left:4px solid #bd7917}.meta{color:#5c6c7e}@media(max-width:700px){.cards{grid-template-columns:repeat(2,1fr)}main{padding:0 12px}section{padding:14px}}
</style><main><h1>新四项 CGM 关联结果总览</h1><p class="meta">运行：%s · 主人群 Model 0 / 2 / 3 · OLS，常规标准误</p>%s
<div class="cards">%s</div>
<p>显著组合指“饮食/物种 × CGM”检验，不是参与者人数；同一物种可对应多个结局。卡片使用家族 FDR &lt; 0.05。</p>
<section><h2>先看统计</h2><p>饮食家族按原四评分16次、EAT13/NOVA4共8次、碳水4次分开；另报告全部28次全局校正。菌群按每个结局校正，并报告四结局全局校正。失败检验不作为不显著或零效应。</p><div class="scroll">%s</div><p><a href="tables/直观统计.csv">下载直观统计 CSV</a></p></section>
%s
<section><h2>Model 2 饮食结果明细</h2><div class="scroll">%s</div><p><a href="tables/diet_primary_associations.csv">下载主分析饮食完整表</a> · <a href="tables/microbiome_primary_associations.csv">下载主分析菌群完整表</a></p></section>
<section><h2>阅读说明</h2><p>红/蓝仅表示正/负关联，不等同于健康/不健康。TIR 为辅助结局。TAR180 的零值集中和比例偏态仍然存在；本轮沿用旧分析模型，不作因果解释。森林图的 Model 2/3 可有不同样本，区分样本筛选与 BMI 调整请看本次运行 reports 下的 same-cohort 表。严格人群结果在原模型完整表中。</p><p>本页是独立关联阶段的图表；尚未生成路径或中介图。统计来自保存的模型结果，不重新拟合，也不重新选择 FDR 范围。</p></section></main></html>''' % (html.escape(run_id), banner,
        "".join('<div class="card">%s<strong>%s</strong></div>' % (html.escape(k), v) for k, v in cards), count_html,
        "".join(sections), diet_html)


def generate_visual_report(run_dir, require_completed=True, synthetic_label=None):
    run_dir = Path(run_dir).resolve()
    if require_completed:
        manifest = json.loads((run_dir / "reports/manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") not in ("completed", "completed_with_model_failures", "completed_with_visualization_failure"):
            raise ValueError("A completed model run is required to generate figures")
    inputs = [run_dir / "models/diet_cgm_all_models.csv", run_dir / "models/microbiome_cgm_all_models.csv"]
    snapshots = [fingerprint(p) for p in inputs]
    diet, micro = [prepare_results(pd.read_csv(p), branch) for p, branch in zip(inputs, ["diet", "microbiome"])]
    stamp = datetime.now(timezone.utc).strftime("visual_%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:6]
    output = run_dir / "visuals" / stamp
    (output / "figures").mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir()
    record = {"status": "running", "source_model_files": snapshots, "run_id": run_dir.name,
              "matplotlib": matplotlib.__version__, "script": fingerprint(Path(__file__)),
              "synthetic_label": synthetic_label}
    try:
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
                             "pdf.fonttype": 42, "svg.fonttype": "none"})
        counts = intuitive_counts(diet, micro)
        counts.to_csv(output / "tables/直观统计.csv", index=False)
        diet.to_csv(output / "tables/diet_primary_associations.csv", index=False)
        micro.to_csv(output / "tables/microbiome_primary_associations.csv", index=False)
        figures = []
        plotters = [("01_diet_heatmap", "饮食关联热图", "同时查看三个模型的效应方向和大小；星号为家族FDR，加号为全局FDR。", lambda: diet_heatmap(diet)),
                    ("02_diet_forest", "饮食效应与95%置信区间", "比较主调整模型与额外BMI调整模型；实心点表示家族FDR显著。", lambda: diet_forest(diet)),
                    ("03_significant_counts", "各结局有多少显著关联", "按模型和结局统计正、负关联数量；每个组合只计一次。", lambda: count_figure(counts))]
        for model, number in [(2, "04"), (3, "05")]:
            plotters.append((number + "_microbiome_volcano_M%d" % model, "菌群火山图：Model %d" % model,
                             "每个点是一个物种，横轴是效应，纵轴是家族FDR；虚线对应q=0.05。", lambda m=model: volcano(micro, m)))
        for stem, title, caption, plotter in plotters:
            save_figure(plotter(), output / "figures", stem, synthetic_label)
            figures.append((stem, title, caption))
        for model, number in [(2, "06"), (3, "07")]:
            figure, selected = micro_heatmap(micro, model)
            stem = number + "_microbiome_heatmap_M%d" % model
            save_figure(figure, output / "figures", stem, synthetic_label)
            selected.to_csv(output / "tables" / (stem + "_selection.csv"), index=False)
            figures.append((stem, "代表菌种热图：Model %d" % model,
                            "最多20个菌种，按至少一个结局显著后的最小家族FDR排序；若没有显著菌种，明确标记为探索性排序。完整名称见对应selection表。"))
        page = output / "结果总览.html"
        page.write_text(build_html(counts, diet, micro, figures, run_dir.name, synthetic_label), encoding="utf-8")
        (output / "图表说明.md").write_text("# 关联结果图表\n\n打开同目录 `结果总览.html` 查看统计和全部图表。\n\n"
                                         "- 每张图有 PNG、PDF、SVG 三种格式。\n"
                                         "- 图内英文标签用于避免服务器中文字体缺失；网页和统计说明为中文。\n"
                                         "- 灰色/NA 为不可估计；没有显著结果时如实显示，不把排序靠前等同于显著。\n"
                                         "- 本文件夹图表由已保存模型表生成，没有重新拟合模型。\n", encoding="utf-8")
        if snapshots != [fingerprint(p) for p in inputs]:
            raise ValueError("Model result files changed during figure generation")
        record.update(status="completed", figure_count=len(figures), report=str(page))
        print("\n=== 直观统计：主分析各模型/结局 ===")
        print(counts.to_string(index=False))
        print("VISUAL_REPORT=" + str(page))
        print("FIGURE_DIR=" + str(output / "figures"))
        return {"status": "completed", "report": str(page), "figure_dir": str(output / "figures"),
                "figure_count": len(figures), "summary": counts.to_dict("records")}
    except Exception as error:
        record.update(status="failed", error="%s: %s" % (type(error).__name__, error))
        raise
    finally:
        plt.close("all")
        (output / "visualization_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate tables and figures from existing model CSVs; never refit models.")
    parser.add_argument("--run-dir", required=True, help="Exact outputs/run_... directory")
    args = parser.parse_args(argv)
    generate_visual_report(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
