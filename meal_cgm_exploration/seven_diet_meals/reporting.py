"""Reports that retain all seven scores, including unestimated and null results."""
import numpy as np
import pandas as pd
from .inputs import DIETS


def summary(run, state, results=None):
    lines = ["# 七项饮食变量与餐次血糖", "", "数据：**{}**；状态：**{}**。".format(run.c["data_kind"], state["status"]), "",
             "使用既有个人饮食 Z 分数。AHEI、AMED、EAT13 沿用项目改良版本；hPDI 使用更正来源；rEDIH 已反向，不再次翻转。NOVA4 和碳水为个人供能比例。",
             "七项均分析并报告，分数没有全部统一为健康方向。", ""]
    if state.get("reason"):
        lines += ["说明：" + state["reason"], ""]
    if state["status"] not in ["COMPLETE", "COMPLETE_WITH_LIMITATIONS", "CHECK_PASS"]:
        lines += ["本目录尚未完整完成，不能按完成结果解释。", ""]
    if results is not None:
        for branch, model, title, unit in [
            ("A", "meal_adjusted", "问题一：个人饮食与平均餐后反应", "每增加一个原始饮食 Z 单位，平均餐后净增量差异，mg/dL"),
            ("B", "carb_interaction", "问题二：饮食评分与个人内碳水斜率差异", "饮食 Z 每增加一单位，餐次碳水每增加 10g 对应的斜率差异，mg/dL")]:
            d = results.loc[results.setting.eq("primary_available") & results.branch.eq(branch)
                            & results.model.eq(model) & results.outcome.eq("mean_increment")]
            lines += ["## " + title, "", unit + "。", "",
                      "| 饮食 | 人数 | 餐数 | 估计 | 95% CI | q（主家族14） | 状态 |", "|---|---:|---:|---:|---|---:|---|"]
            for r in d.to_dict("records"):
                interval = "{:.3f} 至 {:.3f}".format(r["ci_low"], r["ci_high"]) if r["status"] == "ESTIMATED" else r["reason"]
                estimate = "{:.3f}".format(r["estimate"]) if np.isfinite(r["estimate"]) else "未估计"
                q = "{:.4g}".format(r["q_family14"]) if np.isfinite(r["q_family14"]) else "—"
                lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(r["diet"], r["people"], r["meals"], estimate, interval, q, r["status"]))
            lines.append("")
        for setting in ["primary_available", "strict_available", "primary_common7"]:
            g = results.loc[results.setting.eq(setting)]
            lines.append("- {}：{} / {} 个规格成功估计；未估计原因见模型表。".format(setting, int(g.status.eq("ESTIMATED").sum()), len(g)))
    lines += ["", "## 阅读与解释", "",
              "1. tables/00_score_mapping.csv：七项评分来源和版本；00_score_coverage.csv：源表缺人、评分缺值与可匹配人数/餐数；01_score_summary.csv、01_score_correlations.csv：分布、相关及配对人数。",
              "2. tables/01_sample_flow.csv：各模型筛选及损失；每人至少5餐/3日，碳水交互至少10餐/3日且碳水范围至少30g。",
              "3. tables/02_diet_person_models.csv：问题一，两结局、背景/餐次调整模型、三套样本。",
              "4. tables/03_diet_carb_interactions.csv：问题二，两结局、三套样本。",
              "5. tables/04_sensitivity_comparison.csv：主、strict、common7对应估计；05_person_response_descriptives.csv：个人平均餐后表型描述。",
              "6. tables/06_coefficients.csv 和 all_models.csv：完整系数、条件数、常量项、样本哈希；figures/ 为汇总森林图。",
              "", "A 每人一行、每人等权，HC3；解释个人间平均反应差异。背景和餐次调整模型使用同一批人。餐次调整使用平均特征，不能称相同标准餐。",
              "B 为个人固定效应、每人总权重1、按人聚类CR1，包含碳水平方和背景；个人饮食主效应被吸收，只估计其与餐次碳水的交互。斜率是控制总能量等条件下的关联，不是随意添加10g碳水的干预效应。",
              "", "每套样本的主家族14项=A餐次调整净增量7项+B净增量7项；峰值另14项；A背景两结局另14项。未估计规格仍计入计划FDR分母。CI为各项点区间，不是多重校正后的同时区间。",
              "strict/common7 是重叠样本敏感性，不是独立复制。共同样本便于比较，但不能仅用显著数量判断评分优劣，也未正式检验不同评分系数之差。",
              "", "数据来自同一观察窗；这些为新探索，不重新定义原7评分或重跑既有20项CGM分析。没有新增菌群模型或机制检验。",
              "默认current4/new3评分来源继承前期与CGM汇总表匹配及主要结局可用的筛选，覆盖范围不是全部原始饮食参与者；本轮评分匹配损失见00_score_coverage.csv。",
              "private/ 保存分析输入和逐模型成员；manifest.json 含路径与哈希。对外汇报使用摘要、汇总表和图。"]
    (run.out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_reports(run, people, membership):
    eligible = membership.loc[membership.setting.eq("primary_available") & membership.branch.eq("A"), "participant_id"].unique()
    p = people.loc[people.index.isin(eligible), DIETS]
    rows, correlations = [], []
    for diet in DIETS:
        v = p[diet].dropna()
        rows.append(dict(diet=diet, people=len(v), mean=v.mean(), sd=v.std(), minimum=v.min(),
                         p25=v.quantile(.25), median=v.median(), p75=v.quantile(.75), maximum=v.max(),
                         scope="union of primary_available A participants; original score units"))
    for i, a in enumerate(DIETS):
        for b in DIETS[i + 1:]:
            pair = p[[a, b]].dropna()
            rho = pair[a].corr(pair[b], method="spearman") if len(pair) >= 3 and pair[a].nunique() > 1 and pair[b].nunique() > 1 else np.nan
            correlations.append(dict(diet_a=a, diet_b=b, people=len(pair), spearman=rho))
    run.table(pd.DataFrame(rows), "01_score_summary.csv")
    run.table(pd.DataFrame(correlations), "01_score_correlations.csv")


def figures(run, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    title = "SYNTHETIC TEST DATA" if run.c["data_kind"] == "SYNTHETIC" else "HPP observational seven-diet analysis"
    for ax, branch, model, caption in zip(axes, ["A", "B"], ["meal_adjusted", "carb_interaction"],
                                         ["Personal mean response association", "Within-person carb slope difference"]):
        d = results.loc[results.setting.eq("primary_available") & results.branch.eq(branch) & results.model.eq(model)
                        & results.outcome.eq("mean_increment")].set_index("diet")
        for i, diet in enumerate(DIETS):
            r = d.loc[diet]
            if r.status == "ESTIMATED":
                color = "#126b89" if r.q_family14 < .05 else "#727b83"
                ax.plot([r.ci_low, r.ci_high], [i, i], color=color)
                ax.scatter(r.estimate, i, color=color, zorder=3)
            else:
                ax.text(.02, i, "not estimated", transform=ax.get_yaxis_transform(), fontsize=8)
        ax.set_yticks(range(len(DIETS)))
        ax.set_yticklabels(DIETS)
        ax.invert_yaxis()
        ax.axvline(0, color="gray", linestyle="--", linewidth=.8)
        ax.set_title(caption, fontsize=11)
        ax.set_xlabel("mg/dL per original diet Z" if branch == "A" else "mg/dL per 10g carbs per original diet Z", fontsize=9)
        ax.grid(axis="x", alpha=.15)
    fig.suptitle(title + "\nMean net increment; pointwise 95% CI; blue: joint primary-family BH q < 0.05", fontsize=11)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(run.out / "figures" / ("01_seven_diet_main." + ext), dpi=160)
    plt.close(fig)
