"""Aggregate-only reports and plots."""
import numpy as np
import pandas as pd


def summary(run, info, differences=None, prediction_status=None, gains=None):
    lines = ["# HPP 重复餐后续分析", "", "数据标记：**{}**。".format(run.c["data_kind"]),
             "运行状态：**{}**。".format(info["status"]), "",
             "观察性后续探索。不是饮食或菌群的因果检验。", "",
             "时间采用原运行已留档的参与者自报进食时间解释；不代表逐条准确性已证实。", ""]
    if info.get("reason"):
        lines += ["说明：" + str(info["reason"]), ""]
    if info["status"] in ["FAILED", "INTERRUPTED", "RUNNING"]:
        lines += ["当前输出不完整，不可按完成分析解释。", ""]
    lines += ["## 主要重复餐指标", "", "所有差异单位为 mg/dL；均值与中位数不是同一统计量。",
              "95% 区间按参与者整组重抽样，固定当前选中的餐对。", "",
              "| 人群 | 人数 | 餐对 | 平均增量绝对差中位数 | 95% 区间 | 状态 |",
              "|---|---:|---:|---:|---|---|"]
    if differences is not None and not differences.empty:
        focus = differences.loc[differences.outcome.eq("mean_increment") & differences.metric.eq("pair_median_absolute_difference")]
        for r in focus.to_dict("records"):
            interval = "{:.3f} 至 {:.3f}".format(r["ci_low"], r["ci_high"]) if np.isfinite(r["ci_low"]) else "未估计"
            estimate = "{:.3f}".format(r["estimate"]) if np.isfinite(r["estimate"]) else "无配对"
            lines.append("| {} | {} | {} | {} | {} | {} |".format(r["cohort"], r["people"], r["pairs"], estimate, interval, r["status"]))
    lines += ["", "all_descriptive 未做协变量/糖尿病筛选；primary 和 strict 沿用原规则，不要求 NOVA 变化。",
              "完整模板的日期统计与原时刻匹配的互斥餐对口径不同，不混用人数。", "",
              "## 后续日期预测", "",
              "目标：已知参与者、训练中至少出现两个日期的食物模板，在较晚日期的两小时平均绝对血糖。",
              "训练/测试按每人的当地日期划分。所有模型评价同一批完整测试餐；不是新人的预测验证。",
              "整餐日志用于确定模板，因此这是已知食物记录条件下的回顾性时间验证，不是已验证的实时餐前工具。", ""]
    for r in prediction_status or []:
        lines.append("- {}：{}；训练 {} 人/{} 餐，测试 {} 人/{} 餐。{}".format(
            r["cohort"], r["status"], r["train_people"], r["train_meals"], r["test_people"], r["test_meals"], r["reason"]))
    if gains is not None and not gains.empty:
        lines += ["", "| 人群 | 比较：后模型相对前模型 | 每人等权 MAE 改善 | 95% 区间 |",
                  "|---|---|---:|---|"]
        for r in gains.to_dict("records"):
            lines.append("| {} | {} → {} | {:.3f} | {:.3f} 至 {:.3f} |".format(
                r["cohort"], r["previous"], r["newer"], r["mae_improvement"], r["ci_low"], r["ci_high"]))
    lines += ["", "MAE 改善为正表示后模型误差更低；负值表示更差；区间跨零时改善方向不明确。",
              "预测区间固定已训练模型，仅对测试参与者整组重抽样；没有把重新训练的不确定性包含进来。",
              "premeal_persistence 直接用餐前基线预测餐后均值，供判断模型是否超出简单的基线延续。",
              "预测增益可能反映背景血糖延续，不代表食物、餐前血糖或菌群的因果作用。", "",
              "## 文件阅读顺序", "",
              "1. tables/01_cohort_flow.csv、02_template_structure.csv：人群和完整模板重复次数。",
              "2. tables/03_pair_differences.csv：四项差异及参与者聚类区间。",
              "3. tables/03_pair_quantiles.csv、03_pair_counts.csv：复核截图口径及每人贡献。",
              "4. tables/04_prediction_flow.csv、04_prediction_status.csv：日期划分及支持不足造成的损失。",
              "5. tables/04_prediction_metrics.csv、04_prediction_gains.csv：共同测试集误差和增量信息。",
              "6. figures/：汇总图。无相应样本时不生成该图，查看表中的状态。", "",
              "运行完成不代表有显著改善。DESCRIPTIVE_ONLY / NOT_ESTIMATED 表示不能估计相应区间/模型，不是证明没有差异。",
              "没有计算 ICC 或个人稳定性评分：当前正式统计量为重复差异分布及聚类区间。",
              "四项结局和多个比较均为后续探索，没有根据显著性选取最有利结局。",
              "第一轮 NOVA4 主模型及菌群交互保持不变。", "",
              "private/ 保存身份、模板、切分和逐餐预测；manifest.json 保存输入路径。它们留在服务器，汇报优先使用 tables/、figures/ 和本摘要。"]
    (run.out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def figures(run, differences, gains):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    title = "SYNTHETIC TEST DATA" if run.c["data_kind"] == "SYNTHETIC" else "HPP observational follow-up"
    f = differences.loc[differences.cohort.eq("primary") & differences.metric.eq("pair_median_absolute_difference") & differences.estimate.notna()]
    if len(f):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(f.estimate, np.arange(len(f)), color="#246d8a", zorder=3)
        for i, row in enumerate(f.to_dict("records")):
            if np.isfinite(row["ci_low"]):
                ax.plot([row["ci_low"], row["ci_high"]], [i, i], color="#246d8a", linewidth=2)
        ax.set_yticks(np.arange(len(f)))
        ax.set_yticklabels(f.outcome)
        ax.set(xlabel="Median absolute paired difference (mg/dL); cluster 95% CI",
               title=title + "\nPrimary cohort: repeated logged meals")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(run.out / "figures" / ("03_repeated_differences." + ext), dpi=160)
        plt.close(fig)
    if not gains.empty:
        f = gains.loc[gains.cohort.eq("primary")]
        if len(f):
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for i, row in enumerate(f.to_dict("records")):
                ax.plot([row["ci_low"], row["ci_high"]], [i, i], color="#246d8a", linewidth=2)
                ax.scatter(row["mae_improvement"], i, color="#246d8a")
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_yticks(np.arange(len(f)))
            ax.set_yticklabels([r["previous"] + " -> " + r["newer"] for r in f.to_dict("records")])
            ax.set(xlabel="Person-equal MAE improvement (mg/dL); positive = better",
                   title=title + "\nLater-date holdout; conditional cluster 95% CI")
            fig.tight_layout()
            for ext in ["png", "pdf"]:
                fig.savefig(run.out / "figures" / ("04_prediction_gains." + ext), dpi=160)
            plt.close(fig)
