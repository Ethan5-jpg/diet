"""Readable summaries and figures; person-level outputs remain in private/."""
import numpy as np
import pandas as pd


def repeat_pairs(meals):
    eligible = meals.loc[meals.analysis_ready & meals.signature.fillna("").ne("")].copy()
    rows = []
    for _, g in eligible.groupby(["participant_id", "cohort", "research_stage", "signature"]):
        records = g.sort_values("start").to_dict("records")
        used = set()
        for i, a in enumerate(records):
            if i in used:
                continue
            for j in range(i + 1, len(records)):
                b = records[j]
                clock_diff = abs(a["local_hour"] - b["local_hour"])
                clock_diff = min(clock_diff, 24 - clock_diff)
                if j in used or a["local_date"] == b["local_date"] or clock_diff > 1:
                    continue
                rows.append(dict(participant_id=a["participant_id"], meal_a=a["meal_id"], meal_b=b["meal_id"],
                                 iauc_a=a["iauc_net_0_120"], iauc_b=b["iauc_net_0_120"],
                                 delta=b["iauc_net_0_120"] - a["iauc_net_0_120"], clock_difference_hours=clock_diff))
                used.update([i, j])
                break
    return pd.DataFrame(rows, columns=["participant_id", "meal_a", "meal_b", "iauc_a", "iauc_b", "delta", "clock_difference_hours"])


def build_tables(run, meals, connections):
    stages = [("all_eating_episodes", pd.Series(True, index=meals.index)),
              ("meal_amount_identity_timezone_valid", meals.meal_eligible),
              ("plus_valid_cgm", meals.meal_eligible & meals.cgm_valid),
              ("plus_known_isolated_window", meals.meal_eligible & meals.cgm_valid & meals.isolated_window),
              ("plus_no_invalid_diet_timestamps", meals.analysis_ready),
              ("plus_nova_mapping_coverage", meals.analysis_ready & meals.nova_eligible)]
    flow = pd.DataFrame([dict(stage=label, meals=int(mask.sum()),
                              people=meals.loc[mask, "participant_id"].nunique()) for label, mask in stages])
    run.table(flow, "02_meal_flow.csv")
    reasons = meals.groupby("cgm_reason").agg(meals=("meal_id", "size"), people=("participant_id", "nunique")).reset_index()
    run.table(reasons, "02_cgm_window_reasons.csv")
    quality = []
    for field in ["calories_kcal", "carbohydrate_g", "dietary_fiber_g", "duration_minutes", "nova_coverage",
                  "previous_gap_minutes", "next_gap_minutes", "iauc_net_0_120", "peak_rise_0_120"]:
        s = pd.to_numeric(meals[field], errors="coerce")
        quality.append(dict(field=field, observed=int(s.notna().sum()), missing=int(s.isna().sum()),
                            p05=s.quantile(0.05), median=s.median(), p95=s.quantile(0.95)))
    run.table(pd.DataFrame(quality), "02_measure_distributions.csv")
    counts = meals.groupby("participant_id").agg(episodes=("meal_id", "size"),
                                                 valid_cgm=("cgm_valid", "sum"), isolated_valid=("analysis_ready", "sum"))
    run.table(counts.reset_index(), "02_per_person_counts.csv", private=True)
    run.table(counts.describe().reset_index().rename(columns={"index": "statistic"}), "02_per_person_count_distribution.csv")
    if connections:
        conn = pd.DataFrame(connections)
        run.table(conn, "02_connection_quality.csv", private=True)
        numeric = conn.select_dtypes(include=np.number)
        run.table(numeric.describe().reset_index().rename(columns={"index": "statistic"}), "02_connection_quality_distribution.csv")
    good = meals.loc[meals.analysis_ready].copy()
    good["local_time_bin"] = pd.cut(good.local_hour, bins=[0, 6, 12, 18, 24], right=False,
                                     labels=["00-06", "06-12", "12-18", "18-24"])
    time = good.groupby("local_time_bin", observed=False).agg(meals=("meal_id", "size"),
                       people=("participant_id", "nunique"), median_net_iauc=("iauc_net_0_120", "median"),
                       median_carbs=("carbohydrate_g", "median")).reset_index()
    run.table(time, "02_time_of_day_descriptive.csv")
    pairs = repeat_pairs(meals)
    run.table(pairs, "02_repeated_meal_pairs.csv", private=True)
    info = dict(pairs=len(pairs), people=pairs.participant_id.nunique(),
                median_absolute_iauc_difference=pairs.delta.abs().median() if len(pairs) else np.nan,
                median_signed_iauc_difference=pairs.delta.median() if len(pairs) else np.nan,
                interpretation="Descriptive disjoint pairs only; not an ICC or causal timing effect")
    run.table(pd.DataFrame([info]), "02_repeated_meal_summary.csv")
    return flow.to_dict("records")


def example_plot(run, examples):
    if not examples:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(examples), 1, figsize=(9, 2.8 * len(examples)), squeeze=False)
    for i, (ax, example) in enumerate(zip(axes[:, 0], examples), 1):
        x, y, baseline = example["curve"]
        ax.plot(x, y, "o-", color="#266b92", markersize=3, label="CGM curve (endpoints interpolated if needed)")
        ax.axhline(baseline, color="#8a8a8a", linestyle="--", label="Premeal baseline")
        for j, t in enumerate(example["food_minutes"]):
            ax.axvline(t, color="#d99530", alpha=0.5, linewidth=1, label="Logged food time" if j == 0 else None)
        ax.set(title="Example {} | carbs {:.1f} g | observed NOVA4 {:.1f}% | classified {:.0f}%".format(
            i, example["carbs"], example["nova4"], example["coverage"] * 100), xlabel="Minutes from episode start", ylabel="Glucose (mg/dL)")
        ax.set_xlim(-5, 125)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        if i == 1:
            ax.legend(fontsize=7, loc="best")
    provisional = not run.c["time"]["consumption_time_verified"]
    label = "SYNTHETIC TEST DATA" if run.c["data_kind"] == "SYNTHETIC" else "HPP exploratory meal alignment"
    if provisional:
        label += " | consumption timestamp unverified"
    fig.suptitle(label, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for extension in ["png", "pdf"]:
        fig.savefig(run.out / "figures" / ("02_meal_examples." + extension), dpi=140)
    plt.close(fig)


def summary(run):
    lines = ["# HPP 餐次与 CGM 运行摘要", "", "数据标记：**{}**。".format(run.c["data_kind"]), "",
             "本运行是观察性探索，不构成饮食—菌群—血糖的因果中介证明。", ""]
    if not run.c["time"]["consumption_time_verified"]:
        lines += ["**实际进食时间字段尚未确认；餐次指标仅供核查，关联模型不会自动运行。**", ""]
    for stage, result in run.status.items():
        lines += ["## " + stage, "", "状态：**{}**".format(result.get("status", "UNKNOWN")), ""]
        for key in ["reason", "build_ready", "model_input_headers_ready", "consumption_time_verified", "cgm_files",
                    "output_disk_free_bytes", "main_model", "estimated_models", "planned_models", "microbiome_timing"]:
            if key in result:
                lines.append("- {}：{}".format(key, result[key]))
        lines.append("")
    if any(v["status"] in ["RUNNING", "FAILED", "INTERRUPTED"] for v in run.status.values()):
        lines += ["**存在未完成或失败的阶段，其结果表可能只是部分输出，不能按完成结果解释。**",
                  "若进程已停止但仍显示 RUNNING，说明未正常结束，请启动新运行。", ""]
    flow = run.out / "tables" / "02_meal_flow.csv"
    if flow.exists():
        lines += ["## 餐次筛选", "", "| 阶段 | 餐次数 | 人数 |", "|---|---:|---:|"]
        for r in pd.read_csv(flow).to_dict("records"):
            lines.append("| {} | {} | {} |".format(r["stage"], r["meals"], r["people"]))
        lines.append("")
    lines += ["## 阅读顺序", "", "0. tables/00_preflight.csv：依赖、路径和字段快速检查；通过不代表行级数据已合格。",
              "1. tables/01_input_audit.csv：输入覆盖和异常。",
              "2. tables/02_meal_flow.csv、02_cgm_window_reasons.csv：剩余餐次及排除原因。",
              "3. figures/02_meal_examples.png：时间对齐人工检查（若无合格餐则不生成）。",
              "4. tables/03_within_person_models.csv：如已运行，先看 status、人数与区间。",
              "5. tables/03_microbiome_interactions.csv：少数预定菌的探索性交互及对应 FDR。", "",
              "private/ 和 manifest.json 含身份或输入路径，仅在服务器保留；截图汇报优先使用本摘要、tables/ 和 figures/。",
              "运行成功不表示关联显著；NOT_ESTIMATED 表示未能估计，不能解释为没有效应。", ""]
    (run.out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
