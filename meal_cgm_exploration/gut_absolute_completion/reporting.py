"""Three compact pages and evidence-bound tables; no inferred causal narrative."""
import json
from pathlib import Path
import pandas as pd
from hppmeal.common import require, sha256
from gut_absolute_robustness.reporting import effect, number, q_number, TAXA
from .inputs import contract

SHORT = {"baseline_glucose": "Pre", "mean_increment": "Net", "absolute_post_mean": "Abs"}


def header(out, state, n, title):
    return ["KEY {}/3 | {}".format(n, title), "RUN=" + Path(out).name,
            "DATA={} | STATUS={}".format(state["data_kind"], state["status"])]


def pages(out, state, common, original, flow, balance):
    c = contract()
    lines = header(out, state, 1, "来源、样本与本轮完成情况")
    lines += ["两个来源的哈希/成员/旧参照/FDR核对：PASS；输入未变。",
              "上轮原48项复现PASS已核验记录；本轮复用旧估计，新增拟合32个规格。",
              "菌日期先后不筛选；每人等权；三个门槛同时要求餐数和不同日期数。"]
    lines += flow[["setting", "threshold", "people", "meals", "retained_fraction", "meals_p50", "dates_p50"]].to_string(index=False, float_format=lambda x: number(x, 4)).splitlines()
    lines += ["保留/排除者特征差异（不含餐数/日期数；先提示完全分离，否则最大|SMD|）："]
    for (setting, threshold), g in balance.groupby(["setting", "threshold"], sort=False):
        g = g.loc[~g.field.isin(["meals", "dates"])]
        separated = g.loc[g.smd_status.eq("ZERO_VARIANCE_SEPARATION")]
        finite = g.dropna(subset=["standardized_difference"])
        note = "NA"
        if len(separated):
            note = separated.iloc[0]["field"] + "=零方差完全分离（SMD未定义）"
        elif len(finite):
            r = finite.loc[finite.standardized_difference.abs().idxmax()]
            note = r["field"] + "=" + number(abs(r.standardized_difference), 4)
        lines.append("{} {} {}".format(setting, threshold, note))
    lines += ["共同调整：复用40行+新增32行；共{}/72估计完成；新增{}/32。".format(state["common_estimated"], state["new_estimated"]),
              "同X代数核对 {}/24通过；仅程序核对，不是中介检验。".format(state["algebra_pass"]),
              "旧Net主模型的q_family<0.05数/可估计数（每组预定4项，原调整不变）："]
    for setting in c["settings"]:
        values = []
        for t in c["thresholds"]:
            g = original.loc[original.setting.eq(setting) & original.threshold.eq(t["name"]) & original.outcome.eq("mean_increment")]
            values.append("{} {}/{}".format(t["name"], int((g.status.eq("ESTIMATED") & g.q_family.lt(.05)).sum()), int(g.status.eq("ESTIMATED").sum())))
        lines.append(setting + " | " + "; ".join(values))
    lines += ["FIGURES=" + state["figures_status"] + " | 完整表/方法/Results在本次输出目录。",
              "这轮为探索/敏感性；样本相互重叠；private/留服务器；* q=0为数值下溢。"]
    result = ["\n".join(lines) + "\n"]
    for n, setting in enumerate(c["settings"], 2):
        lines = header(out, state, n, setting + "：共同调整Pre/Net/Abs的三个门槛")
        lines += ["B[逐项95%CI];q_panel12；单位mg/dL/原CLR-Z。每样本x门槛固定12项BH。",
                  "同X不含餐前水平/斜率；此Net与原主模型不同；Abs为0-120分钟平均绝对值。"]
        sf = flow.loc[flow.setting.eq(setting)].set_index("threshold")
        lines.append("; ".join("{} N={}".format(t["name"], int(sf.loc[t["name"], "people"])) for t in c["thresholds"]))
        rows = []
        for outcome in c["outcomes"]:
            for taxon in c["taxa"]:
                r = dict(Y=SHORT[outcome], taxon=taxon)
                for t in c["thresholds"]:
                    row = common.loc[common.setting.eq(setting) & common.threshold.eq(t["name"]) & common.taxon.eq(taxon) & common.outcome.eq(outcome)].iloc[0]
                    r[t["name"]] = effect(row, "q_panel12")
                rows.append(r)
        lines += pd.DataFrame(rows).to_string(index=False).splitlines()
        lines += ["Pre复用旧估计，但此处q重新归入12项家族；NE及原因保留在CSV。",
                  "比较方向、量级、区间与人群变化；显著性跨线不等于正式效应差异。",
                  "各组q不控制跨全部样本/门槛的整体FDR。", TAXA]
        result.append("\n".join(lines) + "\n")
    require(len(result) == 3 and all(len(x.splitlines()) <= 38 for x in result), "Summary exceeds three-screen budget")
    return result


def save_pages(out, state, common, original, flow, balance):
    target = Path(out) / "pages"
    target.mkdir(exist_ok=True)
    for i, value in enumerate(pages(out, state, common, original, flow, balance), 1):
        (target / ("0{}_summary.txt".format(i))).write_text(value, encoding="utf-8")


def show(path, root, page=None):
    if str(path) == "latest":
        candidates = sorted(p for p in Path(root).expanduser().glob("gut_absolute_completion_*") if p.is_dir())
        require(candidates, "No completion run exists")
        path = candidates[-1]
    out = Path(path).expanduser().resolve()
    require((out / "manifest.json").is_file(), "Run manifest missing")
    state = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    require(state.get("status") in ["COMPLETE_WITH_LIMITATIONS", "PARTIAL", "COMPLETE_ANALYSIS_FIGURES_UNAVAILABLE"],
            "Latest/requested run is not complete: {}. {}".format(state.get("status"), state.get("error", "")))
    require(state.get("data_kind") in ["REAL", "SYNTHETIC"], "Unverified run label")
    require(page in [None, 1, 2, 3], "Page must be 1, 2 or 3")
    for n in ([page] if page else [1, 2, 3]):
        name = "pages/0{}_summary.txt".format(n)
        p = out / name
        require(p.is_file() and sha256(p) == state.get("result_sha256", {}).get(name), "Missing/changed page: " + name)
        print(p.read_text(encoding="utf-8"), end="\n")
    return state


def write_results(out, kind, common, original, flow):
    lines = ["# Exploratory meal-support completion", "", "DATA=" + kind,
             "Run: " + Path(out).name, "",
             "These analyses were specified after review of the earlier results. They retain four previously selected candidates, overlapping same-visit samples, original exposure scales and equal participant weights. The new common-adjustment Net estimates do not replace the original baseline-adjusted primary models.", "",
             "## Participant support", "", "| Sample | Minimum meals/dates | Participants | Meals | Retained fraction |", "|---|---|---:|---:|---:|"]
    for r in flow.itertuples():
        lines.append("| {} | {} | {} | {} | {:.4f} |".format(r.setting, r.threshold, r.people, r.meals, r.retained_fraction))
    lines += ["", "## Original net-increment reference", "",
              "The original models retain adjustment for pre-meal level and slope, and their original four-test net-increment BH families. Numbers below are q<0.05 findings among estimable models; each group contains four planned tests. A missing estimate is not a null finding.", ""]
    for (setting, threshold), g in original.loc[original.outcome.eq("mean_increment")].groupby(["setting", "threshold"], sort=False):
        estimated = g.status.eq("ESTIMATED")
        hits = g.loc[estimated & g.q_family.lt(.05), "taxon"].tolist()
        lines.append("- {} / {}: {} of {} estimable (4 planned); taxa: {}.".format(setting, threshold, len(hits), int(estimated.sum()), ", ".join(hits) if hits else "none meeting the criterion"))
    lines += ["", "## Common-adjustment absolute post-meal glucose", "",
              "Each estimate is in mg/dL per original CLR-Z, with a pointwise HC3 95% confidence interval. The outcome is the 0–120-minute time-averaged absolute glucose level. q_panel12 refers to all four candidates and all three common-adjustment outcomes within the given sample and threshold. Pre, Net and Abs share covariates that exclude baseline glucose and pre-meal slope.", "",
              "| Sample | Support | Taxon | Beta | 95% CI | p | q_panel12 | Status |", "|---|---|---|---:|---|---:|---:|---|"]
    for r in common.loc[common.outcome.eq("absolute_post_mean")].itertuples():
        fmt = lambda x: number(x, 8)
        lines.append("| {} | {} | {} | {} | [{}, {}] | {} | {} | {} |".format(r.setting, r.threshold, r.taxon, fmt(r.estimate), fmt(r.ci_low), fmt(r.ci_high), fmt(r.p), q_number(r.q_panel12), r.status))
    lines += ["", "All candidates and planned specifications are retained in the accompanying 17-significant-digit CSVs. Displayed q values preserve their side of 0.05; a computed p/q of zero reflects numerical underflow, with zero q displayed as 0*. The combined table contains repeated Pre references across the two adjustment blocks and must not be counted as independent discoveries. These BH families do not control FDR across all thresholds, samples and blocks. Confidence intervals crossing zero and q values use different inferential criteria.", "",
              "Higher meal-support thresholds change participant composition as well as measurement support. Retained/excluded characteristics are provided in selection_balance.csv. Neither persistence nor loss of q<0.05 establishes a formal between-group effect difference. The samples overlap and do not constitute independent replication. The coefficient identity is algebraic, not evidence of mediation. Same-visit timing does not establish causal order.", "",
              "These generated results provide a transparent numerical draft; manuscript interpretation must use effect magnitudes, uncertainty and the original primary result. Individual identifiers are retained only in private outputs."]
    (Path(out) / "RESULTS_en.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
