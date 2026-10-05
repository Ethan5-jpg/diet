"""Three aggregate screens; original precision remains in server tables."""
import json
from pathlib import Path

import pandas as pd

from hppmeal.common import require, sha256
from .inputs import contract

SHORT = {"baseline_glucose": "Pre", "mean_increment": "Net", "absolute_post_mean": "Abs", "peak_rise_0_120": "Peak"}
TAXA = "T023=GGB6012_SGB8640; T046=Clostridium_phoceensis; T068=Eubacterium_rectale; T096=Ruthenibacterium_lactatiformans"


def number(x, digits=3):
    return "NA" if pd.isna(x) else format(float(x), ".{}g".format(digits))


def q_number(x):
    if pd.isna(x):
        return "NA"
    if x == 0:
        return "0*"
    # Preserve the side of 0.05 when a screenshot is the only return channel.
    for digits in [3, 5, 8, 12, 17]:
        shown = number(x, digits)
        if (float(shown) < .05) == (x < .05) and (float(shown) == .05) == (x == .05):
            return shown
    return number(x, 17)


def effect(row, q):
    if row.status != "ESTIMATED":
        return "NE"
    qtext = q_number(row[q])
    return "{}[{},{}];q={}".format(number(row.estimate), number(row.ci_low), number(row.ci_high), qtext)


def header(out, state, page, title):
    return ["KEY {}/3 | {}".format(page, title), "RUN=" + Path(out).name,
            "DATA={} | STATUS={}".format(state["data_kind"], state["status"])]


def pages(out, state, tables):
    c = contract()
    flow, balance = tables["support_flow"], tables["selection_balance"]
    a, b = tables["absolute_models"], tables["support_models"]
    lines = header(out, state, 1, "来源复现、样本流失与完成情况")
    lines += ["原48项模型与共同成员复现：PASS。原输入只读；菌日期先后不筛选；不加两周5%门槛。",
              "阈值同时要求合格餐数/不同日期数；仅筛人，保留其全部原合格餐均值。"]
    public = flow[["setting", "threshold", "people", "meals", "retained_fraction", "meals_p50", "dates_p50"]]
    lines += public.to_string(index=False, float_format=lambda x: number(x, 4)).splitlines()
    lines += ["", "保留/排除者特征差异：优先提示零方差完全分离，否则显示最大|SMD|。",
              "不含餐数/日期数；不可计算时NA；不证明选择偏差消除。"]
    for setting in c["settings"]:
        for threshold in ["m10d5", "m20d7"]:
            group = balance.loc[balance.setting.eq(setting) & balance.threshold.eq(threshold) &
                                ~balance.field.isin(["meals", "dates"])]
            separated = group.loc[group.smd_status.eq("ZERO_VARIANCE_SEPARATION")]
            defined = group.dropna(subset=["standardized_difference"])
            if len(separated):
                note = separated.iloc[0]["field"] + "=零方差完全分离（共{}项；SMD未定义）".format(len(separated))
            elif len(defined):
                r = defined.loc[defined.standardized_difference.abs().idxmax()]
                note = r["field"] + "=" + number(abs(r.standardized_difference), 4)
            else:
                note = "NA"
            lines.append("{} {} {}".format(setting, threshold, note))
    n_a = int(a.status.eq("ESTIMATED").sum())
    new_b = b.loc[b.threshold.ne("m5d3")]
    identity = tables["algebra_check"].status.eq("PASS_ALGEBRA_ONLY").sum()
    lines += ["", "A估计 {}/24（含8个旧Pre参照）；B新增估计 {}/48，另保留原24餐次参照。".format(n_a, int(new_b.status.eq("ESTIMATED").sum())),
              "同X系数代数核对 {}/8可核对且通过；不是中介检验或独立证据。".format(identity),
              "A/B均为新增探索或敏感性；原Net主结果不改；strict和门槛样本相互重叠。",
              "private/保留服务器；完整CI/p/q/NE原因见tables/。* q=0为数值下溢。"]
    page1 = "\n".join(lines) + "\n"

    lines = header(out, state, 2, "A：绝对餐后水平与餐前/净增量的共同调整比较")
    lines += ["B[逐项95%CI];q_panel12；单位mg/dL/原CLR-Z；每组4菌x3结局为新12项家族。",
              "三结局同X，不含餐前水平或斜率；这里的Net不同于原调餐前的主模型。",
              "Abs=同批餐Pre+Net，为0-120分钟平均绝对血糖，不是120分钟单点值。"]
    rows = []
    for outcome in c["absolute_outcomes"]:
        for taxon in c["taxa"]:
            row = dict(Y=SHORT[outcome], taxon=taxon)
            for s in c["settings"]:
                r = a.loc[a.setting.eq(s) & a.taxon.eq(taxon) & a.outcome.eq(outcome)].iloc[0]
                row[s] = effect(r, "q_panel12")
            rows.append(row)
    lines += pd.DataFrame(rows).to_string(index=False).splitlines()
    lines += ["Pre系数是原参照，但此处q使用新12项家族；不要与旧q混写。",
              "不得相加不同调整集的旧系数或CI；不计算中介/解释比例。", TAXA]
    page2 = "\n".join(lines) + "\n"

    lines = header(out, state, 3, "B：原模型在三个最低餐数/日期门槛下的结果")
    lines += ["B[逐项95%CI];原家族q：Net4项、Pre+Peak8项；各组另有q_meal12见CSV。",
              "此处Net/Peak沿用原模型的餐前水平和斜率调整；每个门槛均每人等权。"]
    for s in c["settings"]:
        sf = flow.loc[flow.setting.eq(s)].set_index("threshold")
        lines.append(s + " | " + "; ".join("{} N={}".format(t["name"], int(sf.loc[t["name"], "people"])) for t in c["thresholds"]))
        rows = []
        for outcome in c["original_meal_outcomes"]:
            for taxon in c["taxa"]:
                row = dict(Y=SHORT[outcome], taxon=taxon)
                for t in c["thresholds"]:
                    r = b.loc[b.setting.eq(s) & b.threshold.eq(t["name"]) & b.taxon.eq(taxon) & b.outcome.eq(outcome)].iloc[0]
                    row[t["name"]] = effect(r, "q_family")
                rows.append(row)
        lines += pd.DataFrame(rows).to_string(index=False).splitlines()
    lines += ["比较效应/区间及人群变化；跨0.05不同不等于效应差异；NE不等于零效应。"]
    page3 = "\n".join(lines) + "\n"
    result = [page1, page2, page3]
    require(all(len(x.splitlines()) <= 38 for x in result), "Summary exceeds three-screen line budget")
    return result


def save_pages(out, state, tables):
    folder = Path(out) / "pages"
    folder.mkdir(exist_ok=True)
    for n, text in enumerate(pages(out, state, tables), 1):
        (folder / ("0{}_summary.txt".format(n))).write_text(text, encoding="utf-8")


def resolve_run(path, output_root):
    if str(path) != "latest":
        return Path(path).expanduser().resolve()
    candidates = sorted(p for p in Path(output_root).expanduser().glob("gut_absolute_robustness_*") if p.is_dir())
    require(candidates, "No gut_absolute_robustness runs")
    return candidates[-1]


def show(path, output_root, page=None):
    out = resolve_run(path, output_root)
    require((out / "manifest.json").is_file(), "Run manifest missing: " + str(out))
    state = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    require(state.get("data_kind") in ["REAL", "SYNTHETIC"], "Unverified run data label")
    require(state.get("status") in ["COMPLETE_WITH_LIMITATIONS", "PARTIAL"],
            "Latest/requested run is not complete: {}. {}".format(state.get("status"), state.get("error", "")))
    require(page in [None, 1, 2, 3], "Page must be 1, 2 or 3")
    chosen = [page] if page else [1, 2, 3]
    for n in chosen:
        name = "pages/0{}_summary.txt".format(n)
        p = out / name
        require(p.is_file() and sha256(p) == state.get("result_sha256", {}).get(name), "Missing/changed summary page: " + name)
        print(p.read_text(encoding="utf-8"), end="\n")
    return state
