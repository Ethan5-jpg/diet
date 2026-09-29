"""Only three key pages; detailed outputs remain on the server."""
import json
from pathlib import Path
import pandas as pd
from hppmeal.common import require, sha256
from .inputs import contract


def number(value):
    if pd.isna(value):
        return "NA"
    return "{:.3g}".format(float(value))


def qtext(value):
    return "<1e-300*" if value == 0 else number(value)


def cell(row, ci=True, q12=False):
    if row.status != "ESTIMATED":
        return "NE"
    text = number(row.estimate)
    if ci:
        text += "[{},{}]".format(number(row.ci_low), number(row.ci_high))
    text += ";q=" + qtext(row.q_family)
    if q12:
        text += ";q12=" + qtext(row.q_meal12)
    return text


def pages(run, result, state):
    c = contract()
    tables = result["models"]
    (run.out / "pages").mkdir(exist_ok=True)

    def intro(index, title):
        return ["KEY {}/3 | {}".format(index, title), "RUN=" + run.out.name,
                "DATA={} | STATUS={}".format(run.c["data_kind"], state["status"]), ""]

    lines = intro(1, "状态、共同样本和菌采样时间")
    lines += ["主检验: 4菌 × 餐后平均净增量；次要: 餐前水平/峰值升幅。", "沿用旧候选与原CLR-Z；新增中介/交互模型=0；同队列探索。",
        "估计完成 {}/48；主检验 {}/8（主样本4+strict4）。".format(state["estimated"], state["primary_estimated"]), "",
        result["sample"][["setting", "people", "meals", "meals_p50", "dates_p50"]].to_string(index=False), "",
        "筛选流程:", result["flow"].to_string(index=False), "",
        "日期已知/未知；相对所纳入餐的日期范围（不是整个CGM监测期）:",
        result["timing"][["setting", "known_date", "unknown_date", "ambiguous", "before", "within", "after", "distant_over_30d"]].to_string(index=False),
        "连接不匹配/缺少CGM连接: {}人、{}餐。".format(result["mismatch"]["people"], result["mismatch"]["meals"]),
        "日期缺失不填补；多次采样未匹配时不选第一条；时间先后不作因果解释。",
        "T023=GGB6012_SGB8640; T046=Clostridium_phoceensis",
        "T068=Eubacterium_rectale; T096=Ruthenibacterium_lactatiformans"]
    if len(tables.loc[tables.status.ne("ESTIMATED")]):
        lines += ["NE原因见 tables/models.csv；保留全部计划检验分母。"]
    bodies = [lines]
    lines = intro(2, "主样本：菌与餐次表型")
    lines += ["每人等权，原CLR-Z增加1单位对应mg/dL差异；逐项95%CI。",
        "q: 净增量4项家族；餐前+峰值8项家族；q12为全部12餐次检验。",
        "均调背景、hPDI/NOVA4、日能量/饮酒及餐情境；净/峰另调餐前水平/趋势。", ""]
    selected = tables.loc[tables.setting.eq("primary_common")]
    for outcome, title in [("mean_increment", "主：餐后平均净增量"), ("baseline_glucose", "次：餐前水平"), ("peak_rise_0_120", "次：峰值升幅")]:
        lines.append(title)
        for taxon in c["taxa"]:
            row = selected.loc[selected.taxon.eq(taxon["id"]) & selected.outcome.eq(outcome)].iloc[0]
            lines.append("{} N={} {}".format(taxon["id"], int(row.people), cell(row, q12=True)))
        lines.append("")
    lines += ["餐前水平不是空腹血糖；平均净增量可为负。", "不能凭一项显著另一项不显著，断言环节不同；未做恢复率或因果中介。"]
    bodies.append(lines)
    lines = intro(3, "strict敏感性与整体CGM参照")
    lines += ["strict与主样本重叠；下面各列B[95%CI];q，均为mg/dL/原CLR-Z。"]
    selected = tables.loc[tables.setting.eq("strict_common")]
    compact = []
    for taxon in c["taxa"]:
        g = selected.loc[selected.taxon.eq(taxon["id"])].set_index("outcome")
        compact.append(dict(taxon=taxon["id"], N=int(g.iloc[0].people), Net=cell(g.loc["mean_increment"]),
            Pre=cell(g.loc["baseline_glucose"]), Peak=cell(g.loc["peak_rise_0_120"])))
    lines += [pd.DataFrame(compact).to_string(index=False), "", "主样本同人/同连接的整体CGM参照：B;q，原CGM Z/原CLR-Z。",
        "仅调背景+hPDI/NOVA4+日能量/饮酒；q为12项参照家族。"]
    selected = tables.loc[tables.setting.eq("primary_common")]
    compact = []
    for taxon in c["taxa"]:
        g = selected.loc[selected.taxon.eq(taxon["id"])].set_index("outcome")
        compact.append(dict(taxon=taxon["id"], Mean=cell(g.loc["cgm_mean_z"], False),
            CV=cell(g.loc["cgm_cv_z"], False), SDw=cell(g.loc["cgm_sdw_z"], False)))
    lines += [pd.DataFrame(compact).to_string(index=False), "",
        "参照为本轮重拟合，非旧中介b或ab；不能与餐次系数直接比大小。",
        "整体CGM与餐次可能共享读数；不作独立验证或机制解释比例。",
        "所有CI、strict参照、NE原因和诊断留 tables/；个体数据留 private/。",
        "* 数值下溢显示，不表示真实概率等于0。只需回传这3页。"]
    bodies.append(lines)
    for i, lines in enumerate(bodies, 1):
        (run.out / "pages" / "{:02d}.txt".format(i)).write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary(run, state):
    text = "# 菌群与餐次血糖衔接\n\nDATA={}\n\nSTATUS={}\n\n{}\n\n".format(
        run.c["data_kind"], state["status"], state.get("reason", ""))
    text += "只回传 pages/01.txt、02.txt、03.txt。完整表和private个体信息保留服务器。\n"
    (run.out / "SUMMARY.md").write_text(text, encoding="utf-8")


def show(value, output_root, page=None):
    if value == "latest":
        dirs = sorted(p for p in Path(output_root).expanduser().glob("gut_meal_bridge_v2_*") if p.is_dir())
        require(dirs, "No gut_meal_bridge_v2 runs found")
        path = dirs[-1]
    else:
        path = Path(value).expanduser().resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    state = manifest["status"]["gut_meal_bridge_v2"]
    require(state["status"] in ["COMPLETE", "COMPLETE_WITH_LIMITATIONS"],
            "Selected run is not complete: {} ({}) {}".format(path.name, state["status"], state.get("reason", "")))
    require(manifest["data_kind"] in ["REAL", "SYNTHETIC"], "Unverified data label")
    wanted = [page] if page else [1, 2, 3]
    for i in wanted:
        name = "pages/{:02d}.txt".format(i)
        p = path / name
        require(p.is_file() and sha256(p) == state.get("result_sha256", {}).get(name), "Missing/changed result page: " + name)
        print(p.read_text(encoding="utf-8"))
