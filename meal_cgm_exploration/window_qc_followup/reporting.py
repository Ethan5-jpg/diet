"""Four aggregate pages; participant records remain in private/."""
import json
from pathlib import Path

import pandas as pd

from hppmeal.common import require, sha256
from seven_diet_meals.inputs import DIETS


def number(x):
    if pd.isna(x):
        return "NA"
    return "{:.4g}".format(float(x))


def cell(row, qfield):
    if row.status != "ESTIMATED":
        return "NE"
    return "{}[{},{}];q={}".format(number(row.estimate), number(row.ci_low), number(row.ci_high), number(row[qfield]))


def build_pages(run, state, diet=None, gut=None):
    (run.out / "pages").mkdir(exist_ok=True)

    def intro(i, title):
        return ["KEY {}/4 | {}".format(i, title), "RUN=" + run.out.name,
                "DATA={} | STATUS={}".format(run.c["data_kind"], state["status"]), ""]

    def read(name):
        path = run.out / "tables" / name
        return pd.read_csv(path) if path.is_file() else None

    def add_table(lines, name, cols=None):
        table = read(name)
        if table is not None:
            lines.append((table[cols] if cols else table).to_string(index=False, float_format=lambda x: "{:.4g}".format(x)))

    lines = intro(1, "两周覆盖、缺失率与筛选")
    lines += ["CGM: 14×24小时/15分钟=1344槽；严格小于阈值，无插值填补。",
              "饮食: 14个参考日历日期的无记录日比例；不是漏餐率。",
              "默认<5%饮食要求14/14日期均有记录；<10%需13/14，<20%需12/14。",
              "主阈值: CGM<0.05; diet<{}。".format(state["diet_main_threshold"])]
    add_table(lines, "00_scope_and_anchor.csv", ["cgm_population_scope", "window_anchor", "missing_before_first_record_identifiable"])
    add_table(lines, "01_coverage_population.csv", ["people", "connections", "audited_connections", "no_cgm_people", "multiple_connection_people", "first_record_proxy_connections", "known_plan_connections"])
    add_table(lines, "01_coverage_population.csv", ["audited_connections_with_any_cgm_missing", "audited_connections_with_zero_cgm_missing", "audited_connections_with_all_14_diet_days"])
    add_table(lines, "01_coverage_population.csv", ["connections_without_auditable_window", "connections_with_records_but_unmatched_window"])
    lines += ["", "缺失率分组（参与者可有多个连接，连接与人数请分开）:"]
    add_table(lines, "01_missing_rate_distribution.csv")
    lines += ["", "缺失率分布（逐连接，非参与者加权）:"]
    add_table(lines, "01_coverage_quantiles.csv")
    lines += ["", "阈值流失（strict的糖尿病/A10完整资格仅在模型阶段追加）:"]
    add_table(lines, "02_threshold_flow.csv", ["setting", "raw_auditable_people", "cgm_pass_people", "diet_pass_people", "joint_pass_people", "in_window_people", "in_window_analysis_ready_meals"])
    lines += ["", "原模型复现: " + state.get("reproduction", "NOT_RUN"),
              "CGM首记录是代理起点时，首记录之前缺失未知；尾部仍按336小时计缺失。",
              "无法确定窗口的缺失率记NA；有明确窗口而没有读数才计100%缺失。",
              "多连接不拼接；时区和无法解析饮食时间单独审计。",
              "日表有记录+事件有记录仍不能确认同日无漏餐。", state.get("reason", "")]
    bodies = [lines]

    lines = intro(2, "质量主样本：七评分与餐次反应")
    if diet is None:
        lines += ["本次仅检查/覆盖审计，未估计模型。下一步运行--stage all。"]
    else:
        lines += ["B[逐项95%CI];q。A: mg/dL/原饮食Z；B: mg/dL/10g碳水/原饮食Z。",
            "均为已确定主阈值qc_primary；下面展示A餐食调整、B交互，背景模型保留表中。",
            "q按原三家族各14项；表中另有q_all42。保留全部7评分和阴性结果。"]
        for outcome, title in [("mean_increment", "餐后平均净增量"), ("peak_rise_0_120", "峰值升幅")]:
            compact = []
            for score in DIETS:
                group = diet.loc[diet.setting.eq("qc_primary") & diet.diet.eq(score) & diet.outcome.eq(outcome)]
                a = group.loc[group.model.eq("meal_adjusted")].iloc[0]
                b = group.loc[group.branch.eq("B")].iloc[0]
                compact.append(dict(diet=score, A_N=int(a.people), A=cell(a, "q_family14"), B_N=int(b.people), B=cell(b, "q_family14")))
            lines += ["", title, pd.DataFrame(compact).to_string(index=False)]
        lines += ["评分方向沿用旧评分；rEDIH/NOVA4方向与健康评分不同。",
                  "固定原Z；未在质量筛选后重新标准化。NE及全部背景规格见diet_models.csv。"]
    bodies.append(lines)

    lines = intro(3, "同访视窗口：候选菌与餐前餐后")
    if gut is None:
        lines += ["本次未估计模型。老师确认的同访视窗口解释保存在analysis_spec.json。"]
    else:
        lines += ["四候选菌是旧同队列候选；B[95%CI];q，mg/dL/原CLR-Z。",
            "净增量4项家族，餐前+峰值8项；表中另有全部餐次q12/每组q24。"]
        for label in ["qc_primary", "qc_strict"]:
            compact = []
            for taxon in ["T023", "T046", "T068", "T096"]:
                group = gut.loc[gut.setting.eq(label) & gut.taxon.eq(taxon)].set_index("outcome")
                compact.append(dict(taxon=taxon, N=int(group.iloc[0].people),
                    Net=cell(group.loc["mean_increment"], "q_family"),
                    Pre=cell(group.loc["baseline_glucose"], "q_family"),
                    Peak=cell(group.loc["peak_rise_0_120"], "q_family")))
            lines += ["", label, pd.DataFrame(compact).to_string(index=False)]
        lines += ["", "采样日期相对CGM窗口（不按前后筛选）:"]
        add_table(lines, "04_stool_timing.csv", ["setting", "people", "known_window_and_stool_date", "unknown_or_unmatched", "before", "within", "after", "gap_after_window_median"])
        lines += ["采样日期未知者保留；唯一访视记录匹配不等于测序样本身份验证。",
                  "T023=GGB6012_SGB8640; T046=Clostridium_phoceensis;",
                  "T068=Eubacterium_rectale; T096=Ruthenibacterium_lactatiformans。",
                  "允许CGM后交样的同窗关联；不称基线预测、因果中介或机制验证。"]
    bodies.append(lines)

    lines = intro(4, "阈值敏感性、整体CGM参照与完成情况")
    if diet is not None:
        compact = []
        for score in DIETS:
            row = dict(diet=score)
            for label in ["source_primary", "qc_primary", "qc_strict", "qc10", "qc20"]:
                r = diet.loc[diet.setting.eq(label) & diet.diet.eq(score) & diet.outcome.eq("mean_increment") & diet.model.eq("meal_adjusted")].iloc[0]
                row[label] = "N={};{}".format(int(r.people), cell(r, "q_family14"))
            compact.append(row)
        lines += ["七评分A餐食调整净增量：筛选前、主质量、strict、10%、20%", pd.DataFrame(compact).to_string(index=False), ""]
        compact = []
        for taxon in ["T023", "T046", "T068", "T096"]:
            row = dict(taxon=taxon)
            for label in ["source_primary", "qc_primary", "qc10", "qc20"]:
                r = gut.loc[gut.setting.eq(label) & gut.taxon.eq(taxon) & gut.outcome.eq("mean_increment")].iloc[0]
                row[label] = "N={};{}".format(int(r.people), cell(r, "q_family"))
            compact.append(row)
        lines += ["四菌主结局阈值敏感性:", pd.DataFrame(compact).to_string(index=False), ""]
        compact = []
        for taxon in ["T023", "T046", "T068", "T096"]:
            group = gut.loc[gut.setting.eq("qc_primary") & gut.taxon.eq(taxon)].set_index("outcome")
            compact.append(dict(taxon=taxon, Mean=cell(group.loc["cgm_mean_z"], "q_family"),
                CV=cell(group.loc["cgm_cv_z"], "q_family"), SDw=cell(group.loc["cgm_sdw_z"], "q_family")))
        lines += ["冻结的同连接整体CGM参照：原CGM Z/CLR-Z，q为12项家族。",
            "这些指标可能含336h之外读数；不是本轮重新计算的固定两周指标。",
            pd.DataFrame(compact).to_string(index=False), "",
            "新模型估计: diet {}/168; gut {}/96。".format(state["diet_new_estimated"], state["gut_new_estimated"]),
            "NE原因/完整CI与q、背景模型、筛选差异见tables/。",
            "严格/宽松样本重叠；不可按q挑阈值。系数变化不是机制比例。"]
    else:
        lines += ["仅完成输入检查或覆盖审计，没有新的关联结果。"]
    lines += ["只回传这4页；private/包含个体记录，留在服务器。",
              "小数显示经过舍入，正式写作使用tables/全精度；p/q=0为数值下溢。"]
    bodies.append(lines)
    for i, lines in enumerate(bodies, 1):
        (run.out / "pages" / "{:02d}.txt".format(i)).write_text("\n".join(lines) + "\n", encoding="utf-8")
    text = "# 两周质量与同访视关联\n\nDATA={}\n\nSTATUS={}\n\n{}\n\n".format(run.c["data_kind"], state["status"], state.get("reason", ""))
    text += "终端回传pages/01–04；个体信息留private/。缺失率阈值在本轮结果产生前固定。\n"
    (run.out / "SUMMARY.md").write_text(text, encoding="utf-8")


def show(value, root, page=None):
    if value == "latest":
        folders = sorted(p for p in Path(root).expanduser().glob("window_qc_followup_*") if p.is_dir())
        require(folders, "No window_qc_followup runs")
        path = folders[-1]
    else:
        path = Path(value).expanduser().resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    state = manifest["status"]["window_qc_followup"]
    require(state["status"] in ["CHECK_PASS", "AUDIT_COMPLETE", "COMPLETE_WITH_LIMITATIONS", "PARTIAL"],
            "Latest selected run not displayable: {} {}".format(state["status"], state.get("reason", "")))
    require(manifest["data_kind"] in ["REAL", "SYNTHETIC"], "Unverified data label")
    for i in [page] if page else range(1, 5):
        name = "pages/{:02d}.txt".format(i)
        file = path / name
        require(file.is_file() and sha256(file) == state.get("result_sha256", {}).get(name), "Changed/missing result page: " + name)
        print(file.read_text(encoding="utf-8"), flush=True)
