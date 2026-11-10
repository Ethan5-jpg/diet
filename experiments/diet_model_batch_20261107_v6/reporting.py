"""A short, deterministic summary; full model results remain in summary.json."""

from experiment import INTERACTIONS, model_names


def compact_summary(result, config):
    score = {(row["model"], row["repeat"]): row for row in result["scores"]}
    models = model_names(config)

    def values(name):
        return [score[(name, repeat)]["R2"] for repeat in range(2)]

    ranked = sorted(models, key=lambda name: min(values(name)), reverse=True)
    baseline = values("v4_hgb_oral")
    shown = list(dict.fromkeys(["v4_hgb_oral", *ranked[:3], "kernel_main"]))
    lines = [
        "# V6 结果摘要",
        "",
        "纤维密度 → 平均血糖（mg/dL）；发现组两套家庭划分。",
        "",
        "| 模型 | 第一轮 R² | 第二轮 R² | ΔR² 对基线：第一轮 | 第二轮 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in shown:
        a, b = values(name)
        lines.append(
            "| %s | %.6f | %.6f | %+.6f | %+.6f |"
            % (name, a, b, a - baseline[0], b - baseline[1])
        )
    lines += [
        "",
        "展示前三个新管线，按两轮中较低的 R² 排序；另列基线与匹配核主模型。",
        "",
        "| 显式交互 | ΔR² 对 kernel_main：第一轮 | 第二轮 |",
        "|---|---:|---:|",
    ]
    positive = 0
    for name in INTERACTIONS:
        changes = [
            next(
                row["delta_R2"]
                for row in result["comparisons"]
                if row["model"] == name and row["repeat"] == repeat
            )
            for repeat in range(2)
        ]
        positive += all(value > 0 for value in changes)
        lines.append("| %s | %+.6f | %+.6f |" % (name, *changes))
    reached = [
        "是" if any(min(values(name)) >= cutoff for name in models) else "否"
        for cutoff in (0.2, 0.3)
    ]
    lines += [
        "",
        "同一新管线两轮均≥0.20：%s；均≥0.30：%s。交互两轮均为正：%d/3。"
        % (*reached, positive),
        "两轮共享参与者，仍为发现集探索；阈值按原始精度判断。完整结果保存在 summary.json。",
    ]
    return "\n".join(lines) + "\n"
