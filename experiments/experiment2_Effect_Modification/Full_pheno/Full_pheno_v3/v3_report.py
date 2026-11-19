"""Read-only reports from sealed aggregate results."""

from collections import defaultdict
from statistics import median

import v3_design as d


def pair(row):
    return row["exposure"], row["endpoint"]


def summarize(summary, task_records):
    candidates = [r for task in task_records for r in task.library if task.pair != d.ANCHOR]
    points = defaultdict(list)
    for r in candidates:
        points[pair(r)].append(r)
    peaks = []
    for (x, e), rows in points.items():
        previous = max((r for r in rows if r["source"] == "v2"), key=lambda r: r["delta_r2_pp"])
        expanded = max(rows, key=lambda r: r["delta_r2_pp"])
        peaks.append(dict(exposure=x, endpoint=e, v2_delta_pp=previous["delta_r2_pp"],
                          v2_recipe=previous["recipe"], expanded_delta_pp=expanded["delta_r2_pp"],
                          expanded_recipe=expanded["recipe"]))
    relative = defaultdict(list)
    for r in summary["comparisons"]:
        if r["reference"] == "fixed_pc5" and pair(r) != d.ANCHOR:
            relative[r["workflow"]].append(r)
    ranks = []
    for method in d.WORKFLOWS:
        rows = [r for r in summary["results"] if r["workflow"] == method and pair(r) != d.ANCHOR]
        inc = [r["delta_pp"] for r in rows]
        ranks.append(dict(workflow=method, positive=sum(v > d.EPS_PP for v in inc),
                          good=sum(v >= d.MIN_GAIN_PP-d.EPS_PP for v in inc),
                          large=sum(v >= d.LARGE_GAIN_PP-d.EPS_PP for v in inc),
                          loss=sum(v <= -d.MIN_GAIN_PP+d.EPS_PP for v in inc),
                          median_pp=median(inc),
                          adjusted_positive=sum(r["low_pp"] > d.EPS_PP for r in rows),
                          corrected_beat_pc5=sum(r["low_pp"] > d.EPS_PP for r in relative[method])))
    ranks.sort(key=lambda r: (-r["good"], -r["positive"], r["loss"], -r["median_pp"], r["workflow"]))
    summary.update(ranking=ranks, library_peaks=peaks)
    return candidates


def format_score(r):
    share = "NA" if r["share_pct"] is None else f'{r["share_pct"]:+.2f}%'
    return f'{r["delta_pp"]:+.4f} [{r["low_pp"]:+.4f},{r["high_pp"]:+.4f}] share={share}'


def render(summary):
    results = {(pair(r), r["workflow"]): r for r in summary["results"]}
    against_selection = {pair(r): r for r in summary["comparisons"] if r["reference"] == "select"}
    anchor = results[d.ANCHOR, d.PRIMARY]
    lines = [
        f'FULL_PHENO_V3_SCREEN | 18 pairs | library={len(d.LIBRARY)} | fixed-prediction inference',
        f'ANCHOR fiber/D_Glu | N={anchor["people"]} HH={anchor["households"]} | '
        f'R2 {anchor["r2_off"]:.5f}->{anchor["r2_on"]:.5f} | {format_score(anchor)} | '
        f'all workflows use {d.ANCHOR_RECIPE}',
        f'OTHER17_WORKFLOWS | primary={d.PRIMARY} | gains are R2 percentage points',
        'workflow | >0 | >=.10 | >=.25 | <=-.10 | median | adjusted >0 | adjusted beats fixed PC5',
    ]
    for row in summary["ranking"]:
        lines.append(f'{row["workflow"]:16} | {row["positive"]:2}/17 | {row["good"]:2}/17 | {row["large"]:2}/17'
                     f' | {row["loss"]:2}/17 | {row["median_pp"]:+.4f} | {row["adjusted_positive"]:2}'
                     f' | {row["corrected_beat_pc5"]:2}')
    lines.append('OTHER17_PAIRS | v2/v3 peak (descriptive) | fixed PC5 | primary pp[95%] share | active | primary-select')
    for row in sorted(summary["library_peaks"], key=lambda r: -r["v2_delta_pp"]):
        key = pair(row)
        label = ("fiber" if key[0] == "fiber_density" else key[0]) + "/" + key[1]
        prim = results[key, d.PRIMARY]
        pc = results[key, "fixed_pc5"]
        diff = against_selection[key]
        lines.append(f'{label:30} | {row["v2_delta_pp"]:+.4f}/{row["expanded_delta_pp"]:+.4f}'
                     f' | {pc["delta_pp"]:+.4f} | {format_score(prim)} | {prim["active_folds"]}/5'
                     f' | {diff["difference_pp"]:+.4f} [{diff["low_pp"]:+.4f},{diff["high_pp"]:+.4f}]')
    glu = next(r for r in summary["diet_overlap"] if r["endpoint"] == "D_Glu")
    if glu["fiber_pc5_delta_pp"] is not None:
        lines.append(f'COMMON_GLU_PC5 | N={glu["people"]} HH={glu["households"]} | '
                     f'fiber/UPF pp={glu["fiber_pc5_delta_pp"]:+.4f}/{glu["upf_pc5_delta_pp"]:+.4f}')
    lines.extend(['Peak library scores are outcome-wise exploration. Workflow weights are learned inside outer training.',
                  'FULL_PHENO_V3_SCREEN_END | workflow_scores.csv, workflow_comparisons.csv, '
                  'library_screen.csv, fold_choices.csv, diet_overlap.csv'])
    return "\n".join(lines) + "\n"
