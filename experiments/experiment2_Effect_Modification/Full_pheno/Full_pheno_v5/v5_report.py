"""Report breadth, fixed-library leads and the two matched ratios."""

from collections import defaultdict
from statistics import median

import v5_design as d
from source import check


def pair(row):
    return row["exposure"], row["endpoint"]


def summarize(summary, tasks):
    fixed = [row for t in tasks for row in t.fixed]
    groups, comparisons, recipes = defaultdict(list), defaultdict(list), defaultdict(list)
    peaks = {}
    for row in fixed:
        key = pair(row)
        if key not in peaks or row["delta_pp"] > peaks[key]["delta_pp"]:
            peaks[key] = row
        recipes[row["recipe"]].append(row["delta_pp"])
    fixed_rank = []
    for name, values in recipes.items():
        if len(values) == 17:
            fixed_rank.append(dict(recipe=name, positive=sum(x > d.EPS_PP for x in values),
                                   good=sum(x >= d.MIN_GAIN_PP-d.EPS_PP for x in values),
                                   loss=sum(x <= -d.MIN_GAIN_PP+d.EPS_PP for x in values),
                                   median_pp=median(values)))
    fixed_rank.sort(key=lambda r: (-r["good"], -r["positive"], r["loss"], -r["median_pp"], r["recipe"]))
    for row in summary["rows"]:
        groups[row["workflow"]].append(row)
    for row in summary["comparisons"]:
        if pair(row) != d.ANCHOR:
            comparisons[row["workflow"]].append(row)
    rank = []
    for method in d.WORKFLOWS:
        rows = groups[method]
        other = [r for r in rows if pair(r) != d.ANCHOR]
        check(len({pair(r) for r in rows}) == len(rows) == 18 and len(other) == 17, "V5_INCOMPLETE_REPORT")
        gain = [r["delta_pp"] for r in other]
        rank.append(dict(workflow=method, pos18=sum(r["delta_pp"] > d.EPS_PP for r in rows),
                         pos17=sum(x > d.EPS_PP for x in gain),
                         good17=sum(x >= d.MIN_GAIN_PP-d.EPS_PP for x in gain),
                         big17=sum(x >= .25-d.EPS_PP for x in gain),
                         loss17=sum(x <= -d.MIN_GAIN_PP+d.EPS_PP for x in gain),
                         median17=median(gain), peak17=max(gain),
                         adjusted=sum(r["low_pp"] > d.EPS_PP for r in other),
                         beats_half=sum(r["difference_pp"] > d.EPS_PP for r in comparisons[method]),
                         beats_half_adjusted=sum(r["low_pp"] > d.EPS_PP for r in comparisons[method]),
                         full_share_positive=sum((r["full_share_pct"] or 0.) > 0. for r in other),
                         diet_share_positive=sum((r["diet_share_pct"] or 0.) > 0. for r in other)))
    rank.sort(key=lambda r: (-r["good17"], -r["pos17"], r["loss17"], -r["median17"], r["workflow"]))
    share = {}
    for style in ("full_share_pct", "diet_share_pct"):
        usable = [r for r in summary["rows"] if pair(r) != d.ANCHOR
                  and r["delta_pp"] >= d.MIN_GAIN_PP-d.EPS_PP and r[style] is not None]
        share[style] = max(usable, key=lambda r: r[style], default=None)
    summary.update(ranking=rank, fixed_ranking=fixed_rank, fixed_peaks=list(peaks.values()), share_peaks=share)
    return fixed


def percent(value):
    return "NA" if value is None else f"{value:+.2f}%"


def render(s):
    anchor = next(r for r in s["rows"] if pair(r) == d.ANCHOR and r["workflow"] == d.PRIMARY)
    lines = [
        f"FULL_PHENO_V5_SCREEN | 18 pairs | library={len(d.LIBRARY)} | discovery only",
        f'ANCHOR fiber/D_Glu | {d.ANCHOR_RECIPE} | delta={anchor["delta_pp"]:+.4f} pp'
        f' | full share={percent(anchor["full_share_pct"])} | diet share={percent(anchor["diet_share_pct"])}',
        "WORKFLOWS | method | pos18 | pos17 | >=.10 | >=.25 | <=-.10 | median | peak | adj+"
        " | beats half / adj | positive full/diet share (from pos17 excludes anchor)",
    ]
    for r in s["ranking"]:
        lines.append(f'{r["workflow"]:18} | {r["pos18"]:2}/18 | {r["pos17"]:2}/17 | {r["good17"]:2}'
                     f' | {r["big17"]:2} | {r["loss17"]:2} | {r["median17"]:+.4f} | {r["peak17"]:+.4f}'
                     f' | {r["adjusted"]:2} | {r["beats_half"]:2}/{r["beats_half_adjusted"]:2}'
                     f' | {r["full_share_positive"]:2}/{r["diet_share_positive"]:2}')
    ref = {pair(r): r for r in s["comparisons"] if r["workflow"] == d.PRIMARY}
    lines.append(f"PRIMARY_PAIRS | {d.PRIMARY} | delta [95% simultaneous] pp | vs half pp | "
                 "diet total pp | full share | diet share")
    for r in s["rows"]:
        if r["workflow"] == d.PRIMARY:
            x, e = pair(r)
            dif = ref[x, e]
            name = ("fiber" if x == "fiber_density" else x) + "/" + e
            lines.append(f'{name:30} | {r["delta_pp"]:+.4f} [{r["low_pp"]:+.4f},{r["high_pp"]:+.4f}]'
                         f' | {dif["difference_pp"]:+.4f} | {r["total_diet_pp"]:+.4f}'
                         f' | {percent(r["full_share_pct"])} | {percent(r["diet_share_pct"])}')
    lines.append("EXPLORATORY_FIXED_PEAKS | pair-specific max pp")
    for r in sorted(s["fixed_peaks"], key=lambda a: -a["delta_pp"])[:5]:
        lines.append(f'{r["exposure"]}/{r["endpoint"]} | {r["delta_pp"]:+.4f} | {r["recipe"]}')
    lines.append("EXPLORATORY_FIXED_COVERAGE | complete other17 | >=.10/positive/loss | median pp")
    for r in s["fixed_ranking"][:3]:
        lines.append(f'{r["recipe"]} | {r["good"]}/{r["positive"]}/{r["loss"]} | {r["median_pp"]:+.4f}')
    for style in ("full_share_pct", "diet_share_pct"):
        r = s["share_peaks"][style]
        prefix = f"CHECKED_SHARE_PEAK | {style} | other17 | delta >= +0.10 pp"
        if r is None:
            lines.append(prefix + " | NA")
        else:
            lines.append(f'{prefix} | {r["exposure"]}/{r["endpoint"]} | {r["workflow"]} | '
                         f'{percent(r[style])} | delta={r["delta_pp"]:+.4f}')
    lines.append("FULL_PHENO_V5_SCREEN_END | workflow_scores.csv, workflow_comparisons.csv, "
                 "library_screen.csv, fold_choices.csv")
    return "\n".join(lines) + "\n"
