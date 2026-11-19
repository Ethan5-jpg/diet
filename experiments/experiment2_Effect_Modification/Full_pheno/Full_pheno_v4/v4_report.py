"""Compare workflows by absolute gain and coverage, with both share definitions."""

from collections import defaultdict
from statistics import median

import v4_design as d
from source import check


def pair(row):
    return row["exposure"], row["endpoint"]


def summarize(result, tasks):
    fixed = [row for t in tasks for row in t.fixed]
    peaks = {}
    fixed_by_recipe = defaultdict(list)
    for row in fixed:
        key = pair(row)
        fixed_by_recipe[row["recipe"]].append(row["delta_pp"])
        if key not in peaks or row["delta_pp"] > peaks[key]["delta_pp"]:
            peaks[key] = row
    fixed_ranking = []
    for recipe, gains in fixed_by_recipe.items():
        if len(gains) == 17:
            fixed_ranking.append(dict(recipe=recipe, positive=sum(x > d.EPS_PP for x in gains),
                                      good=sum(x >= d.MIN_GAIN_PP-d.EPS_PP for x in gains),
                                      loss=sum(x <= -d.MIN_GAIN_PP+d.EPS_PP for x in gains),
                                      median_pp=median(gains)))
    fixed_ranking.sort(key=lambda r: (-r["good"], -r["positive"], r["loss"], -r["median_pp"], r["recipe"]))
    groups, differences = defaultdict(list), defaultdict(list)
    for row in result["rows"]:
        groups[row["workflow"]].append(row)
    for row in result["comparisons"]:
        if pair(row) != d.ANCHOR:
            differences[row["workflow"]].append(row)
    ranking = []
    for method in d.WORKFLOWS:
        full = groups[method]
        other = [row for row in full if pair(row) != d.ANCHOR]
        check(len({pair(row) for row in full}) == len(full) == 18 and len(other) == 17, "V4_INCOMPLETE_REPORT")
        delta = [row["delta_pp"] for row in other]
        ranking.append(dict(
            workflow=method, pos18=sum(row["delta_pp"] > d.EPS_PP for row in full),
            pos17=sum(x > d.EPS_PP for x in delta),
            good17=sum(x >= d.MIN_GAIN_PP-d.EPS_PP for x in delta),
            large17=sum(x >= d.LARGER_GAIN_PP-d.EPS_PP for x in delta),
            loss17=sum(x <= -d.MIN_GAIN_PP+d.EPS_PP for x in delta),
            median17=median(delta), peak17=max(delta),
            adjusted_positive17=sum(row["low_pp"] > d.EPS_PP for row in other),
            adjusted_beat_pc5=sum(row["low_pp"] > d.EPS_PP for row in differences[method]),
            pos_full_share17=sum((row["full_share_pct"] or 0.) > 0. for row in other),
            pos_diet_share17=sum((row["diet_share_pct"] or 0.) > 0. for row in other),
        ))
    ranking.sort(key=lambda row: (-row["good17"], -row["pos17"], row["loss17"], -row["median17"], row["workflow"]))
    share_peaks = {}
    for field in ("full_share_pct", "diet_share_pct"):
        eligible = [r for r in result["rows"] if pair(r) != d.ANCHOR
                    and r["delta_pp"] >= d.MIN_GAIN_PP-d.EPS_PP and r[field] is not None]
        share_peaks[field] = max(eligible, key=lambda row: row[field], default=None)
    result.update(ranking=ranking, fixed_peaks=list(peaks.values()), fixed_ranking=fixed_ranking,
                  share_peaks=share_peaks)
    return fixed


def percent(value):
    return "NA" if value is None else f"{value:+.2f}%"


def render(summary):
    lookup = {(pair(row), row["workflow"]): row for row in summary["rows"]}
    anchor = lookup[d.ANCHOR, d.PRIMARY]
    lines = [
        f"FULL_PHENO_V4_SCREEN | 18 pairs | {len(d.LIBRARY)} settings | discovery only",
        f'ANCHOR fiber/D_Glu | {d.ANCHOR_RECIPE} | delta={anchor["delta_pp"]:+.4f} pp '
        f'| full share={percent(anchor["full_share_pct"])} | diet share={percent(anchor["diet_share_pct"])}',
        "WORKFLOWS | method | pos18 | pos17 | >=.10 | >=.25 | <=-.10 | median | peak | "
        "adj+ | beats PC5 adj | positive full/diet share (columns from pos17 exclude anchor)",
    ]
    for r in summary["ranking"]:
        lines.append(f'{r["workflow"]:19} | {r["pos18"]:2}/18 | {r["pos17"]:2}/17 | {r["good17"]:2}'
                     f' | {r["large17"]:2} | {r["loss17"]:2} | {r["median17"]:+.4f}'
                     f' | {r["peak17"]:+.4f} | {r["adjusted_positive17"]:2} | {r["adjusted_beat_pc5"]:2}'
                     f' | {r["pos_full_share17"]:2}/{r["pos_diet_share17"]:2}')
    lines.append(f"PRIMARY_PAIRS | {d.PRIMARY} | R2 no-diet/off/on | delta [95% simultaneous] pp | "
                 "diet total pp | full share | diet share | fold wins")
    for row in summary["rows"]:
        if row["workflow"] != d.PRIMARY:
            continue
        label = ("fiber" if row["exposure"] == "fiber_density" else row["exposure"]) + "/" + row["endpoint"]
        lines.append(f'{label:30} | {row["r2_no_diet"]:+.5f}/{row["r2_off"]:+.5f}/{row["r2_on"]:+.5f}'
                     f' | {row["delta_pp"]:+.4f} [{row["low_pp"]:+.4f},{row["high_pp"]:+.4f}]'
                     f' | {row["total_diet_pp"]:+.4f} | {percent(row["full_share_pct"])}'
                     f' | {percent(row["diet_share_pct"])} | {row["folds_positive"]}/5')
    lines.append("EXPLORATORY_FIXED_PEAKS | strongest other-pair point estimates, each pair chooses its own setting")
    for r in sorted(summary["fixed_peaks"], key=lambda x: -x["delta_pp"])[:5]:
        lines.append(f'{r["exposure"]}/{r["endpoint"]} | {r["delta_pp"]:+.4f} pp | {r["recipe"]}')
    lines.append("EXPLORATORY_FIXED_COVERAGE | same setting across all other 17 | >=.10/positive/loss | median pp")
    for r in summary["fixed_ranking"][:3]:
        lines.append(f'{r["recipe"]} | {r["good"]}/{r["positive"]}/{r["loss"]} | {r["median_pp"]:+.4f}')
    for field in ("full_share_pct", "diet_share_pct"):
        row = summary["share_peaks"][field]
        label = f"CHECKED_SHARE_PEAK | {field} | other 17 | delta >= +0.10 pp"
        if row is None:
            lines.append(label + " | NA")
        else:
            lines.append(f'{label} | {row["exposure"]}/{row["endpoint"]} | {row["workflow"]}'
                         f' | {percent(row[field])} | delta={row["delta_pp"]:+.4f} pp')
    lines.append("FULL_PHENO_V4_SCREEN_END | workflow_scores.csv, workflow_comparisons.csv, "
                 "library_screen.csv, fold_choices.csv")
    return "\n".join(lines) + "\n"
