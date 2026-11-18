"""Paired household inference for the same model in both discovery repartitions."""

from collections import Counter
import json

import numpy as np

import v8_scores as stats
import v9_ablation
import v9_cv as cv
import v9_fixed
import v9_grid as grid


def ordered_predictions(source, folds):
    groups = np.asarray(source.groups).astype(str)
    batches = []
    for r in (0, 1):
        outcome = source.reference["y"][r]
        eligible = np.isfinite(outcome)
        parts = []
        for f in range(5):
            a, m = folds[r, f]
            expected = np.flatnonzero(eligible & (source.reference["outer"][r] == f))
            if (
                m["names"] != list(cv.NAMES)
                or set(m["choices"]) != set(cv.NAMES)
                or (m["repeat"], m["fold"], m["people"]) != (r, f, len(expected))
                or not np.array_equal(a["rows"], expected)
                or not np.array_equal(a["groups"], groups[expected])
                or not np.array_equal(a["y"], outcome[expected])
                or a["on"].shape != (len(expected), len(cv.NAMES))
                or a["off"].shape != a["on"].shape
                or a["setting_off"].shape != (len(expected), len(grid.FAMILIES))
                or a["setting_on"].shape != (len(expected), len(grid.KEYS))
            ):
                raise ValueError("Fold does not match the frozen discovery rows, outcomes or models")
            cv.verify_fold_choices(a, m)
            parts.append(a)
        order = np.argsort(np.concatenate([a["rows"] for a in parts]))
        batch = {
            k: np.concatenate([a[k] for a in parts])[order]
            for k in ("rows", "groups", "y", "off", "on", "setting_off", "setting_on")
        }
        if not np.array_equal(batch["rows"], np.flatnonzero(eligible)):
            raise ValueError("Outer folds must cover every eligible row exactly once")
        batches.append(batch)
    return batches


def parent_comparisons(batches, names, parents, draws, seed):
    """Bootstrap child and its named parents together on identical household draws."""
    subset = sorted(set(parents).union(*(set(pp) for pp in parents.values())))
    idx = [names.index(n) for n in subset]
    inputs = [{**b, "off": b["off"][:, idx], "on": b["on"][:, idx]} for b in batches]
    point, boot = stats.paired_bootstrap(inputs, draws, seed)
    reliable = len(boot) >= np.ceil(0.95 * draws)
    cache = {
        (name, r): stats.ratios(point[r, :, i], boot[:, r, :, i], reliable)
        for i, name in enumerate(subset)
        for r in (0, 1)
    }
    out = []
    for child, members in parents.items():
        c = subset.index(child)
        for parent in members:
            p = subset.index(parent)
            for r in (0, 1):
                value, ratios, _ = cache[child, r]
                base, reference, _ = cache[parent, r]
                gain = 100.0 * (point[r, 2, c] - point[r, 2, p])
                dist = 100.0 * (boot[:, r, 2, c] - boot[:, r, 2, p])
                out.append(
                    dict(
                        child=child,
                        parent=parent,
                        repeat=r,
                        ratio_difference_pp=None if value is None or base is None else value - base,
                        ratio_difference_ci95_pp=(
                            stats.interval(ratios - reference)
                            if ratios is not None and reference is not None
                            else None
                        ),
                        delta_r2_difference_pp=float(gain),
                        delta_r2_difference_ci95_pp=stats.interval(dist) if reliable else None,
                    )
                )
    return out


def summarize(source, folds, draws):
    panels = ordered_predictions(source, folds)
    result = stats.compare_routes(panels, cv.NAMES, grid.FAMILIES, grid.ANCHOR, draws)
    fixed, critical, valid = v9_fixed.evaluate(
        panels, grid.KEYS, cv.OFF_INDEX, stats.TARGET_PCT, draws, stats.BOOT_SEED
    )
    if valid != result["valid_draws"]:
        raise ArithmeticError("Household resamples differ across the two screens")
    result["simultaneous_critical_r2_pp"] = critical
    result["joint_threshold_columns"] = len(cv.NAMES) + len(grid.KEYS)
    result["fixed_settings"] = fixed
    result["fixed_ranking_by_minimum"] = sorted(
        (k for k, v in fixed.items() if v["minimum_ratio_pct"] is not None),
        key=lambda k: -fixed[k]["minimum_ratio_pct"],
    )
    result["fixed_both_above_point"] = [k for k, v in fixed.items() if v["above_target_both_point"]]
    result["fixed_both_above_adjusted"] = [k for k, v in fixed.items() if v["above_target_both_adjusted"]]
    result["best_stable_fixed_setting"] = next(
        (k for k in result["fixed_ranking_by_minimum"] if fixed[k]["stable_both"]), None
    )
    for name, data in result["routes"].items():
        # The V8 statistics API names its comparator V7. Here it is V8's PC4 gate.
        data["minimum_ratio_vs_v8_pp"] = data.pop("minimum_ratio_vs_v7_pp")
        for r, s in enumerate(data["repeats"]):
            for key in tuple(s):
                if "_vs_v7" in key:
                    s[key.replace("_vs_v7", "_vs_v8")] = s.pop(key)
            s["simultaneous_lower_margin_r2_pp"] = (
                None if critical is None else s["threshold_margin_r2_pp"] - critical
            )
            choices = [folds[r, f][1]["choices"][name] for f in range(5)]
            s["active_folds"] = sum(c["enabled"] for c in choices)
            s["choices"] = dict(
                Counter(json.dumps(c, sort_keys=True, separators=(",", ":")) for c in choices)
            )
        data["above_target_both_adjusted"] = (
            all(s["simultaneous_lower_margin_r2_pp"] > stats.EPS for s in data["repeats"])
            if data["stable_both"] and critical is not None
            else None
        )
    result["both_above_adjusted"] = [
        k for k, v in result["routes"].items() if v["above_target_both_adjusted"]
    ]
    result[
        "inference"
    ] += " Joint threshold includes every fixed setting and every reported selection procedure."
    result["parent_comparisons"] = parent_comparisons(
        panels, cv.NAMES, grid.PARENTS, draws, stats.BOOT_SEED
    )
    result["common_main_ablation"] = v9_ablation.summarize(panels, draws, stats.BOOT_SEED)
    result["reference"] = grid.ANCHOR
    result["primary"] = cv.PRIMARY
    result["fixed_family_count"] = len(grid.FAMILIES)
    result["fixed_setting_reference"] = cv.FIXED
    result["stable_both_above_point"] = [
        n for n in result["both_above_point"] if result["routes"][n]["stable_both"]
    ]
    return result


def compact(summary):
    def number(value):
        return "NA" if value is None else f"{value:+.3f}"

    def ci(limits):
        return "NA" if limits is None else "[" + ",".join(number(v) for v in limits) + "]"

    ranking = summary["exploratory_ranking_by_minimum"]
    replay = summary["v8_replay"]
    print("EXP2_V9_REPLAY", json.dumps(replay, ensure_ascii=True))
    print(
        f"EXP2_V9_SCREEN_BEGIN | N={summary['people']} HH={summary['households']} | "
        f"same recipe above {summary['benchmark_pct']}% in BOTH household repartitions"
    )
    print(
        "  route | ratio% r0/r1 | smaller % | gain over V8 smaller (pp) | delta R2 r0/r1 (pp) | both point/adjusted"
    )
    for n in dict.fromkeys((grid.ANCHOR, cv.FIXED, *cv.WORKFLOWS, *ranking[:6])):
        row = summary["routes"][n]
        a, b = row["repeats"]
        print(
            f"  {n} | {number(a['ratio_pct'])}/{number(b['ratio_pct'])} | {number(row['minimum_ratio_pct'])} | "
            f"{number(row['minimum_ratio_vs_v8_pp'])} | {number(a['delta_r2_pp'])}/{number(b['delta_r2_pp'])} "
            f"| {row['above_target_both_point']}/{row['above_target_both_adjusted']}"
        )
    for n in dict.fromkeys((cv.PRIMARY, *ranking[:1])):
        for row in summary["routes"][n]["repeats"]:
            print(
                f"  r{row['repeat']} {n} off/on R2={row['off_r2']:.5f}/{row['on_r2']:.5f} "
                f"ratio CI%={ci(row['ratio_ci95_pct'])} delta R2 CI pp={ci(row['delta_r2_ci95_pp'])} "
                f"vs PC4 ratio pp={number(row['ratio_vs_v8_pp'])} {ci(row['ratio_vs_v8_ci95_pp'])} "
                f"| {row['ratio_status']}"
            )
    winners = summary["both_above_point"]
    print(
        f"EXP2_V9_BOTH | point={len(winners)} stable_point={len(summary['stable_both_above_point'])} "
        f"adjusted={len(summary['both_above_adjusted'])} / {summary['screened_route_count']} | "
        f"best_stable_family={summary['best_stable_family'] or 'none'}"
    )
    print("EXP2_V9_POINT_NAMES (up to 12)", json.dumps(winners[:12]))
    print("EXP2_V9_ADJUSTED_NAMES (up to 12)", json.dumps(summary["both_above_adjusted"][:12]))
    fixed = summary["fixed_settings"]
    ranked_fixed = summary["fixed_ranking_by_minimum"]
    first_stable = summary["best_stable_fixed_setting"]
    supported = set(summary["fixed_both_above_adjusted"])
    first_supported = next((k for k in ranked_fixed if k in supported), None)
    print(
        f"EXP2_V9_FIXED_BOTH | same setting every fold | point={len(summary['fixed_both_above_point'])} "
        f"stable_point={sum(fixed[k]['stable_both'] for k in summary['fixed_both_above_point'])} "
        f"adjusted={len(summary['fixed_both_above_adjusted'])} / {len(fixed)} | "
        f"best stable={first_stable or 'none'} | best adjusted={first_supported or 'none'}"
    )
    display = list(ranked_fixed[:4])
    for name in (first_stable, first_supported):
        if name and name not in display:
            display.append(name)
    for k in display:
        row = fixed[k]
        a, b = row["repeats"]
        print(
            f"  {k} | ratio% {number(a['ratio_pct'])}/{number(b['ratio_pct'])} | "
            f"delta R2 pp {number(a['delta_r2_pp'])}/{number(b['delta_r2_pp'])} | "
            f"both {row['above_target_both_point']}/{row['above_target_both_adjusted']} | "
            f"{a['ratio_status']}/{b['ratio_status']}"
        )
    if first_stable:
        top = fixed[first_stable]
        a, b = top["repeats"]
        print(
            f"  {first_stable} | off/on r0={a['off_r2']:.5f}/{a['on_r2']:.5f} "
            f"r1={b['off_r2']:.5f}/{b['on_r2']:.5f} | smaller ratio% "
            f"{number(top['minimum_ratio_pct'])} CI {ci(top['minimum_ratio_ci95_pct'])}"
        )
    name = "factor_gate_light_curve"
    print("EXP2_V9_COMBINATION | PC4 gates + lighter ecology penalty + fermentation dose curve")
    rows = [row for row in summary["parent_comparisons"] if row["child"] == name]
    for parent in grid.PARENTS[name]:
        a, b = sorted((row for row in rows if row["parent"] == parent), key=lambda row: row["repeat"])
        print(
            f"  vs {parent} | ratio change pp {number(a['ratio_difference_pp'])}/{number(b['ratio_difference_pp'])} "
            f"| delta R2 change pp {number(a['delta_r2_difference_pp'])}/{number(b['delta_r2_difference_pp'])}"
        )
    audit = summary["common_main_ablation"]
    a, b = audit["repeats"]
    print(
        f"EXP2_V9_COMMON_MAIN | identical off across 8 cells | penalty={audit['fixed_penalty']} | "
        f"full minus core R2 pp r0/r1 = {number(a['full_minus_core_r2_pp'])}/{number(b['full_minus_core_r2_pp'])}"
    )
    for name in audit["block_order"]:
        x, y = a["blocks"][name], b["blocks"][name]
        print(
            f"  {name} | average addition R2 pp r0/r1 = {number(x['order_average_r2_pp'])} "
            f"{ci(x['order_average_simultaneous_ci95_pp'])} / {number(y['order_average_r2_pp'])} "
            f"{ci(y['order_average_simultaneous_ci95_pp'])}"
        )
    print(
        "EXP2_V9_SCREEN_END | discovery only | all_routes.csv, fixed_settings.csv, parent_comparisons.csv",
        flush=True,
    )
