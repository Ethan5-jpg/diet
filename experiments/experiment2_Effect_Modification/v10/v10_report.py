"""Paired, household-resampled comparisons of the identical recipe in both repeats."""

import numpy as np
import v8_scores as stats
from v9_ablation import order_average
from v10_audit import evaluate
from v9_report import parent_comparisons
from v10_cv import adaptive_pairs
import v10_grid as grid

ALL = (*grid.KEYS, *grid.ADAPTIVE)
OFF_NAMES = (*grid.FAMILIES, *grid.ADAPTIVE)
OFF = np.r_[grid.OFF_INDEX, np.arange(len(grid.ADAPTIVE)) + len(grid.FAMILIES)].astype(int)
INDEX = {k: i for i, k in enumerate(ALL)}


def assemble(source, folds):
    batches = []
    groups = np.asarray(source.groups).astype(str)
    for r in (0, 1):
        outcome = source.reference["y"][r]
        eligible = np.isfinite(outcome)
        parts = []
        for f in range(5):
            a, m = folds[r, f]
            rows = np.flatnonzero(eligible & (source.reference["outer"][r] == f))
            n = len(rows)
            if (
                (m["repeat"], m["fold"], m["people"]) != (r, f, n)
                or m.get("paired_column_sha256") != grid.COLUMN_SHA
                or m.get("additive_models_checked") != len(grid.FAMILIES)
                or not np.array_equal(a["rows"], rows)
                or not np.array_equal(a["groups"], groups[rows])
                or not np.array_equal(a["y"], outcome[rows])
            ):
                raise ValueError("Outer fold identity changed")
            if a["setting_off"].shape != (n, len(grid.FAMILIES)) or a["setting_on"].shape != (
                n,
                len(grid.KEYS),
            ):
                raise ValueError("Wrong setting dimensions")
            for actual, expected in zip(
                (a["adaptive_off"], a["adaptive_on"]),
                adaptive_pairs(a["setting_off"], a["setting_on"], m["choices"]),
                strict=True,
            ):
                if not np.array_equal(actual, expected):
                    raise ValueError("Adaptive forecast differs from its frozen inner choice")
            parts.append(a)
        order = np.argsort(np.concatenate([a["rows"] for a in parts]))

        def merged(k):
            return np.concatenate([a[k] for a in parts])[order]

        batch = {k: merged(k) for k in ("rows", "groups", "y")}
        if not np.array_equal(batch["rows"], np.flatnonzero(eligible)):
            raise ValueError("Incomplete eligible outer predictions")
        # Fixed and adaptive procedures enter the same simultaneous threshold calculation exactly once.
        batch["setting_off"] = np.column_stack((merged("setting_off"), merged("adaptive_off")))
        batch["setting_on"] = np.column_stack((merged("setting_on"), merged("adaptive_on")))
        batch["on"] = batch["off"] = np.empty((len(order), 0))
        batches.append(batch)
    return batches


def subset(batches, names):
    ids = [INDEX[n] for n in names]
    return [
        dict(
            rows=b["rows"],
            y=b["y"],
            groups=b["groups"],
            off=b["setting_off"][:, OFF[ids]],
            on=b["setting_on"][:, ids],
        )
        for b in batches
    ]


def ablation(batches, draws):
    names = [grid.key(n, 0.01) for n in grid.CELLS]
    selected = subset(batches, names)
    if any(not np.allclose(p["off"], p["off"][:, :1], rtol=0, atol=2e-10) for p in selected):
        raise ValueError("Common-main factorial contains different additive models")
    point, boot = stats.paired_bootstrap(selected, draws, grid.SEED)
    est = order_average(100 * point[:, 2], 3)
    reliable = len(boot) >= np.ceil(0.95 * draws)
    radius = None
    if reliable:
        samples = order_average(100 * boot[:, :, 2], 3)
        radius = float(np.quantile(np.max(np.abs(samples - est), axis=(1, 2)), 0.95))
    return {
        label: [
            dict(
                repeat=r,
                mean_increment_r2_pp=float(est[r, j]),
                simultaneous_ci95_pp=(
                    None if radius is None else [float(est[r, j] - radius), float(est[r, j] + radius)]
                ),
            )
            for r in (0, 1)
        ]
        for j, label in enumerate(grid.BLOCKS)
    }


def ranked(rows, names):
    return sorted((n for n in names if rows[n]["stable_both"]), key=lambda n: -rows[n]["minimum_ratio_pct"])


def common_main_comparison(batches, draws):
    """Same additive prediction for five surfaces; conditional paired CIs against its anchor."""
    names = [grid.key(n, 0.01) for n in grid.FITS if grid.GROUPS[n] == "common_background"]
    anchor = grid.key("controlled_anchor", 0.01)
    selected = subset(batches, names)
    if any(not np.allclose(p["off"], p["off"][:, :1], rtol=0, atol=2e-10) for p in selected):
        raise ValueError("The five score shapes do not share the declared off predictions")
    return parent_comparisons(
        selected, names, {n: (anchor,) for n in names if n != anchor}, draws, grid.SEED
    )


def crossed_comparison(batches, draws):
    """Two controlled 2x2 comparisons; one conditional band across 16 estimates."""
    names = [grid.key(n, grid.CROSS_PENALTY) for cells in grid.CROSS_CELLS.values() for n in cells]
    selected = subset(batches, names)
    for panel in selected:
        for i in range(0, len(names), 4):
            if not np.allclose(panel["off"][:, i : i + 4], panel["off"][:, i : i + 1], atol=2e-10, rtol=0):
                raise ValueError("Crossed cells have different off predictions")
    point, boot = stats.paired_bootstrap(selected, draws, grid.SEED)
    groups = tuple(grid.CROSS_CELLS)

    def contrasts(gains):
        x = np.reshape(gains, (*gains.shape[:-1], len(groups), 4))
        base, dose, marker, both = np.moveaxis(x, -1, 0)
        return np.stack((dose - base, marker - base, both - base, (both - marker) - (dose - base)), axis=-1)

    point_est = contrasts(100 * point[:, 2])
    radius = None
    if len(boot) >= np.ceil(0.95 * draws):
        deviations = np.abs(contrasts(100 * boot[:, :, 2]) - point_est)
        radius = float(np.quantile(np.max(deviations, axis=(1, 2, 3)), 0.95))
    labels = ("dose_minus_base", "marker_minus_base", "both_minus_base", "excess_over_sum_of_single_gains")
    return {
        g: {
            label: [
                dict(
                    repeat=r,
                    delta_r2_pp=float(z),
                    simultaneous_ci95_pp=None if radius is None else [float(z - radius), float(z + radius)],
                )
                for r, z in enumerate(point_est[:, j, k])
            ]
            for k, label in enumerate(labels)
        }
        for j, g in enumerate(groups)
    }


def summarize(source, folds, draws):
    batches = assemble(source, folds)
    rows, critical, valid = evaluate(batches, ALL, OFF, OFF_NAMES, grid.TARGET, draws, grid.SEED)
    control = rows[grid.REFERENCE]
    for name, row in rows.items():
        for s, ref in zip(row["repeats"], control["repeats"], strict=True):
            s["off_change_vs_v9_r2_pp"] = 100 * (s["off_r2"] - ref["off_r2"])
            s["on_change_vs_v9_r2_pp"] = 100 * (s["on_r2"] - ref["on_r2"])
            s["gain_change_vs_v9_r2_pp"] = s["delta_r2_pp"] - ref["delta_r2_pp"]
        row["group"] = grid.GROUPS[name.rsplit("@", 1)[0]] if name in grid.INDEX else "inner_workflow"
        row["fixed"] = name in grid.INDEX
    fixed = ranked(rows, grid.KEYS)
    fixed_hits = [k for k in grid.KEYS if rows[k]["above_target_both_point"]]
    stable_hits = [k for k in fixed if rows[k]["above_target_both_point"]]
    adjusted = [k for k in grid.KEYS if rows[k]["above_target_both_adjusted"]]
    equal_or_better_gain = [
        k for k in stable_hits if all(s["gain_change_vs_v9_r2_pp"] >= -1e-10 for s in rows[k]["repeats"])
    ]
    strong_hits = [
        k
        for k in equal_or_better_gain
        if all(s["on_change_vs_v9_r2_pp"] >= -1e-10 for s in rows[k]["repeats"])
    ]
    winners = list(
        dict.fromkeys(
            [grid.PRIMARY, *grid.ADAPTIVE, *fixed[:3], *equal_or_better_gain[:1], *strong_hits[:1]]
        )
    )
    winners = [n for n in winners if n != grid.REFERENCE]
    members = [grid.REFERENCE, *winners]
    comparison = parent_comparisons(
        subset(batches, members), members, {n: (grid.REFERENCE,) for n in winners}, draws, grid.SEED
    )
    return dict(
        target_pct=grid.TARGET,
        scope="discovery only",
        people=len(batches[0]["rows"]),
        households=len(np.unique(batches[0]["groups"])),
        draws=draws,
        valid_draws=valid,
        simultaneous_critical_r2_pp=critical,
        settings=len(grid.KEYS),
        adaptive=len(grid.ADAPTIVE),
        fixed_both_point=fixed_hits,
        fixed_both_point_stable=stable_hits,
        fixed_both_adjusted=adjusted,
        stable_hits_with_gain_at_least_v9_both=equal_or_better_gain,
        stable_hits_with_gain_and_full_at_least_v9_both=strong_hits,
        ranked_stable_fixed=fixed,
        routes=rows,
        vs_anchor=comparison,
        common_background=common_main_comparison(batches, draws),
        anchor_ablation=ablation(batches, draws),
        crossed_combinations=crossed_comparison(batches, draws),
        paired_point_audit=dict(
            checked=len(ALL) * 2,
            fixed_checked=2 * len(grid.KEYS),
            outer_fold_family_checks=len(folds) * len(grid.FAMILIES),
            expected_off="by family name",
            completed=True,
        ),
        inner_population=[
            dict(repeat=r, fold=f, eligible=m["inner_people"], common=m["inner_common_people"])
            for (r, f), (_, m) in sorted(folds.items())
        ],
    )


def fmt(value, digits=3):
    return "NA" if value is None else f"{value:+.{digits}f}"


def fmt_ratio(value):
    if value is not None and 0 < abs(value - grid.TARGET) < 0.0005:
        return f"{value:+.9f} (gap {value-grid.TARGET:+.2e})"
    return fmt(value)


def compact(result):
    print("EXP2_V10_REPLAY", result["v9_replay"], flush=True)
    print(
        f"EXP2_V10_SCREEN | N={result['people']} HH={result['households']} | same discovery people | strict target >10.00% in both"
    )
    ranks = result["ranked_stable_fixed"]
    audit = result["paired_point_audit"]
    print(
        f"EXP2_V10_PAIR_AUDIT | fold-by-family off checks={audit['outer_fold_family_checks']}; direct person-level recalculations: fixed={audit['fixed_checked']} total={audit['checked']}"
    )
    print("type | setting | ratio % r0/r1 | smaller % | delta R2 pp r0/r1 | above both point/adjusted")
    unique_families = {}
    for n in ranks:
        unique_families.setdefault(result["routes"][n]["off_family"], n)
    shown = [
        grid.REFERENCE,
        grid.PRIMARY,
        *grid.ADAPTIVE,
        *ranks[:2],
        *list(unique_families.values())[:4],
        *result["stable_hits_with_gain_at_least_v9_both"][:1],
        *result["stable_hits_with_gain_and_full_at_least_v9_both"][:1],
    ]
    for n in dict.fromkeys(shown):
        row = result["routes"][n]
        scores = row["repeats"]
        ratio = "/".join(fmt_ratio(s["ratio_pct"]) for s in scores)
        delta = "/".join(fmt(s["delta_r2_pp"]) for s in scores)
        print(
            f"  {'fixed' if row['fixed'] else 'adaptive'} | {n} | {ratio} | {fmt_ratio(row['minimum_ratio_pct'])} | {delta} | {row['above_target_both_point']}/{row['above_target_both_adjusted']}"
        )
    best = ranks[0] if ranks else None
    print(
        f"EXP2_V10_BOTH | fixed_total={result['settings']} point={len(result['fixed_both_point'])} stable_point={len(result['fixed_both_point_stable'])} adjusted={len(result['fixed_both_adjusted'])} stable_hits_with_delta_at_least_v9_both={len(result['stable_hits_with_gain_at_least_v9_both'])} best_stable={best}"
    )
    print(
        f"EXP2_V10_GAIN_CHECK | above 10% with stable denominators and both delta and full R2 at least V9 in each split: {len(result['stable_hits_with_gain_and_full_at_least_v9_both'])}"
    )
    if result["fixed_both_point"]:
        hits = sorted(result["fixed_both_point"], key=lambda n: -result["routes"][n]["minimum_ratio_pct"])
        print("EXP2_V10_POINT_HITS | top same fixed settings:", ", ".join(hits[:5]))
    if result["fixed_both_adjusted"]:
        print("EXP2_V10_ADJUSTED |", ", ".join(result["fixed_both_adjusted"][:6]))
    if best:
        for s in result["routes"][best]["repeats"]:
            ci = s["ratio_ci95_pct"]
            print(
                f"EXP2_V10_BEST r{s['repeat']} | off/on R2={s['off_r2']:.6f}/{s['on_r2']:.6f} | ratio {fmt(s['ratio_pct'])}% [{fmt(ci[0])},{fmt(ci[1])}] | change vs V9 off/on/gain (R2 pp)={fmt(s['off_change_vs_v9_r2_pp'])}/{fmt(s['on_change_vs_v9_r2_pp'])}/{fmt(s['gain_change_vs_v9_r2_pp'])}"
            )
        for s in result["vs_anchor"]:
            if s["child"] == best:
                print(
                    f"EXP2_V10_VS_ANCHOR r{s['repeat']} | ratio difference pp={fmt(s['ratio_difference_pp'])} CI95={s['ratio_difference_ci95_pp']}"
                )
    print("EXP2_V10_ABLATION | order-averaged extra R2 (pp), r0/r1")
    for n, p in result["anchor_ablation"].items():
        print("  " + n + " " + "/".join(fmt(s["mean_increment_r2_pp"]) for s in p))
    for g, cols in result["crossed_combinations"].items():
        text = " | ".join(
            k + " " + "/".join(fmt(s["delta_r2_pp"]) for s in cols[k])
            for k in ("both_minus_base", "excess_over_sum_of_single_gains")
        )
        print(f"EXP2_V10_CROSSED {g} | {text} (R2 pp r0/r1; family bands in summary.json)")
    print(
        "EXP2_V10_SCREEN_END | fixed_settings.csv, workflows.csv, vs_v9.csv, summary.json | discovery screen; household intervals condition on fitted predictions."
    )
