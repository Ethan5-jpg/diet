"""Compact ASCII aggregate reports; no individual records leave TRE_ONLY."""

import argparse
import json
from pathlib import Path

import numpy as np

from bridge import Memo, metric
from registry import INTERACTIONS, REFERENCES, names, specifications

ROOT = Path(__file__).resolve().parent


def warning(note):
    return bool(
        note.get("warnings")
        or note.get("converged") is False
        or note.get("selected_inner_warning")
        or note.get("background_warning")
        or note.get("contributing_fit_warnings")
    )


def comparison(y, prediction, reference, groups):
    difference = (y - reference) ** 2 - (y - prediction) ** 2
    _, index = np.unique(groups, return_inverse=True)
    total = np.bincount(index, weights=difference)
    size, g = np.bincount(index), len(total)
    mean = float(difference.mean())
    variance = float(np.var(y))
    se = float(np.sqrt(g / (g - 1) * np.sum((total - size * mean) ** 2)) / len(y))
    return dict(
        delta_R2=mean / variance,
        conditional_household_interval=[(mean - 1.96 * se) / variance, (mean + 1.96 * se) / variance],
    )


def aggregate(config, completed):
    if len(completed) != config["outer_repeats"] * config["outer_folds"]:
        raise ValueError("Partial batches have no pooled report")
    scores, gains, modifiers, quality, selections, ablations = [], [], [], [], [], []
    paired = dict(
        daily_cat_oral="ecohybrid_cat_oral",
        daily_cat_balanced="ecohybrid_cat_balanced",
        gated_soft="convex_core",
        gated_medium="convex_core",
        gated_strong="convex_core",
        tree_meta="ridge_core",
        polynomial_stack="ridge_core",
    )
    for repeat in range(config["outer_repeats"]):
        blocks = [completed[repeat, f] for f in range(config["outer_folds"])]
        rows = np.concatenate([b["rows"] for b in blocks])
        group_sets = [set(b["groups"]) for b in blocks]
        if len(set(rows)) != len(rows) or any(
            group_sets[i] & group_sets[j] for i in range(len(blocks)) for j in range(i)
        ):
            raise ValueError("Duplicate OOF rows or household leakage")
        y, groups = np.concatenate([b["y"] for b in blocks]), np.concatenate([b["groups"] for b in blocks])
        if np.var(y) <= 0 or len(set(groups)) < 2:
            raise ValueError("Nonvariable pooled targets or insufficient households")
        pred = {
            n: np.concatenate([b["predictions"][n] for b in blocks]) for n in names(config) + list(REFERENCES)
        }
        for name, p in pred.items():
            if p.shape != y.shape or not np.isfinite(p).all():
                raise ValueError("Invalid pooled prediction")
            scores.append(dict(model=name, repeat=repeat, N=len(y), **metric(y, p)))
        for name in names(config):
            base = INTERACTIONS.get(name, "v8_ridge_wide")
            gain = comparison(y, pred[name], pred[base], groups)
            wins = sum(
                np.mean((b["y"] - b["predictions"][base]) ** 2)
                > np.mean((b["y"] - b["predictions"][name]) ** 2)
                for b in blocks
            )
            gains.append(
                dict(
                    model=name,
                    repeat=repeat,
                    comparator=base,
                    improved_folds=int(wins),
                    type="explicit_interaction" if name in INTERACTIONS else "total_prediction",
                    **gain,
                )
            )
        for name, base in paired.items():
            ablations.append(
                dict(
                    model=name,
                    comparator=base,
                    repeat=repeat,
                    **comparison(y, pred[name], pred[base], groups),
                )
            )
        for name in INTERACTIONS:
            axis = np.concatenate([b["modifiers"][name] for b in blocks])
            modifiers.append(
                dict(
                    model=name,
                    repeat=repeat,
                    SD_mg_dL=float(axis.std()),
                    active_folds=sum(int(b["active"][name]) for b in blocks),
                    off_folds=sum(int(b["selected"][name] == 0) for b in blocks),
                )
            )
        selections.extend(
            dict(repeat=repeat, fold=f, selected=b["selected"], meta=b["meta"]) for f, b in enumerate(blocks)
        )
        quality.extend(
            dict(repeat=repeat, fold=f, model=n, **note)
            for f, b in enumerate(blocks)
            for n, note in b["outer_diagnostics"].items()
            if warning(note)
        )
    return dict(
        status="COMPLETE",
        scores=scores,
        comparisons=gains,
        ablations=ablations,
        modifiers=modifiers,
        selections=selections,
        fit_quality=quality,
        contrast=config["contrast"],
        confirmation_outcomes_read=False,
        original_input_models=[s["name"] for s in specifications(config) if not s.get("daily")],
        daily_input_models=[s["name"] for s in specifications(config) if s.get("daily")],
        limitations="Discovery exploration; repeated participants; conditional intervals do not cover adaptive model selection or multiple comparisons.",
    )


def completed_results(run, config):
    if not (run / "TRE_ONLY").is_dir():
        return {}
    memo, completed = Memo(run / "TRE_ONLY"), {}
    for r in range(config["outer_repeats"]):
        for f in range(config["outer_folds"]):
            key = "r%d_f%d_result" % (r, f)
            if (memo.root / key).is_dir():
                try:
                    completed[r, f] = memo.get(key)
                except FileNotFoundError:
                    pass
    return completed


def compact(result, config, all_models=False):
    score = {(v["model"], v["repeat"]): v for v in result["scores"]}
    gains = {(v["model"], v["repeat"]): v for v in result["comparisons"]}
    mods = {(v["model"], v["repeat"]): v for v in result["modifiers"]}

    def value(n):
        return [score[n, r]["R2"] for r in range(2)]

    ranked = sorted(names(config), key=lambda n: min(value(n)), reverse=True)
    shown = (
        ranked
        if all_models
        else list(
            dict.fromkeys(
                ranked[:4] + ["ridge_core", "ridge_wide", "ridge_daily", "gated_medium", "tree_meta"]
            )
        )
    )
    flags = {v["model"] for v in result["fit_quality"]}
    daily = set(result["daily_input_models"]) | {"convex_daily", "ridge_daily"}
    lines = [
        "V9 COMPLETE | fiber_density -> D_Glu (mg/dL) | 10/10 household outer folds",
        "N0=%d N1=%d | %d pipelines; repeats share participants"
        % (score["v8_ridge_wide", 0]["N"], score["v8_ridge_wide", 1]["N"], len(names(config))),
        "%-25s %10s %10s %10s %10s %4s" % ("model", "R2_0", "R2_1", "dV8_0", "dV8_1", "flag"),
    ]
    baseline = value("v8_ridge_wide")
    for n in list(REFERENCES) + shown:
        a, b = value(n)
        flag = ("D" if n in daily else "") + ("W" if n in flags else "")
        lines.append(
            "%-25s %10.6f %10.6f %+10.6f %+10.6f %4s"
            % (n, a, b, a - baseline[0], b - baseline[1], flag or "-")
        )
    lines += ["", "EXPLICIT: r null_R2 -> interaction_R2; delta vs matched null; off/active out of 5"]
    for n, base in INTERACTIONS.items():
        for r in range(2):
            lines.append(
                "%-23s r%d %.6f->%.6f d=%+.6f off=%d active=%d"
                % (
                    n,
                    r,
                    score[base, r]["R2"],
                    score[n, r]["R2"],
                    gains[n, r]["delta_R2"],
                    mods[n, r]["off_folds"],
                    mods[n, r]["active_folds"],
                )
            )
    for threshold in (0.2, 0.3):
        passed = [n for n in names(config) if min(value(n)) >= threshold]
        lines.append(
            "Both repeats R2>=%.2f: %d/%d; without fit flags: %d"
            % (threshold, len(passed), len(names(config)), len(set(passed) - flags))
        )
    lines += [
        "D=added FOOD-only day variability; W=fit warning incl. contributing fusion inputs.",
        "Exploratory discovery ranking; no confirmation data. Full aggregates: summary.json.",
    ]
    return "\n".join(lines) + "\n"


def diagnostics(result):
    lines = ["TARGETED DIAGNOSTICS (aggregate only)"]
    for n in sorted({v["model"] for v in result["fit_quality"]}):
        notes = [v for v in result["fit_quality"] if v["model"] == n]
        lines.append("%s flagged=%d/10" % (n, len(notes)))
    lines.append("MATCHED ADDITIONS: delta_R2 repeat0, repeat1")
    paired = {}
    for v in result["ablations"]:
        paired.setdefault((v["model"], v["comparator"]), {})[v["repeat"]] = v["delta_R2"]
    for (n, base), values in paired.items():
        lines.append("%s vs %s: %+.6f %+.6f" % (n, base, values[0], values[1]))
    lines.append("Native units retained; conditional intervals exclude adaptive-search uncertainty.")
    return "\n".join(lines) + "\n"


def show(run, all_models=False, progress=False, fit_notes=False):
    registration = run / "registration.json"
    if not registration.is_file():
        print("NOT_REGISTERED: preflight, loading or hashing may still be running.")
        return
    config = json.loads(registration.read_text())["config"]
    if progress:
        receipts = list((run / "TRE_ONLY").glob("*/done_*.json"))
        count = sum(p.parent.name.endswith("_result") for p in receipts)
        print(
            "RUN=%s receipts=%d outer=%d/10 lock=%s"
            % (run.name, len(receipts), count, "PRESENT" if (run / "RUNNING.lock").exists() else "ABSENT")
        )
        return
    if not (run / "summary.json").is_file():
        print("INCOMPLETE: pooled R2 requires all 10 folds and the final integrity audit.")
        print("Use --progress for checkpoint counts; inspect the running terminal for current stage.")
        return
    result = aggregate(config, completed_results(run, config))
    if result != json.loads((run / "summary.json").read_text()):
        raise ValueError("Summary differs from checksum-verified completed units")
    print("RUN=%s STATUS=COMPLETE" % run.name)
    print(diagnostics(result) if fit_notes else compact(result, config, all_models), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="adaptive_v9_20261108")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--all", action="store_true", dest="all_models")
    modes.add_argument("--progress", action="store_true")
    modes.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    from runtime import valid_id

    valid_id(args.run_id)
    show(ROOT / "runs" / args.run_id, args.all_models, args.progress, args.diagnostics)
