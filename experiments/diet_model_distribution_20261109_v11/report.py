"""Compact ASCII aggregate reports; no individual records leave TRE_ONLY."""

import argparse
import json
from pathlib import Path

import numpy as np

from bridge import Memo, metric
from registry import CORRECTIONS, INTERACTIONS, REFERENCES, additions, names, specifications

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
    paired = additions()
    population = None
    for repeat in range(config["outer_repeats"]):
        blocks = [completed[repeat, f] for f in range(config["outer_folds"])]
        rows = np.concatenate([b["rows"] for b in blocks])
        group_sets = [set(b["groups"]) for b in blocks]
        if len(set(rows)) != len(rows) or any(
            group_sets[i] & group_sets[j] for i in range(len(blocks)) for j in range(i)
        ):
            raise ValueError("Duplicate OOF rows or household leakage")
        current = np.sort(rows)
        if population is not None and not np.array_equal(current, population):
            raise ValueError("Evaluation populations differ between repeats")
        population = current
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
            base = INTERACTIONS.get(name, CORRECTIONS.get(name, "v10_ridge_diverse"))
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
                    type="explicit_interaction"
                    if name in INTERACTIONS
                    else ("predictive_residual_correction" if name in CORRECTIONS else "total_prediction"),
                    **gain,
                )
            )
        microbial_pairs = [
            ("hgb_oral_joint", "hgb_host_joint"),
            ("cat_balanced_joint", "cat_host_joint"),
            ("cat_balanced_joint", "cat_oral_joint"),
        ]
        for name, base in list(paired.items()) + microbial_pairs:
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
        v10_replay_max_error=max(b["v10_replay_max_error"] for b in completed.values()),
        input_blocks={s["name"]: s.get("context", "matched_null_context") for s in specifications(config)},
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
    flags = {v["model"] for v in result["fit_quality"]}

    def value(n):
        return [score[n, r]["R2"] for r in range(2)]

    ranked = sorted(names(config), key=lambda n: min(value(n)), reverse=True)
    shown = (
        ranked
        if all_models
        else list(
            dict.fromkeys(ranked[:5] + ["ridge_new", "positive_new", "ridge_combined", "sparse_combined"])
        )
    )
    lines = [
        "V11 COMPLETE | fiber_density -> D_Glu (mg/dL) | 10/10 household outer folds",
        "N0=%d N1=%d | %d new pipelines; repeats share participants"
        % (score["v8_ridge_wide", 0]["N"], score["v8_ridge_wide", 1]["N"], len(names(config))),
        "%-30s %9s %9s %10s %10s %4s" % ("model", "R2_0", "R2_1", "dV10_0", "dV10_1", "flag"),
    ]
    baseline = value("v10_ridge_diverse")
    for n in list(REFERENCES) + shown:
        a, b = value(n)
        lines.append(
            "%-30s %9.6f %9.6f %+10.6f %+10.6f %4s"
            % (n, a, b, a - baseline[0], b - baseline[1], "W" if n in flags else "-")
        )
    lines += [
        "",
        "EXPLICIT: delta_R2 vs SAME selected background; SD of added 10->15 fiber contrast (mg/dL)",
        "%-24s %10s %10s %9s %8s %8s" % ("model", "dR2_0", "dR2_1", "active0/1", "SD0", "SD1"),
    ]
    for n in INTERACTIONS:
        lines.append(
            "%-24s %+10.6f %+10.6f %4d/%-4d %8.3f %8.3f"
            % (
                n,
                gains[n, 0]["delta_R2"],
                gains[n, 1]["delta_R2"],
                mods[n, 0]["active_folds"],
                mods[n, 1]["active_folds"],
                mods[n, 0]["SD_mg_dL"],
                mods[n, 1]["SD_mg_dL"],
            )
        )
    beats = sum(all(score[n, r]["R2"] > baseline[r] for r in range(2)) for n in names(config))
    lines += [
        "BEATS_V10_BOTH=%d/%d; fit-flag new pipelines=%d"
        % (beats, len(names(config)), len(set(names(config)) & flags))
    ]
    for threshold in (0.2, 0.3):
        passed = [n for n in names(config) if min(value(n)) >= threshold]
        lines.append(
            "Both repeats R2>=%.2f: %d/%d; without fit flags: %d"
            % (threshold, len(passed), len(names(config)), len(set(passed) - flags))
        )
    lines += [
        "V10 bank replay max error=%.3g; prior matched INNER caches used for combined fusion."
        % result["v10_replay_max_error"],
        "Prediction gains do not establish microbial effect modification; discovery exploration only.",
        "No confirmation data. --diagnostics: input profiles, microbial gains and matched nulls.",
    ]
    return "\n".join(lines) + "\n"


def diagnostics(result):
    scores = {(v["model"], v["repeat"]): v["R2"] for v in result["scores"]}
    paired = {(v["model"], v["comparator"], v["repeat"]): v for v in result["ablations"]}
    lines = ["MATCHED INPUT PROFILES: delta_R2 repeat0 repeat1 (same people, algorithm family and grid)"]
    targets = [n for n in additions() if n.startswith(("hgb_oral_", "cat_balanced_"))]
    for n in targets:
        base = additions()[n]
        lines.append(
            "%s vs %s: %+.6f %+.6f"
            % (n, base, paired[n, base, 0]["delta_R2"], paired[n, base, 1]["delta_R2"])
        )
    lines.append("MATCHED MICROBIAL INPUTS / POOLS: delta_R2 repeat0 repeat1")
    for n, base in (
        ("hgb_oral_joint", "hgb_host_joint"),
        ("cat_balanced_joint", "cat_host_joint"),
        ("cat_balanced_joint", "cat_oral_joint"),
        ("convex_balanced", "convex_oral"),
        ("ridge_balanced", "ridge_oral"),
    ):
        lines.append(
            "%s vs %s: %+.6f %+.6f"
            % (n, base, paired[n, base, 0]["delta_R2"], paired[n, base, 1]["delta_R2"])
        )
    lines.append(
        "PREDICTIVE RESIDUAL CORRECTIONS: delta_R2 vs its SAME selected base (not effect modification)"
    )
    for n, base in CORRECTIONS.items():
        lines.append(
            "%s vs %s: %+.6f %+.6f"
            % (n, base, scores[n, 0] - scores[base, 0], scores[n, 1] - scores[base, 1])
        )
    lines.append("EXPLICIT MATCHED NULLS: null R2_0/R2_1 -> fitted R2_0/R2_1")
    for n, base in INTERACTIONS.items():
        lines.append(
            "%s vs %s: %.6f/%.6f -> %.6f/%.6f"
            % (n, base, scores[base, 0], scores[base, 1], scores[n, 0], scores[n, 1])
        )
    flags = sorted({v["model"] for v in result["fit_quality"]})
    lines.append("FIT_FLAGS=%d; full notes in summary.json" % len(flags))
    lines.append("Conditional household intervals in JSON exclude adaptive-search uncertainty.")
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
        print("INCOMPLETE: pooled R2 requires all 10 folds and final integrity audit.")
        print("Use --progress for checkpoint counts; inspect the running terminal for the current stage.")
        return
    result = aggregate(config, completed_results(run, config))
    if result != json.loads((run / "summary.json").read_text()):
        raise ValueError("Summary differs from checksum-verified completed units")
    print("RUN=%s STATUS=COMPLETE" % run.name)
    print(diagnostics(result) if fit_notes else compact(result, config, all_models), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="distribution_v11_20261109")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--all", action="store_true", dest="all_models")
    modes.add_argument("--progress", action="store_true")
    modes.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    from runtime import valid_id

    valid_id(args.run_id)
    show(ROOT / "runs" / args.run_id, args.all_models, args.progress, args.diagnostics)
