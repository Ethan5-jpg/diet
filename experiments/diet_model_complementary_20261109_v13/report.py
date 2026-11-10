"""Compact screenshot reports from native closed checkpoints; no fitting."""

import argparse
import json
from pathlib import Path
import numpy as np
from bridge import Memo, metric
from registry import REFERENCES, additions, names

ROOT = Path(__file__).resolve().parent


def warning(note):
    return bool(
        note.get("warnings") or note.get("selected_inner_warning") or note.get("contributing_fit_warnings")
    )


def comparison(y, p, base, groups):
    delta = (y - base) ** 2 - (y - p) ** 2
    _, g = np.unique(groups, return_inverse=True)
    total = np.bincount(g, weights=delta)
    size = np.bincount(g)
    count = len(total)
    mean = float(delta.mean())
    variance = float(np.var(y))
    se = float(np.sqrt(count / (count - 1) * np.sum((total - size * mean) ** 2)) / len(y))
    return dict(
        delta_R2=mean / variance,
        conditional_household_interval=[(mean - 1.96 * se) / variance, (mean + 1.96 * se) / variance],
    )


def aggregate(config, completed):
    if len(completed) != 10:
        raise ValueError("Partial batches have no pooled report")
    scores, gains, ablations, quality, selections = [], [], [], [], []
    population = None
    for repeat in range(2):
        blocks = [completed[repeat, f] for f in range(5)]
        rows = np.concatenate([b["rows"] for b in blocks])
        sets = [set(b["groups"]) for b in blocks]
        if len(set(rows)) != len(rows) or any(sets[i] & sets[j] for i in range(5) for j in range(i)):
            raise ValueError("Household or row leakage")
        current = np.sort(rows)
        if population is not None and not np.array_equal(current, population):
            raise ValueError("Repeat populations differ")
        population = current
        y = np.concatenate([b["y"] for b in blocks])
        groups = np.concatenate([b["groups"] for b in blocks])
        if not np.isfinite(y).all() or np.var(y) <= 0 or len(set(groups)) < 2:
            raise ValueError("Invalid pooled target")
        pred = {
            n: np.concatenate([b["predictions"][n] for b in blocks]) for n in names(config) + list(REFERENCES)
        }
        for n, p in pred.items():
            if p.shape != y.shape or not np.isfinite(p).all():
                raise ValueError("Invalid pooled prediction")
            scores.append(dict(model=n, repeat=repeat, N=len(y), **metric(y, p)))
        for n in names(config):
            base = "v11_positive_combined"
            wins = sum(
                np.mean((b["y"] - b["predictions"][n]) ** 2) < np.mean((b["y"] - b["predictions"][base]) ** 2)
                for b in blocks
            )
            gains.append(
                dict(
                    model=n,
                    repeat=repeat,
                    comparator=base,
                    improved_folds=int(wins),
                    **comparison(y, pred[n], pred[base], groups),
                )
            )
        for n, base in additions(config).items():
            ablations.append(
                dict(model=n, comparator=base, repeat=repeat, **comparison(y, pred[n], pred[base], groups))
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
        fit_quality=quality,
        selections=selections,
        v11_replay_max_error=max(b["v11_replay_max_error"] for b in completed.values()),
        confirmation_outcomes_read=False,
        limitations="Exploratory prediction; repeats share people. Conditional intervals exclude adaptive-search uncertainty; this is not V12 effect-modification inference.",
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
    scores = {(v["model"], v["repeat"]): v for v in result["scores"]}
    flags = {v["model"] for v in result["fit_quality"]}

    def values(n):
        return [scores[n, r]["R2"] for r in range(2)]

    ranked = sorted(names(config), key=lambda n: min(values(n)), reverse=True)
    shown = (
        ranked
        if all_models
        else list(
            dict.fromkeys(
                ranked[:5] + ["positive_new", "positive_combined", "ridge_combined", "sparse_combined"]
            )
        )
    )
    lines = [
        "V13 COMPLETE | fiber_density -> D_Glu (mg/dL) | 10/10 household outer folds",
        "N0=%d N1=%d | %d new workflows; repeats share participants"
        % (
            scores["v11_positive_combined", 0]["N"],
            scores["v11_positive_combined", 1]["N"],
            len(names(config)),
        ),
        "%-32s %9s %9s %10s %10s %4s" % ("model", "R2_0", "R2_1", "dV11_0", "dV11_1", "flag"),
    ]
    base = values("v11_positive_combined")
    for n in list(REFERENCES) + shown:
        a, b = values(n)
        lines.append(
            "%-32s %9.6f %9.6f %+10.6f %+10.6f %4s"
            % (n, a, b, a - base[0], b - base[1], "W" if n in flags else "-")
        )
    beats = sum(all(values(n)[r] > base[r] for r in range(2)) for n in names(config))
    lines.append(
        "BEATS_V11_BOTH=%d/%d; fit-flag new workflows=%d"
        % (beats, len(names(config)), len(set(names(config)) & flags))
    )
    for threshold in (0.2, 0.3):
        passed = [n for n in names(config) if min(values(n)) >= threshold]
        lines.append(
            "Both repeats R2>=%.2f: %d/%d; without fit flags: %d"
            % (threshold, len(passed), len(names(config)), len(set(passed) - flags))
        )
    lines += [
        "V11 positive-fusion replay max error=%.3g" % result["v11_replay_max_error"],
        "Supplementary prediction only; does not establish microbial effect modification.",
        "No confirmation data. --diagnostics: matched representations/objectives/pools.",
    ]
    return "\n".join(lines) + "\n"


def diagnostics(result):
    lines = ["MATCHED NEW REPRESENTATIONS / OBJECTIVES: delta_R2 repeat0 repeat1"]
    pairs = {}
    for v in result["ablations"]:
        pairs.setdefault((v["model"], v["comparator"]), {})[v["repeat"]] = v["delta_R2"]
    for (n, base), values in pairs.items():
        lines.append("%s vs %s: %+.6f %+.6f" % (n, base, values[0], values[1]))
    lines += [
        "Positive/negative increments are retained. Conditional intervals are in summary.json.",
        "Supervised directions are predictive; no pathway or causal interpretation.",
    ]
    return "\n".join(lines) + "\n"


def show(run, all_models=False, progress=False, fit_notes=False):
    registration = run / "registration.json"
    if not registration.is_file():
        print("NOT_REGISTERED: preflight, loading or hashing may still be running.")
        return
    config = json.loads(registration.read_text())["config"]
    if progress:
        receipts = list((run / "TRE_ONLY").glob("*/done_*.json"))
        count = len(completed_results(run, config))
        print(
            "RUN=%s receipt_files=%d verified_outer=%d/10 lock=%s"
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
    parser.add_argument("--run-id", default="complementary_v13_20261109")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--all", action="store_true", dest="all_models")
    modes.add_argument("--progress", action="store_true")
    modes.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    from runtime import valid_id

    valid_id(args.run_id)
    show(ROOT / "runs" / args.run_id, args.all_models, args.progress, args.diagnostics)
