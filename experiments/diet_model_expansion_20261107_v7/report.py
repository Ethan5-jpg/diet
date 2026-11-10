"""English aggregate-only reports; individual records remain inside TRE_ONLY."""

import argparse
import json
from pathlib import Path

import numpy as np

from bridge import Memo, metric
from registry import INTERACTIONS, META, REFERENCES, names

ROOT = Path(__file__).resolve().parent


def aggregate(config, completed):
    total = config["outer_repeats"] * config["outer_folds"]
    if len(completed) != total:
        raise ValueError("Partial batches have no pooled report")
    scores, gains, modifiers, selections, quality = [], [], [], [], []
    for repeat in range(config["outer_repeats"]):
        blocks = [completed[(repeat, fold)] for fold in range(config["outer_folds"])]
        rows = np.concatenate([b["rows"] for b in blocks])
        groups = [set(b["groups"]) for b in blocks]
        if len(set(rows)) != len(rows) or any(
            groups[i] & groups[j] for i in range(len(groups)) for j in range(i)
        ):
            raise ValueError("Duplicated OOF row or household leakage")
        y = np.concatenate([b["y"] for b in blocks])
        prediction = {
            name: np.concatenate([b["predictions"][name] for b in blocks])
            for name in names(config) + list(REFERENCES)
        }
        if float(np.var(y)) <= 0:
            raise ValueError("Constant pooled outcome")
        for name, p in prediction.items():
            scores.append(dict(model=name, repeat=repeat, N=len(y), **metric(y, p)))
        for name in names(config):
            comparator = INTERACTIONS.get(name, "v6_fusion")
            changes = [
                float(
                    np.mean(
                        (b["y"] - b["predictions"][comparator]) ** 2
                        - (b["y"] - b["predictions"][name]) ** 2
                    )
                )
                for b in blocks
            ]
            gain = float(np.mean((y - prediction[comparator]) ** 2 - (y - prediction[name]) ** 2))
            gains.append(
                dict(
                    model=name,
                    repeat=repeat,
                    comparator=comparator,
                    delta_R2=gain / float(np.var(y)),
                    improved_folds=sum(g > 0 for g in changes),
                    fold_MSE_gains=changes,
                    type="explicit_interaction" if name in INTERACTIONS else "total_prediction",
                )
            )
        for name in INTERACTIONS:
            axis = np.concatenate([b["modifiers"][name] for b in blocks])
            modifiers.append(
                dict(
                    model=name,
                    repeat=repeat,
                    SD_mg_dL=float(axis.std()),
                    active_folds=sum(b["active"][name] for b in blocks),
                )
            )
        selections.extend(
            dict(
                repeat=repeat,
                fold=fold,
                model=b["inner_selector"],
                weights=b["blend_weights"],
                selected=b["selected"],
            )
            for fold, b in enumerate(blocks)
        )
        quality.extend(
            dict(repeat=repeat, fold=fold, model=name, **note)
            for fold, block in enumerate(blocks)
            for name, note in block["outer_diagnostics"].items()
            if note.get("warnings") or note.get("converged") is False
        )
    return dict(
        status="COMPLETE",
        scores=scores,
        comparisons=gains,
        modifiers=modifiers,
        selections=selections,
        fit_quality=quality,
        contrast=config["contrast"],
        confirmation_outcomes_read=False,
        limitations="Discovery development; repeats share participants; outer ranking is exploratory.",
    )


def completed_results(run, config):
    if not (run / "TRE_ONLY").is_dir():
        return {}
    memo, completed = Memo(run / "TRE_ONLY"), {}
    for r in range(config["outer_repeats"]):
        for f in range(config["outer_folds"]):
            folder = memo.root / ("r%d_f%d_result" % (r, f))
            if folder.is_dir():
                try:
                    completed[(r, f)] = memo.get(folder.name)
                except FileNotFoundError:
                    pass
    return completed


def compact(result, config, all_models=False):
    score = {(v["model"], v["repeat"]): v for v in result["scores"]}
    changes = {(v["model"], v["repeat"]): v for v in result["comparisons"]}
    mods = {(v["model"], v["repeat"]): v for v in result["modifiers"]}

    def value(name):
        return [score[name, r]["R2"] for r in range(2)]

    ranked = sorted(names(config), key=lambda n: min(value(n)), reverse=True)
    shown = (
        ranked
        if all_models
        else list(dict.fromkeys(ranked[:5] + list(META) + list(INTERACTIONS.values())))
    )
    flagged = {v["model"] for v in result["fit_quality"]}
    lines = [
        "V7 COMPLETE | fiber_density -> D_Glu (mg/dL) | 10/10 outer folds",
        "N0=%d N1=%d | %d new pipelines | household-grouped CV"
        % (score["v6_fusion", 0]["N"], score["v6_fusion", 1]["N"], len(names(config))),
        "%-28s %10s %10s %10s %10s %4s" % ("model", "R2_0", "R2_1", "dV6_0", "dV6_1", "flag"),
    ]
    for name in list(REFERENCES) + shown:
        a, b = value(name)
        baseline = value("v6_fusion")
        lines.append(
            "%-28s %10.6f %10.6f %+10.6f %+10.6f %4s"
            % (name, a, b, a - baseline[0], b - baseline[1], "W" if name in flagged else "-")
        )
    lines.extend(
        [
            "",
            "EXPLICIT INTERACTION: delta vs matched null (not vs V6)",
            "%-28s %10s %10s %7s %7s %9s %9s"
            % ("model", "dR2_0", "dR2_1", "wins0", "wins1", "SD0", "SD1"),
        ]
    )
    for name, base in INTERACTIONS.items():
        a, b = (changes[name, r] for r in range(2))
        lines.append(
            "%-28s %+10.6f %+10.6f %5d/5 %5d/5 %9.4f %9.4f"
            % (
                name,
                a["delta_R2"],
                b["delta_R2"],
                a["improved_folds"],
                b["improved_folds"],
                mods[name, 0]["SD_mg_dL"],
                mods[name, 1]["SD_mg_dL"],
            )
        )
        lines.append(
            "  null=%s; active=%d/5,%d/5"
            % (base, mods[name, 0]["active_folds"], mods[name, 1]["active_folds"])
        )
    thresholds = [
        sum(min(value(n)) >= threshold for n in names(config)) for threshold in (0.2, 0.3)
    ]
    lines.extend(
        [
            "",
            "Same pipeline passes both repeats: R2>=0.20 %d/%d; R2>=0.30 %d/%d"
            % (thresholds[0], len(names(config)), thresholds[1], len(names(config))),
            "W=fit warning or FM iteration limit; full details retained in summary.json.",
            "Ranking is exploratory. Repeats share participants. No confirmation data read.",
            "Full results: summary.json; private checkpoints/models: TRE_ONLY/.",
        ]
    )
    return "\n".join(lines) + "\n"


def show(run, all_models=False, progress=False):
    registration = run / "registration.json"
    if not registration.is_file():
        print("NOT_REGISTERED: preflight, input loading or hashing may still be running.")
        return
    config = json.loads(registration.read_text())["config"]
    if progress:
        folders = list((run / "TRE_ONLY").glob("*/done_*.json"))
        outer = sum(p.parent.name.endswith("_result") for p in folders)
        print(
            "RUN=%s receipt_files=%d outer_receipts=%d/10 lock=%s"
            % (
                run.name,
                len(folders),
                outer,
                "PRESENT" if (run / "RUNNING.lock").exists() else "ABSENT",
            )
        )
        return
    summary = run / "summary.json"
    if not summary.is_file():
        print("INCOMPLETE: no pooled R2 until all 10 folds and integrity audit finish.")
        print("Use --progress or attach the tmux session for current activity.")
        return
    # Recheck checkpoint hashes instead of trusting a stale aggregate file.
    result = aggregate(config, completed_results(run, config))
    if result != json.loads(summary.read_text()):
        raise ValueError("Summary does not match completed checkpoint records")
    print("RUN=%s STATUS=COMPLETE" % run.name)
    print(compact(result, config, all_models), end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="main_v7_20261107")
    parser.add_argument("--all", action="store_true", dest="all_models")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    from runtime import valid_id

    valid_id(args.run_id)
    show(ROOT / "runs" / args.run_id, args.all_models, args.progress)
