"""Run or report the complete, discovery-only method batch."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from data import load
from experiment import INTERACTIONS, REFERENCES, Engine, model_names
from numerics import metric
from reporting import compact_summary
from storage import Memo, sha, write_json

ROOT = Path(__file__).resolve().parent


def aggregate(config, completed):
    names = model_names(config) + list(REFERENCES)
    scores, increments, modifiers = [], [], []
    for repeat in range(config["outer_repeats"]):
        blocks = [completed[(repeat, f)] for f in range(config["outer_folds"])]
        rows = np.concatenate([block["rows"] for block in blocks])
        if len(set(rows)) != len(rows):
            raise ValueError("Duplicate out-of-fold rows")
        y = np.concatenate([block["y"] for block in blocks])
        pred = {
            name: np.concatenate([block["predictions"][name] for block in blocks])
            for name in names
        }
        for name in names:
            scores.append(dict(repeat=repeat, model=name, **metric(y, pred[name])))
        for name in names[: -len(REFERENCES)]:
            comparator = "kernel_main" if name in INTERACTIONS else "v4_hgb_oral"
            gains = [
                float(
                    np.mean(
                        (b["y"] - b["predictions"][comparator]) ** 2
                        - (b["y"] - b["predictions"][name]) ** 2
                    )
                )
                for b in blocks
            ]
            mse_gain = float(
                np.mean((y - pred[comparator]) ** 2 - (y - pred[name]) ** 2)
            )
            increments.append(
                dict(
                    repeat=repeat,
                    model=name,
                    comparator=comparator,
                    delta_R2=mse_gain / float(np.var(y)),
                    improved_folds=sum(g > 0 for g in gains),
                    fold_MSE_gains=gains,
                    comparison="explicit_interaction"
                    if name in INTERACTIONS
                    else "total_prediction",
                )
            )
        for name in INTERACTIONS:
            axis = np.concatenate([b["modifiers"][name] for b in blocks])
            modifiers.append(
                dict(
                    repeat=repeat,
                    model=name,
                    contrast=config["contrast"],
                    SD_mg_dL=float(axis.std()),
                    nonzero_folds=sum(b["effective_weights"][name] > 0 for b in blocks),
                )
            )
    return dict(
        status="COMPLETE",
        scores=scores,
        comparisons=increments,
        modifiers=modifiers,
        confirmation_outcomes_read=False,
    )


def results(run, config):
    memo, completed = Memo(run / "TRE_ONLY"), {}
    for repeat in range(config["outer_repeats"]):
        for fold in range(config["outer_folds"]):
            try:
                completed[(repeat, fold)] = memo.get("r%d_f%d_result" % (repeat, fold))
            except FileNotFoundError:
                pass
    return completed


def report(run):
    if not (run / "registration.json").is_file():
        print(
            "No registered run yet; input loading may still be in progress. Check the run log."
        )
        return None
    config = json.loads((run / "registration.json").read_text())["config"]
    complete = results(run, config)
    total = config["outer_repeats"] * config["outer_folds"]
    print("Completed outer folds: %d/%d" % (len(complete), total), flush=True)
    if len(complete) != total or not (run / "summary.json").is_file():
        print("Run incomplete or final integrity check pending; no pooled R2 reported.")
        return None
    result = aggregate(config, complete)
    print(compact_summary(result, config), end="")
    return result


def execute(args, loader=load):
    package = ROOT.parent.parent
    if (
        package.name != "collaborator_handoff_20261102_v1"
        or ROOT.parent.name != "experiments"
    ):
        raise ValueError("Upload this folder under the handoff experiments/ directory")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run_id):
        raise ValueError("Invalid run-id")
    run = ROOT / "runs" / args.run_id
    if args.summary:
        return report(run)
    config = json.loads((ROOT / "config.json").read_text())
    if (
        config["confirmation_outcomes_read"]
        or config["outcome"] != "D_Glu"
        or config["exposure"] != "fiber_density"
        or config["outer_seed"] != 20261104
        or config["outer_repeats"] != 2
        or config["outer_folds"] != 5
        or config["inner_folds"] != 3
    ):
        raise ValueError("Unexpected discovery configuration")
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "scipy",
            "pandas",
            "pyarrow",
            "scikit-learn",
            "catboost",
            "xgboost",
        )
    }
    for name, version in config["dependencies"].items():
        if versions[name] != version:
            raise ValueError("Install the pinned requirements first: " + name)
        importlib.import_module(name)
    print("NEW MODEL BATCH: six method groups; discovery data only.", flush=True)
    dataset = loader(package)
    sources = [
        ROOT / name
        for name in (
            "config.json",
            "data.py",
            "experiment.py",
            "learners.py",
            "numerics.py",
            "run.py",
            "reporting.py",
            "storage.py",
        )
    ]
    identity = dict(
        config=config,
        hashes={str(path): sha(path) for path in dataset.paths + sources},
        versions=versions,
        python=sys.version,
        threads=args.threads,
        prior_discovery_selection=True,
    )
    if args.resume:
        if json.loads((run / "registration.json").read_text()) != identity:
            raise ValueError("Code, inputs or environment changed; use a new run-id")
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "registration.json", identity)
    lock = run / "RUNNING.lock"
    with lock.open("x") as stream:
        stream.write("pid=%d\n" % os.getpid())
    memo = Memo(run / "TRE_ONLY", lambda message: print(message, flush=True))
    try:
        count = 0
        for repeat in range(config["outer_repeats"]):
            for fold in range(config["outer_folds"]):
                engine = Engine(config, memo, args.threads)
                before = memo.created
                memo.get(
                    "r%d_f%d_result" % (repeat, fold),
                    lambda: engine.outer(dataset, repeat, fold),
                )
                print("OUTER_COMPLETE r%d f%d" % (repeat, fold), flush=True)
                count += memo.created > before
                if args.limit_outer and count >= args.limit_outer:
                    return None
        if any(sha(path) != digest for path, digest in identity["hashes"].items()):
            raise ValueError("Input/source changed during execution")
        result = aggregate(config, results(run, config))
        if not (run / "summary.json").exists():
            write_json(run / "summary.json", result)
        if not (run / "SUMMARY.md").exists():
            with (run / "SUMMARY.md").open("x", encoding="utf-8") as stream:
                stream.write(compact_summary(result, config))
        report(run)
        print("BATCH_COMPLETE", flush=True)
        return result
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="main_20261107")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--limit-outer", type=int, default=0)
    args = parser.parse_args()
    if args.threads < 1 or args.limit_outer < 0:
        parser.error("Positive threads and nonnegative limit-outer required")
    with threadpool_limits(limits=args.threads):
        execute(args)
