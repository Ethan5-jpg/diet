"""Fixed-seed synthetic calibration of the finite-basis score-inference core.

It does not validate every possible learned-PCA approximation on native data.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats
from threadpoolctl import threadpool_limits

from features import partial_out
from inference import partial_inference
from synthetic import folds


def one(seed, interaction=False, nonlinear=True, omitted_background=False):
    rng = np.random.default_rng(seed)
    n = 1200
    groups = np.arange(n) // 2
    background = rng.normal(size=n)
    markers = rng.normal(size=(n, 4))
    markers[:, 0] += .6 * background
    diet = .6 * markers[:, 0] + .5 * background + rng.normal(size=n)
    main = np.column_stack([np.ones(n), diet, diet ** 2, diet ** 3, markers, markers ** 2,
                            background, background ** 2, diet * background])
    target = diet[:, None] * markers
    theta = np.array([.8, -.7, 0., 0.]) if interaction else np.zeros(4)
    baseline = 4 * markers[:, 0] + 3 * diet + 2 * background
    if nonlinear:
        baseline += 2 * markers[:, 1] ** 2 + 1.2 * diet ** 2 + .8 * diet * background
    noise = 2 * rng.normal(size=n // 2)[groups] + 2 * rng.normal(size=n)
    y = baseline + target @ theta + noise
    if omitted_background:
        main = main[:, :-3]  # Deliberately remove background protection; assumption stress case.
    assignment = folds(groups.astype(str), 5, seed + 10000)
    ry, rt, blocks = np.zeros(n), np.zeros((n, 4)), []
    for fold in range(5):
        train, valid = assignment != fold, assignment == fold
        residual, q, _ = partial_out(main[valid], np.column_stack([y[valid], target[valid]]))
        ry[valid], rt[valid] = residual[:, 0], residual[:, 1:]
        blocks.append(dict(rows=np.flatnonzero(valid), q=q))
    info = partial_inference(ry, rt, groups, np.ones(4), draws=399, seed=seed + 20000, main_blocks=blocks)
    if info["status"] != "OK":
        return dict(available=False, reason=info["reason"])
    covered = [row["coefficient_ci"][0] <= value <= row["coefficient_ci"][1] for row, value in zip(info["markers"], theta)]
    return dict(available=True, reject=info["omnibus"]["p_primary"] < .05,
                diagnostic_joint_reject=info["omnibus"]["p_multiplier"] < .05, coverage=covered,
                estimate=[row["coefficient"] for row in info["markers"]], theta=theta.tolist())


def run(repetitions=200):
    root = Path(__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("features.py", "inference.py", "calibrate.py", "synthetic.py")}
    scenarios = [
        ("strong_main_effects_only", False, False, False),
        ("nonlinear_correlated_main_effects_only", False, True, False),
        ("known_interaction", True, True, False),
        ("omitted_background_assumption_stress", False, True, True),
    ]
    results = {}
    for j, (name, interaction, nonlinear, omitted) in enumerate(scenarios):
        values = [one(20261112 + 100000 * j + i, interaction, nonlinear, omitted) for i in range(repetitions)]
        available = [v for v in values if v["available"]]
        count = sum(v["reject"] for v in available)
        ci = stats.binomtest(count, len(available)).proportion_ci(method="wilson")
        results[name] = dict(repetitions=repetitions, available=len(available), rejection_rate=count / len(available),
                             unavailable=repetitions-len(available),
                             unavailable_reasons=sorted({v["reason"] for v in values if not v["available"]}),
                             rejection_rate_mc95=[ci.low, ci.high], coverage=np.mean([v["coverage"] for v in available], axis=0).tolist(),
                             mean_estimate=np.mean([v["estimate"] for v in available], axis=0).tolist(),
                             diagnostic_joint_rejection_rate=float(np.mean([v["diagnostic_joint_reject"] for v in available])),
                             true_theta=available[0]["theta"], deliberately_misspecified=omitted)
        print("SYNTHETIC %s reps=%d rejection=%.3f MC95=[%.3f,%.3f]" % (name, repetitions, count / len(available), ci.low, ci.high), flush=True)
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest for name, digest in source_hashes.items()):
        raise RuntimeError("Calibration source changed during execution")
    return dict(
        evidence_source="SYNTHETIC_CALIBRATION_ONLY", seed=20261112, n_per_replication=1200, household_size=2,
        score_draws=399, scenarios=results,
        limitations="Fixed finite-basis simulation, not a proof for all nuisance errors, learned PCs, heavy tails or historical discovery selection. Omitted-background case deliberately violates approximation assumptions.",
        native_participant_inputs_read=False,
        source_sha256=source_hashes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--output", default="research_records/verification/calibration.json")
    args = parser.parse_args()
    if args.repetitions < 20:
        parser.error("Use at least 20 replications")
    with threadpool_limits(limits=1):
        value = run(args.repetitions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
