"""Read-only scalar diagnostics for the stopped V13 r0_f3 replay check."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))

import numpy as np
from threadpoolctl import threadpool_limits
from bridge import Memo


def diagnose(base):
    base = Path(base)
    run = base / "runs/complementary_v13_20261109"
    reg = json.loads((run / "registration.json").read_text())
    oldrun = base.parent / "diet_model_distribution_20261109_v11/runs/distribution_v11_20261109"
    oldreg = json.loads((oldrun / "registration.json").read_text())
    v10run = base.parent / "diet_model_context_20261108_v10/runs" / oldreg["config"]["v10_reference_run"]
    for path in (run, oldrun, v10run):
        if not (path / "TRE_ONLY").is_dir():
            raise FileNotFoundError("Required existing cache: " + str(path))
    current, previous, older = [Memo(p / "TRE_ONLY") for p in (run, oldrun, v10run)]
    print("V13_REPLAY_DIAG: read-only; no fits, file changes or individual values")
    print(
        "THREADS V13=%s V11=%s; V13_lock=%s"
        % (reg.get("threads"), oldreg.get("threads"), (run / "RUNNING.lock").exists())
    )
    finished = []
    for r in range(2):
        for f in range(5):
            key = "r%d_f%d_result" % (r, f)
            if (current.root / key).is_dir():
                try:
                    value = current.get(key)
                except FileNotFoundError:
                    continue
                finished.append(value["v11_replay_max_error"])
    print("COMPLETED_OUTER=%d/10 previous_max_error=%.12g" % (len(finished), max(finished, default=0)))
    tag = "r0_f3"
    new = current.get(tag + "_meta_fit")["v11_replay"]
    original_prior = current.get(tag + "_prior_inner")
    prior = original_prior["replay"]
    old = previous.get(tag + "_meta_fit")["positive_combined"]
    value, v10 = previous.get(tag + "_result"), older.get(tag + "_result")
    print(
        "OUTER_ALIGNMENT=%s; INNER_CHOICES_V11=%s V10=%s"
        % (
            np.array_equal(value["rows"], v10["rows"]) and np.array_equal(value["y"], v10["y"]),
            all(value["selected"].get(n) == j for n, j in original_prior["choices"]["v11"].items()),
            all(v10["selected"].get(n) == j for n, j in original_prior["choices"]["v10"].items()),
        )
    )
    originals = {"old__" + n: v10["predictions"][n] for n in oldreg["config"]["legacy_members"]}
    originals.update(value["predictions"])
    match = new.members == ["bank__" + n for n in old.members]
    print("r0_f3 members_match=%s members=%d" % (match, len(new.members)))
    if not match:
        print("STOP: member order/name mismatch; do not relax replay tolerance")
        return
    predictions = {"bank__" + n: originals[n] for n in old.members}
    a, b = new.model, old.model

    def delta(x, y):
        return float(np.max(abs(np.asarray(x) - np.asarray(y))))

    print("PENALTY rebuilt=%s saved=%s" % (new.diagnostic.get("penalty"), old.diagnostic.get("penalty")))
    print("TARGET mean_delta=%.12g scale_delta=%.12g" % (abs(a.mean - b.mean), abs(a.scale - b.scale)))
    print(
        "INPUT center_max_delta=%.12g scale_max_delta=%.12g"
        % (delta(a.scaler.mean_, b.scaler.mean_), delta(a.scaler.scale_, b.scaler.scale_))
    )
    print(
        "FIT coef_max_delta=%.12g intercept_delta=%.12g"
        % (delta(a.model.coef_, b.model.coef_), delta(a.model.intercept_, b.model.intercept_))
    )
    target = value["predictions"]["positive_combined"]
    with threadpool_limits(limits=reg.get("threads", 2)):
        rebuilt, saved = new.predict(predictions), old.predict(originals)
        print(
            "REBUILT max_pred_error=%.12g rmse_difference=%.12g"
            % (delta(rebuilt, target), np.sqrt(np.mean((rebuilt - target) ** 2)))
        )
        print("SAVED_V11_MODEL max_pred_error=%.12g" % delta(saved, target))
        print("PRIOR_vs_META max_pred_error=%.12g" % delta(prior.predict(predictions), rebuilt))
    print("END_DIAG")


if __name__ == "__main__":
    diagnose(Path.cwd())
