cd /home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_adaptive_20261108_v9 &&
/home/ec2-user/.venvs/hpp_v8_20261108/bin/python - <<'PY'
"""Read recorded aggregate diagnostics only; no model fitting or private rows."""

import json
from pathlib import Path

from registry import INTERACTIONS, META, REFERENCES

result = json.loads(Path("runs/adaptive_v9_20261108/summary.json").read_text())
if result.get("status") != "COMPLETE":
    raise SystemExit("The recorded batch is not complete")
scores = {(v["model"], v["repeat"]): v["R2"] for v in result["scores"]}
models = sorted({n for n, r in scores} - set(REFERENCES))
flags = {v["model"] for v in result["fit_quality"]}
daily = set(result["daily_input_models"]) | {"convex_daily", "ridge_daily"}
better = sum(all(scores[n, r] > scores["v8_ridge_wide", r] for r in (0, 1)) for n in models)
print("BEATS_V8_BOTH=%d/%d; FIT_FLAG_PIPELINES=%d" % (better, len(models), len(flags)))
atomic = [n for n in models if n not in META and n not in INTERACTIONS]
print("TOP ATOMIC PIPELINES (ranked by lower repeat): model R2_0 R2_1 flag")
for name in sorted(atomic, key=lambda n: min(scores[n, 0], scores[n, 1]), reverse=True)[:6]:
    flag = ("D" if name in daily else "") + ("W" if name in flags else "")
    print("%-26s %.6f %.6f %s" % (name, scores[name, 0], scores[name, 1], flag or "-"))
gains = {(v["model"], v["repeat"]): v for v in result["comparisons"]}
print("GAINS VS V8: conditional household intervals; wins out of 5")
for name in ("convex_wide", "ridge_wide", "ridge_core"):
    for r in (0, 1):
        item = gains[name, r]
        low, high = item["conditional_household_interval"]
        print("%s r%d d=%+.6f CI=[%+.6f,%+.6f] wins=%d/5" % (name, r, item["delta_R2"], low, high, item["improved_folds"]))
paired = {(v["model"], v["repeat"]): v for v in result["ablations"]}
print("MATCHED ADDITIONS: delta_R2 repeat0 repeat1")
for name in ("daily_cat_oral", "daily_cat_balanced", "gated_soft", "gated_medium", "gated_strong"):
    a, b = paired[name, 0], paired[name, 1]
    print("%s vs %s: %+.6f %+.6f" % (name, a["comparator"], a["delta_R2"], b["delta_R2"]))
print("D=extra FOOD inputs; W=fit flag. Intervals exclude adaptive-search uncertainty.")
PY
