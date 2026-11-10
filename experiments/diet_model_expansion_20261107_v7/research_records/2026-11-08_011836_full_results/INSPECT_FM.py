import json
from collections import Counter
from pathlib import Path
from bridge import Memo

run = Path("runs/main_v7_20261107")
s = json.loads((run / "summary.json").read_text())
cfg = json.loads((run / "registration.json").read_text())["config"]
if s["status"] != "COMPLETE" or not (run / "TRE_ONLY").is_dir():
    raise SystemExit("Completed run required")
j = cfg["blend_members"].index("factorization_balanced")
print("FM QUALITY: outer folds only")
for name in ("factorization_oral", "factorization_balanced"):
    rows = [v for v in s["fit_quality"] if v["model"] == name]
    failed = sum(v.get("converged") is False for v in rows)
    print("%s flagged=%d/10 optimizer_not_converged=%d/10" % (name, len(rows), failed))
    messages = Counter(m for v in rows for m in [v.get("message", "")] + v.get("warnings", []) if m)
    for message, count in messages.items():
        text = " ".join(str(message).split()).encode("ascii", "backslashreplace").decode()
        print("  n=%d %s" % (count, text[:160]))
memo = Memo(run / "TRE_ONLY")
for repeat in range(cfg["outer_repeats"]):
    rows = sorted((v for v in s["selections"] if v["repeat"] == repeat), key=lambda v: v["fold"])
    weights = [float(v["weights"][j]) for v in rows]
    coefs = []
    for fold in range(cfg["outer_folds"]):
        key = "r%d_f%d_result" % (repeat, fold)
        if not (memo.root / key).is_dir():
            raise SystemExit("Missing completed fold: " + key)
        stack = memo.get(key)["stack"]
        coef = stack.named_steps["ridge"].coef_[j] / stack.named_steps["standardscaler"].scale_[j]
        coefs.append(float(coef))
    print("r%d convex_FM_weights: " % repeat + " ".join("%.6f" % x for x in weights))
    print("r%d ridge_FM_rawcoefs: " % repeat + " ".join("%.6f" % x for x in coefs))
print("Model-level diagnostics only; no refitting or individual records printed.")
