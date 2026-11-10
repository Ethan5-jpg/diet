"""Read saved aggregate JSON only; no fitting or individual-record access."""
import json
from pathlib import Path
from statistics import mean
from registry import INTERACTIONS

run = Path("runs/overnight_v8_20261108")
s = json.loads((run / "summary.json").read_text())
cfg = json.loads((run / "registration.json").read_text())["config"]
assert s["status"] == "COMPLETE" and s["confirmation_outcomes_read"] is False
score = {(v["model"], v["repeat"]): v["R2"] for v in s["scores"]}
mods = {(v["model"], v["repeat"]): v for v in s["modifiers"]}
gains = {(v["model"], v["repeat"]): v for v in s["comparisons"]}
flagged = {v["model"] for v in s["fit_quality"]}
atomic = [f"{name}_{site}" for name in cfg["families"] for site in ("oral", "balanced")]
top = sorted(atomic, key=lambda n: min(score[n, 0], score[n, 1]), reverse=True)[:5]
print("TOP NON-STACKED PIPELINES        R2_0       R2_1 flag")
for n in top:
    print(f"{n:28s} {score[n,0]:.6f}   {score[n,1]:.6f} {'W' if n in flagged else '-'}")
print("EXPLICIT: r null_R2 -> interaction_R2; delta; off=case0; active=nonzero contrast")
for n, base in INTERACTIONS.items():
    for r in (0, 1):
        off = sum(v["selected"][n] == 0 for v in s["selections"] if v["repeat"] == r)
        print(f"{n:25s} r{r} {score[base,r]:.6f}->{score[n,r]:.6f} "
              f"d={gains[n,r]['delta_R2']:+.2e} off={off}/5 active={mods[n,r]['active_folds']}/5")
print("CONDITIONAL HOUSEHOLD CI: selection uncertainty is not covered")
for r in (0, 1):
    g = gains["ridge_wide", r]
    lo, hi = g["conditional_household_interval"]
    print(f"ridge_wide vs V6 r{r}: dR2={g['delta_R2']:+.6f} CI=[{lo:+.6f},{hi:+.6f}]")
for r in (0, 1):
    fits = [v["meta"]["convex_with_fm"] for v in s["selections"] if v["repeat"] == r]
    weights = [v["weights"][v["members"].index("fm_safe_balanced")] for v in fits]
    bad = sum(v["model"] == "fm_safe_balanced" and v["repeat"] == r for v in s["fit_quality"])
    print(f"FM r{r}: mean_weight={mean(weights):.6f} max_weight={max(weights):.6f} flagged={bad}/5")
print("UPF joint-minus-wide stack dR2:", *(f"{score['ridge_joint',r]-score['ridge_wide',r]:+.6f}" for r in (0,1)))
