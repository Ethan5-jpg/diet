/home/ec2-user/miniconda/envs/pheno/bin/python -B - <<'HPP_VIEW'
import json
from pathlib import Path

p = Path("/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_model_batch_20261107_v6/runs/console_20261107/summary.json")
if not p.is_file():
    raise SystemExit("Summary not found: " + str(p))
d = json.loads(p.read_text(encoding="utf-8"))
if d.get("status") != "COMPLETE":
    raise SystemExit("Run incomplete; no scores displayed")
s = {(x["model"], x["repeat"]): x for x in d["scores"]}
names = sorted({name for name, _ in s})
b = [s[("v4_hgb_oral", r)]["R2"] for r in (0, 1)]
print("RUN=console_20261107 STATUS=COMPLETE MODELS=%d" % len(names))
print("N0=%d N1=%d; total-model dR2 uses v4_hgb_oral" % tuple(s[("v4_hgb_oral", r)]["n"] for r in (0, 1)))
print("%-23s %9s %9s %10s %10s" % ("model", "R2_0", "R2_1", "dR2_0", "dR2_1"))
for name in names:
    a, z = [s[(name, r)]["R2"] for r in (0, 1)]
    print("%-23s %9.6f %9.6f %+10.6f %+10.6f" % (name, a, z, a-b[0], z-b[1]))
c = {(x["model"], x["repeat"]): x for x in d["comparisons"]}
print("\nPAIRED GAINS: ref=HGB(v4_hgb_oral) or KMAIN(kernel_main)")
print("%-23s %5s %10s %10s %7s %7s" % ("model", "ref", "dR2_0", "dR2_1", "wins0/5", "wins1/5"))
for name in ("fusion", "kernel_linear", "kernel_spline", "kernel_supervised"):
    a, z = [c[(name, r)] for r in (0, 1)]
    assert a["comparator"] == z["comparator"], "Comparator mismatch"
    ref = {"v4_hgb_oral": "HGB", "kernel_main": "KMAIN"}[a["comparator"]]
    print("%-23s %5s %+10.6f %+10.6f %5d/5 %5d/5" % (name, ref, a["delta_R2"], z["delta_R2"], a["improved_folds"], z["improved_folds"]))
m = {(x["model"], x["repeat"]): x for x in d["modifiers"]}
print("\nINTERACTION: active=nonzero effective-weight folds; SD unit=mg/dL")
print("%-23s %9s %9s %10s %10s" % ("model", "active0/5", "active1/5", "SD_0", "SD_1"))
for name in ("kernel_linear", "kernel_spline", "kernel_supervised"):
    a, z = [m[(name, r)] for r in (0, 1)]
    print("%-23s %7d/5 %7d/5 %10.6f %10.6f" % (name, a["nonzero_folds"], z["nonzero_folds"], a["SD_mg_dL"], z["SD_mg_dL"]))
print("Discovery results; repeats share participants. No refitting performed.")
HPP_VIEW
