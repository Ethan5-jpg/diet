import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from bridge import Memo

base = Path.cwd()
run = base / "runs/complementary_v13_20261109"
v11 = base.parent / "diet_model_distribution_20261109_v11/runs/distribution_v11_20261109"
reg = json.loads((run / "registration.json").read_text())
oldreg = json.loads((v11 / "registration.json").read_text())
v10 = base.parent / "diet_model_context_20261108_v10/runs" / oldreg["config"]["v10_reference_run"]
for folder in (run, v11, v10):
    assert (folder / "TRE_ONLY").is_dir(), "Required existing cache is missing"
a, b, c = [Memo(folder / "TRE_ONLY") for folder in (run, v11, v10)]
tag = "r0_f3"
new = a.get(tag + "_meta_fit")["v11_replay"]
old = b.get(tag + "_meta_fit")["positive_combined"]
prior = a.get(tag + "_prior_inner")
result, older = b.get(tag + "_result"), c.get(tag + "_result")
print("READ_ONLY_DIAG r0_f3; no fitting or file changes")
print("THREADS V13=%s V11=%s" % (reg.get("threads"), oldreg.get("threads")))
match = new.members == ["bank__" + n for n in old.members]
print("MEMBERS_MATCH=%s" % match)
print("CHOICES_MATCH=%s" % all(result["selected"].get(n) == j for n, j in prior["choices"]["v11"].items()))
assert match, "Member order differs; retain existing caches"
original = {"old__" + n: older["predictions"][n] for n in oldreg["config"]["legacy_members"]}
original.update(result["predictions"])
inputs = {"bank__" + n: original[n] for n in old.members}
x, y = new.model, old.model
print("PENALTY rebuilt=%s saved=%s" % (new.diagnostic.get("penalty"), old.diagnostic.get("penalty")))
for label, left, right in (
    ("TARGET_MEAN", x.mean, y.mean),
    ("TARGET_SCALE", x.scale, y.scale),
    ("INPUT_MEAN", x.scaler.mean_, y.scaler.mean_),
    ("INPUT_SCALE", x.scaler.scale_, y.scaler.scale_),
    ("COEFFICIENT", x.model.coef_, y.model.coef_),
):
    print("%s_MAX_DIFF=%.12g" % (label, np.max(abs(np.asarray(left) - np.asarray(right)))))
with threadpool_limits(limits=reg.get("threads", 2)):
    rebuilt, saved = new.predict(inputs), old.predict(original)
    target = result["predictions"]["positive_combined"]
    print("REBUILT_PRED_ERROR=%.12g" % np.max(abs(rebuilt - target)))
    print("SAVED_MODEL_PRED_ERROR=%.12g" % np.max(abs(saved - target)))
    print("PRIOR_META_PRED_ERROR=%.12g" % np.max(abs(prior["replay"].predict(inputs) - rebuilt)))
print("END_DIAG")
