"""Read-only, outcome-free aggregate background inspection after V11 completion.

The existing loader opens discovery Y only to recover the unchanged eligibility
and folds. This script never summarizes Y values or fits an outcome model.
"""

import json
from pathlib import Path

import numpy as np

from bridge import load
from features import TARGET_NAMES, ecology
from runtime import check_config, environment, v11_completion

ROOT = Path(__file__).resolve().parent


def render(dataset):
    eligible = np.isfinite(dataset.reference["y"][0])
    rows = np.flatnonzero(eligible)
    diet = dataset.index.fiber_density.to_numpy(float)[rows]
    markers = np.column_stack([ecology(dataset.raw[site][rows])[:, [1, 3]] for site in ("oral", "gut")])
    lines = [
        "BACKGROUND ONLY | no outcome model, no new scientific result",
        "N=%d households=%d | same people in two repeats" % (len(rows), dataset.index.iloc[rows].split_unit_id.nunique()),
        "fiber g/1000 kcal Q05/Q25/Q50/Q75/Q95: " + " ".join("%.3f" % v for v in np.quantile(diet, [.05, .25, .5, .75, .95])),
        "marker                 Q25       Q75   corr_fiber",
    ]
    for j, name in enumerate(TARGET_NAMES):
        q = np.quantile(markers[:, j], [.25, .75])
        corr = np.corrcoef(markers[:, j], diet)[0, 1] if markers[:, j].std() > 0 else np.nan
        lines.append("%-20s %9.4f %9.4f %12.4f" % (name, q[0], q[1], corr))
    lines += ["Fiber Q10/Q90 within marker tails (descriptive, no exclusions):", "marker               tail     N     fiber_Q10     fiber_Q90"]
    for j, name in enumerate(TARGET_NAMES):
        q25, q75 = np.quantile(markers[:, j], [.25, .75])
        for label, mask in (("Q1", markers[:, j] <= q25), ("Q4", markers[:, j] >= q75)):
            low, high = np.quantile(diet[mask], [.1, .9])
            lines.append("%-20s %3s %5d %13.3f %13.3f" % (name, label, mask.sum(), low, high))
    corr = np.corrcoef(markers.T)
    lines += ["Marker correlations: oral H/dom=%.3f gut H/dom=%.3f oral/gut H=%.3f" % (corr[0, 1], corr[2, 3], corr[0, 2]),
              "Person averages are not strict diet/CGM-day alignment; sample-date semantics unverified.",
              "No Y summaries, participant rows, V11 model scores or input/result changes."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    package = ROOT.parent.parent
    if package.name != "collaborator_handoff_20261102_v1":
        raise ValueError("Use this read-only helper in the established server handoff")
    cfg = json.loads((ROOT / "config.json").read_text())
    check_config(cfg)
    environment(cfg)
    v11_completion(cfg)
    print(render(load(package, cfg)), end="")
