"""TRE-only matched comparison of two-species composition added to retained estimators."""
import os
for env in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[env] = "1"

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import re
import sys

import numpy as np
import composition as comp

ROOT = Path(__file__).resolve().parent
STABILITY = ROOT.parent/"fiber_stability_20261118_v1"
PARENT = STABILITY/"runs_TRE_ONLY/discovery-1"
PARENT_RELEASE = "f81a0768cf93b183ff2a82c8cbb817048f9a822f"
sys.path.insert(0, str(STABILITY))
try:
    spec = spec_from_file_location("hpp_stability_archive", STABILITY/"run.py")
    s = module_from_spec(spec)
    spec.loader.exec_module(s)
finally:
    sys.path.pop(0)
old = s.prev.old
surface = old.base.surface
require, read, digest = s.require, s.read, s.digest


def extra_marker(parent):
    daily, index, raw, taxa, _, _, src = old.prior.read(old.base.PACKAGE, read(old.prior.DAILY/"config.json"))
    require(all(parent["sources"].get(k) == v for k, v in src.items()), "Inputs differ")
    pop, counts = np.unique(daily.input_row.to_numpy(int), return_counts=True)
    pop = pop[counts >= 4]
    total, matches, panel = old.base.panel.build_score(raw["gut"][pop], taxa["gut"])
    two = comp.scores_without_bromii(panel, matches, old.base.panel.MASS_UNIT)
    _, fam = np.unique(index.iloc[pop].split_unit_id.astype(str), return_inverse=True)
    return total, two, fam


def parts(smoke):
    return [(2, 0)] if smoke else [(r, f) for r in range(2, 10) for f in range(5)]


def run_fold(inputs, folds, omit, r, f, parent_hash):
    flow, roles, ctx, eco, scores, fam, _, pcs_for = inputs
    held = folds[r] == f
    train = ~held
    require(not set(fam[train]) & set(fam[held]), "Household overlap")
    pcs = pcs_for(train)
    path = PARENT/f"r{r}_f{f}.npz"
    meta = read(path.with_suffix(".json"))
    require(meta["sha"] == digest(path) and meta["registration"] == parent_hash, "Parent fold changed")
    payload = {}
    with np.load(path, allow_pickle=False) as archive:
        for i, (k, other) in enumerate(zip(s.DIRS, reversed(s.DIRS))):
            target, aux = roles[k], roles[other]
            full, _ = s.prev.ext.predictions(surface, old.pattern, target, aux, eco, pcs, scores, ctx[k], train, held)
            args = (eco, pcs, scores, omit, train, held)
            main, own = comp.correction(surface, target, target, *args)
            _, opposite = comp.correction(surface, aux, target, *args)
            retained = full[:, (3, 5)[i]]
            preds = comp.assemble(full[:, 0], retained, main, own, opposite)
            totals = old.pattern.loss_totals(target[0], preds, train, held, fam, flow["households"])
            anchor = (5, 6)[i]
            # Stability checkpoints have count, denominator, six SSEs then six absolute-error sums.
            n = len(comp.NAMES)
            np.testing.assert_allclose(totals[:, [0, 1, 2, 3, 2+n, 3+n]],
                                       archive[k][:, [0, 1, 2, anchor, 8, anchor+6]], rtol=2e-8, atol=2e-7)
            payload[k] = totals
    return payload


def collect(out, reg):
    sha = digest(out/"registration.json")
    groups = {}
    for r, f in parts(reg["smoke"]):
        path = out/f"r{r}_f{f}.npz"
        require(read(path.with_suffix(".json")) == dict(sha=digest(path), registration=sha), "Fold receipt differs")
        group = groups.setdefault(r, {})
        with np.load(path, allow_pickle=False) as data:
            require(set(data.files) == set(s.DIRS), "Missing direction")
            for k in s.DIRS:
                a = data[k]
                require(a.shape == (reg["flow"]["households"], 2+2*len(comp.NAMES)), "Unexpected losses")
                group[k] = group.get(k, 0) + a
    n = reg["flow"]["households"]
    if not reg["smoke"]:
        counts = groups[2][s.DIRS[0]][:, 0]
        require(min(counts) > 0 and counts.sum() == reg["flow"]["people"], "Incomplete validation")
        require(all(np.array_equal(a[:, 0], counts) for g in groups.values() for a in g.values()), "Households differ")
    weights = np.ones((400 if reg["smoke"] else 6000, n))
    rng = np.random.default_rng(2026111861)
    for row in weights[1:]:
        row[:] = np.bincount(rng.integers(n, size=n), minlength=n)
    return {k:comp.summarize([g[k] for g in groups.values()], weights) for k in s.DIRS}


def stage(reg, results):
    lines = ["# Two-species composition on disjoint dates", "", f"Release {reg['commit']}. Cohort {reg['flow']}.",
             "Primary direction is early diet to late glucose. The primary extension is shared_half.",
             "Each retained model and its no-interaction comparator match the prior household losses.",
             "Relative gain is the mean of 100 times (model R2 minus off R2) divided by off R2 in each repeat.",
             "Matched rows also adjust for the extra marker main effect. Each delta uses its corresponding retained model.",
             "Intervals resample families with shared weights across repeats, conditional on fitted predictions.",
             "Selection criterion for replacing a retained model is 7/8 wins, nonnegative MAE gain and a positive six-comparison lower bound.",
             "", "|Direction|Route|R2|Relative %|Delta R2|Wins|Adjusted lower|Adjusted upper|MAE gain|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, routes in results.items():
        for name in comp.NAMES:
            v = routes[name]
            lo, hi = v["simultaneous"]
            pct = "undefined" if v["relative_pct"] is None else f"{v['relative_pct']:.4f}"
            lines.append(f"|{k}|{name}|{v['mean_r2']:.7f}|{pct}|{v['delta']:+.7f}|{v['wins']}|{lo:+.7f}|{hi:+.7f}|{v['mae_gain']:+.6f}|")
    for k, routes in results.items():
        passed = [name for name in comp.TESTED if not reg["smoke"] and routes[name]["wins"] >= 7
                  and routes[name]["mae_gain"] >= 0 and routes[name]["simultaneous"][0] > 0]
        lines.append("")
        lines.append(f"{k} extensions passing the prespecified stability check: {', '.join(passed) or 'none'}.")
        lines.append("")
    return "\n".join(lines)+"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.run) and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Bad run or release")
    out = ROOT/"runs_TRE_ONLY"/args.run
    if args.verify:
        reg = read(out/"registration.json")
        require(reg["commit"] == args.commit and old.receipt(out) == read(out/"COMPLETE.json"), "Completion differs")
        require(all(digest(Path(p)) == value for p, value in reg["sources"].items()), "Sources differ")
        summary = collect(out, reg)
        require(summary == read(out/"summary.json") and stage(reg, summary) == (out/"stage_record.md").read_text(), "Readback differs")
        print("COMPOSITION_READBACK_PASS", len(parts(reg["smoke"])), flush=True)
        return
    require(not out.exists(), "Use a new run directory")
    parent = read(PARENT/"registration.json")
    require(parent["commit"] == PARENT_RELEASE and old.receipt(PARENT) == read(PARENT/"COMPLETE.json"), "Parent differs")
    sources = dict(parent["sources"])
    require(all(digest(Path(p)) == value for p, value in sources.items()), "Parent inputs differ")
    sources.update({str(p):digest(p) for p in (*ROOT.glob("*.py"), STABILITY/"run.py", STABILITY/"stability.py",
                                               PARENT/"COMPLETE.json", PARENT/"registration.json")})
    total, omit, fam = extra_marker(parent)
    inputs = old.inputs(parent)
    flow, _, _, _, scores, family, original, _ = inputs
    require(np.array_equal(family, fam) and np.array_equal(total, scores[:, 0]) and np.isfinite(omit).all(), "Marker alignment")
    folds = s.load_folds(family, original)
    reg = dict(commit=args.commit, flow=flow, smoke=args.smoke, sources=sources, routes=list(comp.NAMES),
               primary="shared_half", primary_direction=s.DIRS[0], multiplicity=6, common_family_draws=5999 if not args.smoke else 399)
    out.mkdir(parents=True)
    s.io.write(out/"registration.json", reg)
    fingerprint = digest(out/"registration.json")
    for r, f in parts(args.smoke):
        payload = run_fold(inputs, folds, omit, r, f, digest(PARENT/"registration.json"))
        buf = BytesIO()
        np.savez_compressed(buf, **payload)
        path = out/f"r{r}_f{f}.npz"
        path.write_bytes(buf.getvalue())
        s.io.write(path.with_suffix(".json"), dict(sha=digest(path), registration=fingerprint))
        print("COMPOSITION_FOLD", r, f, flush=True)
    result = collect(out, reg)
    s.io.write(out/"summary.json", result)
    (out/"stage_record.md").write_text(stage(reg, result))
    s.io.write(out/"COMPLETE.json", old.receipt(out))
    print("COMPOSITION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
