"""TRE runner for prespecified split-date fusion comparisons with paired additive controls."""
import os
for env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[env] = "1"

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import re
import sys
import numpy as np
import fusion

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent/"fiber_panel_composition_20261118_v1"
ARCH = PREV/"runs_TRE_ONLY/discovery-1"
PARENT_RELEASE = "610c05beccbf0db0586441e1ef28c5f9ca1a9b51"
INSPIRATION = "39ba6400dec767147e916a092e1913f3713c9386"
sys.path.insert(0, str(PREV))
try:
    spec = spec_from_file_location("date_composition_reference", PREV/"run.py")
    prev = module_from_spec(spec)
    spec.loader.exec_module(prev)
finally:
    sys.path.pop(0)
s, old = prev.s, prev.old
require, read, digest = prev.require, prev.read, prev.digest


def markers(reg):
    daily, index, raw, taxa, _, _, src = old.prior.read(old.base.PACKAGE, read(old.prior.DAILY/"config.json"))
    require(all(reg["sources"].get(k) == v for k, v in src.items()), "Source archive mismatch")
    pop, counts = np.unique(daily.input_row.to_numpy(int), return_counts=True)
    pop = pop[counts >= 4]
    panelmod = old.base.panel
    total, mapping, panel = panelmod.build_score(raw["gut"][pop], taxa["gut"])
    omit = prev.comp.scores_without_bromii(panel, mapping, panelmod.MASS_UNIT)
    balances = fusion.panel_views(panel, mapping, panelmod.MASS_UNIT)
    _, fam = np.unique(index.iloc[pop].split_unit_id.astype(str), return_inverse=True)
    return total, omit, balances, fam


def evaluate(inputs, folds, omit, balances, r, f, reg_sha):
    flow, roles, ctx, eco, scores, family, _, get_pcs = inputs
    held = folds[r] == f
    train = ~held
    require(not set(family[train]) & set(family[held]), "Family overlap")
    pcs = get_pcs(train)
    path = ARCH/f"r{r}_f{f}.npz"
    require(read(path.with_suffix(".json")) == dict(sha=digest(path), registration=reg_sha), "Historical fold mismatch")
    data = {}
    with np.load(path, allow_pickle=False) as ref:
        for i, (k, other) in enumerate(zip(s.DIRS, reversed(s.DIRS))):
            target, aux = roles[k], roles[other]
            full, _ = s.prev.ext.predictions(prev.surface, old.pattern, target, aux, eco, pcs, scores, ctx[k], train, held)
            anchor = full[:, (3, 5)[i]]
            history_column = prev.comp.NAMES.index("retained")
            if i:
                _, own = prev.comp.correction(prev.surface, target, target, eco, pcs, scores, omit, train, held)
                anchor = anchor+own
                history_column = prev.comp.NAMES.index("own_full")
            additive, delta = fusion.additions(prev.surface, target, eco, pcs, scores, balances, omit, ctx[k], train, held, bool(i))
            pred = fusion.assemble(full[:, 0], anchor, additive, delta)
            totals = old.pattern.loss_totals(target[0], pred, train, held, family, flow["households"])
            n, m = len(fusion.NAMES), len(prev.comp.NAMES)
            j = 2+history_column
            np.testing.assert_allclose(totals[:, [0, 1, 2, 3, 2+n, 3+n]], ref[k][:, [0, 1, 2, j, 2+m, j+m]],
                                       rtol=2e-8, atol=2e-7)
            data[k] = totals
    return data


def collect(out, reg):
    sha = digest(out/"registration.json")
    groups = {}
    for r, f in prev.parts(reg["smoke"]):
        path = out/f"r{r}_f{f}.npz"
        require(read(path.with_suffix(".json")) == dict(sha=digest(path), registration=sha), "Fold receipt mismatch")
        group = groups.setdefault(r, {})
        with np.load(path, allow_pickle=False) as data:
            require(set(data.files) == set(s.DIRS), "Missing direction")
            for k in s.DIRS:
                a = data[k]
                require(a.shape == (reg["flow"]["households"], 2+2*len(fusion.NAMES)), "Invalid layout")
                group[k] = group.get(k, 0)+a
    if not reg["smoke"]:
        counts = groups[2][s.DIRS[0]][:, 0]
        require(min(counts) > 0 and counts.sum() == reg["flow"]["people"], "Incomplete population")
        require(all(np.array_equal(a[:, 0], counts) for g in groups.values() for a in g.values()), "Mismatched people")
    families = reg["flow"]["households"]
    weights = np.ones((reg["draws"]+1, families))
    rng = np.random.default_rng(2026111871)
    for w in weights[1:]:
        w[:] = np.bincount(rng.integers(families, size=families), minlength=families)
    return {k:fusion.summarize([g[k] for g in groups.values()], weights) for k in s.DIRS}


def stage(reg, summary):
    lines = ["# Fiber and gut microbiome fusion on disjoint observation dates", "",
             f"Release {reg['commit']}. Models adapted from {INSPIRATION}. Cohort {reg['flow']}.",
             "Forward anchor is transfer25. Reverse anchor is the previous exploratory transfer_spread plus own_full two-species correction.",
             "Primary extension is Shannon plus two panel balances in the forward direction. Other fixed recipes are discovery comparisons.",
             "Frozen models score the original additive background. Matched sensitivities add the same new main effects to every on/off pair.",
             "The relative gain uses off R2. The contribution fraction uses on R2. Ratios are calculated per repeat before averaging.",
             "Worst fraction is the minimum across eight household partitions of the same people. Intervals condition on trained predictions.",
             "Selection requires at least seven repeat wins, nonnegative MAE improvement, a positive adjusted lower limit and better R2 than the historical anchor.",
             "", "|Direction|Model|R2|Gain/off %|Fraction/on %|Worst fraction %|Delta vs paired anchor|Wins|Adjusted interval|MAE gain|",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    def pct(x):
        return "undefined" if x is None else f"{x:.4f}"
    for k in s.DIRS:
        for name in fusion.NAMES:
            v = summary[k][name]
            lo, hi = v["interval"]
            lines.append(f"|{k}|{name}|{v['r2']:.7f}|{pct(v['gain_pct'])}|{pct(v['fraction_pct'])}|{pct(v['worst_fraction'])}|"
                         f"{v['delta']:+.7f}|{v['wins']}|[{lo:+.7f}, {hi:+.7f}]|{v['mae_gain']:+.6f}|")
    for k in s.DIRS:
        passed = [n for n, v in summary[k].items() if n.removeprefix("matched_") in fusion.METHODS and not reg["smoke"]
                  and v["wins"] >= 7 and v["mae_gain"] >= 0 and v["interval"][0] > 0 and v["delta_historical"] > 0]
        lines.extend(("", f"Passed for {k}: {', '.join(passed) or 'none'}."))
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
        require(reg["commit"] == args.commit and old.receipt(out) == read(out/"COMPLETE.json"), "Completion receipt mismatch")
        require(all(digest(Path(p)) == v for p, v in reg["sources"].items()), "Source changed")
        summary = collect(out, reg)
        require(read(out/"summary.json") == summary and (out/"stage_record.md").read_text() == stage(reg, summary), "Readback failed")
        print("FUSION_READBACK_PASS", len(prev.parts(reg["smoke"])), flush=True)
        return
    require(not out.exists(), "Choose a fresh run ID")
    parent = read(ARCH/"registration.json")
    require(parent["commit"] == PARENT_RELEASE and old.receipt(ARCH) == read(ARCH/"COMPLETE.json"), "Parent receipt mismatch")
    sources = dict(parent["sources"])
    require(all(digest(Path(p)) == v for p, v in sources.items()), "Source mismatch")
    sources.update({str(p):digest(p) for p in (*ROOT.glob("*.py"), ARCH/"registration.json", ARCH/"COMPLETE.json")})
    total, omit, balances, fam = markers(parent)
    inputs = old.inputs(parent)
    flow, _, _, _, scores, family, orig, _ = inputs
    require(np.array_equal(total, scores[:, 0]) and np.array_equal(fam, family) and np.isfinite(balances).all(), "Unaligned panel")
    folds = s.load_folds(family, orig)
    out.mkdir(parents=True)
    reg = dict(commit=args.commit, smoke=args.smoke, flow=flow, sources=sources, recipes=list(fusion.NAMES),
               draws=399 if args.smoke else 11999, comparisons=fusion.CONTRASTS,
               primary="early_to_late/shannon_balances", inspiration=INSPIRATION)
    s.io.write(out/"registration.json", reg)
    sha = digest(out/"registration.json")
    reg_sha = digest(ARCH/"registration.json")
    for r, f in prev.parts(args.smoke):
        data = evaluate(inputs, folds, omit, balances, r, f, reg_sha)
        path = out/f"r{r}_f{f}.npz"
        buf = BytesIO()
        np.savez_compressed(buf, **data)
        path.write_bytes(buf.getvalue())
        s.io.write(path.with_suffix(".json"), dict(registration=sha, sha=digest(path)))
        print("FUSION_FOLD", r, f, flush=True)
    summary = collect(out, reg)
    s.io.write(out/"summary.json", summary)
    (out/"stage_record.md").write_text(stage(reg, summary))
    s.io.write(out/"COMPLETE.json", old.receipt(out))
    print("FUSION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
