"""Evaluate daily fiber threshold exposure and dispersion on the archived disjoint-date study."""
import os

for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[var] = "1"

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import re
import sys

import numpy as np
import pattern

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "fiber_surface_dates_20261118_v1"
ARCHIVE = BASE / "runs_TRE_ONLY/discovery-2"
RELEASE = "0186606e16114fdccc490c1b5a896f5b3337aee3"
PINS = {"run.py":"8252e5acf24f36a34893e9c305db3dfbdd7a1ff47e3ab4f665bc9a9c8ff55641",
        "surface.py":"9b4e0f0252999b6ba8f8b293cc138529867662dea066215b917b9bc080482022"}
DIRS = ("early_to_late", "late_to_early")
SEED = 2026111802

sys.path.insert(0, str(BASE))
try:
    spec = spec_from_file_location("archived_surface", BASE / "run.py")
    base = module_from_spec(spec)
    spec.loader.exec_module(base)
finally:
    sys.path.pop(0)
dates, prior = base.dates, base.prior
digest, read, io, require = base.digest, base.read, base.io, base.require


def inputs(registered):
    daily, index, raw, taxa, folds, clr, src = prior.read(base.PACKAGE, read(prior.DAILY/"config.json"))
    require(all(registered["sources"].get(k)==v for k,v in src.items()), "Daily sources differ")
    cols = list(dates.COLUMNS)
    values = daily[cols].to_numpy(float)
    augmented = np.column_stack((values, pattern.daily_moments(values[:,cols.index("fiber")])))
    pop, periods, early, late = dates.period_means(daily.input_row.to_numpy(int), daily.date.to_numpy(dtype="datetime64[D]"), augmented)
    groups, families = np.unique(index.iloc[pop].split_unit_id.astype(str), return_inverse=True)
    flow = dict(people=len(pop), households=len(groups), early=int(early.sum()), late=int(late.sum()), days=int(early.sum()+late.sum()))
    require(flow == registered["flow"], "Unexpected study population")
    roles = dates.roles({k:v[:,:len(cols)] for k,v in periods.items()})
    context = {}
    for key,counts in zip(DIRS, (early,late)):
        p = periods[dates.PAIRINGS[key][0]]
        context[key] = pattern.pattern(p[:,1], p[:,len(cols):], counts)
    ecology = np.column_stack([prior.raw_ecology(raw[s][pop]) for s in ("oral","gut")])
    panel = base.panel.build_score(raw["gut"][pop], taxa["gut"])[0]
    scores = np.column_stack((panel, ecology[:,2]))
    def pcs(train):
        return prior.site_scores(raw,taxa,index,pop,train,clr,5)["both_pc"]
    return flow, roles, context, ecology, scores, families, folds[:,pop], pcs


def collect(out, reg):
    fingerprint = digest(out/"registration.json")
    totals = {}
    for r,f in base.parts(reg["smoke"]):
        path = out/f"r{r}_f{f}.npz"
        require(read(path.with_suffix(".json")) == dict(sha=digest(path), registration=fingerprint), "Changed fold")
        part = totals.setdefault(r, {})
        with np.load(path, allow_pickle=False) as data:
            require(set(data.files)==set(DIRS), "Unexpected directions")
            for k in DIRS:
                a = data[k]
                require(a.shape==(reg["flow"]["households"],2+2*len(pattern.ROUTES)), "Unexpected loss shape")
                part[k] = part.get(k,0) + a
    if not reg["smoke"]:
        for k in DIRS:
            first, second = totals[0][k][:,0], totals[1][k][:,0]
            require(np.array_equal(first,second) and min(first)>0 and first.sum()==reg["flow"]["people"], "Repeat household alignment changed")
    return pattern.summarize(totals, SEED, 399 if reg["smoke"] else 5999)


def stage(summary, reg):
    lines = ["# Daily fiber pattern and three-species score", "",
             f"Release {reg['commit']}. Population {reg['flow']}.",
             "Primary uses early diet and late glucose. Reverse dates use the same people.",
             "Original routes reproduce the archived baseline and surface. Matched routes include threshold gap and SD main effects.",
             "Relative gain uses each route's no-interaction baseline. Absolute R2 differences support comparisons across baselines.",
             "", "| Repeat | Direction | Route | R2 | Relative % | MAE (mg/dL) | R2 vs original fixed | R2 vs matched fixed |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r, dirs in summary["repeats"].items():
        for key,routes in dirs.items():
            for name,v in routes.items():
                pct = v["relative_pct"]
                pct = "undefined" if pct is None else f"{pct:.3f}"
                lines.append(f"| {r} | {key} | {name} | {v['r2']:.7f} | {pct} | {v['mae']:.5f} | {v['vs_original']:+.7f} | {v['vs_matched']:+.7f} |")
    lines += ["", "Repeat-averaged paired R2 gains versus matched fixed, six-comparison intervals, and MAE reduction in mg/dL."]
    for key,vals in summary["paired"].items():
        for name in pattern.NEW:
            v = vals[name]
            lo,hi = v["simultaneous"]
            lines.append(f"- {key} {name} {v['delta_r2']:+.7f} [{lo:+.7f}, {hi:+.7f}], MAE gain {v['mae_gain']:+.5f}.")
        v = vals["original_gain"]
        lines.append(f"- {key} original fixed gain over original off {v['delta_r2']:+.7f}, conditional 95% CI {v['ci95']}.")
    return "\n".join(lines)+"\n"


def receipt(out):
    return {p.name:digest(p) for p in [out/"summary.json",out/"stage_record.md",out/"registration.json",*sorted(out.glob("r*_f*.json"))]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--commit",required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--verify",action="store_true")
    args = p.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.run) and re.fullmatch(r"[0-9a-f]{40}", args.commit), "Bad run or release")
    out = ROOT/"runs_TRE_ONLY"/args.run
    if args.verify:
        reg = read(out/"registration.json")
        require(reg["commit"]==args.commit and read(out/"COMPLETE.json")==receipt(out), "Bad release or completion receipt")
        require(all(digest(Path(k))==v for k,v in reg["sources"].items()), "Source hash changed")
        require(collect(out,reg)==read(out/"summary.json"), "Summary readback differs")
        io.emit("PATTERN_READBACK_PASS", folds=len(base.parts(reg["smoke"])))
        return
    require(not out.exists(), "Use a fresh run directory")
    require(all(digest(BASE/k)==v for k,v in PINS.items()), "Fixed surface code changed")
    base.verify(ARCHIVE, RELEASE)
    parent = read(ARCHIVE/"registration.json")
    flow, roles, context, eco, scores, family, folds, pcs_for = inputs(parent)
    sources = dict(parent["sources"])
    sources.update({str(p):digest(p) for p in (*ROOT.glob("*.py"),ARCHIVE/"COMPLETE.json",ARCHIVE/"registration.json")})
    reg = dict(commit=args.commit, smoke=args.smoke, flow=flow, sources=sources, seed=SEED, primary=DIRS[0],
               routes=pattern.ROUTES, secondary=DIRS[1], primary_endpoint="paired change in outer R2",
               people_with_days_on_both_sides_of_knot={k:int((ctx[:,0]>1e-7).sum()) for k,ctx in context.items()})
    out.mkdir(parents=True)
    io.write(out/"registration.json", reg)
    fingerprint = digest(out/"registration.json")
    io.emit("PATTERN_START", **flow)
    for r,f in base.parts(args.smoke):
        held = folds[r]==f
        train = ~held
        require(not set(family[train]) & set(family[held]), "Household overlap")
        pcs = pcs_for(train)
        arrays = {}
        with np.load(ARCHIVE/f"r{r}_f{f}.npz",allow_pickle=False) as archived:
            for k in DIRS:
                pred = pattern.predictions(base.surface, roles[k],eco,pcs,scores,context[k],train,held)
                arrays[k] = pattern.loss_totals(roles[k][0], pred,train,held,family,flow["households"])
                # Archived columns: denominator, off, Shannon, fixed, tuned, joint.
                np.testing.assert_allclose(arrays[k][:,[1,2,3]], archived[k][:,[0,1,3]], rtol=2e-8, atol=2e-7)
        buf = BytesIO()
        np.savez_compressed(buf, **arrays)
        path = out/f"r{r}_f{f}.npz"
        path.write_bytes(buf.getvalue())
        io.write(path.with_suffix(".json"),dict(sha=digest(path),registration=fingerprint))
        io.emit("PATTERN_FOLD", repeat=r, fold=f)
    summary = collect(out,reg)
    io.write(out/"summary.json",summary)
    (out/"stage_record.md").write_text(stage(summary,reg))
    io.write(out/"COMPLETE.json",receipt(out))
    io.emit("PATTERN_COMPLETE", smoke=args.smoke)


if __name__ == "__main__":
    main()
