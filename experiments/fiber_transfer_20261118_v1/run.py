"""Test anchored spread, opposite-direction transfer and robust corrections on the frozen household folds."""
import os

for var in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[var] = "1"

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import re
import sys

import numpy as np
import extensions as ext

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/"fiber_daily_pattern_20261118_v1"
ARCHIVE = PREVIOUS/"runs_TRE_ONLY/discovery-1"
PARENT_RELEASE = "a5135065e7f820e46e1b1b0fb2e942e4ec916543"
PINS = {"run.py":"5a62d1f1fca780b698f48a1cdf577ad4b0684f882cd81046b074fa2e189c2f12",
        "pattern.py":"76ca9b8a3d5460d15914ee0edb207f910e28662e858947a70fd59fe5772057ee"}
SEED = 2026111803

sys.path.insert(0,str(PREVIOUS))
try:
    spec = spec_from_file_location("archived_daily_pattern",PREVIOUS/"run.py")
    old = module_from_spec(spec)
    spec.loader.exec_module(old)
finally:
    sys.path.pop(0)
require, digest, read, io = old.require, old.digest, old.read, old.io
DIRS = old.DIRS


def verify_parent():
    require(all(digest(PREVIOUS/k)==v for k,v in PINS.items()),"Parent implementation changed")
    reg = read(ARCHIVE/"registration.json")
    require(reg["commit"]==PARENT_RELEASE and not reg["smoke"],"Unexpected parent")
    require(old.receipt(ARCHIVE)==read(ARCHIVE/"COMPLETE.json"),"Invalid parent receipt")
    require(all(digest(Path(p))==v for p,v in reg["sources"].items()),"Parent inputs changed")
    require(old.collect(ARCHIVE,reg)==read(ARCHIVE/"summary.json"),"Archived prediction check failed")
    return reg


def collect(out, reg):
    fingerprint = digest(out/"registration.json")
    totals = {}
    for r,f in old.base.parts(reg["smoke"]):
        path = out/f"r{r}_f{f}.npz"
        require(read(path.with_suffix(".json"))==dict(sha=digest(path),registration=fingerprint),"Changed checkpoint")
        group = totals.setdefault(r,{})
        with np.load(path,allow_pickle=False) as data:
            require(set(data.files)==set(DIRS),"Missing direction")
            for k in DIRS:
                a = data[k]
                require(a.shape==(reg["flow"]["households"],2+2*len(ext.ROUTES)),"Unexpected household data shape")
                group[k] = group.get(k,0) + a
    if not reg["smoke"]:
        for k in DIRS:
            a, b = (totals[r][k][:,0] for r in (0,1))
            require(np.array_equal(a,b) and min(a)>0 and sum(a)==reg["flow"]["people"],"Unmatched households")
    return ext.summarize(old.pattern,totals,SEED,399 if reg["smoke"] else 7999)


def report(reg, res):
    lines = ["# Interaction-only extensions of the three-species fiber model","",
             f"Release {reg['commit']}. Cohort {reg['flow']}. All comparisons use identical target-direction main effects.",
             "The transfer weight and incremental spread weight are each 0.25. Huber cutoff is 1.345 robust scale units.",
             "Primary uses earlier diet and later glucose. The reverse direction uses the same households.",
             "Relative percent is 100 times the difference in R2 over positive no-interaction R2.",
             "", "| Repeat | Direction | Model | R2 | Relative % | R2 vs fixed | MAE (mg/dL) |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r,ds in res["repeats"].items():
        for k,routes in ds.items():
            for name,v in routes.items():
                pct = "undefined" if v["relative_pct"] is None else f"{v['relative_pct']:.3f}"
                lines.append(f"| {r} | {k} | {name} | {v['r2']:.7f} | {pct} | {v['vs_fixed']:+.7f} | {v['mae']:.5f} |")
    lines += ["", "Paired mean gain over fixed. Shared household bootstrap conditional on discovery predictions.",
              "Intervals control ten comparisons in this batch. Positive MAE gain indicates smaller absolute error."]
    for k,models in res["paired"].items():
        for name,v in models.items():
            lo,hi = v["simultaneous"]
            lines.append(f"- {k} {name} R2 {v['delta_r2']:+.7f} [{lo:+.7f}, {hi:+.7f}], MAE gain {v['mae_gain']:+.5f}.")
    return "\n".join(lines)+"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True)
    p.add_argument("--commit",required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--verify",action="store_true")
    args = p.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]+",args.run) and re.fullmatch(r"[0-9a-f]{40}",args.commit),"Invalid run or release")
    out = ROOT/"runs_TRE_ONLY"/args.run
    if args.verify:
        reg = read(out/"registration.json")
        require(reg["commit"]==args.commit and old.receipt(out)==read(out/"COMPLETE.json"),"Invalid receipt or release")
        require(all(digest(Path(p))==v for p,v in reg["sources"].items()),"Registered inputs differ")
        require(collect(out,reg)==read(out/"summary.json"),"Summary changed")
        io.emit("TRANSFER_READBACK_PASS",folds=len(old.base.parts(reg["smoke"])))
        return
    require(not out.exists(),"Use a new run ID")
    parent = verify_parent()
    flow, roles, ctx, eco, scores, fam, folds, pcs_for = old.inputs(parent)
    sources = dict(parent["sources"])
    sources.update({str(p):digest(p) for p in (*ROOT.glob("*.py"),ARCHIVE/"COMPLETE.json",ARCHIVE/"registration.json")})
    reg = dict(commit=args.commit, smoke=args.smoke, sources=sources, flow=flow, seed=SEED,
               primary=DIRS[0], routes=ext.ROUTES, share=ext.SHARE, huber_constant=1.345)
    out.mkdir(parents=True)
    io.write(out/"registration.json",reg)
    sha = digest(out/"registration.json")
    io.emit("TRANSFER_START",**flow)
    for r,f in old.base.parts(args.smoke):
        held = folds[r]==f
        train = ~held
        require(not set(fam[train]) & set(fam[held]),"Household crosses folds")
        pcs = pcs_for(train)
        results = {}
        with np.load(ARCHIVE/f"r{r}_f{f}.npz",allow_pickle=False) as ref:
            for k,other in zip(DIRS,reversed(DIRS)):
                pred, reference = ext.predictions(old.base.surface,old.pattern,roles[k],roles[other],
                                                  eco,pcs,scores,ctx[k],train,held)
                args_loss = (train,held,fam,flow["households"])
                expected = old.pattern.loss_totals(roles[k][0],reference,*args_loss)
                np.testing.assert_allclose(expected,ref[k],rtol=2e-8,atol=2e-7)
                results[k] = old.pattern.loss_totals(roles[k][0],pred,*args_loss)
        path = out/f"r{r}_f{f}.npz"
        buffer = BytesIO()
        np.savez_compressed(buffer,**results)
        path.write_bytes(buffer.getvalue())
        io.write(path.with_suffix(".json"),dict(sha=digest(path),registration=sha))
        io.emit("TRANSFER_FOLD",repeat=r,fold=f)
    result = collect(out,reg)
    io.write(out/"summary.json",result)
    (out/"stage_record.md").write_text(report(reg,result))
    io.write(out/"COMPLETE.json",old.receipt(out))
    io.emit("TRANSFER_COMPLETE",smoke=args.smoke)


if __name__=="__main__":
    main()
