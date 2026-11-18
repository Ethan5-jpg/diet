"""Re-audit two original runs, then test frozen and training-selected corrections on eight fresh splits."""
import os

for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import re
import sys

import numpy as np
import stability as st

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent/"fiber_transfer_20261118_v1"
ARCH = PREV/"runs_TRE_ONLY/discovery-1"
RELEASE = "3bee922303df97ce1a408858c749082f4ea5c2ea"
sys.path.insert(0,str(PREV))
try:
    spec = spec_from_file_location("transfer_reference",PREV/"run.py")
    prev = module_from_spec(spec)
    spec.loader.exec_module(prev)
finally:
    sys.path.pop(0)

require, digest, read, io = prev.require, prev.digest, prev.read, prev.io
DIRS = prev.DIRS
COLUMNS = [0,1,2,3,5]


def parts(smoke):
    return [(0,0),(2,0)] if smoke else [(r,f) for r in range(2+st.NEW_REPEATS) for f in range(5)]


def verify_archive():
    reg = read(ARCH/"registration.json")
    require(reg["commit"]==RELEASE and not reg["smoke"],"Unexpected reference release")
    require(prev.old.receipt(ARCH)==read(ARCH/"COMPLETE.json"),"Reference receipt differs")
    require(all(digest(Path(p))==v for p,v in reg["sources"].items()),"Reference source differs")
    require(prev.collect(ARCH,reg)==read(ARCH/"summary.json"),"Reference readback differs")
    return reg


def load_folds(fam, original):
    folds = list(original)
    folds.extend(st.partition(fam,st.SEED+i) for i in range(st.NEW_REPEATS))
    partitions = set()
    for arr in folds:
        groups = tuple(sorted(tuple(np.flatnonzero(arr==f)) for f in range(5)))
        require(groups not in partitions,"Duplicate train/validation partitions")
        partitions.add(groups)
    return folds


def fit_pair(inputs,train,held):
    flow, roles, context, eco, scores, family, _, pcs_for = inputs
    require(not set(family[train]) & set(family[held]),"Household crosses training and validation")
    pcs = pcs_for(train)
    out = {}
    for k,other in zip(DIRS,reversed(DIRS)):
        full,_ = prev.ext.predictions(prev.old.base.surface,prev.old.pattern,roles[k],roles[other],
                                      eco,pcs,scores,context[k],train,held)
        out[k] = full
    return out


def run_fold(inputs, folds, r, f, n):
    _, roles, _, _, _, fam, _, _ = inputs
    held = folds[r]==f
    train = ~held
    loss = {k:np.zeros((n,4)) for k in DIRS}
    seen = np.zeros(len(fam),int)
    for it, iv in prev.old.base.surface.inner_roles(fam,train,st.SEED+8000+r*5+f):
        seen[iv] += 1
        fitted = fit_pair(inputs,it,iv)
        for k in DIRS:
            pred = fitted[k][:,COLUMNS[1:]]
            err = roles[k][0][iv,None]-pred
            loss[k] += st.by_family(err**2,fam[iv],n)
    require(np.array_equal(seen,train.astype(int)),"Incomplete inner validation")
    options = {k:st.choose(loss[k][np.unique(fam[train])]) for k in DIRS}
    fitted = fit_pair(inputs,train,held)
    payload = {}
    for k in DIRS:
        y = roles[k][0]
        if r<2:
            with np.load(ARCH/f"r{r}_f{f}.npz",allow_pickle=False) as ref:
                replay = prev.old.pattern.loss_totals(y,fitted[k],train,held,fam,n)
                np.testing.assert_allclose(replay,ref[k],rtol=2e-8,atol=2e-7)
        payload[k] = st.checkpoint(y[held],y[train].mean(),fitted[k][:,COLUMNS],fam[held],n,options[k])
    return payload,options


def collect(out,reg):
    totals, gates = {}, {}
    sha = digest(out/"registration.json")
    smoke, flow = reg["smoke"], reg["flow"]
    for r,f in parts(smoke):
        path = out/f"r{r}_f{f}.npz"
        meta = read(path.with_suffix(".json"))
        require(meta["sha"]==digest(path) and meta["registration"]==sha,"Fold receipt differs")
        group = totals.setdefault(r,{})
        phase = "archived" if r<2 else "fresh"
        counts = gates.setdefault(phase,{k:Counter() for k in DIRS})
        with np.load(path,allow_pickle=False) as data:
            require(set(data.files)==set(DIRS),"Missing direction")
            for k in DIRS:
                a = data[k]
                require(a.shape==(flow["households"],19),"Loss schema differs")
                group[k] = group.get(k,0) + a
                counts[k][st.ROUTES[1+meta["options"][k]]] += 1
    if not smoke:
        ref = totals[0][DIRS[0]][:,0]
        require(min(ref)>0 and ref.sum()==flow["people"],"Reference population differs")
        require(all(np.array_equal(ref,arr[:,0]) for g in totals.values() for arr in g.values()),"Unmatched repeats")
    n = flow["households"]
    rng = np.random.default_rng(st.SEED+90122)
    w = np.ones((400 if smoke else 6000,n))
    for row in w[1:]:
        row[:] = np.bincount(rng.integers(n,size=n),minlength=n)
    result = {}
    for phase in ("archived","fresh"):
        subset = [g for r,g in sorted(totals.items()) if (r<2)==(phase=="archived")]
        result[phase] = {k:st.summarize([g[k] for g in subset],w,phase=="fresh" and not smoke) for k in DIRS}
        for k in DIRS:
            result[phase][k]["gate_counts"] = dict(gates[phase][k])
    return result


def report(reg,res):
    lines = ["# Stability of the fiber and three-species corrections", "", f"Release {reg['commit']}. Cohort {reg['flow']}.",
             "Fresh repeats use eight new household partitions of the same discovery participants.",
             "The primary direction throughout the project is early diet to late glucose.",
             "Each correction has a fixed weight of 0.25. The gate chooses fixed, spread, transfer or both using three inner household folds.",
             "A within-cohort stability pass requires seven wins in eight repeats, nonnegative mean MAE improvement and a positive four-comparison interval lower limit.",
             "Confidence intervals condition on the fitted out-of-fold predictions. Influence trimming removes the 1% of households with largest absolute average SSE contributions.",
             "", "| Phase | Direction | Model | Mean delta R2 | Wins | Adjusted interval | MAE gain | Delta after trim | Stable |",
             "|---|---|---|---:|---:|---|---:|---:|---|"]
    for phase, ds in res.items():
        for k, entry in ds.items():
            for name,v in entry["results"].items():
                interval = str([round(i,7) for i in v["simultaneous"]]) if name in ("both","gate") else "descriptive"
                lines.append(f"| {phase} | {k} | {name} | {v['delta']:+.7f} | {v['wins']} | {interval} | "
                             f"{v['mae_gain']:+.5f} | {v['drop_top_1pct']:+.7f} | {v['stable']} |")
    lines += ["","Exact held-out squared error decomposition in R2 units"]
    for phase,ds in res.items():
        for k,entry in ds.items():
            vals = {n:round(v,7) for n,v in entry["components"].items()}
            lines.append(f"- {phase} {k} {vals}; inner gate counts {entry['gate_counts']}.")
    return "\n".join(lines)+"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--commit",required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--verify",action="store_true")
    args = p.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}",args.commit) and re.fullmatch(r"[A-Za-z0-9_-]+",args.run),"Invalid ID")
    out = ROOT/"runs_TRE_ONLY"/args.run
    if args.verify:
        reg = read(out/"registration.json")
        require(reg["commit"]==args.commit and prev.old.receipt(out)==read(out/"COMPLETE.json"),"Invalid release or receipt")
        require(all(digest(Path(p))==v for p,v in reg["sources"].items()),"Registered source changed")
        require(collect(out,reg)==read(out/"summary.json"),"Recalculated summary differs")
        io.emit("STABILITY_READBACK_PASS",folds=len(parts(reg["smoke"])))
        return
    require(not out.exists(),"Use a fresh run ID")
    parent = verify_archive()
    inputs = prev.old.inputs(parent)
    flow, _, _, _, _, fam, original, _ = inputs
    folds = load_folds(fam,original)
    sources = dict(parent["sources"])
    sources.update({str(p):digest(p) for p in (*ROOT.glob("*.py"),ARCH/"registration.json",ARCH/"COMPLETE.json")})
    reg = dict(commit=args.commit,smoke=args.smoke,flow=flow,sources=sources,seed=st.SEED,fresh_repeats=st.NEW_REPEATS,
               routes=st.ROUTES,criterion="7/8 positive, mean MAE gain >= 0 and multiplicity adjusted lower CI > 0")
    out.mkdir(parents=True)
    io.write(out/"registration.json",reg)
    sha = digest(out/"registration.json")
    io.emit("STABILITY_START",**flow)
    for r,f in parts(args.smoke):
        payload,options = run_fold(inputs,folds,r,f,flow["households"])
        path = out/f"r{r}_f{f}.npz"
        b = BytesIO()
        np.savez_compressed(b,**payload)
        path.write_bytes(b.getvalue())
        io.write(path.with_suffix(".json"),dict(registration=sha,sha=digest(path),options=options))
        io.emit("STABILITY_FOLD",repeat=r,fold=f)
    res = collect(out,reg)
    io.write(out/"summary.json",res)
    (out/"stage_record.md").write_text(report(reg,res))
    io.write(out/"COMPLETE.json",prev.old.receipt(out))
    io.emit("STABILITY_COMPLETE",smoke=args.smoke)


if __name__=="__main__":
    main()
