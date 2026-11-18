"""Rank existing split-date panel models on the same eight archived household partitions."""
import os
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from io import BytesIO
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"runs_TRE_ONLY/discovery-1"
NAMES = ("off","fixed","shannon","tuned","joint","pattern_off","pattern_fixed","daily_hinge","full_spread",
         "hinge_spread","spread25","transfer25","huber","transfer_spread","combined_huber","gate")
MATCHED = {"pattern_fixed","daily_hinge","full_spread","hinge_spread"}


def rank(repeats):
    r2 = np.array([1-a.sum(axis=0)[1:]/a[:,0].sum() for a in repeats])
    if not np.isfinite(r2).all():
        raise ValueError("Nonfinite scores")
    results = {}
    for j,name in enumerate(NAMES):
        if name in ("off","pattern_off"):
            continue
        off = NAMES.index("pattern_off" if name in MATCHED else "off")
        pct = None if (r2[:,off]<=0).any() else float(np.mean(100*(r2[:,j]-r2[:,off])/r2[:,off]))
        results[name] = dict(mean_r2=float(r2[:,j].mean()),relative_pct=pct,r2_by_repeat=r2[:,j].tolist())
    return sorted(results,key=lambda n:-results[n]["mean_r2"]),results


def self_test():
    n=len(NAMES)
    a=np.full((4,n+1),6.)
    a[:,0]=10.
    a[:,1+NAMES.index("pattern_off")]=8.
    a[:,1+NAMES.index("transfer25")]=3.
    a[:,1+NAMES.index("full_spread")]=4.
    order,results=rank([a,a])
    assert order[0]=="transfer25" and len(order)==n-2
    assert np.isclose(results["full_spread"]["relative_pct"],200)
    print("RANK_SYNTHETIC_PASS")


def parts(smoke):
    return [(0,0),(2,0)] if smoke else [(r,f) for r in range(2,10) for f in range(5)]


def replay(s,inputs,folds,r,f,reg_sha):
    flow,roles,ctx,eco,scores,fam,_,pcs_for=inputs
    held=folds[r]==f
    train=~held
    s.require(not set(fam[train]) & set(fam[held]),"Household overlap")
    old=s.prev.old
    dirs={k:roles[k] for k in s.DIRS}
    outer_pcs=pcs_for(train)
    def pcs(mask):
        return outer_pcs if np.array_equal(mask,train) else pcs_for(mask)
    panel,_=old.base.fold_results(dirs,eco,scores,fam,held,pcs,old.base.SEED+100*r+f)
    path=SRC/f"r{r}_f{f}.npz"
    meta=s.read(path.with_suffix(".json"))
    s.require(meta["sha"]==s.digest(path) and meta["registration"]==reg_sha,"Archive fold mismatch")
    out={}
    with np.load(path,allow_pickle=False) as archive:
        for k,other in zip(s.DIRS,reversed(s.DIRS)):
            pred,pattern=s.prev.ext.predictions(old.base.surface,old.pattern,roles[k],roles[other],eco,
                                                outer_pcs,scores,ctx[k],train,held)
            y=roles[k][0]
            args=(train,held,fam,flow["households"])
            ext=old.pattern.loss_totals(y,pred,*args)
            pat=old.pattern.loss_totals(y,pattern,*args)
            a=archive[k]
            # Reconstruct original, transfer, spread and combination SSEs from the exact inputs.
            np.testing.assert_allclose(ext[:,[0,1,2,3,4,5,7]],a[:,:7],rtol=2e-8,atol=2e-7)
            np.testing.assert_allclose(panel[k][:,[0,1,3]],a[:,[1,2,3]],rtol=2e-8,atol=2e-7)
            if r==0:
                with np.load(old.base.ROOT/"runs_TRE_ONLY/discovery-2"/f"r{r}_f{f}.npz",allow_pickle=False) as prev:
                    np.testing.assert_allclose(panel[k],prev[k],rtol=2e-8,atol=2e-7)
            out[k]=np.column_stack((a[:,1],ext[:,2:4],panel[k][:,[2,4,5]],pat[:,4:9],
                                    ext[:,4:9],a[:,7]))
            s.require(out[k].shape==(flow["households"],1+len(NAMES)),"Route mapping differs")
    return out


def collect(s,out,reg):
    groups={}
    for r,f in parts(reg["smoke"]):
        path=out/f"r{r}_f{f}.npz"
        s.require(s.read(path.with_suffix(".json"))=={"sha":s.digest(path),"registration":s.digest(out/"registration.json")},"Fold changed")
        group=groups.setdefault(r,{})
        with np.load(path,allow_pickle=False) as data:
            for k in s.DIRS:
                group[k]=group.get(k,0)+data[k]
    result={}
    for k,retained in zip(s.DIRS,("transfer25","transfer_spread")):
        repeats=[g[k] for g in groups.values()]
        order,rows=rank(repeats)
        valid=[n for n in order if rows[n]["relative_pct"] is not None]
        by_pct=sorted(valid,key=lambda n:-rows[n]["relative_pct"])
        best=order[0]
        # Descriptive paired interval for the observed top scorer minus the retained route.
        ib,ir=(1+NAMES.index(n) for n in (best,retained))
        n=repeats[0].shape[0]
        rng=np.random.default_rng(2026111851)
        weights=np.ones((400 if reg["smoke"] else 4000,n))
        for w in weights[1:]:
            w[:]=np.bincount(rng.integers(n,size=n),minlength=n)
        diff=np.mean([(weights@(a[:,ir]-a[:,ib]))/(weights@a[:,0]) for a in repeats],axis=0)
        result[k]=dict(rank_r2=order,rank_relative=by_pct,scores=rows,retained=retained,
                       best_minus_retained=float(diff[0]),descriptive_ci=np.quantile(diff[1:],[.025,.975]).tolist())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--run")
    p.add_argument("--commit")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--verify",action="store_true")
    arg=p.parse_args()
    if arg.self_test:
        self_test()
        return
    sys.path.insert(0,str(ROOT))
    import run as s
    s.require(re.fullmatch(r"[0-9a-f]{40}",arg.commit or "") and re.fullmatch(r"[a-zA-Z0-9_-]+",arg.run or ""),"Bad ID")
    out=ROOT/"diagnostics_TRE_ONLY"/arg.run
    parent=s.read(SRC/"registration.json")
    s.require(parent["commit"]=="f81a0768cf93b183ff2a82c8cbb817048f9a822f" and s.prev.old.receipt(SRC)==s.read(SRC/"COMPLETE.json"),"Stability archive differs")
    reg=dict(commit=arg.commit,smoke=arg.smoke,routes=list(NAMES),parent=s.digest(SRC/"COMPLETE.json"),code=s.digest(Path(__file__)))
    if arg.verify:
        reg["smoke"]=s.read(out/"registration.json")["smoke"]
        s.require(reg==s.read(out/"registration.json") and collect(s,out,reg)==s.read(out/"summary.json"),"Readback differs")
        s.require(s.read(out/"COMPLETE.json")=={"summary":s.digest(out/"summary.json"),"registration":s.digest(out/"registration.json")},"Invalid receipt")
        print("RANK_READBACK_PASS",len(parts(reg["smoke"])))
        return
    s.require(not out.exists(),"Use a fresh run ID")
    s.require(all(s.digest(Path(p))==v for p,v in parent["sources"].items()),"Parent inputs changed")
    inputs=s.prev.old.inputs(parent)
    folds=s.load_folds(inputs[5],inputs[6])
    out.mkdir(parents=True)
    s.io.write(out/"registration.json",reg)
    sha=s.digest(out/"registration.json")
    src_sha=s.digest(SRC/"registration.json")
    for r,f in parts(arg.smoke):
        payload=replay(s,inputs,folds,r,f,src_sha)
        path=out/f"r{r}_f{f}.npz"
        buf=BytesIO()
        np.savez_compressed(buf,**payload)
        path.write_bytes(buf.getvalue())
        s.io.write(path.with_suffix(".json"),dict(sha=s.digest(path),registration=sha))
        s.io.emit("RANK_FOLD",repeat=r,fold=f)
    s.io.write(out/"summary.json",collect(s,out,reg))
    s.io.write(out/"COMPLETE.json",dict(summary=s.digest(out/"summary.json"),registration=sha))
    print("RANK_COMPLETE")

if __name__=="__main__":
    main()
