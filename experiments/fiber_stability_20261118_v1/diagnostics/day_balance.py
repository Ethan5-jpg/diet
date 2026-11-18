"""Assess the frozen combination by date-count parity using existing household loss checkpoints."""
import os

for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key]="1"

import argparse
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY_RELEASE = "f81a0768cf93b183ff2a82c8cbb817048f9a822f"


def strata(family, days):
    size = np.bincount(family)
    odd = np.bincount(family,weights=days%2)
    return {"equal_days":odd==0,"extra_late":odd==size,"mixed":(odd>0)&(odd<size)}


def gain(a, w):
    # Denominator is the sum of the archived outer-training-mean reference errors in this subgroup.
    return (w @ (a[:,3]-a[:,6]))/(w @ a[:,1])


def self_test():
    family = np.array([0,0,1,1,2,2])
    days = np.array([4,6,5,7,4,7])
    groups = strata(family,days)
    np.testing.assert_array_equal(np.argmax(np.stack(list(groups.values())),axis=1),[0,1,2])
    a = np.zeros((3,7))
    a[:,1],a[:,3],a[:,6] = 4,[4,5,6],[3,7,7]
    np.testing.assert_allclose(gain(a,np.ones((1,3))),[-2/12])
    print("DAY_BALANCE_SYNTHETIC_PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--commit")
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    sys.path.insert(0,str(ROOT))
    import run
    require, digest, read, io = run.require,run.digest,run.read,run.io
    require(re.fullmatch(r"[0-9a-f]{40}",args.commit or ""),"Expected release")
    src = ROOT/"runs_TRE_ONLY/discovery-1"
    out = ROOT/"diagnostics_TRE_ONLY/date_balance-1"
    reg = read(src/"registration.json")
    require(reg["commit"]==STUDY_RELEASE and not reg["smoke"] and
            run.prev.old.receipt(src)==read(src/"COMPLETE.json"),"Invalid primary run")
    dep = run.prev.old
    daily,index,_,_,_,_,source = dep.prior.read(dep.base.PACKAGE,read(dep.prior.DAILY/"config.json"))
    require(all(reg["sources"].get(p)==v for p,v in source.items()),"Date sources differ")
    rows = daily.input_row.to_numpy(int)
    pop,_,early,late = dep.dates.period_means(rows,daily.date.to_numpy(dtype="datetime64[D]"),np.ones((len(rows),1)))
    _,family = np.unique(index.iloc[pop].split_unit_id.astype(str),return_inverse=True)
    sizes = np.bincount(family)
    groups = strata(family,early+late)
    require(len(pop)==reg["flow"]["people"] and len(sizes)==reg["flow"]["households"] and
            early.sum()==reg["flow"]["early"] and late.sum()==reg["flow"]["late"],"Date alignment differs")
    repeats = []
    for r in range(2,10):
        tally = {}
        for f in range(5):
            path = src/f"r{r}_f{f}.npz"
            require(digest(path)==read(path.with_suffix(".json"))["sha"],"Checkpoint changed")
            with np.load(path,allow_pickle=False) as data:
                for k in run.DIRS:
                    tally[k] = tally.get(k,0)+data[k]
        require(all(np.array_equal(t[:,0],sizes) for t in tally.values()),"Family order/size differs")
        repeats.append(tally)
    summary = {}
    directional = {}
    for j,(name,mask) in enumerate(groups.items()):
        summary[name] = dict(households=int(mask.sum()),people=int(sizes[mask].sum()))
        if mask.sum()<25:
            continue
        n = mask.sum()
        w = np.ones((2000,n))
        rng = np.random.default_rng(2026111880+j)
        for row in w[1:]:
            row[:] = np.bincount(rng.integers(n,size=n),minlength=n)
        direction = []
        for k in run.DIRS:
            estimates = np.stack([gain(t[k][mask],w) for t in repeats])
            avg = estimates.mean(axis=0)
            summary[name][k] = dict(delta=float(avg[0]),ci95=np.quantile(avg[1:],[.025,.975]).tolist(),
                                    wins=int((estimates[:,0]>0).sum()))
            direction.append(avg)
        diff = direction[1]-direction[0]
        directional[name] = diff
        summary[name]["reverse_minus_forward"] = dict(delta=float(diff[0]),ci95=np.quantile(diff[1:],[.025,.975]).tolist())
    interaction = directional["extra_late"]-directional["equal_days"]
    summary["direction_gap_extra_minus_equal"] = dict(delta=float(interaction[0]),ci95=np.quantile(interaction[1:],[.025,.975]).tolist())
    result = dict(commit=args.commit,script=digest(Path(__file__)),source_receipt=digest(src/"COMPLETE.json"),groups=summary)
    if args.verify:
        require(result==read(out/"summary.json"),"Date balance readback differs")
        print("DAY_BALANCE_READBACK_PASS")
    else:
        out.mkdir(parents=True,exist_ok=False)
        io.write(out/"summary.json",result)
        print("DAY_BALANCE_COMPLETE")


if __name__=="__main__":
    main()
