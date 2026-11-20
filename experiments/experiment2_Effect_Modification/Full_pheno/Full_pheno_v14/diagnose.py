"""Summarize prediction losses by diet category and observed coverage."""

from collections import Counter
from statistics import median

from run_v2 import require
from taxonomy import GROUPS


def shared_rows(partitions, method, field="diet"):
    scores=[]
    for p in partitions:
        data=[r for r in p["rows"] if r[field]==method]
        idx={r["endpoint"]:r for r in data}
        require(len(idx)==len(data),"V14_DIAGNOSTIC_DUPLICATE")
        scores.append(idx)
    common=sorted(set.intersection(*(set(p) for p in scores)))
    return {key:tuple(p[key] for p in scores) for key in common}


def main_losses(rows, bins):
    """Both/all partition signs on the same set of outcomes."""
    counts={}
    for e,values in rows.items():
        tag=bins[e]
        x=counts.setdefault(tag,Counter())
        x["outcomes"]+=1
        x["main_negative"]+=int(all(r["diet_main_pp"]<0. for r in values))
        x["main_positive"]+=int(all(r["diet_main_pp"]>0. for r in values))
        x["total_negative"]+=int(all(r["diet_total_pp"]<0. for r in values))
    return {k:dict(v) for k,v in sorted(counts.items())}


def adjudicate_history(v12, v13, domain):
    fixed=shared_rows(v13,"fixed_half")
    samples={}
    for e,rows in fixed.items():
        households=min(x["families"] for x in rows)
        samples[e]="under_750" if households<750 else "750_to_1499" if households<1500 else "1500_plus"
    class_sizes={m:main_losses(shared_rows(v12,m),domain) for m in GROUPS}
    before=main_losses(fixed,domain)
    return dict(previous_full_by_domain=before, previous_full_by_households=main_losses(fixed,samples),
                previous_category_by_domain=class_sizes, common_outcomes=len(fixed))


def diary_comparison(results,domain):
    """Compare the same phenotypes supported in both strata and partitions."""
    group=("3_to_9_days","10_plus_days")
    arms=[shared_rows([dict(rows=r["diary"]) for r in results],k,"stratum") for k in group]
    common=set(arms[0])&set(arms[1])
    require(bool(common),"V14_NO_SHARED_DIARY_STRATA")
    summaries={}
    for labels,score in zip(group,arms,strict=True):
        scope={k:score[k] for k in sorted(common)}
        statistics={}
        for tag in ("all",*sorted({domain[k] for k in common})):
            part=[row for e,row in scope.items() if tag=="all" or domain[e]==tag]
            statistics[tag]=dict(outcomes=len(part),main_negative=sum(all(r["diet_main_pp"]<0. for r in z) for z in part),
                                 main_positive=sum(all(r["diet_main_pp"]>0. for r in z) for z in part),
                                 medians=[median(z[i]["diet_main_pp"] for z in part) for i in range(len(results))])
        summaries[labels]=statistics
    return summaries
