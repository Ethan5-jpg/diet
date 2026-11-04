"""Cross-fold species weights and module membership audits, only inside TRE."""
import itertools
import json
import pickle
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from audit_v4 import ROOT, record


def jaccard(a,b):
    return len(a & b)/len(a | b) if a or b else 0.


def main(kind):
    out=ROOT/(kind+'_TRE_ONLY')
    states=[pickle.loads((out/f'r{r}f{f}.pkl').read_bytes())[1] for r in range(2) for f in range(5)]
    pairs=list(itertools.combinations(range(10),2))
    if kind=='pc':
        # Missing reference entries are never recoded to zero.
        df=pd.DataFrame([s['log_weights'] for s in states])
        cols=states[0]['clr']['source_features']
        df=df.reindex(columns=cols)
        df.to_csv(out/'aligned_log_weights.csv',index_label='outer_split')
        statuses=[]
        for s in states:
            ref=np.asarray(s['clr']['reference_indices'])
            kept=set(ref[s['output_indices']])
            statuses.append(['output' if j in kept else 'reference_only' if j in ref else 'prevalence_filtered' for j in range(len(cols))])
        pd.DataFrame(statuses,columns=cols).to_csv(out/'taxon_status.csv',index=False)
        tops=[set(row.dropna().abs().nlargest(10).index) for _,row in df.iterrows()]
        overlap=[];ranks=[];directions=[]
        for i,j in pairs:
            x,y=df.iloc[i].align(df.iloc[j])
            both=x.notna() & y.notna()
            ranks.append(spearmanr(x[both].abs(),y[both].abs()).statistic)
            common=tops[i]&tops[j]
            directions.extend(np.sign(x[list(common)])==np.sign(y[list(common)]))
            overlap.append(jaccard(tops[i],tops[j]))
        core=set.intersection(*tops)
        err=max(t['reconstruction_error'] for t in states)
        text=(f'Max reconstruction error {err:.3g} mg/dL. Top-10 median pair Jaccard {np.median(overlap):.3f} '
              f'(range {min(overlap):.3f} to {max(overlap):.3f}); {len(core)} taxa top ten in all splits. '
              f'Median absolute-weight rank correlation {np.median(ranks):.3f}; shared top-ten pair sign agreement '
              f'{np.mean(directions):.3f}. Private files: aligned_log_weights.csv, taxon_status.csv, r*f*.pkl.')
    else:
        pools=[[set(m) for m in t['members']] for t in states]
        counts=[len(p) for p in pools]; matches=[]
        for i,pool in enumerate(pools):
            for j,m in enumerate(pool):
                sims=[];gains=[]
                for k,p in enumerate(pools):
                    if k==i:
                        sims.append(1.);gains.append(states[i]['ablation_gain'][j]);continue
                    vals=[jaccard(m,v) for v in p]
                    jj=int(np.argmax(vals)) if vals else None
                    sims.append(vals[jj] if vals else 0.)
                    gains.append(states[k]['ablation_gain'][jj] if vals and vals[jj]>=.6 else None)
                stable=sum(v>=.6 for v in sims)>=9
                byround=[[g for g in gains[q:q+5] if g is not None] for q in (0,5)]
                pos=all(len(a)>=4 and np.mean(a)>0 for a in byround)
                matches.append(dict(split=i,members=sorted(m),stable=stable,positive=pos,
                                    jaccard=sims,ablation_gain=gains))
        (out/'aligned_modules.json').write_text(json.dumps(matches))
        rs=[json.loads((out/f'round{r}.json').read_text()) for r in range(2)]
        bs=[json.loads((ROOT/'pc_TRE_ONLY'/f'round{r}.json').read_text()) for r in range(2)]
        gates=[z['delta_R2']>0 and sum(v>0 for v in z['fold_MSE_gain'])>=3 and z['min_support']>=10 and z['all_marginal'] for z in rs]
        passed=[m for m in matches if m['stable'] and m['positive'] and all(gates)]
        ratios=[z['delta_R2']/b['delta_R2'] if b['delta_R2']>0 else None for z,b in zip(rs,bs)]
        text=(f'Modules per split {min(counts)}..{max(counts)}. Stable split-module occurrences '
              f'{sum(m["stable"] for m in matches)}/{len(matches)}; retention of B increment {ratios}; '
              f'model gates {gates}; all-gate occurrences {len(passed)}. See aligned_modules.json. '
              'Ablation holds the fitted background and other coefficients fixed; correlated members need not be individually necessary.')
    record(kind+' interpretation',text);print(text,flush=True)

if __name__=='__main__':main(sys.argv[1])
