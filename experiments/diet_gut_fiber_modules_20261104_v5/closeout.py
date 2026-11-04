"""TRE-only final readback and date-window audit. Reads no upstream outcome values."""
import hashlib
import io
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from audit_v4 import ROOT, V4, record


def main():
    pkg=ROOT.parents[1]
    x=pq.ParquetFile(pkg/'inputs/discovery_v2/discovery_covariates_food_TRE_ONLY.parquet').read(
        columns=['participant_id','cohort','research_stage','input_row','final_role']).to_pandas()
    assert x.final_role.eq('discovery').all()
    stages=[]
    with np.load(io.BytesIO((V4/'results/TRE_ONLY/final_predictions.npz').read_bytes())) as old:
        for r in range(2):
            frames={}
            for kind in ('pc','module'):
                out=ROOT/(kind+'_TRE_ONLY')
                parts=[pickle.loads((out/f'r{r}f{i}.pkl').read_bytes())[0] for i in range(5)]
                frames[kind]=pd.concat(parts).sort_values('row').reset_index(drop=True)
                d=frames[kind]
                rows=d.row.to_numpy()
                np.testing.assert_array_equal(rows,np.flatnonzero(np.isfinite(old['y'][r])))
                np.testing.assert_array_equal(d.y,old['y'][r,rows])
                np.testing.assert_array_equal(d.fold,old['outer'][r,rows])
                gain=(np.mean((d.null-d.y)**2)-np.mean((d.pred-d.y)**2))/np.var(d.y)
                summary=json.loads((out/f'round{r}.json').read_text())
                np.testing.assert_allclose(gain,summary['delta_R2'])
                assert summary['all_marginal']
                stages.append(f'Round {r+1} {kind}: delta R2 {gain:+.6f}, fold directions '+
                              ''.join('+' if e>0 else '-' for e in summary['fold_MSE_gain']))
            np.testing.assert_allclose(frames['pc'].null,frames['module'].null,atol=1e-10,rtol=0)
            np.testing.assert_array_equal(frames['pc'].family,frames['module'].family)
        mask=np.isfinite(old['y'])
        stages.append(f'Eligible sets shared {np.sum(mask[0]&mask[1])} of {mask.sum(1).tolist()}.')
        eligible=x.loc[mask[0]&mask[1]].copy()
    keys=['participant_id','cohort','research_stage']
    cols=['food_first_date','food_last_date','Y_first_date','Y_last_date','G_sample_date','O_sample_date']
    path=pkg.parent/'analysis/D_Glu/fiber_density/base_table_TRE_ONLY.parquet'
    # Only visit metadata from the discovery people used in paired evaluation.
    dates=pq.read_table(path,columns=keys+cols,filters=[('participant_id','in',eligible.participant_id.tolist()),
                       ('cohort','=',eligible.cohort.unique().item()),
                       ('research_stage','=',eligible.research_stage.unique().item())],pre_buffer=False).to_pandas()
    dates=eligible[keys].merge(dates,on=keys,how='left',validate='one_to_one')
    dates[cols]=dates[cols].apply(pd.to_datetime,errors='coerce')
    known=dates[cols[:4]].notna().all(1)
    overlap=(dates.food_first_date<=dates.Y_last_date)&(dates.Y_first_date<=dates.food_last_date)
    timing=f'Date envelopes overlap for {int((known&overlap).sum())}/{int(known.sum())} jointly eligible with windows known (envelopes do not prove same recorded day).'
    for col in ('G_sample_date','O_sample_date'):
        seen=dates[[col,'Y_first_date','Y_last_date']].notna().all(1)
        relative=[int(((dates[col]<dates.Y_first_date)&seen).sum()),
                  int((dates[col].between(dates.Y_first_date,dates.Y_last_date)&seen).sum()),
                  int(((dates[col]>dates.Y_last_date)&seen).sum())]
        timing+=f' {col} before/within/after CGM window {relative}, known {seen.sum()}.'
    record('Phenotype and timing', 'W14_operational_contract.json and W14 candidate mapping: D_Glu is glucose_visit_equal_day_mean, mg/dL. Same baseline visit; qualified calendar days weighted equally; no required food overlap. '+timing)
    weights=pd.read_csv(ROOT/'pc_TRE_ONLY/aligned_log_weights.csv',index_col=0)
    common=set.intersection(*[set(v.dropna().abs().nlargest(10).index) for _,v in weights.iterrows()])
    stages.append('Common top-ten IDs: '+str(sorted(common))+'. Not a single-taxon validation.')
    stages+=['Retain the continuous gut PC10 score; the module gate failed at predictive transfer despite many recurring member sets. No representative-taxon fit or oral/joint module screen. Confirmation remains sealed.',
             'Restricted outputs: paired_v4_TRE_ONLY.json, pc_TRE_ONLY/, module_TRE_ONLY/, frozen_pc_TRE_ONLY/rule.json; upstream sample dates are descriptive, exact collection times and ingestion order are unknown.']
    digest=hashlib.sha256((ROOT/'frozen_pc_TRE_ONLY/rule.json').read_bytes()).hexdigest()
    stages.append('Frozen score SHA256 '+digest+'. Source version '+sys.argv[1]+'.')
    record('Final audit and decision','\n'.join(stages))
    print(timing+'\n'+'\n'.join(stages[:5])+'\nCLOSEOUT_PASS',flush=True)

if __name__=='__main__':main()
