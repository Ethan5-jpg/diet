"""Fit the designated PC score once, using only discovery. No confirmation loader."""
import io
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from run import (ROOT, V4, inputs, pcs, backmap, fit_split, grid, v4, Background,
                 record, config)


from apply_score import score_new


def main():
    data=inputs();x,raw,taxa,yraw=data
    out=ROOT/'frozen_pc_TRE_ONLY';out.mkdir(exist_ok=False)
    # Refit penalty on pooled development using household CV, never used for reported metrics.
    losses=[]
    for f in range(5):
        val=x.cv_fold.to_numpy()==f
        t=fit_split(data,~val,val,'pc',grid);va=t['va']
        losses.append([np.mean((t['y'][va]-t['fits'][p]['prediction'][va])**2) for p in grid])
    penalty=grid[np.argmin(np.mean(losses,axis=0))]
    z={};pc={};states={};pca={}
    for site in ('oral','gut'):
        m=v4.MicrobiomeCLR().fit(raw[site],taxa[site],x.participant_id.tolist(),site=site)
        t=m.transform(raw[site],taxa[site],site=site)
        z[site]=t['values']; states[site]=m.state
        pc[site],pca[site]=pcs(z[site],t['supported'])
    fiber=x.fiber_density.to_numpy(float)
    good=x.n_fiber_days.ge(3).to_numpy() & np.isfinite(fiber)
    quant=np.quantile(fiber[good],[.1,.5,.9]);u=(fiber-quant[1])/5;h=v4.rcs(u,(quant-quant[1])/5)
    ok=np.isfinite(yraw)
    phen=v4.PhenotypeOutliers().fit(yraw[ok],np.flatnonzero(ok).tolist(),endpoint='D_Glu')
    y=phen.transform(yraw,endpoint='D_Glu')['values']
    tr=good & np.isfinite(y) & np.isfinite(x.age_calendar_years.to_numpy(float)) & x.sex_code.isin([0,1]).to_numpy() & np.isfinite(x.mean_daily_energy_fiber.to_numpy(float))
    for a in z.values():tr &= np.isfinite(a).all(1)
    host,_=v4.host_design(x,tr,u,h)
    bg=Background(np.column_stack([host,pc['oral'][:,:5],pc['gut'][:,:5]]),
                  np.column_stack([z['oral'],z['gut']]),y,tr)
    coef=bg.interaction(u[:,None]*pc['gut']).ridge(penalty)['interaction_coef']
    center=np.mean(pc['gut'][tr]@coef)
    ref,b,c=backmap(states['gut'],pca['gut'],coef,center)
    rule=dict(version=1,scope='discovery-trained candidate for confirmation',target='D_Glu',
              contrast=config['contrast'],unit='mg/dL',input='same gut abundance units and taxon identifiers as discovery',
              species=list(np.array(taxa['gut'])[ref]),zero_replacement=states['gut']['pseudocounts'],
              weights=b.tolist(),intercept=float(c),lambda_interaction=penalty)
    path=out/'rule.json';path.write_text(json.dumps(rule,allow_nan=False))
    read=json.loads(path.read_text());test=score_new(raw['gut'],taxa['gut'],read)
    np.testing.assert_allclose(test[tr],(pc['gut']@coef-center)[tr],rtol=0,atol=1e-8)
    np.testing.assert_allclose(score_new(raw['gut'][:,::-1],taxa['gut'][::-1],read)[tr],test[tr],rtol=0,atol=0)
    assert np.isnan(score_new(np.zeros(len(taxa['gut'])),taxa['gut'],read)).all()
    import hashlib
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    record('Candidate frozen',f'Gut PC10 score frozen in {path.relative_to(ROOT)}, SHA256 {digest}. '
           f'Full discovery refit n={tr.sum()}, lambda={penalty}. Raw-score readback, column reordering and unsupported sample tests passed. '
           'This is a centered modification score. Positive values denote a larger modeled change relative to the training mean. '
           'It is not an intervention estimate. Confirmation outcomes unopened.')
    print('FROZEN_PC_RULE_PASS',digest,flush=True)

if __name__=='__main__':main()
