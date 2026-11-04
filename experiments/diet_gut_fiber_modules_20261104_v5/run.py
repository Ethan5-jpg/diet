"""Nested discovery experiment. Run PC stage first, then module stage."""
import os
for var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[var]='1'
import sys
import hashlib
import io
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from audit_v4 import ROOT, V4, record
from features import pcs, backmap, modules

pkg=ROOT.parents[1]
v1=ROOT.parent/'diet_response_heterogeneity_20261103_v1_attempt02'
v3=ROOT.parent/'diet_representation_screen_20261103_v3'
sys.path.insert(0,str(V4))
import run_nested as v4
v4.dependencies(pkg,v1/'code',v3)
from core import Background, cluster_fit

config=json.loads((ROOT/'config.json').read_text())
grid=config['grid']


def inputs():
    data=pkg/'inputs/discovery_v2'
    x=v4.read_table(data/'discovery_covariates_food_TRE_ONLY.parquet')
    assert len(x)==4427 and x.final_role.eq('discovery').all() and x.participant_id.is_unique
    np.testing.assert_array_equal(x.input_row,np.arange(len(x)))
    assert x.split_unit_id.notna().all() and x.groupby('split_unit_id').cv_fold.nunique().max()==1
    yd=v4.read_table(data/'D_Glu_raw_Y_TRE_ONLY.parquet')
    rows=yd.input_row.to_numpy(int)
    assert len(set(rows))==len(rows) and yd.Y_unit.dropna().unique().tolist()==['mg/dL']
    for k in ('participant_id','cohort','research_stage'):
        if k in yd: np.testing.assert_array_equal(yd[k],x.iloc[rows][k])
    y=np.full(len(x),np.nan); y[rows]=yd.Y_value
    taxa,raw={},{}
    for s in ('oral','gut'):
        taxa[s]=json.loads((data/(s+'_taxa.json')).read_text())
        with np.load(io.BytesIO((data/(s+'_raw_TRE_ONLY.npz')).read_bytes())) as a:
            raw[s]=a['abundance']
        assert raw[s].shape==(len(x),len(taxa[s]))
    return x,raw,taxa,y


def fit_split(data,train,valid,kind,penalties):
    x,raw,taxa,yraw=data
    d,y,tr,va,state=v4.context(x,raw,taxa,yraw,train,valid)
    host=d['host']; zo=d['oral'][:,host.shape[1]:]; zg=d['oral_gut'][:,d['oral'].shape[1]:]
    po,_=pcs(zo,train); pg,pca=pcs(zg,train)
    bg=Background(np.column_stack([host,po[:,:5],pg[:,:5]]),np.column_stack([zo,zg]),y,tr)
    null=bg.null()['prediction']
    st=state['CLR']['gut']; diet=state['diet']
    u=(x.fiber_density.to_numpy()-diet['center'])/diet['scale']
    h=v4.rcs(u,diet['knots'])
    members=[]
    if kind=='pc': scores=pg
    else: scores,members=modules(zg,st,train)
    inter=bg.interaction(u[:,None]*scores) if scores.shape[1] else None
    fits={p:inter.ridge(p) if inter else dict(prediction=null,interaction_coef=np.empty(0)) for p in penalties}
    return locals()


def save_fold(out,r,f,kind,p,loss,t,old):
    y,null,va,tr,u,h,scores=[t[k] for k in ('y','null','va','tr','u','h','scores')]
    x,raw,taxa,_=t['data']
    fit=t['fits'][p]; pred=fit['prediction']; coef=fit['interaction_coef']
    # Fixed contrast delta u = 1. Train center is removed for modification dispersion.
    delta=(config['contrast'][1]-config['contrast'][0])/t['diet']['scale']
    linear=scores@coef*delta
    center=linear[tr].mean()
    axis=linear-center; sd=axis[tr].std()
    g=x.split_unit_id.to_numpy()
    if r==0:
        np.testing.assert_allclose(null[va],old.matched_null_prediction.to_numpy()[va],rtol=0,atol=2e-8)
        if kind=='pc':
            np.testing.assert_allclose(t['fits'][0.1]['prediction'][va],old.ridge_PC10_gut_prediction.to_numpy()[va],rtol=0,atol=2e-8)
    # Out-of-fold predictions are compared only within the identical v4 target/role.
    st=t['st']; foldinfo=dict(penalty=float(p),inner_grid_mean_losses=loss.tolist(),clr=st,
        fiber=t['diet'],training_score_center=float(center),training_score_SD=float(sd),coef=coef)
    if kind=='pc':
        ref,b,intercept=backmap(st,t['pca'],coef*delta,center)
        ar=raw['gut'][:,ref]; logs=np.log(np.where(ar==0,st['pseudocounts'],ar))
        err=float(np.max(abs((logs@b+intercept-axis)[tr|va])))
        assert err<1e-8
        ids=np.array(taxa['gut'])[ref]
        foldinfo.update(log_weights=dict(zip(ids,b)),score_intercept=intercept,pca=t['pca'],
                        reconstruction_error=err,output_indices=st['output_indices'])
    else:
        ids=np.array(taxa['gut'])[st['reference_indices']][st['output_indices']]
        foldinfo['members']=[list(ids[m]) for m in t['members']]
        foldinfo['ablation_gain']=[]
        for j,c in enumerate(coef):
            partial=u[va]*scores[va,j]*c
            err=y[va]-pred[va]
            foldinfo['ablation_gain'].append(float(np.mean((err+partial)**2-err**2)))
    quant=np.quantile(axis[tr],[.25,.5,.75])
    strata=np.searchsorted(quant,axis[va])
    fiber=x.fiber_density.to_numpy()
    support=[[len(set(g[va][(strata==j)&(abs(fiber[va]-a)<=1)])) for a in config['contrast']] for j in range(4)]
    foldinfo.update(support_households=support,contrast_in_marginal_support=bool(np.quantile(fiber[tr],.05)<=10 and np.quantile(fiber[tr],.95)>=15))
    frame=pd.DataFrame(dict(row=x.input_row[va],fold=f,family=g[va],y=y[va],null=null[va],pred=pred[va],
        u=u[va],h=h[va],axis=axis[va],score=axis[va]/sd if sd>1e-12 else np.nan))
    with (out/f'r{r}f{f}.pkl').open('xb') as stream:
        pickle.dump((frame,foldinfo),stream)
    record(f'{kind} round {r+1} fold {f+1}',f'n={len(frame)}; lambda={p}; MSE improvement={np.mean((frame.y-frame.null)**2-(frame.y-frame.pred)**2):.6f}; checkpoint {out.name}/r{r}f{f}.pkl.')


def summary(out,r):
    parts=[pickle.loads((out/f'r{r}f{f}.pkl').read_bytes()) for f in range(5)]
    df=pd.concat([x for x,_ in parts]); assert df.row.is_unique
    dy=(df.y-df.null)**2-(df.y-df.pred)**2
    direction=dy.groupby(df.fold).mean()
    slope=None
    if np.isfinite(df.score).all():
        design=np.column_stack([np.ones(len(df)),df.u,df.h,df.score,df.u*df.score,np.eye(5)[df.fold][:,1:]])
        b,_,ci=cluster_fit(df.y-df.null,design,df.family)
        slope=[float(b[4]),*ci[4].tolist()]
    result=dict(n=len(df),households=int(df.family.nunique()),delta_R2=float(dy.mean()/df.y.var(ddof=0)),
        mse_gain=float(dy.mean()),MSE_null=float(np.mean((df.y-df.null)**2)),MSE_model=float(np.mean((df.y-df.pred)**2)),
        fold_MSE_gain=direction.tolist(),calibration=slope,contrast_SD=float(df.axis.std(ddof=0)),
        min_support=int(min(np.min(t['support_households']) for _,t in parts)),
        all_marginal=all(t['contrast_in_marginal_support'] for _,t in parts))
    (out/f'round{r}.json').write_text(json.dumps(result,indent=2))
    record(out.name+f' round {r+1} summary',json.dumps(result))
    print(out.name,r+1,'gain',round(result['delta_R2'],6),'MSE_gain',round(result['mse_gain'],4),
          'better',int((direction>0).sum()),'beta',slope,'SD',round(result['contrast_SD'],4),
          'support_min',result['min_support'],flush=True)


def main(kind):
    assert kind in ('pc','module')
    if kind=='module': assert (ROOT/'pc_TRE_ONLY/round1.json').exists()
    out=ROOT/(kind+'_TRE_ONLY');out.mkdir(exist_ok=True)
    reg=json.loads((V4/'results/registration.json').read_text())
    watched=[(Path(k),v) for k,v in reg['source_and_input_sha256'].items()
             if '/inputs/discovery_v2/' in k or Path(k).name in
             ('preprocessing_v1_20261030.py','run_pilot.py','core.py','run_nested.py','run_screen.py')]
    assert len(watched)>=11
    for path,digest in watched:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, path.name
    data=inputs(); x=data[0]; groups=x.split_unit_id.astype(str).to_numpy()
    old=v4.read_table(v1/'results/held_predictions_and_contrasts_TRE_ONLY.parquet')
    assert len(old)==len(x)
    if 'input_row' in old: np.testing.assert_array_equal(old.input_row,x.input_row)
    with np.load(io.BytesIO((V4/'results/TRE_ONLY/final_predictions.npz').read_bytes())) as a:
        outer=a['outer']; yy=a['y']
    for r,folds in enumerate((x.cv_fold.to_numpy(),v4.folds(groups,5,config['seed']))):
        np.testing.assert_array_equal(outer[r],folds)
        for f in range(5):
            if (out/f'r{r}f{f}.pkl').exists(): continue
            train=folds!=f
            inner=np.full(len(x),-1); inner[train]=v4.folds(groups[train],3,config['seed']+100*r+f+1)
            losses=[]
            for k in range(3):
                t=fit_split(data,train&(inner!=k),train&(inner==k),kind,grid)
                va=t['va']
                losses.append([np.mean((t['y'][va]-t['fits'][p]['prediction'][va])**2) for p in grid])
                del t
            loss=np.mean(losses,axis=0); p=grid[np.argmin(loss)]
            t=fit_split(data,train,~train,kind,set([p,.1]) if kind=='pc' and r==0 else [p])
            np.testing.assert_array_equal(t['va'],(~train)&np.isfinite(yy[r]))
            np.testing.assert_array_equal(t['y'][t['va']],yy[r,t['va']])
            save_fold(out,r,f,kind,p,loss,t,old)
            print(kind,r,f,'complete',flush=True)
            del t
        summary(out,r)

if __name__=='__main__': main(sys.argv[1])
