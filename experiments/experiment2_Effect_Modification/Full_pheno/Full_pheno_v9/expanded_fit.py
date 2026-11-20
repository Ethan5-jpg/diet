"""Prediction arms with verified additional diet features and a retained v8 control."""

import numpy as np

import protocol as p
from breadth import basis, fit_profile
from diet_audit import Fit, ridge_diet
from diet_inventory import encode_blocks
from exposure_models import ANCHOR, inputs
from feature_policies import BLOCKS, FIXED, extend, pick
from interaction_fit import Repartition
from models import create_design, dose_basis, predict, products
from run_precision import features
from source import Outcome, check
from v3_inputs import InnerContexts, read_sites
from v4_models import no_diet_main


RETAINED = ("fiber_density","D_Liver_viscosity")
PROTECTED = {ANCHOR, RETAINED}


def gut_update(gut, main, residual):
    n = len(residual)
    inv = np.linalg.pinv(main[:n],rcond=p.RANK_TOL)
    pc5 = fit_profile(gut,main,inv,residual).fit(1.)
    pc2 = fit_profile(gut[:,:2],main,inv,residual).fit(1.)
    return 1.37*(.45*pc5+.35*pc2)


def ridge_projected(z, residual, penalty):
    n = len(residual)
    training = z[:n]
    gram = training.T@training/n
    rhs = training.T@residual/n
    return z@np.linalg.solve(gram+penalty*np.eye(z.shape[1]),rhs)


def fit_endpoint(study,x,e,y,fold,raw,views,precision,panels):
    clean,tr,held = inputs(study,x,e,y,fold)
    check(not np.intersect1d(study.family[tr],study.family[held]).size,"V9_FAMILY_OVERLAP")
    rows,n = np.r_[tr,held],len(tr)
    check(n>=30 and len(held)>=10,"V9_INSUFFICIENT_FOLD_SUPPORT")
    target = clean[tr]
    d = create_design(study,x,fold,rows,n)
    dose = dose_basis(raw[rows],n)[0]
    base = no_diet_main(study,rows,n,d.main,dose)
    base_inv = np.linalg.pinv(base[:n],rcond=p.RANK_TOL)
    nodiet = base[n:]@(base_inv@target)
    if (x,e)==ANCHOR:
        q = predict(d,target)
        return Fit(held,clean[held],nodiet,np.repeat(q[:,:1],len(FIXED),axis=1),
                   np.repeat(q[:,2,None],len(FIXED),axis=1))

    new = precision[rows,None]
    old_terms = np.column_stack((dose,products(dose,d.main[:,1:3])))
    old,old_projection = ridge_diet(base,base_inv,old_terms,target,(.1,.5,1.,1.,1.,1.))
    linear,new_projection = ridge_diet(base,base_inv,new,target,(.1,))
    off = .5*(old+linear)
    main = np.column_stack((base,old_projection,new_projection))
    inv = np.linalg.pinv(main[:n],rcond=p.RANK_TOL)
    gut = products(new,views["gut_pc5"][rows])
    ref = off[n:]+gut_update(gut,main,target-off[:n])
    gut_base = gut-main@(inv@gut[:n])
    b,c,extra = [off[n:]],[ref],None
    for key in (*BLOCKS,"all"):
        z = panels[key][rows]
        if z.shape[1]==0:
            b.append(off[n:]); c.append(ref)
            continue
        z = z-main@(inv@z[:n])
        z_inv = np.linalg.pinv(z[:n],rcond=p.RANK_TOL)
        gut_z = gut_base-z@(z_inv@gut_base[:n])
        correction = ridge_projected(z,target-off[:n],1.0 if key=="all" else .50)
        off2 = off+correction
        residual = target-off2[:n]
        pc5 = ridge_projected(gut_z,residual,1.)[n:]
        pc2 = ridge_projected(gut_z[:,:2],residual,1.)[n:]
        on2 = off2[n:]+1.37*(.45*pc5+.35*pc2)
        b.append(off2[n:]); c.append(on2)
        if key=="all":
            # Only credit the new interactions beyond the retained gut direction.
            raw_extra = products(panels[key][rows],views["gut_pc5"][rows,:2])
            adjusted_extra = raw_extra-main@(inv@raw_extra[:n])
            adjusted_extra -= z@(z_inv@adjusted_extra[:n])
            gut_inv = np.linalg.pinv(gut_z[:n],rcond=p.RANK_TOL)
            adjusted_extra -= gut_z@(gut_inv@adjusted_extra[:n])
            extra = .35*ridge_projected(adjusted_extra,residual,1.)[n:]
    b.append(b[-1]); c.append(c[-1] if extra is None else c[-1]+extra)
    b,c = np.column_stack(b),np.column_stack(c)
    check(np.isfinite(b).all() and np.isfinite(c).all(),"V9_NONFINITE_PREDICTIONS")
    return Fit(held,clean[held],nodiet,b,c)


def fit_split(study,fold,x,outcomes,days,values):
    bg = study.backgrounds[fold]
    raw,views = basis(study,x,fold)
    precise,_ = features(x,bg,raw[:,0],days)
    panels,support = encode_blocks(values,bg.role&bg.supported)
    records = {e:fit_endpoint(study,x,e,v,fold,raw,views,precise["person_clipped"],panels) for e,v in outcomes.items()}
    return records,support


def run_partition(study,outcomes,days,values,calibration,history,seed,progress,adaptive):
    sites = read_sites(study) if adaptive or seed is not None else None
    working = study if seed is None else Repartition(study,sites,calibration,seed,progress)
    nested = InnerContexts(working,sites) if adaptive else None
    done = {pair:[] for pair in p.MODES["all"]}
    support,decisions = [],[]
    for outer in p.FOLDS:
        choices = {}
        if adaptive:
            masked = {e:Outcome(np.where(working.fold!=outer,v.y,np.nan),v.unit) for e,v in outcomes.items()}
            inner = {pair:[] for pair in done}
            for val in p.FOLDS:
                if val==outer:
                    continue
                ctx = nested.split(outer,val)
                for x in p.EXPOSURES:
                    if x=="UPF":
                        ctx,_ = calibration.inner(ctx,val,100*(seed or p.BOOTSTRAP_SEED)+10*outer+val)
                    fits,_ = fit_split(ctx,val,x,masked,days,values)
                    for e,f in fits.items():
                        inner[x,e].append(f)
            choices = {k:pick(v) for k,v in inner.items() if k not in PROTECTED}
        for x in p.EXPOSURES:
            fitted,admitted = fit_split(working,outer,x,outcomes,days,values)
            support.append(dict(fold=outer,exposure=x,blocks=admitted))
            for e,result in fitted.items():
                key = x,e
                j,stop = choices.get(key,(0,"keep"))
                h = "keep" if key in PROTECTED else history.get(key,"keep")
                done[key].append(extend(result,j,stop,h))
                decisions.append(dict(fold=outer,exposure=x,endpoint=e,selected=FIXED[j],gate=stop,historical=h))
        progress("V9_FOLD_DONE",dict(seed=seed,fold=outer,groups=len(done)))
    return done,support,decisions
