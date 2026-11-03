"""Read saved aggregate HPP model tables; never refit or read participant tables."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path('/home/ec2-user/Desktop/HPP/Data')
DG = ROOT / 'diet_microbiome_glucose_analysis'
O = DG / '4New_cgm/outputs'
FOLLOW = DG / '6Step20_23_followup/outputs/followup_20260920T044755_767120Z'
sources = []

def read(p):
    p = Path(p)
    if p.stat().st_size > 80_000_000:
        raise ValueError('Aggregate table exceeds audit memory budget: ' + str(p))
    d = pd.read_csv(p)
    sources.append(dict(path=str(p), rows=len(d), bytes=p.stat().st_size,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return d

def flag(s):
    if s.isna().any():
        raise ValueError('Missing audit flag: ' + str(s.name))
    x = s.astype(str).str.lower()
    if not x.isin(['true', 'false', '1', '0', '1.0', '0.0']).all():
        raise ValueError('Unexpected flag encoding: ' + str(s.name))
    return x.isin(['true', '1', '1.0'])

def fmt(x):
    return format(float(x), '.7g')

assert '/envs/pheno/' in sys.executable, sys.executable
F = read(O / 'paper20_mediation_threeway_robustness/reports/18f_threeway_path_robustness.csv')
M = read(FOLLOW / 'tables/21_all1490_three_model_results.csv')
K = ['exposure', 'outcome_field', 'species']
assert not F.duplicated(K).any()
assert not M.duplicated(['model'] + K).any()
assert set(M.model) == {'primary', 'strict', 'protocol'}, set(M.model)
r = {t: flag(F['robust_' + t]) for t in ['primary', 'strict', 'protocol']}
for t in r:
    assert (r[t] == (flag(F['significant_' + t]) & flag(F['consistent_' + t]))).all()
keep = r['primary'] & r['strict'] & r['protocol']
kept = F.loc[keep].set_index(K)
Q = M.pivot(index=K, columns='model', values='FDR_BH_global_all_mediation')
assert not Q.isna().any().any()
global3 = Q.reindex(kept.index).lt(.05).all(axis=1)
sections = {}
s = ['CORE: saved aggregate tables, no refit or FDR recalculation',
     'PYTHON ' + sys.executable,
     f'CANDIDATES {len(F)}; MODEL_ROWS {len(M)}; DUPLICATE_KEYS 0']
for t in r:
    s.append(f'{t}: positive_direction_consistent={int(r[t].sum())}')
s.append(f'PRIMARY retained strict={int((r["primary"] & r["strict"]).sum())}; protocol={int((r["primary"] & r["protocol"]).sum())}')
s.append(f'THREEWAY {len(kept)} paths; {kept.reset_index().species.nunique()} exact species')
s.append(f'GLOBAL_FDR_ALL3 within frozen THREEWAY: {int(global3.sum())}/{len(kept)}')
s.append('DIET primary_positive -> threeway_retained')
for e in sorted(F.exposure.unique()):
    z = F.exposure.eq(e)
    s.append(f'{e:18} {int((z & r["primary"]).sum()):4} -> {int((z & keep).sum()):4}')
delta = (M.a_diet_to_microbiome * M.b_microbiome_to_cgm - M.indirect_effect).abs()
s.append('MAX abs(a*b-indirect) ' + fmt(delta.max()))
assert delta.max() < 1e-8
n_bad = 0
for t in r:
    z = M.loc[M.model.eq(t)].set_index(K).reindex(F.set_index(K).index)
    n_bad += int((z.N.to_numpy() != F['N_' + t].to_numpy()).sum())
s.append(f'N_RECORD_MISMATCHES {n_bad}/{len(M)} (saved records, not new cohort reconstruction)')
assert n_bad == 0
P = M.loc[M.model.eq('primary')].set_index(K).reindex(kept.index)
s.append(f'RETAINED primary total_abs<=.01: {int(P.total_effect.abs().le(.01).sum())}; proportion>1: {int(P.mediated_proportion.gt(1).sum())}')
sections['core'] = s

s = ['PATHS: primary estimates; a,b,I are original standardized model units',
     'diet outcome N a b indirect [CI95] within_q global_q']
for name, diets, outcomes in [('Eubacterium_rectale', ['hPDI','NOVA4'], ['cgm_cv','cgm_modd','cgm_sdw']),
                             ('Ruthenibacterium_lactatiformans', ['AHEI','AMED'], ['cgm_mean'])]:
    z = P.reset_index()
    z = z[z.species.str.contains(name, regex=False) & z.exposure.isin(diets) & z.outcome_field.isin(outcomes)]
    assert z.species.nunique() == 1, (name, z.species.nunique())
    s.append(name)
    for row in z.sort_values(['exposure','outcome_field']).itertuples():
        s.append(f'{row.exposure} {row.outcome_field} N={int(row.N)} a={fmt(row.a_diet_to_microbiome)} b={fmt(row.b_microbiome_to_cgm)}')
        s.append(f' I={fmt(row.indirect_effect)} [{fmt(row.indirect_CI95_lower)},{fmt(row.indirect_CI95_upper)}] q={fmt(row.FDR_BH_within_score_outcome)} qG={fmt(row.FDR_BH_global_all_mediation)}')
    keys = z.set_index(K).index
    s.append(f'ALL listed paths global-threeway passed: {int(global3.reindex(keys).sum())}/{len(keys)}')
sections['paths'] = s

s = ['LEGS: recount original full model records']
D = read(O / 'paper20_diet_cgm_20260919T034331Z/models/15_paper20_diet_cgm_all_models.csv')
d = D.loc[D.analysis_set.eq('primary') & D.model.eq(3)]
s.append(f'DIET_CGM rows={len(D)}; primary_M3={len(d)}; role_FDR_positive={int(flag(d.significant_role_05).sum())}')
G = read(O / 'paper20_direct_gate_20260919T035650Z/reports/15b_all140_context_gate_audit.csv')
s.append(f'DIRECT_GATE contexts={len(G)}; bridge_eligible={int(flag(G.bridge_context_eligible).sum())}')
W = read(O / 'paper20_microbiome_mwas_20260919T040500Z/models/16_paper20_microbiome_all_models.csv')
w = W[W.set_name.eq('primary_m2')]
s.append(f'GUT_CGM rows={len(W)}; computed={int(W.status.eq("computed").sum())}; primary_M2={len(w)}; family_FDR_positive={int(flag(w.significant_family_05).sum())}')
s.append('CURRENT four-score intersection: uses corrected hPDI, not seven-score intersection')
for model, prefix in [(2,'04'), (3,'05')]:
    positive, negative = [], []
    for score in ['AHEI','AMED','hPDI','rEDIH']:
        if score == 'hPDI':
            p = DG / f'outputs/reports/new_diet_extension/hpdi_correction_audit/downstream/microbiome/15b3_corrected_hPDI_MWAS_model{model}.csv'
        else:
            p = ROOT / f'diet_microbiome_analysis/reports/{prefix}_MWAS_{score}_model{model}.csv'
        a = read(p)
        assert len(a) == 379 and not a.species.duplicated().any()
        good = a.FDR.lt(.05)
        positive.append(set(a.loc[good & a.beta.gt(0), 'species']))
        negative.append(set(a.loc[good & a.beta.lt(0), 'species']))
        s.append(f'M{model} {score:6} significant={int(good.sum())}; positive={int((good & a.beta.gt(0)).sum())}; negative={int((good & a.beta.lt(0)).sum())}')
    shared = set.intersection(*positive) | set.intersection(*negative)
    s.append(f'M{model} shared4 same-direction={len(shared)}')
sections['legs'] = s

out = DG / 'audits' / ('saved_results_check_' + datetime.now().strftime('%Y%m%dT%H%M%SZ'))
out.mkdir(parents=True, exist_ok=False)
report = dict(status='PASS_SAVED_TABLE_CHECKS', python=sys.executable,
              scope='Saved aggregate records only; no raw participant data, refit, bootstrap or FDR recalculation',
              sources=sources, sections=sections)
(out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
for key, lines in sections.items():
    (out / (key + '.txt')).write_text('\n'.join(lines) + '\n')
for i, (key, lines) in enumerate(sections.items()):
    print('\n'.join(lines), flush=True)
    if i < len(sections)-1:
        input('NEXT_SECTION: press Enter (audit files already saved)')
print('AUDIT_DIR=' + str(out), flush=True)
print('AUDIT_COMPLETE; source tables unchanged', flush=True)
