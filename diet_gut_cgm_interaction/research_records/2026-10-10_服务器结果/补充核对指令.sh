conda activate pheno && python - <<'PY'
from pathlib import Path
import hashlib
import json
import pandas as pd

run = Path('/home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/outputs/run_20261010T085601_096554Z_4d43e0')
meta = json.loads((run / 'manifest.json').read_text())
assert meta['data_kind'] == 'SERVER_INPUTS', 'Not the expected real-data run'
assert meta['status'] == 'COMPLETE', 'Run is not COMPLETE'
for rel in ['tables/all_interactions.csv', 'tables/conditional_diet_slopes.csv']:
    actual = hashlib.sha256((run / rel).read_bytes()).hexdigest()
    assert actual == meta['output_sha256'][rel], 'File changed: ' + rel
r = pd.read_csv(run / 'tables/all_interactions.csv')
s = pd.read_csv(run / 'tables/conditional_diet_slopes.csv')
fmt = lambda x: '{:.6g}'.format(x)

print('CHECK 1/2 |', meta['data_kind'], '|', meta['version'], '|', meta['status'])
print('Saved hashes verified: interactions + slopes')
one = r[r.model.eq('protocol_adjusted') & r.exposure.eq('NOVA4') & r.taxon.eq('T068') & r.outcome.eq('CV')]
assert len(one) == 1 and one.iloc[0]['status'] == 'OK', 'Expected one valid disputed row'
print('\nNOVA4 x T068 / protocol_adjusted / CV: verify screenshot transcription')
cols = ['n', 'interaction_beta_native', 'interaction_ci_low_native', 'interaction_ci_high_native', 'q_focus']
names = {'interaction_beta_native': 'Beta', 'interaction_ci_low_native': 'L95', 'interaction_ci_high_native': 'U95', 'q_focus': 'qFocus'}
print(one[cols].rename(columns=names).to_string(index=False, float_format=lambda x: '{:.10g}'.format(x)))
a = one.iloc[0]
gap = a.interaction_beta_native - (a.interaction_ci_low_native + a.interaction_ci_high_native) / 2
print('Beta minus CI midpoint = {:.12g}'.format(gap))

p = r[r.model.eq('primary') & r.exposure.eq('NOVA4') & r.taxon.eq('T068') & r.outcome.isin(['CV', 'SDw'])].sort_values('outcome')
assert len(p) == 2 and p.status.eq('OK').all(), 'Expected both primary targets'
print('\nPRIMARY double-IQR interaction contrast; CV=percentage points; SDw=mg/dL')
cols = ['outcome', 'n', 'double_iqr_beta_native', 'double_iqr_ci_low_native', 'double_iqr_ci_high_native', 'double_iqr_status']
names = {'double_iqr_beta_native': 'Contrast', 'double_iqr_ci_low_native': 'L95', 'double_iqr_ci_high_native': 'U95', 'double_iqr_status': 'IQRstatus'}
print(p[cols].rename(columns=names).to_string(index=False, float_format=fmt))
print('\nPRIMARY support and model diagnostics')
cols = ['outcome', 'support_min_tail_n', 'condition_number', 'max_leverage', 'interaction_delta_r2']
names = {'support_min_tail_n': 'MinTailN', 'condition_number': 'CondNum', 'max_leverage': 'MaxLev', 'interaction_delta_r2': 'DeltaR2'}
print(p[cols].rename(columns=names).to_string(index=False, float_format=fmt))

print('\nCHECK 2/2 | EXTENSION slopes at microbial Q25 and Q75 |', meta['data_kind'])
print('Per original Diet Z; pointwise 95% CI; CV=pp, Mean/SDw=mg/dL')
print('Selected by existing primary qAll<0.05 across 84 tests; no new fits or FDR')
ext = r[r.model.eq('primary') & r.family.eq('extension') & r.status.eq('OK') & r.q_all.lt(.05)].sort_values(['exposure', 'taxon', 'outcome'])
rows = []
for _, a in ext.iterrows():
    t = s[s.model.eq('primary') & s.exposure.eq(a.exposure) & s.taxon.eq(a.taxon) & s.outcome.eq(a.outcome)]
    row = {'Diet': 'Carb_pct' if a.exposure == 'Carbohydrate_pct' else a.exposure, 'Taxon': a.taxon, 'Y': a.outcome, 'N': int(a.n)}
    for q in [25, 75]:
        u = t[t.microbial_quantile.eq(q)]
        assert len(u) == 1, 'Missing or duplicated conditional slope'
        b = u.iloc[0]
        row['Q' + str(q) + ' slope [CI]'] = '{:.3g} [{:.3g},{:.3g}]'.format(b.beta_native, b.ci_low_native, b.ci_high_native)
    rows.append(row)
if rows:
    print(pd.DataFrame(rows).to_string(index=False))
else:
    print('No significant extension tests in this run')
print('Extension rows:', len(rows), '| Carb_pct = Carbohydrate_pct')
PY
