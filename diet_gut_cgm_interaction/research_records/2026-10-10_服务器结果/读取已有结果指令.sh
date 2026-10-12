conda activate pheno && python - <<'PY'
from pathlib import Path
import json
import pandas as pd

run = Path('/home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction/outputs/run_20261010T085601_096554Z_4d43e0')
meta = json.loads((run / 'manifest.json').read_text())
r = pd.read_csv(run / 'tables/all_interactions.csv')
s = pd.read_csv(run / 'tables/conditional_diet_slopes.csv')
fmt = lambda x: '{:.4g}'.format(x)

print('FOLLOWUP 1/2 | NOVA4 x T068 |', meta['data_kind'], '|', meta['version'])
t = r[r.exposure.eq('NOVA4') & r.taxon.eq('T068') & r.outcome.isin(['CV', 'SDw'])]
cols = ['model', 'outcome', 'n', 'interaction_beta_native', 'interaction_ci_low_native', 'interaction_ci_high_native', 'q_focus']
names = {'interaction_beta_native': 'Beta', 'interaction_ci_low_native': 'L95', 'interaction_ci_high_native': 'U95', 'q_focus': 'qFocus'}
print(t[cols].rename(columns=names).to_string(index=False, float_format=fmt))
print('\nPRIMARY DIET SLOPES at microbial Q25/Q50/Q75; native units per Diet Z')
u = s[s.model.eq('primary') & s.exposure.eq('NOVA4') & s.taxon.eq('T068') & s.outcome.isin(['CV', 'SDw'])]
print(u[['outcome', 'microbial_quantile', 'beta_native', 'ci_low_native', 'ci_high_native']].rename(columns={'microbial_quantile': 'Q', 'beta_native': 'Slope', 'ci_low_native': 'L95', 'ci_high_native': 'U95'}).to_string(index=False, float_format=fmt))
expected = {('NOVA4', 'T068', 'CV'), ('NOVA4', 'T068', 'SDw')}
groups = r[(r.model != 'primary') & r.family.eq('focus')].groupby('model')
same = len(groups) == 6 and all(set(zip(g.loc[g.q_focus.lt(.05), 'exposure'], g.loc[g.q_focus.lt(.05), 'taxon'], g.loc[g.q_focus.lt(.05), 'outcome'])) == expected for _, g in groups)
print('Same two significant targets in every sensitivity:', same)

print('\nFOLLOWUP 2/2 | EXTENSION PRIMARY qAll<0.05 |', meta['data_kind'])
v = r[r.model.eq('primary') & r.family.eq('extension') & r.q_all.lt(.05)].sort_values(['exposure', 'taxon', 'outcome'])
cols = ['exposure', 'taxon', 'outcome', 'n', 'interaction_beta_native', 'interaction_ci_low_native', 'interaction_ci_high_native', 'q_all']
names['q_all'] = 'qAll'
print(v[cols].rename(columns=names).to_string(index=False, float_format=fmt))
print('Extension significant tests:', len(v))
PY
