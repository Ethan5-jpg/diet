conda activate pheno && python - <<'PY'
from pathlib import Path
import hashlib
import json
import pandas as pd

base = Path('/home/ec2-user/Desktop/HPP/Data/diet_gut_cgm_interaction')
old = base / 'outputs/run_20261010T085601_096554Z_4d43e0'
new = base / 'extension_sensitivity/outputs_extension_sensitivity/extension_20261010T105930_496220Z_ff8085'
om = json.loads((old / 'manifest.json').read_text())
nm = json.loads((new / 'manifest.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert om['data_kind'] == nm['data_kind'] == 'SERVER_INPUTS'
assert om['status'].startswith('COMPLETE') and nm['status'].startswith('COMPLETE')
assert Path(nm['reference_run']).resolve() == old.resolve()
for name, digest in nm['reference_file_sha256'].items():
    assert sha(old / name) == digest, 'Reference changed: ' + name
for run, meta, name in [(old, om, 'tables/all_interactions.csv'),
                        (new, nm, 'tables/all_interactions.csv'),
                        (new, nm, 'tables/reference_hits_stability.csv')]:
    assert sha(run / name) == meta['output_sha256'][name], 'Saved result changed: ' + str(run / name)

print('CHECK | rEDIH x T096 -> CV | SERVER_INPUTS | saved hashes verified')
for label, run, q in [('OLD_PRIMARY', old, 'q_all'), ('NEW_PRIMARY', new, 'q_all84')]:
    frame = pd.read_csv(run / 'tables/all_interactions.csv')
    row = frame[frame.model.eq('primary') & frame.exposure.eq('rEDIH') & frame.taxon.eq('T096') & frame.outcome.eq('CV')]
    assert len(row) == 1, 'Expected one primary row'
    cols = ['n', 'interaction_beta_native', 'interaction_ci_low_native', 'interaction_ci_high_native', 'interaction_p', q]
    rename = {'interaction_beta_native':'Beta', 'interaction_ci_low_native':'L95', 'interaction_ci_high_native':'U95', 'interaction_p':'p', q:'qAll84'}
    print(label)
    print(row[cols].rename(columns=rename).to_string(index=False, float_format=lambda x: '{:.10g}'.format(x)))
frame = pd.read_csv(new / 'tables/reference_hits_stability.csv')
row = frame[frame.exposure.eq('rEDIH') & frame.taxon.eq('T096') & frame.outcome.eq('CV')]
assert len(row) == 1, 'Expected one stability row'
print('Frozen reference qAll84={:.10g}; reproduced tests={}'.format(row.reference_q_all84.iloc[0], nm['reference_reproduced_tests']))
print('Historical 17:10 transcription=0.01872; 19:01 screenshot qBase=0.0487')
PY
