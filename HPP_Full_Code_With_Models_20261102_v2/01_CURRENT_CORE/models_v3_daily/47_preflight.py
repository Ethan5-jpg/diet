"""HPP v3 resumption preflight: metadata/roles only; no scientific Y reads.

Run only inside the authorized TRE, with the pheno interpreter. The browser
transfer wrapper records its actual exit separately. No prior artifact is edited.
"""
from pathlib import Path
import datetime
import hashlib
import json
import sys

import pyarrow.parquet as pq

P = Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
B = P / 'data_qc/w20_simplified_20261101_v2'
O = B / 'results'
L = Path('/home/ec2-user/multisystem_runtime_20261021/w20_simplified_20261101_v2')
S = Path(__file__).resolve().parent
assert Path(sys.executable).resolve() == Path('/home/ec2-user/miniconda/envs/pheno/bin/python').resolve()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

expected_sources = json.loads((S / 'expected_dependency_hashes.json').read_text())
for path, expected in expected_sources.items():
    assert sha(path) == expected, 'Dependency changed: ' + path

exits = {}
for tag in ['41', '42', '43', '44', '45', '46']:
    exits[tag] = int((L / (tag + '_actual_exit_code')).read_text().strip())
assert set(exits.values()) == {0}, exits

verified = json.loads((O / 'delivery/verified_status.json').read_text())
assert verified['CV_metric_readbacks'] == 36
assert verified['score_formula_readbacks'] == 36
assert verified['confirmation_Y_reads_this_rerun'] == 0
manifest = json.loads((O / 'delivery/artifact_manifest_TRE_ONLY.json').read_text())
lookup = {entry['path']: entry for entry in manifest}
protected = json.loads((B / 'protected_v1_hashes.json').read_text())
for path, expected in protected.items():
    assert sha(path) == expected, 'Protected v1 artifact changed: ' + path

critical = [B / 'roles_v2_full_TRE_ONLY.parquet',
            B / 'discovery_covariates_food_TRE_ONLY.parquet',
            O / 'models/D_Glu/fiber_density/scores_and_predictor.json',
            O / 'models/D_Glu/fiber_density/selected_CV_losses_TRE_ONLY.npz']
for key in ['fold0', 'fold1', 'fold2', 'fold3', 'fold4', 'full']:
    critical.append(O / 'fold_preprocessing' / key / 'transformed_discovery_TRE_ONLY.npz')
    for site in ['oral', 'gut']:
        critical.append(O / 'fold_preprocessing' / key / (site + '_transform.json'))
    critical.append(O / 'fold_preprocessing' / key / 'diet_basis.json')
for path in critical:
    assert str(path) in lookup, 'No verified hash for ' + str(path)
    assert sha(path) == lookup[str(path)]['sha256'], 'Changed v2 artifact: ' + str(path)

K = ['participant_id', 'cohort', 'research_stage']
roles = pq.ParquetFile(B / 'roles_v2_full_TRE_ONLY.parquet', pre_buffer=False).read(
    columns=K + ['split_unit_id', 'final_role']).to_pandas(ignore_metadata=True)
assert not roles.duplicated(K).any()
assert roles.groupby('split_unit_id').final_role.nunique().max() == 1
X = pq.ParquetFile(B / 'discovery_covariates_food_TRE_ONLY.parquet', pre_buffer=False).read(
    columns=K + ['split_unit_id', 'cv_fold', 'final_role']).to_pandas(ignore_metadata=True)
assert X.final_role.eq('discovery').all()
assert X.groupby('split_unit_id').cv_fold.nunique().max() == 1
assert set(X.participant_id) == set(roles.loc[roles.final_role.eq('discovery'), 'participant_id'])
assert not set(X.participant_id) & set(roles.loc[roles.final_role.eq('confirmation_reserved'), 'participant_id'])

paths = [
    P / 'analysis/Daily_CGM/fiber_density/base_table_TRE_ONLY.parquet',
    P / 'analysis/shared/cgm_daily_TRE_ONLY.parquet',
    P / 'analysis/shared/cgm_visit_phenotypes_TRE_ONLY.parquet',
    P / 'data_qc/preprocessing_20261030_v1/food_qc_v1/food_daily_primary_TRE_ONLY.parquet',
    P / 'data_qc/preprocessing_20261030_v1/analysis_ready_v1/Daily_CGM/fiber_density/table_TRE_ONLY.parquet',
]
schemas = []
for path in paths:
    entry = {'path': str(path), 'exists': path.is_file()}
    if path.is_file():
        pf = pq.ParquetFile(path, pre_buffer=False)
        entry.update(bytes=path.stat().st_size, rows=pf.metadata.num_rows,
                     columns=[{'name': f.name, 'type': str(f.type)} for f in pf.schema_arrow])
        # Footer/schema only: no values, including reserved scientific Y, read.
    schemas.append(entry)

report = {
    'status': 'preflight_metadata_and_provenance_verified',
    'time_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'python': sys.executable, 'actual_v2_exits': exits,
    'verified_dependencies': len(expected_sources),
    'protected_v1_artifacts': len(protected), 'verified_v2_critical_artifacts': len(critical),
    'role_counts': {str(k): int(v) for k, v in roles.final_role.value_counts().items()},
    'discovery_people': len(X), 'discovery_units': int(X.split_unit_id.nunique()),
    'new_scientific_Y_values_read': 0, 'new_model_fits': 0,
    'confirmation_Y_values_read': 0, 'schemas': schemas,
    'next': 'Resolve daily unit/date fields from verified schema, then run bounded discovery comparison.'
}
destination = S / '47_preflight_report.json'
assert not destination.exists(), 'Previous preflight exists: inspect it before rerunning.'
with destination.open('x') as f:
    json.dump(report, f, indent=2)
    f.write('\n')
print(json.dumps(report, indent=2), flush=True)
