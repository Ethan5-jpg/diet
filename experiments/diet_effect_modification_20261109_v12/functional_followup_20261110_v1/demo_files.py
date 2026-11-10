"""Standalone generated-file fixtures; never opens native participant data."""
import hashlib, json, sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
K = ['participant_id', 'cohort', 'research_stage']
F = ['humann_aggregated_pathway_abundance_pathway_level_arrow', 'humann_aggregated_pathway_coverage_pathway_level_arrow']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_arrow(path, rows, feature_order=None):
    table = pa.Table.from_pylist(rows)
    if feature_order is not None:
        table = table.select(feature_order)
    with pa.OSFile(str(path), 'wb') as stream:
        with ipc.new_file(stream, table.schema, options=ipc.IpcWriteOptions(compression='zstd')) as writer:
            writer.write_table(table)

def fixture(base):
    raw = base/'hpp_datasets'
    oral = raw/'oral_microbiome'
    project = base/'project'
    package = project/'collaborator_handoff_20261102_v1'
    private = package/'experiments/diet_nested_stability_20261103_v4/results/TRE_ONLY'
    inputs = package/'inputs/discovery_v2'
    for path in (oral/'aggregated', private, inputs, project/'analysis/shared', project/'config'):
        path.mkdir(parents=True, exist_ok=True)
    dates = {pid:datetime(2020, 1, 1)+timedelta(days=pid-101) for pid in range(101, 107)}
    source = [dict(participant_id=pid, cohort='10k', research_stage='00_00_visit' if pid < 104 else 'undefined', array_index=0, collection_date=dates[pid], **{F[0]:'aggregated/abundance.arrow', F[1]:'aggregated/coverage.arrow'}) for pid in range(101, 106)]
    src_path = oral/'oral_microbiome.parquet'
    pq.write_table(pa.Table.from_pylist(source), src_path)
    cov_path = inputs/'discovery_covariates_food_TRE_ONLY.parquet'
    covariates = [dict(participant_id=pid, cohort='10k', research_stage='00_00_visit', input_row=i, final_role='discovery', split_unit_id='PRIVATE_HH_'+str(0 if pid < 103 else pid)) for i,pid in enumerate(range(101,106))]
    pq.write_table(pa.Table.from_pylist(covariates), cov_path)
    y = np.full((2,5), np.nan)
    y[:,:3] = 987654321.012345
    reference = private/'final_predictions.npz'
    np.savez(reference, y=y, PRIVATE_UNUSED_SCORE=np.array([123456789.765432]))
    registration = private.parent/'registration.json'
    registration.write_text(json.dumps({'source_and_input_sha256':{str(cov_path):sha(cov_path)}}))
    reg_path = project/'analysis/shared/sample_registry_TRE_ONLY.parquet'
    registry = []
    for i,row in enumerate(source):
        registry.append(dict({k:row[k] for k in K}, site='oral', sample_lock_status='locked_W13_independent_full_source_verification_passed', linked_index_row_ordinal=i+1, primary_index_source_path=str(src_path), primary_index_source_sha256=sha(src_path)))
    registry.append(dict(registry[0], site='gut', primary_index_source_path='PRIVATE_GUT_SOURCE'))
    pq.write_table(pa.Table.from_pylist(registry), reg_path)
    lock = project/'config/w13_sample_selection_lock.json'
    lock.write_text(json.dumps({'sample_registry_sha256':sha(reg_path)}))
    paths = [oral/'aggregated/abundance.arrow', oral/'aggregated/coverage.arrow']
    for j,path in enumerate(paths):
        rows = []
        for pid in [106,103,105,101,102,104]:
            rows.append(dict(participant_id=pid, cohort='10k', research_stage='undefined', array_index=0, collection_date=dates[pid], **{
                'PWY-1234:demo':float(pid-100)/(10 if j else 1),
                'GLYCOLYSIS-PWY:demo':None if pid==101 and j==0 else float('nan') if pid==102 and j==0 else float(pid-99)/(10 if j else 1),
                'VALID-OTHER-FEATURE':0., 'UNMAPPED':1000., 'UNINTEGRATED:total':2000.}))
        write_arrow(path, rows)
    (base/'.python_path').write_text(sys.executable+'\n')
    return dict(raw=raw, oral=oral, project=project, package=package, source=src_path, cov=cov_path, registry=reg_path, lock=lock, reference=reference, registration=registration, paths=paths)
