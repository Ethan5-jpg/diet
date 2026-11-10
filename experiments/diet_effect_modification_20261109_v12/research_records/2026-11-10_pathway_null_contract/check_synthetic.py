"""Generated-file checks only. Never load native HPP inputs."""
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT/'inspect_pathway_null_contract.sh'
BODY = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
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


def edit_arrow(state, which, update):
    path = state['paths'][which]
    with pa.memory_map(str(path), 'r') as stream:
        rows = ipc.open_file(stream).read_all().to_pylist()
    update(rows)
    write_arrow(path, rows)


def edit_registry(state, update, renew_lock=True):
    path = state['registry']
    rows = pq.ParquetFile(path).read().to_pylist()
    update(rows)
    pq.write_table(pa.Table.from_pylist(rows), path)
    if renew_lock:
        state['lock'].write_text(json.dumps({'sample_registry_sha256': sha(path)}))


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


def scenario(name, mutate=None, instrument=False):
    with tempfile.TemporaryDirectory(prefix='hpp_fixed_pathway_synthetic_') as tmp:
        base = Path(tmp)
        state = fixture(base)
        if mutate:
            mutate(state)
        args = [str(state['raw']), str(state['project']), '3', '5', '2']
        if instrument:
            prelude = '''import numpy as np
original_load = np.load
class Guard:
    def __init__(self, real): self.real = real
    def __enter__(self): return self
    def __exit__(self, *args): self.real.close()
    def __getitem__(self, name):
        assert name == 'y', 'Unexpected old model score access'
        return self.real[name]
def load(*args, **kwargs): return Guard(original_load(*args, **kwargs))
np.load = load
'''
            if instrument == 'file_change':
                prelude += '''import builtins
from pathlib import Path
import sys
old_print = builtins.print
def changed_print(*args, **kwargs):
    if args and str(args[0]).startswith('DESCRIPTIVE_QC:'):
        path = Path(sys.argv[1])/'oral_microbiome/aggregated/abundance.arrow'
        with path.open('ab') as stream: stream.write(b'\\x00')
    old_print(*args, **kwargs)
builtins.print = changed_print
'''
            result = subprocess.run([sys.executable, '-B', '-']+args, input=prelude+'\nexec(compile('+repr(BODY)+',"fixed_feasibility","exec"))\n', cwd=base, capture_output=True, text=True, timeout=70)
        else:
            result = subprocess.run(['bash', str(SCRIPT)]+args, cwd=base, capture_output=True, text=True, timeout=70)
        assert result.returncode == 0 and not result.stderr, (name, result.stderr)
        out = result.stdout
        for private in ('PRIVATE_', str(base), '987654321', '123456789', 's3://', 'Traceback'):
            assert private not in out, (name, private, out)
        return out


def main():
    results, previews = [], {}
    def verify(name, out, *expected):
        assert all(value in out for value in expected), (name, expected, out)
        results.append(name)
    good = scenario('valid')
    previews['valid'] = good
    verify('fixed_original_mask_and_input_checksum', good, 'fixed_people=3 households=2 same_two_repeat_mask=True discovery_input_sha_verified=True')
    verify('locked_registry_and_source_records_verified', good, 'registry_sha_verified=True reasons={"source_record_verified": 3}')
    verify('placeholder_visit_retained_and_unique_person_date_used', good, 'FIXED_LINK target=3 linked=3 reasons={"source_record_metadata_agree": 3}')
    verify('all_source_dates_compared_by_identity_not_counts', good, 'shared_unique_people=5 cohort_equal=5 array_equal=5 calendar_date_equal=5 timestamp_equal=5 source_record_metadata_compatible=5')
    verify('special_totals_excluded_without_dropping_other_features', good, 'numeric_total=5 special_totals_excluded=2 candidate_columns=3')
    verify('null_and_nonnull_nan_separately_counted', good, 'QC_CELLS abundance cells=9 arrow_null=1 nonnull_nan=1 inf=0 negative=0')
    verify('extra_people_not_added_to_fixed_qc', good, 'QC_PROFILES abundance fixed_people=3', 'PAIR_NUMERIC fixed_people=3 common_finite_cells=7 exact_equal_common_finite_cells=3')
    verify('no_feature_zero_imputation', good, 'no_finite=0 no_positive_observed=0 all_observed_zero=0 constant_observed_features=2 all_missing_features=0')
    out = scenario('only_y_availability', instrument=True)
    verify('instrumented_only_old_y_array_loaded_no_model_scores', out, 'PATHWAY_NULL_CONTRACT_DONE')
    def input_changed(s):
        rows = pq.ParquetFile(s['cov']).read().to_pylist()
        rows[0]['cohort'] = 'PRIVATE_CHANGED_COHORT'
        pq.write_table(pa.Table.from_pylist(rows), s['cov'])
    verify('discovery_input_checksum_change_stops', scenario('input_changed', input_changed), 'FEASIBILITY_STOP DISCOVERY_INPUT_CHECKSUM_MISMATCH')
    verify('registry_checksum_change_stops', scenario('registry_changed', lambda s:edit_registry(s,lambda r:r[0].update(site='other'),renew_lock=False)), 'FEASIBILITY_STOP LOCKED_REGISTRY_CHECKSUM_MISMATCH')
    def membership_changed(s):
        y = np.full((2,5), np.nan)
        y[:,:2] = 1.
        np.savez(s['reference'], y=y)
    verify('fixed_population_not_replaced', scenario('membership_changed', membership_changed), 'FEASIBILITY_STOP FIXED_COUNTS_CHANGED')
    def repeats_differ(s):
        y = np.full((2,5),np.nan)
        y[0,:3] = 1.
        y[1,1:4] = 1.
        np.savez(s['reference'], y=y)
    verify('different_repeat_memberships_rejected', scenario('repeat_changed', repeats_differ), 'FEASIBILITY_STOP FIXED_MEMBERSHIP_CHANGED')
    cases = [
        ('source_path', lambda r:r[0].update(primary_index_source_path='PRIVATE_WRONG_PATH'), 'source_path_mismatch'),
        ('source_sha', lambda r:r[0].update(primary_index_source_sha256='0'*64), 'source_checksum_mismatch'),
        ('zero_ordinal', lambda r:r[0].update(linked_index_row_ordinal=0), 'source_ordinal_invalid'),
        ('fractional_ordinal', lambda r:r[0].update(linked_index_row_ordinal=1.5), 'source_ordinal_invalid'),
        ('wrong_ordinal', lambda r:r[0].update(linked_index_row_ordinal=2), 'source_key_or_uniqueness_mismatch'),
        ('unlocked', lambda r:r[0].update(sample_lock_status='provisional'), 'registry_not_locked'),
        ('registry_duplicate', lambda r:r.append(dict(r[0])), 'registry_missing_or_ambiguous'),
    ]
    for name, change, reason in cases:
        out = scenario(name, lambda s,c=change:edit_registry(s,c))
        verify(name+'_blocks_all_fixed_qc', out, '"'+reason+'": 1', 'QC_SKIPPED: complete fixed-population source-record linkage not established; no subset replacement.')
        assert 'QC_CELLS' not in out
    def wrong_date(rows):
        next(r for r in rows if r['participant_id']==101)['collection_date'] += timedelta(days=1)
    out = scenario('wrong_date', lambda s:edit_arrow(s,0,wrong_date))
    verify('date_mismatch_held_not_joined_by_person_only', out, 'calendar_date_equal=4 timestamp_equal=4', 'FIXED_LINK target=3 linked=2', 'QC_SKIPPED:')
    previews['wrong_date'] = out
    def missing_date(rows):
        next(r for r in rows if r['participant_id']==101)['collection_date'] = None
    verify('missing_identity_date_not_invented', scenario('missing_date',lambda s:edit_arrow(s,0,missing_date)), 'FIXED_LINK target=3 linked=2', 'QC_SKIPPED:')
    def different_array(rows):
        next(r for r in rows if r['participant_id']==101)['array_index'] = 9
    verify('array_disagreement_not_silently_overridden', scenario('wrong_array',lambda s:edit_arrow(s,0,different_array)), 'array_equal=4', 'FIXED_LINK target=3 linked=2', 'QC_SKIPPED:')
    def valid_conflicting_stage(rows):
        next(r for r in rows if r['participant_id']==101)['research_stage'] = 'different_visit'
    verify('valid_conflicting_stage_not_overridden', scenario('stage_conflict',lambda s:edit_arrow(s,0,valid_conflicting_stage)), '"function_metadata_conflict_or_date_missing": 1', 'FIXED_LINK target=3 linked=2', 'QC_SKIPPED:')
    def duplicate_person(rows):
        rows.append(dict(next(r for r in rows if r['participant_id']==101)))
    verify('function_duplicate_person_not_arbitrarily_deduplicated', scenario('duplicate_person',lambda s:edit_arrow(s,0,duplicate_person)), '"function_person_missing_or_nonunique": 1', 'QC_SKIPPED:')
    def same_day_different_time(rows):
        next(r for r in rows if r['participant_id']==101)['collection_date'] += timedelta(hours=3)
    verify('calendar_evidence_not_arbitrary_timestamp_or_lag_gate', scenario('same_day',lambda s:edit_arrow(s,0,same_day_different_time)), 'calendar_date_equal=5 timestamp_equal=4 source_record_metadata_compatible=5', 'QC_PROFILES abundance fixed_people=3')
    def all_missing_feature(rows):
        for r in rows:
            r['VALID-OTHER-FEATURE'] = None
    out = scenario('all_missing_typed_feature', lambda s:edit_arrow(s,0,all_missing_feature))
    verify('untyped_null_field_not_misrepresented_as_numeric_feature', out, 'numeric_total=4 special_totals_excluded=2 candidate_columns=2', 'PAIR_FEATURE_LABEL_SETS_EQUAL=False')
    def typed_missing(s):
        path = s['paths'][0]
        with pa.memory_map(str(path), 'r') as stream:
            table = ipc.open_file(stream).read_all()
        rows = table.to_pylist()
        for r in rows:
            r['VALID-OTHER-FEATURE'] = None
        result = pa.Table.from_pylist(rows, schema=table.schema)
        with pa.OSFile(str(path), 'wb') as stream:
            with ipc.new_file(stream, result.schema) as writer:
                writer.write_table(result)
    verify('typed_all_missing_feature_retained_as_missing', scenario('typed_missing',typed_missing), 'QC_CELLS abundance cells=9 arrow_null=4 nonnull_nan=1', 'all_missing_features=1')
    out = scenario('changed_after_metadata',instrument='file_change')
    verify('changed_function_file_blocks_stale_mapping_qc', out, 'FEASIBILITY_STOP FUNCTION_CHANGED_SINCE_METADATA')
    assert 'QC_CELLS' not in out
    def inf_negative(rows):
        next(r for r in rows if r['participant_id']==101)['PWY-1234:demo'] = float('inf')
        next(r for r in rows if r['participant_id']==102)['PWY-1234:demo'] = -2.
    verify('nonfinite_and_negative_values_preserved_in_qc', scenario('inf_negative',lambda s:edit_arrow(s,0,inf_negative)), 'arrow_null=1 nonnull_nan=1 inf=1 negative=1')
    def mismatch_label(rows):
        for row in rows:
            row['PRIVATE_RENAMED_FEATURE'] = row.pop('VALID-OTHER-FEATURE')
    verify('different_feature_sets_not_numerically_compared', scenario('labels_differ',lambda s:edit_arrow(s,1,mismatch_label)), 'PAIR_FEATURE_LABEL_SETS_EQUAL=False')
    def bad_footer(s):
        s['paths'][0].write_bytes(b'ARROW1'+b'\x00'*4+(33*1024**2).to_bytes(4,'little')+b'ARROW1')
    verify('footer_guard_preserved', scenario('footer_guard',bad_footer), 'TABLE_UNRESOLVED IPC_FOOTER_BUDGET', 'QC_SKIPPED:')
    def huge_file(s):
        with s['paths'][0].open('wb') as stream:
            stream.truncate(65*1024**2)
    verify('physical_file_budget_preserved', scenario('file_guard',huge_file), 'TABLE_UNRESOLVED IPC_FILE_BUDGET', 'QC_SKIPPED:')
    def remote(s):
        rows = pq.ParquetFile(s['source']).read().to_pylist()
        for row in rows:
            row[F[0]] = 's3://PRIVATE_BUCKET/PRIVATE_KEY'
        pq.write_table(pa.Table.from_pylist(rows),s['source'])
        edit_registry(s,lambda rows:[r.update(primary_index_source_sha256=sha(s['source'])) for r in rows if r['site']=='oral'])
    verify('remote_pointer_not_requested', scenario('remote',remote), 'TABLE_UNRESOLVED FUNCTION_POINTER_OR_PATH_AMBIGUOUS', 'QC_SKIPPED:')
    def path_ambiguity(s):
        dest = s['raw']/'aggregated/abundance.arrow'
        dest.parent.mkdir()
        dest.write_bytes(s['paths'][0].read_bytes())
    verify('multiple_local_candidates_not_arbitrarily_selected', scenario('ambiguous_path',path_ambiguity), 'TABLE_UNRESOLVED FUNCTION_POINTER_OR_PATH_AMBIGUOUS', 'QC_SKIPPED:')

    def blank_profile(s, missing_special=False, source_qc=False):
        for which in (0,1):
            def update(rows, which=which):
                for row in rows:
                    if which == 1:
                        row['UNMAPPED'] = 1.
                        row['UNINTEGRATED:total'] = 1.
                    if row['participant_id'] == 101:
                        for name in ('PWY-1234:demo','GLYCOLYSIS-PWY:demo','VALID-OTHER-FEATURE'):
                            row[name] = None
                        row['UNMAPPED'] = None if missing_special else (0. if which == 0 else 1.)
                        row['UNINTEGRATED:total'] = None if missing_special else (0. if which == 0 else 1.)
            edit_arrow(s,which,update)
        if source_qc:
            rows = pq.ParquetFile(s['source']).read().to_pylist()
            rows[0].update(raw_read_count=100., trimmed_read_count=80., metaphlan4_aligned_read_count=50., excessive_duplicate_reads=False, humann_pathway_abundance_tsv='s3://PRIVATE_SAMPLE_FILE', humann_pathway_coverage_tsv='s3://PRIVATE_SAMPLE_FILE')
            pq.write_table(pa.Table.from_pylist(rows),s['source'])
            edit_registry(s,lambda registry:[row.update(primary_index_source_sha256=sha(s['source'])) for row in registry if row['site']=='oral'])
    out = scenario('blank_with_completion_markers',lambda s:blank_profile(s,source_qc=True))
    previews['blank_markers'] = out
    verify('whole_null_profile_distinguished_from_within_profile_gaps', out, 'arrow_null_masks_equal=True whole_candidate_null_in_either_table=1 with_paired_finite_candidates=2')
    verify('special_zero_and_one_markers_reported_as_counts', out, 'SPECIAL_STATUS abundance special_columns=2 fixed_rows_all_special_finite=3 populated_rows_all_special_finite=2 blank_rows_all_special_finite=1 blank_rows_all_special_zero=1 blank_rows_all_special_one=0', 'BLANK_SPECIAL coverage UNMAPPED group_n=1 finite=1 missing_or_nonfinite=0 positive=1 zero=0 one=1 negative=0')
    verify('small_blank_group_qc_no_individual_values', out, 'BLANK_READ_QC raw_read_count finite_numeric=1 positive=1 zero=0 negative=0', 'BLANK_READ_QC excessive_duplicate_reads observed=1 true_flags=0')
    verify('individual_pointer_presence_not_remote_access', out, 'BLANK_POINTER humann_pathway_abundance_tsv nonempty=1/1; no file access inferred or attempted.')
    out = scenario('blank_no_completion_markers',lambda s:blank_profile(s,missing_special=True))
    verify('whole_null_special_values_not_implicitly_zero_or_processed', out, 'BLANK_SPECIAL abundance UNMAPPED group_n=1 finite=0 missing_or_nonfinite=1 positive=0 zero=0 one=0 negative=0')
    verify('absent_read_qc_not_certified_as_pass', out, 'BLANK_SOURCE_QC group_n=1 available_technical_fields=[]')
    def coverage_zero(rows):
        next(row for row in rows if row['participant_id']==101)['PWY-1234:demo'] = 0.
    out = scenario('observed_coverage_zero',lambda s:edit_arrow(s,1,coverage_zero))
    verify('observed_coverage_zero_distinguished_from_null', out, 'coverage_explicit_zero_cells=4 coverage_zero_with_positive_abundance_cells=1')
    def scale_tags(s):
        path = s['paths'][0]
        with pa.memory_map(str(path),'r') as stream:
            table = ipc.open_file(stream).read_all()
        table = table.replace_schema_metadata({b'units':b'PRIVATE_UNRECOGNIZED_UNIT',b'normalization':b'CPM',b'PRIVATE_CONFIG':b'PRIVATE_CONFIG_VALUE'})
        with pa.OSFile(str(path),'wb') as stream:
            with ipc.new_file(stream,table.schema) as writer:
                writer.write_table(table)
    verify('scale_metadata_whitelist_and_unknown_values_suppressed', scenario('scale_metadata',scale_tags), 'SCALE_METADATA abundance explicit_entries=2 recognized={"normalization:cpm": 1} unrecognized_entries=1')
    verify('no_unit_inference_or_data_filling', out, 'HPP aggregation certification', 'No filling, filtering, rescaling or fitting.')
    compile(BODY, str(SCRIPT), 'exec')
    assert subprocess.run(['bash','-n',str(SCRIPT)],capture_output=True).returncode == 0
    results.append('bash_and_python_syntax_valid')
    report = dict(status='passed', synthetic_only=True, real_server_execution='pending_user', check_count=len(results), checks=results, script_sha256=sha(SCRIPT), scope='Generated Arrow/Parquet/NPZ/JSON only; no native HPP data, new outcome associations or models.')
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'synthetic_preview.txt').write_text('SYNTHETIC ONLY; NOT HPP RESULTS\n\n'+previews['blank_markers']+'\nSYNTHETIC DATE-MISMATCH CASE\n'+previews['wrong_date'])
    print(json.dumps(report))


if __name__ == '__main__':
    main()
