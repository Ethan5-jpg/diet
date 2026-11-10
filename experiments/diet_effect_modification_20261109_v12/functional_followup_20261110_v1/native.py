"""Only native reader. All participant data and derived arrays stay inside TRE."""
import contextlib
import hashlib
import io
import json
import runpy
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from frozen import modules
from functional import FixedMemberTarget


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def load(project, raw_root, config):
    project, raw_root = Path(project), Path(raw_root)
    root = Path(__file__).resolve().parent
    frozen = modules()
    if frozen.upload_check.verify(frozen.root) != 51:
        raise ValueError('Original V12 source verification failed')
    versions = frozen.runtime.environment(frozen.config)
    gate = frozen.runtime.readiness(frozen.config, prior_summary_reviewed=True)
    argv, handler = sys.argv[:], signal.getsignal(signal.SIGALRM)
    output = io.StringIO()
    try:
        sys.argv = [str(root/'audit_driver.py'), str(raw_root), str(project), str(config['original_people']), str(config['source_input_rows']), str(config['original_households'])]
        with contextlib.redirect_stdout(output):
            namespace = runpy.run_path(str(root/'audit_driver.py'), run_name='__hpp_function_audit__')
    except SystemExit as error:
        print('SOURCE_AUDIT_INCOMPLETE: ' + ' | '.join(output.getvalue().splitlines()[-3:]), flush=True)
        raise RuntimeError('Source audit did not complete') from error
    finally:
        sys.argv = argv
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)
    if not namespace.get('complete') or 'PATHWAY_NULL_CONTRACT_DONE' not in output.getvalue():
        print('SOURCE_AUDIT_INCOMPLETE: ' + ' | '.join(output.getvalue().splitlines()[-3:]), flush=True)
        raise RuntimeError('Source audit did not establish complete original records')
    records, matrices = namespace['records'], namespace['matrices']
    if records['abundance']['columns'] != records['coverage']['columns']:
        raise ValueError('Function candidate labels changed')
    labels = records['abundance']['columns']
    if len(labels) != config['candidate_fields']:
        raise ValueError('Candidate field count changed')
    null_masks = namespace['ARROW_NULL_MASKS']
    if not np.array_equal(null_masks['abundance'], null_masks['coverage']):
        raise ValueError('Paired function reporting masks changed')
    for name, values in matrices.items():
        if not np.array_equal(np.isnan(values), null_masks[name]) or np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
            raise ValueError('Non-null invalid functional values')
    coverage = matrices['coverage']
    if np.any(coverage[np.isfinite(coverage)] > 1+1e-8):
        raise ValueError('Coverage range changed')
    package = project/'collaborator_handoff_20261102_v1'
    dataset = frozen.bridge.load(package, frozen.config)
    original_target = dataset.api.PhenotypeOutliers
    dataset.api.PhenotypeOutliers = lambda:FixedMemberTarget(original_target())
    original = np.isfinite(dataset.reference['y'])
    if not np.array_equal(original, namespace['eligible']):
        raise ValueError('V12 membership differs from audited source')
    if not dataset.index[namespace['K']].reset_index(drop=True).equals(namespace['x'][namespace['K']].reset_index(drop=True)):
        raise ValueError('Native index keys differ from audit')
    positions = namespace['target'].input_row.to_numpy(int)
    usable = np.isfinite(matrices['abundance']).any(1) & np.isfinite(matrices['coverage']).any(1)
    if len(positions) != config['original_people'] or int(usable.sum()) != config['functional_people']:
        raise ValueError('Predefined functional population changed')
    available = np.zeros(len(dataset.index), bool)
    available[positions[usable]] = True
    signals = {}
    for role in ('abundance', 'coverage'):
        values = np.full((len(dataset.index), len(labels)), np.nan)
        values[positions] = matrices[role]
        signals[role] = values
    source_path = namespace['src_path']
    source_digest = namespace['src_hash']
    if sha(source_path) != source_digest:
        raise ValueError('Source index changed since audit')
    technical_names = ['raw_read_count', 'metaphlan4_aligned_read_count']
    with pq.ParquetFile(source_path, pre_buffer=False) as pf:
        present = [n for n in technical_names if n in pf.schema_arrow.names]
        metadata = pf.read(columns=present, use_threads=False).to_pydict() if present else {}
    technical = np.full((len(dataset.index), 3), np.nan)
    for j,name in enumerate(technical_names):
        if name in metadata:
            for output_row, source_row in zip(positions, namespace['approved']):
                value = metadata[name][source_row]
                try:
                    value = float(value)
                    if np.isfinite(value) and value >= 0:
                        technical[output_row,j] = value
                except (TypeError, ValueError, OverflowError):
                    pass
    ratio_ok = np.isfinite(technical[:,0]) & (technical[:,0] > 0) & np.isfinite(technical[:,1])
    technical[ratio_ok,2] = technical[ratio_ok,1]/technical[ratio_ok,0]
    paths = dataset.paths + frozen.bridge.adapter_sources()
    paths += [source_path, namespace['reg_path'], project/'config/w13_sample_selection_lock.json', namespace['v4']/'registration.json']
    paths += [v['path'] for v in records.values()]
    paths += [frozen.root/n for n in ('config.json','features.py','models.py','engine.py','bridge.py','runtime.py','upload_check.py','upload_manifest.json')]
    paths += [Path(gate['v11_summary'])]
    hashes = {str(Path(p)):sha(p) for p in dict.fromkeys(paths)}
    if hashes[str(source_path)] != source_digest:
        raise ValueError('Source changed during native loading')
    return SimpleNamespace(dataset=dataset, frozen=frozen, available=available, signals=signals, technical=technical,
        labels=labels, source_hashes=hashes, versions=versions, audit_text=output.getvalue(), synthetic_only=False,
        audit=dict(original_people=len(positions), functional_people=int(available.sum()), excluded_whole_candidate_null=int((~usable).sum()),
                   functional_households=int(dataset.index.loc[available,'split_unit_id'].nunique()), technical_fields=present,
                   partial_null_nonreporting_is_assumption=True, provider_abundance_unit_verified=False,
                   confirmation_outcomes_read=False, independent_confirmation=False))
