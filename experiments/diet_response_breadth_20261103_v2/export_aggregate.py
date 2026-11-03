"""Collect all aggregate results and inventory every non-public artifact.

Only the explicit aggregate allowlist is read. Participant predictions, target
arrays, checkpoints and fitted estimators (including stored training rows) are
not included in the public bundle. Run after postrun_audit.py succeeds.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib


def run(root):
    root = Path(root).resolve()
    assert json.loads((root/'postrun_audit.json').read_text())['status'] == 'PASS'
    names = [
        'results/summary.json', 'results/model_comparison.csv',
        'results/grid_scores.json', 'results/diagnostics.json',
        'results/registration.json', 'source_sha256.json', 'deployment.json',
        'logs/execution.json', 'logs/actual_exit_code.txt', 'logs/run.log',
        'postrun_audit.json',
    ] + [f'results/fold{i}_encoding.json' for i in range(5)]
    files = {name: (root/name).read_text() for name in names}
    manifest = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith('public_export/'):
            continue
        entry = dict(path=relative, bytes=path.stat().st_size)
        if relative in names or path.suffix in ('.py', '.md') or path.name == 'source_sha256.json':
            entry['classification'] = 'source_or_aggregate'
            entry['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif 'fits_TRE_ONLY' in path.parts:
            entry['classification'] = 'fitted_artifact_not_publicly_exported'
            entry['reason'] = 'Family includes SVR support vectors and PLS per-training-row scores; no blanket pickle release.'
        elif 'TRE_ONLY' in path.name:
            entry['classification'] = 'participant_level_not_publicly_exported'
            entry['reason'] = 'Contains aligned participant targets, predictions, modifiers, or household/row identifiers.'
        else:
            entry['classification'] = 'operational_not_in_aggregate_allowlist'
        manifest.append(entry)
    bundle = dict(format='hpp_aggregate_export_v1', files=files, artifact_inventory=manifest,
                  public_repository='Ethan5-jpg/diet', includes_individual_health_records=False)
    raw = (json.dumps(bundle, ensure_ascii=False, separators=(',', ':'))+'\n').encode()
    directory = root/'public_export'
    directory.mkdir(exist_ok=False)
    (directory/'aggregate_results.json').write_bytes(raw)
    (directory/'model_comparison.csv').write_text(files['results/model_comparison.csv'])
    token = base64.b64encode(zlib.compress(raw, 9)).decode()
    (directory/'aggregate_transfer.b64').write_text(token+'\n')
    print('PUBLIC_AGGREGATE_SHA256', hashlib.sha256(raw).hexdigest())
    print('PUBLIC_AGGREGATE_BYTES', len(raw), 'TRANSFER_CHARS', len(token))
    print('PUBLIC_AGGREGATE_BEGIN')
    print(token)
    print('PUBLIC_AGGREGATE_END')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment')
    run(parser.parse_args().experiment)
