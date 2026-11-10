#!/usr/bin/env bash
# Source-index pointer metadata only; no individual values are printed.
hpp_meta_settings=/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12/.python_path
hpp_meta_python=python
if [ -s "$hpp_meta_settings" ]; then
    IFS= read -r hpp_meta_python < "$hpp_meta_settings"
fi
"$hpp_meta_python" -B - "$@" <<'PY'
import json, os, re, signal, sys
from collections import Counter
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def expired(signum, frame):
    print('TIME_LIMIT_REACHED; pointer check incomplete.', flush=True)
    raise SystemExit(0)

signal.signal(signal.SIGALRM, expired)
signal.alarm(60)
R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/ec2-user/studies/hpp_datasets')
allowed_root = Path(os.path.realpath(R))
bad = {'', 'undefined', 'none', 'null', 'nan', 'na', 'n/a', 'unknown', '<na>'}
functional = re.compile(r'humann|pathway|gene.?famil|metacyc|kegg|cazy|dbcan|eggnog|uniref', re.I)
sealed = re.compile(r'confirm|holdout|heldout|held_out|w22|w23', re.I)
uri = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.-]*://')

def resolve(raw, module):
    if uri.match(raw):
        return 'URI_NOT_FETCHED', None
    p = Path(raw)
    options = [p] if p.is_absolute() else [module/p, R/p, R.parent/p, module/'abundance'/p]
    tested = False
    for path in options:
        path = Path(os.path.realpath(path))
        if not path.is_relative_to(allowed_root) or sealed.search(str(path)):
            continue
        tested = True
        if path.is_file():
            return 'LOCAL_FILE', path
    return ('LOCAL_NOT_LOCATED' if tested else 'OUTSIDE_ALLOWED_SCOPE'), None

print('FUNCTION_POINTER_CHECK: source-index rows, NOT 3343-person coverage.', flush=True)
print('Pointer values stay in TRE; no outcomes, abundance rows, remote fetches or writes.', flush=True)
for site in ('oral', 'gut'):
    module = R / (site + '_microbiome')
    try:
        with pq.ParquetFile(module / (site + '_microbiome.parquet'), pre_buffer=False) as pf:
            fields = [f.name for f in pf.schema_arrow if functional.search(f.name) and (pa.types.is_string(f.type) or pa.types.is_large_string(f.type))]
            print('INDEX', site, 'source_rows=' + str(pf.metadata.num_rows), 'functional_string_fields=' + str(len(fields)), flush=True)
            if not fields:
                print('  NO_MATCHING_FIELDS_IN_THIS_INDEX; other sources not excluded.', flush=True)
                continue
            data = pf.read(columns=fields, use_threads=False).to_pydict()
        values = {name: [v.strip() for v in data[name] if isinstance(v, str) and v.strip().lower() not in bad] for name in fields}
        chosen = []
        for group in ('gene', 'pathway_abundance', 'pathway_coverage'):
            candidates = sorted([name for name in fields if group in name.lower()], key=lambda name: ('microbe' in name.lower(), not name.lower().endswith('parquet'), name))
            if not candidates:
                continue
            name = next((name for name in candidates if values[name]), candidates[0])
            chosen.append(name)
            valid = values[name]
            unique = list(dict.fromkeys(valid))
            print('FIELD', site, name, 'nonempty_rows=' + str(len(valid)), 'distinct_pointers=' + str(len(unique)), flush=True)
            status, sample_file = Counter(), None
            for raw in unique[:3]:
                label, found = resolve(raw, module)
                status[label] += 1
                if sample_file is None and found is not None:
                    sample_file = found
            print('  FIRST_UP_TO_3_POINTERS', json.dumps(dict(status), sort_keys=True), flush=True)
            if sample_file is not None:
                if sample_file.suffix.lower() == '.parquet':
                    with pq.ParquetFile(sample_file, pre_buffer=False) as probe:
                        print('  ONE_FILE_HEADER rows=%d columns=%d' % (probe.metadata.num_rows, len(probe.schema_arrow)), flush=True)
                else:
                    print('  ONE_FILE_EXISTS bytes=%d; body not read.' % sample_file.stat().st_size, flush=True)
        if not chosen:
            print('  OTHER_FUNCTIONAL_FIELDS', json.dumps(fields[:6]), 'more=' + str(max(0, len(fields)-6)), flush=True)
    except (OSError, ValueError, TypeError) as error:
        print('INDEX_OR_PROBE_UNAVAILABLE', site, type(error).__name__, flush=True)
print('Nonempty pointer != usable measurement; sampled files != complete availability.', flush=True)
print('Locked-specimen matching, 3343-person overlap, units and functional QC remain UNVERIFIED.', flush=True)
print('FUNCTION_POINTER_CHECK_DONE', flush=True)
signal.alarm(0)
PY
