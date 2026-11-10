#!/usr/bin/env bash
# Follow-up: named directories only, never recursive, no row values.
hpp_meta_settings=/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12/.python_path
hpp_meta_python=python
if [ -s "$hpp_meta_settings" ]; then
    IFS= read -r hpp_meta_python < "$hpp_meta_settings"
fi
"$hpp_meta_python" -B - "$@" <<'PY'
import json, os, re, signal, sys
from pathlib import Path
import pyarrow.parquet as pq

def expired(signum, frame):
    print('TIME_LIMIT_REACHED; metadata check incomplete.', flush=True)
    raise SystemExit(0)

signal.signal(signal.SIGALRM, expired)
signal.alarm(30)
R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/ec2-user/studies/hpp_datasets')
sealed = re.compile(r'confirm|holdout|heldout|held_out|w22|w23', re.I)
want = re.compile(r'microbio|abundance|humann|path|gene|uniref|metacyc|kegg|cazy|dbcan|eggnog|function|metabol|annot|readme|dict|schema|docs', re.I)
processing = re.compile(r'humann|pathway|pathabundance|gene.?famil|gene.?abundance|uniref|metacyc|kegg|cazy|dbcan|eggnog|function|metaphlan|wgs|annot', re.I)
date_word = re.compile(r'(?:^|_)(?:date|timestamp|time)(?:_|$)', re.I)
print('METADATA_TOPLEVEL_ONLY: no recursion, participant rows, outcomes or writes.', flush=True)

for site in ('oral', 'gut'):
    path = R / (site + '_microbiome') / (site + '_microbiome.parquet')
    try:
        with pq.ParquetFile(path, pre_buffer=False) as pf:
            fields = [name for name in pf.schema_arrow.names if processing.search(name)]
            print('PROCESSING_FIELDS', site, json.dumps(fields[:10]), 'more=' + str(max(0, len(fields)-10)), flush=True)
            comments = []
            for field in pf.schema_arrow:
                if date_word.search(field.name):
                    for key in (b'description', b'comment'):
                        value = (field.metadata or {}).get(key)
                        if value:
                            comments.append([field.name, value.decode('utf-8', errors='replace')[:180]])
            print('DATE_FIELD_COMMENTS', site, json.dumps(comments[:3], ensure_ascii=True) if comments else 'UNAVAILABLE', flush=True)
    except (OSError, ValueError, TypeError):
        print('HEADER_UNAVAILABLE', site, flush=True)

folders = [R, R/'gut_microbiome', R/'oral_microbiome', R/'gut_microbiome/abundance', R/'oral_microbiome/abundance']
for folder in folders:
    hits, count, limited = [], 0, False
    try:
        with os.scandir(folder) as listing:
            for number, entry in enumerate(listing, 1):
                if number > 500:
                    limited = True
                    break
                count = number
                if sealed.search(entry.name) or entry.name.startswith('.') or entry.is_symlink():
                    continue
                if want.search(entry.name):
                    kind = 'DIR' if entry.is_dir(follow_symlinks=False) else 'FILE'
                    hits.append([kind, entry.name])
        print('TOPLEVEL', str(folder), 'checked=' + str(count), 'limited=' + str(limited), flush=True)
        print('  MATCHES', json.dumps(sorted(hits)[:8], ensure_ascii=True), 'more=' + str(max(0, len(hits)-8)), flush=True)
    except OSError:
        print('DIRECTORY_UNAVAILABLE', str(folder), flush=True)

print('Names and pointer fields do not establish functional measurements or participant overlap.', flush=True)
print('METADATA_TOPLEVEL_DONE', flush=True)
signal.alarm(0)
PY
