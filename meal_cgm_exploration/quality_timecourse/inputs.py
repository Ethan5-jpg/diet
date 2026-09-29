"""Pinned, read-only inputs; exact reuse of original scoring contracts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytz
from hppmeal.common import require, sha256, read_csv, norm, unique, write_json
from seven_diet_meals import inputs as old
from . import VERSION

PACKAGE = Path(__file__).resolve().parents[1]
MODULE = Path(__file__).resolve().parent


def bundle():
    f = MODULE / 'MANIFEST.json'
    b = json.loads(f.read_text(encoding='utf-8'))
    require(b['version'] == VERSION, 'Quality package version mismatch')
    result = {str(f): sha256(f)}
    required = {'inputs.py', 'coverage.py', 'analysis.py', 'reporting.py', 'run_analysis.py',
                'run_quality_timecourse.sh', '__init__.py', 'README.md', 'PLAN.md'}
    require(required.issubset(b['files']), 'Incomplete quality upload manifest')
    for name, digest in b['files'].items():
        p = (MODULE / name).resolve()
        require(MODULE in p.parents and p.is_file() and sha256(p) == digest, 'Changed/incomplete upload: ' + name)
        result[str(p)] = digest
    for name, digest in b['dependencies'].items():
        p = (PACKAGE / name).resolve()
        require(PACKAGE in p.parents and p.is_file() and sha256(p) == digest, 'Frozen dependency differs: ' + name)
        result[str(p)] = digest
    result.update(old.bundle())
    return result


def inspect(meal_source, seven_source, root_override, output, diary_override=None):
    meal_source, seven_source = Path(meal_source).resolve(), Path(seven_source).resolve()
    manifest = seven_source / 'manifest.json'
    spec_file = seven_source / 'analysis_spec.json'
    results_file = seven_source / 'tables/all_models.csv'
    prior = json.loads(manifest.read_text(encoding='utf-8'))
    spec = json.loads(spec_file.read_text(encoding='utf-8'))
    state = prior['status'].get('seven_diets', {})
    require(state.get('status') in ['COMPLETE', 'COMPLETE_WITH_LIMITATIONS'], 'Seven-diet run must be complete')
    require(spec['version'] == '2026.09.25.2' and spec.get('stage') == 'all', 'Requires seven_diet_meals 2026.09.25.2 completed run')
    require(not any(x.get('status') in ['FAILED', 'RUNNING', 'INTERRUPTED'] for x in prior['status'].values()), 'Seven-diet run incomplete')
    root = Path(root_override or prior['root']).expanduser().resolve()
    old_root = Path(prior['root'])
    def relocate(value):
        p = Path(value)
        try:
            return root / p.relative_to(old_root)
        except ValueError:
            return p
    source_config = {k: str(relocate(spec['sources'][k])) for k in ['current4', 'new3', 'corrected_hpdi']}
    config_file = Path(output) / 'source_paths.json'
    write_json(config_file, source_config)
    root, c, paths, expected, audit = old.inspect(meal_source, str(root), str(config_file))
    require(c == prior['config'] and c['data_kind'] == prior['data_kind'], 'Original/seven configuration differs')
    pinned = {x['path']: x['sha256'] for x in prior['inputs']}
    for key in ['current4', 'new3', 'corrected_hpdi', 'covariates', 'meals']:
        original = spec['sources'][key]
        require(original in pinned and sha256(paths[key]) == pinned[original], 'Input differs from seven-diet run: ' + key)
    original_manifest = str(Path(spec['original_meal_run']) / 'manifest.json')
    require(original_manifest in pinned and sha256(meal_source / 'manifest.json') == pinned[original_manifest], 'Meal run does not match seven-diet provenance')
    cache = meal_source / 'private/cache.sqlite'
    meal_manifest = json.loads((meal_source / 'manifest.json').read_text())
    digest = meal_manifest['status']['audit'].get('cache_sha256')
    require(digest and cache.is_file() and sha256(cache) == digest, 'Original cache missing/changed; restore original private/cache.sqlite')
    require(not any(Path(str(cache) + suffix).exists() and Path(str(cache) + suffix).stat().st_size > 0
                    for suffix in ['-wal', '-journal']), 'Cache has live database sidecar; finish original run first')
    require(c['cgm']['post_minutes'] == 120 and c['cgm']['sampling_minutes'] == 15, 'Requires original 120-minute windows and 15-minute CGM')
    diary = Path(diary_override).expanduser().resolve() if diary_override else root / 'Transfer/diet_logging/diet_logging.csv'
    for p in [manifest, spec_file, results_file, diary]:
        require(p.is_file(), 'Missing input: ' + str(p))
        expected[str(p)] = sha256(p)
    expected[str(cache)] = digest
    paths.update(cache=cache, diary=diary, previous_results=results_file)
    return root, c, paths, expected, audit


TZ_NAMES = {x.lower(): x for x in pytz.all_timezones}


def zone(value):
    if pd.isna(value):
        return None
    return TZ_NAMES.get(str(value).strip().lower())


def load_diary(run, path):
    columns = ['participant_id', 'cohort', 'research_stage', 'collection_date', 'food_items', 'total_logging_days', 'timezone']
    d = read_csv(path, columns, usecols=columns)
    for col in ['participant_id', 'cohort', 'research_stage']:
        d[col] = norm(d[col])
    d = d.loc[d.cohort.eq(run.c['cohort']) & d.research_stage.eq(run.c['research_stage'])].copy()
    require(len(d) > 0, 'No target-visit diary rows')
    text = norm(d.collection_date)
    # Daily metadata is a local calendar date, not a UTC timestamp to be shifted.
    valid_format = text.str.match(r'^\d{4}-\d{2}-\d{2}(?:[ T]00:00:00(?:\.0+)?)?$', na=False)
    require(valid_format.all(), 'Diary collection_date must be a local date or midnight date string')
    d['local_date'] = text.str.slice(0, 10)
    require(pd.to_datetime(d.local_date, format='%Y-%m-%d', errors='coerce').notna().all(), 'Invalid diary calendar date')
    unique(d, ['participant_id', 'local_date'], 'diary participant-date')
    d['food_items'] = old.strict_number(d.food_items, 'food_items')
    require(d.food_items.notna().all() and d.food_items.ge(0).all() and (d.food_items % 1).eq(0).all(), 'Invalid diary food_items count')
    d['total_logging_days'] = old.strict_number(d.total_logging_days, 'total_logging_days')
    require(d.total_logging_days.dropna().ge(0).all() and (d.total_logging_days.dropna() % 1).eq(0).all(), 'Invalid total_logging_days')
    d['diary_zone'] = d.timezone.map(zone)
    counts = d.groupby('participant_id').agg(actual_dates=('local_date', 'nunique'), total_values=('total_logging_days', 'nunique'),
                                             reported_total=('total_logging_days', 'first'))
    require(counts.total_values.le(1).all(), 'Inconsistent total_logging_days within participant')
    run.table(pd.DataFrame([dict(rows=len(d), people=len(counts), missing_or_unknown_timezone=int(d.diary_zone.isna().sum()),
        reported_total_missing_people=int(counts.reported_total.isna().sum()),
        reported_total_disagrees_people=int((counts.reported_total.notna() & counts.reported_total.ne(counts.actual_dates)).sum()),
        at_least_7=int(counts.actual_dates.ge(7).sum()), at_least_10=int(counts.actual_dates.ge(10).sum()), at_least_14=int(counts.actual_dates.ge(14).sum()))]), '01_diary_audit.csv')
    return d[['participant_id', 'local_date', 'food_items', 'diary_zone']]
