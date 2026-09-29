"""Read and pin previous outputs. Never edit source exports."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hppmeal.common import require, sha256, read_csv, header, norm, unique, select_participant_scope
from seven_diet_meals import inputs as seven
from quality_timecourse import inputs as quality
from . import VERSION

MODULE = Path(__file__).resolve().parent
PACKAGE = MODULE.parent
DEFAULT_CGM = 'cgm_deal/cgm论文新增17指标/outputs/run_20260919T030229_482714Z_c1a0f480/data/07_cgm_paper_extended_phenotypes.csv'
CGM20 = [
    ('Mean', 'cgm_mean'), ('Median', 'cgm_median'), ('IQR', 'cgm_iqr'), ('GMI', 'cgm_gmi'),
    ('CV', 'cgm_cv'), ('MAD', 'cgm_mad'), ('MAG', 'cgm_mag'), ('MODD', 'cgm_modd'),
    ('SD_ROC', 'cgm_sd_roc'), ('SDhhmm', 'cgm_sdhhmm'), ('SDw', 'cgm_sdw'),
    ('SDwsh', 'cgm_sdwsh'), ('TAR140', 'cgm_above_140'), ('TAR180', 'cgm_above_180'),
    ('HBGI', 'cgm_hbgi'), ('LBGI', 'cgm_lbgi'), ('ADRR', 'cgm_adrr'), ('COGI', 'cgm_cogi'),
    ('GRADE', 'cgm_grade'), ('GRADE_eugly', 'cgm_grade_eugly')]


def bundle():
    f = MODULE / 'MANIFEST.json'
    manifest = json.loads(f.read_text(encoding='utf-8'))
    require(manifest['version'] == VERSION, 'Nutrition package version mismatch')
    require({'inputs.py', 'analysis.py', 'reporting.py', 'run_analysis.py', 'run_nutrition_bridge.sh',
             '__init__.py', 'README.md', 'PLAN.md', 'REVIEW.md'}.issubset(manifest['files']), 'Incomplete upload manifest')
    expected = {str(f): sha256(f)}
    for name, digest in manifest['files'].items():
        p = (MODULE / name).resolve()
        require(MODULE in p.parents and p.is_file() and sha256(p) == digest, 'Changed/incomplete upload: ' + name)
        expected[str(p)] = digest
    expected.update(quality.bundle())
    # Include every executable dependency of the frozen meal package as well.
    for name, digest in manifest['dependencies'].items():
        p = (PACKAGE / name).resolve()
        require(PACKAGE in p.parents and p.is_file() and sha256(p) == digest, 'Frozen dependency differs: ' + name)
        expected[str(p)] = digest
    return expected


def inspect(quality_run, root_override, output, cgm_override=None):
    source = Path(quality_run).expanduser().resolve()
    mf, sf = source / 'manifest.json', source / 'analysis_spec.json'
    prior = json.loads(mf.read_text(encoding='utf-8'))
    spec = json.loads(sf.read_text(encoding='utf-8'))
    state = prior['status'].get('quality_timecourse', {})
    require(state.get('status') in ['COMPLETE', 'COMPLETE_WITH_LIMITATIONS'], 'Quality run must be complete')
    require(spec['version'] == '2026.09.25.1' and spec['data_kind'] == prior['data_kind'], 'Quality version/data label mismatch')
    require(not any(s.get('status') in ['FAILED', 'RUNNING', 'INTERRUPTED'] for s in prior['status'].values()), 'Quality run incomplete')
    root = Path(root_override or prior['root']).expanduser().resolve()
    old_root = Path(prior['root'])

    def relocate(value):
        p = Path(value)
        try:
            return root / p.relative_to(old_root)
        except ValueError:
            return p

    root, c, paths, expected, audit = quality.inspect(relocate(spec['meal_run']), relocate(spec['seven_run']), root, output)
    require(c == prior['config'], 'Quality configuration differs')
    pinned = {str(relocate(x['path'])): x['sha256'] for x in prior['inputs']}
    for key in ['meals', 'covariates', 'current4', 'new3', 'corrected_hpdi', 'cache', 'diary', 'previous_results']:
        p = str(paths[key])
        require(p in pinned and sha256(p) == pinned[p], 'Input differs from quality run: ' + key)
    for key, rel in [('quality_models', 'tables/05_quality_models.csv'),
                     ('quality_meals', 'private/03_meal_quality_timecourse.csv'),
                     ('quality_members', 'private/04_joint_membership.csv')]:
        p = source / rel
        digest = state.get('result_sha256', {}).get(rel)
        require(digest and p.is_file() and sha256(p) == digest, 'Quality output changed/missing: ' + rel)
        paths[key] = p
        expected[str(p)] = digest
    for p in [mf, sf]:
        expected[str(p)] = sha256(p)
    cg = Path(cgm_override).expanduser().resolve() if cgm_override else root / DEFAULT_CGM
    require(cg.is_file(), 'Original CGM table not found: {} ; supply --cgm-file with the original finalized table'.format(cg))
    names = header(cg)
    needed = ['participant_id', 'cohort', 'research_stage', 'connection_id'] + [f + '_z' for _, f in CGM20]
    require(set(needed).issubset(names), 'CGM table missing required identity/original20 Z columns: ' + str(sorted(set(needed) - set(names))))
    expected[str(cg)] = sha256(cg)
    paths['whole_cgm'] = cg
    return root, c, paths, expected, audit


def attach(meals, raw, qm):
    for frame, label in [(raw, 'positive meal export'), (qm, 'quality meal export')]:
        for col in ['participant_id', 'meal_id', 'connection_id']:
            frame[col] = norm(frame[col])
        unique(frame, ['meal_id'], label)
    require(set(qm.meal_id) == set(meals.meal_id), 'Quality/source ready meal membership differs')
    d = meals.merge(qm[['participant_id', 'meal_id', 'connection_id', 'joint']], on=['participant_id', 'meal_id'],
                    validate='one_to_one', how='left', suffixes=('', '_quality'))
    require(d.connection_id.notna().all(), 'Quality meal identity mismatch')
    # Frozen seven loader already includes connection_id; verify it if present.
    if 'connection_id_quality' in d:
        require(norm(d.connection_id).eq(d.connection_id_quality).all(), 'Quality/source connection differs')
        d = d.drop(columns='connection_id_quality')
    values = norm(d.joint).str.lower()
    require(values.isin(['true', 'false']).all(), 'Invalid/missing quality joint flag')
    d['joint'] = values.eq('true')
    raw['positive_mean'] = seven.strict_number(raw.iauc_positive_0_120, 'positive iAUC') / 120
    d = d.merge(raw[['participant_id', 'meal_id', 'connection_id', 'positive_mean']].rename(columns={'connection_id': 'raw_connection'}),
                on=['participant_id', 'meal_id'], how='left', validate='one_to_one')
    require(d.connection_id.eq(d.raw_connection).all(), 'Original/quality meal connection differs')
    d = d.drop(columns='raw_connection')
    valid = d.positive_mean.notna()
    require(d.loc[valid, 'positive_mean'].ge(-1e-8).all(), 'Negative positive iAUC in ready meals')
    require((d.loc[valid, 'positive_mean'] + 1e-7 >= d.loc[valid, 'mean_increment']).all(), 'Positive iAUC smaller than net iAUC')
    return d


def load(run, paths):
    meals, people = seven.load(run, paths)
    names = header(paths['meals'])
    cols = ['participant_id', 'meal_id', 'connection_id']
    if 'iauc_positive_0_120' in names:
        cols.append('iauc_positive_0_120')
    raw = read_csv(paths['meals'], cols, usecols=cols)
    if 'iauc_positive_0_120' not in raw:
        raw['iauc_positive_0_120'] = np.nan
    qm = read_csv(paths['quality_meals'])
    meals = attach(meals, raw, qm)
    run.table(pd.DataFrame([dict(ready_meals=len(meals), positive_column_present='iauc_positive_0_120' in names,
        positive_missing_meals=int(meals.positive_mean.isna().sum()), joint_meals=int(meals.joint.sum()))]), '01_positive_input_audit.csv')
    cg = select_participant_scope(run, read_csv(paths['whole_cgm']), 'whole_cgm')
    cg['connection_id'] = norm(cg.connection_id)
    require(cg.connection_id.notna().all(), 'Missing whole CGM connection ID')
    for _, f in CGM20:
        cg[f + '_z'] = seven.strict_number(cg[f + '_z'], f + '_z')
    run.table(pd.DataFrame([dict(label=l, field=f, source_column=f + '_z', source_valid=int(cg[f + '_z'].notna().sum()),
        meaning='mean within-day SD across days' if l == 'SDw' else l) for l, f in CGM20]), '01_cgm20_mapping.csv')
    members = read_csv(paths['quality_members'])
    for col in ['participant_id', 'meal_id']:
        members[col] = norm(members[col])
    return meals, people, cg, pd.read_csv(paths['quality_models']), members
