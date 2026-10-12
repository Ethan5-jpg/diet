"""Read aggregate files only. Never imports or calls the fitting engine."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['exposure', 'taxon', 'outcome']
MODELS = ['strict', 'protocol_same_sample', 'protocol_adjusted', 'quadratic',
          'covariate_interactions', 'common_all7_sample']
CODES = ['S', 'P', 'A', 'Q', 'C', 'U']
OUTCOMES = ['Mean', 'CV', 'SDw']
TAXA = ['T023', 'T046', 'T068', 'T096']
SPECIES = {'T023': 'GGB6012_SGB8640', 'T046': 'C. phoceensis',
           'T068': 'E. rectale', 'T096': 'R. lactatiformans'}
FULL_SPECIES = {'T023':'GGB6012_SGB8640','T046':'Clostridium phoceensis',
                'T068':'Eubacterium rectale','T096':'Ruthenibacterium lactatiformans'}
DIETS = ['hPDI', 'NOVA4', 'AHEI', 'AMED', 'rEDIH', 'EAT13', 'Carbohydrate_pct']
SHORT = {'Carbohydrate_pct': 'Carb %'}
UNITS = {'Mean': 'mg/dL', 'CV': 'percentage points', 'SDw': 'mg/dL'}
BETA = 'interaction_beta_native'
LOW = 'interaction_ci_low_native'
HIGH = 'interaction_ci_high_native'
MODEL_FIELDS = KEY + ['species', 'family', 'model', 'status', 'reason', 'n', 'unit',
    BETA, 'interaction_se_native', LOW, HIGH, 'interaction_p', 'q_focus24', 'q_all84',
    'double_iqr_beta_native', 'double_iqr_ci_low_native', 'double_iqr_ci_high_native',
    'double_iqr_status', 'parameters', 'residual_df', 'covariates', 'condition_number',
    'max_leverage', 'high_leverage_n', 'r2', 'interaction_delta_r2', 'support_min_tail_n',
    'limited_joint_support']
SLOPE_FIELDS = KEY + ['species', 'model', 'family', 'n', 'unit', 'microbial_quantile',
                     'microbial_original_z', 'beta_native', 'ci_low_native', 'ci_high_native',
                     'p_pointwise', 'slope_note']


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def true(series):
    return series.map(lambda v: str(v).lower() in ('true', 'yes', '1'))


def public(frame, columns):
    return frame.loc[:, [c for c in columns if c in frame]].copy()


def normalize(frame):
    return frame.rename(columns={'q_all': 'q_all84', 'q_focus': 'q_focus24',
                                 'ci_low_native': LOW, 'ci_high_native': HIGH})


def load_preview(package):
    path = package / 'screenshot_sources/snapshot.json'
    data = read_json(path)
    bundle = {k: pd.DataFrame(v) for k, v in data['tables'].items()}
    for frame in bundle.values():
        if 'taxon' in frame:
            frame['species'] = frame.taxon.map(FULL_SPECIES)
    bundle['meta'] = data['meta']
    bundle['meta']['source_files'] = {str(path): sha(path)}
    return bundle


def checked_csv(root, manifest, name, fingerprints):
    path = root / name
    require(name in manifest.get('output_sha256', {}), 'Missing saved hash: ' + name)
    require(path.is_file() and sha(path) == manifest['output_sha256'][name], 'Saved result changed: ' + str(path))
    fingerprints[str(path)] = sha(path)
    return pd.read_csv(path)


def grid(frame, keys, expected, label):
    actual = list(frame[keys].itertuples(index=False, name=None))
    require(len(actual) == len(set(actual)) and set(actual) == set(expected), label + ': missing/duplicate/unexpected rows')


def load_server(old, new, allow_synthetic=False):
    old, new = old.resolve(), new.resolve()
    om, nm = read_json(old/'manifest.json'), read_json(new/'manifest.json')
    kind = 'SYNTHETIC' if allow_synthetic else 'SERVER_INPUTS'
    require(om.get('data_kind') == nm.get('data_kind') == kind, 'Data kind mismatch (synthetic data cannot be published as real)')
    require(om.get('version') == '2026-10-10.2' and nm.get('version') == '2026-10-10.ext1', 'Unsupported run versions')
    require(all(m.get('status', '').startswith('COMPLETE') for m in (om, nm)), 'Incomplete analysis run')
    require(om['config'] == nm['config'], 'Old/new configuration mismatch')
    require(Path(nm['reference_run']).resolve() == old, 'Follow-up refers to another original run')
    fingerprints = {str(old/'manifest.json'): sha(old/'manifest.json'), str(new/'manifest.json'): sha(new/'manifest.json')}
    for name, digest in nm['reference_file_sha256'].items():
        require(sha(old/name) == digest, 'Original reference changed: ' + name)
        fingerprints[str(old/name)] = digest
    original = normalize(checked_csv(old, om, 'tables/all_interactions.csv', fingerprints))
    current = checked_csv(new, nm, 'tables/all_interactions.csv', fingerprints)
    slopes = checked_csv(new, nm, 'tables/conditional_diet_slopes.csv', fingerprints)
    stability = checked_csv(new, nm, 'tables/reference_hits_stability.csv', fingerprints)
    summary = checked_csv(new, nm, 'tables/retention_by_specification.csv', fingerprints)
    fdr = checked_csv(new, nm, 'tables/fdr_families.csv', fingerprints)
    reproduction = checked_csv(new, nm, 'tables/reference_reproduction.csv', fingerprints)
    require(om['config']['alpha'] == 0.05, 'This paper specification requires alpha=0.05')
    expected = list(itertools.product(DIETS, TAXA, OUTCOMES))
    grid(current, KEY+['model'], [k+(m,) for k in expected for m in ['primary']+MODELS], 'Follow-up grid')
    old_models = MODELS[:-1]+['common_focus_sample']
    grid(original, KEY+['model'], [k+(m,) for k in expected for m in ['primary']+(old_models if k[0] in DIETS[:2] else [])], 'Original grid')
    require(len(reproduction) == 204 and true(reproduction.matches).all(), '204-row reproduction failed')
    # Re-read and independently compare saved coefficients, identities and q, not just a status label.
    a = original[~original.model.eq('common_focus_sample')].set_index(KEY+['model']).sort_index()
    b = current.set_index(KEY+['model']).loc[a.index]
    require((a.sample_sha256 == b.sample_sha256).all() and (a.status == b.status).all(), 'Reference sample/status mismatch')
    for c in ['n', BETA, LOW, HIGH, 'interaction_se_native', 'interaction_p']:
        require(np.allclose(a[c], b[c], rtol=1e-7, atol=1e-10, equal_nan=True), 'Reference coefficient mismatch: '+c)
    prim = a.index.get_level_values('model') == 'primary'
    require(np.allclose(a.loc[prim, 'q_all84'], b.loc[prim, 'q_all84'], rtol=1e-7, atol=1e-10, equal_nan=True), 'Reference q mismatch')
    whole = fdr[fdr.family.eq('all84')]
    require(len(whole) == 7 and set(whole.model) == set(['primary']+MODELS) and whole.planned_tests.eq(84).all(), 'Wrong FDR denominator')
    primary = original[original.model.eq('primary')].copy()
    selected = primary[primary.family.eq('extension') & primary.status.eq('OK') & primary.q_all84.lt(.05)]
    grid(stability, KEY, selected[KEY].itertuples(index=False, name=None), 'Frozen extension target set')
    ix = current.set_index(KEY+['model'])
    for _, r in stability.iterrows():
        key = tuple(r[k] for k in KEY)
        base = selected.set_index(KEY).loc[key]
        require(np.isclose(r.reference_q_all84, base.q_all84), 'Frozen q differs')
        count = 0
        for m in MODELS:
            row = ix.loc[key+(m,)]
            require(np.isclose(r[m+'__q_all84'], row.q_all84, equal_nan=True), 'Stability q differs')
            keep = row.status == 'OK' and row.q_all84 < .05 and row[BETA]*base[BETA] > 0
            count += int(keep)
        require(count == int(r.retained_sensitivity_count) and (count == 6) == (str(r.retained_all_six).lower() == 'true'), 'Retained count differs')
    # N belongs to the model; conditional slopes themselves are not subgroup regressions.
    slopes = slopes.merge(current[KEY+['model','n']], on=KEY+['model'], validate='many_to_one')
    meta = dict(mode='SYNTHETIC_TEST' if allow_synthetic else 'SERVER_FULL_PRECISION',
                source_files=fingerprints, original_run=old.name, followup_run=new.name,
                old_status=om['status'], new_status=nm['status'], reference_reproduced=204,
                actual_fits=int(current.status.eq('OK').sum()), planned_fits=588,
                counts=om.get('input_summary', {}), corrections='19:10 historical q verified in saved tables',
                limitations=nm.get('limitations', []))
    return dict(primary=primary, original=original, extension=current[current.family.eq('extension')].copy(),
                all_followup=current, slopes=slopes, stability=stability, summary=summary,
                fdr=fdr, reproduction=reproduction, meta=meta)


def recheck_sources(bundle):
    require(all(sha(p) == digest for p, digest in bundle['meta']['source_files'].items()), 'Source files changed during export')
