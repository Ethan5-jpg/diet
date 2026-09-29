"""Synthetic validation; no real HPP data in fixtures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from hppmeal.common import Run, sha256, write_json
from hppmeal.models import bh_planned
from seven_diet_meals.inputs import (PACKAGE, DIETS, inspect, load, record_expected, unchanged, bundle)
from seven_diet_meals.models import (linear_fit, features, prepare, run_models, aggregate_people)
from seven_diet_meals.run_analysis import main


def synthetic(n=160, meals_each=12, seed=52):
    rng = np.random.RandomState(seed)
    ids = ['{:06d}'.format(i) for i in range(n)]
    score = rng.normal(size=(n, 7))
    people = pd.DataFrame(score, columns=DIETS, index=pd.Index(ids, name='participant_id'))
    people['age_years'] = rng.uniform(30, 65, n)
    people['bmi'] = rng.uniform(19, 34, n)
    people['sex'] = rng.choice(['Female', 'Male'], n)
    people['education_level'] = rng.choice(['1', '2', '3'], n)
    people['smoking_status'] = rng.choice(['never', 'past', 'current'], n)
    people['cgm_device_type'] = rng.choice(['Libre', 'Other'], n)
    people['sleep_duration_hours_day'] = rng.uniform(5, 9, n)
    people['physical_activity_met_h_week'] = rng.uniform(5, 70, n)
    people['vitamin_use'] = rng.randint(0, 2, n)
    people['hormone_use'] = rng.randint(0, 2, n)
    people['alcohol_intake_g_day'] = rng.uniform(0, 30, n)
    people['mean_daily_energy_kcal'] = rng.uniform(1500, 2800, n)
    people['primary_eligible'] = True
    people['strict_eligible'] = True
    rows = []
    for i, pid in enumerate(ids):
        intercept = 7 + 2.0 * score[i, 0] + rng.normal()
        for j in range(meals_each):
            carb = rng.uniform(10, 100)
            energy = carb * 4 + rng.uniform(70, 350)
            baseline = rng.uniform(75, 110)
            increment = intercept + (.4 + .3 * score[i, 0]) * carb / 10 + rng.normal(0, 1)
            hour = [8, 13, 19][j % 3] + rng.uniform(-.3, .3)
            start = 1704067200 + (j // 3) * 86400 + hour * 3600
            rows.append(dict(participant_id=pid, cohort='10k', research_stage='00_00_visit', meal_id=pid + '_' + str(j),
                             local_date='2024-01-{:02d}'.format(1 + j // 3), start=start, local_hour=hour,
                             analysis_ready=True, duration_minutes=rng.uniform(5, 25), carbohydrate_g=carb,
                             calories_kcal=energy, baseline_glucose=baseline, pre_slope_per_min=rng.normal(0, .15),
                             previous_gap_minutes=rng.uniform(150, 600), iauc_net_0_120=120 * increment,
                             peak_rise_0_120=increment * 1.7 + rng.normal(), mean_increment=increment, mean_glucose=increment + baseline))
    return pd.DataFrame(rows), people


def make_source(tmp_path, n=160, meals_each=12):
    root = tmp_path / 'Data'
    source = root / 'meal_cgm_exploration/outputs/original run'
    (source / 'private').mkdir(parents=True)
    (root / 'SYNTHETIC_TEST_ONLY.json').write_text('{}')
    c = json.loads((PACKAGE / 'config/default.json').read_text())
    c['data_kind'] = 'SYNTHETIC'
    c['time']['consumption_time_verified'] = True
    meals, people = synthetic(n, meals_each)
    meals.to_csv(source / 'private/02_meals.csv', index=False)
    cov = people.reset_index().drop(columns=DIETS + ['mean_daily_energy_kcal', 'primary_eligible', 'strict_eligible'])
    cov['cohort'] = '10k'
    cov['cgm_research_stage'] = '00_00_visit'
    cov['exclude_diabetes_or_a10'] = 0.0
    cov.loc[cov.index % 17 == 0, 'exclude_diabetes_or_a10'] = 1.0
    cov.loc[cov.index % 13 == 0, 'exclude_diabetes_or_a10'] = np.nan
    cp = root / c['inputs']['covariates']
    cp.parent.mkdir(parents=True)
    cov.to_csv(cp, index=False)
    config = json.loads((PACKAGE / 'seven_diet_meals/sources.json').read_text())
    frames = {
        'current4': people[['AHEI', 'AMED', 'rEDIH']].rename(columns=lambda x: x + '_z').assign(hPDI_z=999),
        'corrected_hpdi': people[['hPDI']].rename(columns={'hPDI': 'hpdi_score_energy_adjusted_z'}),
        'new3': people[['EAT13', 'NOVA4', 'Carbohydrate_pct']].rename(columns=lambda x: x + '_z').join(people[['mean_daily_energy_kcal']])}
    for label, frame in frames.items():
        p = root / config[label]
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.reset_index().to_csv(p, index=False)
    core = {str(p.relative_to(PACKAGE)): sha256(p) for folder in ['hppmeal', 'scripts'] for p in sorted((PACKAGE / folder).glob('*.py'))}
    write_json(source / 'code_fingerprint.json', core)
    write_json(source / 'manifest.json', dict(data_kind='SYNTHETIC', config=c, root=str(root),
                status=dict(audit=dict(status='PASS'), build=dict(status='PASS', meals_sha256=sha256(source / 'private/02_meals.csv')),
                            analyze=dict(status='PASS')), inputs=[dict(label='covariates', path=str(cp), sha256=sha256(cp))]))
    return root, source


def source_loaded(tmp_path):
    root, source = make_source(tmp_path)
    root, c, paths, expected, audit = inspect(source, root, PACKAGE / 'seven_diet_meals/sources.json')
    run = Run(root, tmp_path / 'new', c)
    record_expected(run, expected)
    meals, people = load(run, paths)
    return root, source, run, meals, people, paths, expected


def test_hc3_matches_independent_matrix_reference():
    rng = np.random.RandomState(5)
    X = pd.DataFrame(rng.normal(size=(150, 2)), columns=['score', 'age'])
    y = 2 + 1.3 * X.score - .4 * X.age + rng.normal(size=150) * (1 + X.age.abs())
    rows, info = linear_fit(y, X, 'score')
    design = np.column_stack([np.ones(150), X])
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ y
    residual = y - design @ beta
    h = np.einsum('ij,jk,ik->i', design, bread, design)
    scores = design * (residual / (1 - h)).to_numpy()[:, None]
    covariance = bread @ scores.T @ scores @ bread
    row = next(x for x in rows if x['term'] == 'score')
    assert row['estimate'] == pytest.approx(beta[1], abs=1e-10)
    assert row['se'] == pytest.approx(np.sqrt(covariance[1, 1]), abs=1e-10)


def test_fe_matches_explicit_weighted_dummy_cluster_reference():
    rng = np.random.RandomState(9)
    ids = np.repeat(np.arange(60), rng.randint(6, 12, 60))
    n = len(ids)
    dietary = rng.normal(size=60)
    carb = rng.normal(size=n)
    X = pd.DataFrame({'carb': carb, 'interaction': carb * dietary[ids]})
    y = rng.normal(size=60)[ids] + .8 * carb + .5 * X.interaction + rng.normal(0, .2, n)
    weights = 1 / np.bincount(ids)[ids]
    rows, diagnostic = linear_fit(y, X, 'interaction', ids, weights)
    design = np.column_stack([np.eye(60)[ids], X]) * np.sqrt(weights[:, None])
    target = y.to_numpy() * np.sqrt(weights)
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ target
    resid = target - design @ beta
    scores = np.zeros((60, design.shape[1]))
    np.add.at(scores, ids, design * resid[:, None])
    cov = 60 / 59 * (n - 1) / (n - design.shape[1]) * bread @ scores.T @ scores @ bread
    row = next(x for x in rows if x['term'] == 'interaction')
    assert row['estimate'] == pytest.approx(beta[-1], abs=1e-9)
    assert row['se'] == pytest.approx(np.sqrt(cov[-1, -1]), abs=1e-9)
    assert row['estimate'] == pytest.approx(.5, abs=.05)
    assert diagnostic['absorbed_effects'] == 60


def test_person_constant_score_cannot_be_estimated_in_fe():
    ids = np.repeat(np.arange(60), 6)
    X = pd.DataFrame({'score': ids.astype(float), 'carb': np.tile(np.arange(6), 60)})
    with pytest.raises(ValueError, match='constant/unidentifiable'):
        linear_fit(np.random.RandomState(3).normal(size=len(ids)), X, 'score', ids, np.ones(len(ids)) / 6)


def test_rank_deficiency_is_explicit():
    x = np.linspace(-1, 1, 100)
    with pytest.raises(ValueError, match='Rank-deficient'):
        linear_fit(x, pd.DataFrame({'score': x, 'copy': x * 2}), 'score')


def test_person_aggregation_does_not_upweight_frequent_meals():
    meals, people = synthetic(80)
    base = features(meals).merge(people.reset_index(), on='participant_id')
    many = pd.concat([base] + [base.loc[base.participant_id.eq('000000')]] * 9, ignore_index=True)
    a, b = aggregate_people(base), aggregate_people(many)
    assert np.allclose(a.mean_increment, b.mean_increment)
    assert len(a) == len(b) == 80


def test_common_sample_identical_and_available_sample_specific():
    meals, people = synthetic(80)
    people.loc['000000', 'AHEI'] = np.nan
    d = features(meals)
    a, _ = prepare(d, people, 'AHEI', 'primary_available', 'A')
    b, _ = prepare(d, people, 'AMED', 'primary_available', 'A')
    assert a.participant_id.nunique() == 79 and b.participant_id.nunique() == 80
    samples = [prepare(d, people, x, 'primary_common7', 'A')[0] for x in DIETS]
    assert all(set(x.meal_id) == set(samples[0].meal_id) for x in samples)


def test_interaction_support_requires_meals_dates_and_carb_range():
    meals, people = synthetic(80)
    meals.loc[meals.participant_id.eq('000000'), 'carbohydrate_g'] = 30
    meals.loc[meals.participant_id.eq('000001'), 'local_date'] = '2024-01-01'
    meals = meals.loc[~(meals.participant_id.eq('000002') & meals.meal_id.str.endswith(('_9', '_10', '_11')))]
    d, _ = prepare(features(meals), people, 'AHEI', 'primary_available', 'B')
    assert not set(['000000', '000001', '000002']) & set(d.participant_id)


def test_original_z_preserved_corrected_hpdi_used(tmp_path):
    root, source, run, meals, people, paths, expected = source_loaded(tmp_path)
    original = pd.read_csv(paths['corrected_hpdi'], dtype={'participant_id': str}).set_index('participant_id')
    assert np.allclose(people.loc[original.index, 'hPDI'], original.hpdi_score_energy_adjusted_z)
    assert not people.hPDI.eq(999).any()
    assert meals.participant_id.str.len().eq(6).all()
    assert np.allclose(meals.mean_increment, meals.iauc_net_0_120 / 120)


def test_missing_covariate_row_excluded_unknown_flag_only_primary(tmp_path):
    root, source = make_source(tmp_path)
    old = json.loads((source / 'manifest.json').read_text())
    cp = root / old['config']['inputs']['covariates']
    cov = pd.read_csv(cp, dtype={'participant_id': str})
    cov = cov.loc[cov.participant_id.ne('000001')]
    cov.to_csv(cp, index=False)
    old['inputs'][0]['sha256'] = sha256(cp)
    write_json(source / 'manifest.json', old)
    root, c, paths, expected, audit = inspect(source, root, PACKAGE / 'seven_diet_meals/sources.json')
    run = Run(root, tmp_path / 'new', c)
    _, people = load(run, paths)
    assert not people.loc['000001', 'primary_eligible']
    assert people.loc['000013', 'primary_eligible']
    assert not people.loc['000013', 'strict_eligible']


def test_changed_original_meals_rejected(tmp_path):
    root, source = make_source(tmp_path)
    with (source / 'private/02_meals.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='differs from original'):
        inspect(source, root, PACKAGE / 'seven_diet_meals/sources.json')


def test_change_after_validation_rejected(tmp_path):
    root, source = make_source(tmp_path)
    root, c, paths, expected, audit = inspect(source, root, PACKAGE / 'seven_diet_meals/sources.json')
    with paths['current4'].open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='between validation'):
        record_expected(Run(root, tmp_path / 'out', c), expected)


def test_change_after_loading_rejected(tmp_path):
    root, source, run, meals, people, paths, expected = source_loaded(tmp_path)
    with paths['corrected_hpdi'].open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='changed during'):
        unchanged(run)


@pytest.mark.parametrize('problem', ['duplicate', 'wrong_visit', 'wrong_cgm_visit', 'conflicting_visits', 'invalid_score'])
def test_bad_score_table_rejected(tmp_path, problem):
    root, source = make_source(tmp_path)
    config = json.loads((PACKAGE / 'seven_diet_meals/sources.json').read_text())
    path = root / config['current4']
    score = pd.read_csv(path, dtype=str)
    if problem == 'duplicate':
        score = pd.concat([score, score.iloc[[0]]])
    elif problem == 'wrong_visit':
        score['research_stage'] = '01_00_visit'
    elif problem in ['wrong_cgm_visit', 'conflicting_visits']:
        score['cgm_research_stage'] = '01_00_visit'
        if problem == 'conflicting_visits':
            score['research_stage'] = '00_00_visit'
    else:
        score.loc[0, 'AHEI_z'] = 'bad'
    score.to_csv(path, index=False)
    root, c, paths, expected, audit = inspect(source, root, PACKAGE / 'seven_diet_meals/sources.json')
    with pytest.raises(ValueError):
        load(Run(root, tmp_path / 'new', c), paths)


def test_score_coverage_distinguishes_absent_and_missing(tmp_path):
    root, source = make_source(tmp_path)
    config = json.loads((PACKAGE / 'seven_diet_meals/sources.json').read_text())
    path = root / config['current4']
    score = pd.read_csv(path, dtype={'participant_id': str})
    score = score.loc[score.participant_id.ne('000000')].copy()
    score.loc[score.participant_id.eq('000001'), 'AHEI_z'] = np.nan
    score['cohort'] = '10k'
    score['cgm_research_stage'] = '00_00_visit'
    score.to_csv(path, index=False)
    root, c, paths, expected, audit = inspect(source, root, PACKAGE / 'seven_diet_meals/sources.json')
    run = Run(root, tmp_path / 'new', c)
    load(run, paths)
    coverage = pd.read_csv(run.out / 'tables/00_score_coverage.csv').set_index('diet')
    assert coverage.loc['AHEI', 'ready_meal_people'] == 160
    assert coverage.loc['AHEI', 'ready_meal_people_absent_from_source'] == 1
    assert coverage.loc['AHEI', 'ready_meal_people_present_but_score_missing'] == 1
    assert coverage.loc['AHEI', 'ready_meal_people_with_score'] == 158
    assert coverage.loc['AHEI', 'ready_meals_with_score'] == 158 * 12
    assert coverage.loc['AMED', 'ready_meal_people_with_score'] == 159
    assert coverage.loc['hPDI', 'ready_meal_people_with_score'] == 160
    scope = pd.read_csv(run.out / 'tables/00_current4_scope.csv')
    assert scope.loc[0, 'scope'] == 'explicit columns'


def test_fdr_keeps_unestimated_planned_denominator():
    values = [.01] + [np.nan] * 13
    result = bh_planned(values)
    assert result[0] == pytest.approx(.14)
    assert np.isnan(result[1:]).all()


def test_no_support_reports_every_planned_model():
    meals, people = synthetic(10)
    result, flow, coef, membership, desc = run_models(meals, people, lambda x: None)
    assert len(result) == 126
    assert result.status.eq('NOT_ESTIMATED').all()
    assert result.groupby(['setting', 'family']).size().eq(14).all()
    assert result.q_family14.isna().all()


def test_models_known_signal_and_same_background_samples():
    meals, people = synthetic(180)
    result, flow, coef, membership, desc = run_models(meals, people, lambda x: None)
    assert len(result) == 126 and result.status.eq('ESTIMATED').all(), result.loc[result.status.ne('ESTIMATED'), 'reason'].tolist()
    r = result.loc[result.setting.eq('primary_available') & result.diet.eq('AHEI') & result.branch.eq('B') & result.outcome.eq('mean_increment')].iloc[0]
    assert r.estimate == pytest.approx(.3, abs=.08)
    assert r.q_family14 < .05
    for _, g in result.groupby(['setting', 'diet', 'branch']):
        assert g.meal_sha256.nunique() == 1


def test_end_to_end_source_unchanged_and_all_scores_reported(tmp_path):
    root, source = make_source(tmp_path)
    before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
    out = tmp_path / 'results'
    rc = main(['--run-dir', str(source), '--data-root', str(root), '--output-root', str(out)])
    assert rc == 0
    run = next(out.iterdir())
    state = json.loads((run / 'manifest.json').read_text())['status']['seven_diets']
    assert state['status'] == 'COMPLETE'
    assert state['estimated_models'] == 126
    assert before == {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
    table = pd.read_csv(run / 'tables/all_models.csv')
    assert set(table.diet) == set(DIETS)
    for p in (run / 'tables').glob('*.csv'):
        assert 'participant_id' not in pd.read_csv(p, nrows=0).columns
    assert (run / 'figures/01_seven_diet_main.png').is_file()
    assert 'SYNTHETIC' in (run / 'SUMMARY.md').read_text()


@pytest.mark.parametrize('problem', ['few_people', 'no_scores', 'no_ready_meals'])
def test_unestimable_cli_exports_readable_tables(tmp_path, problem):
    root, source = make_source(tmp_path, n=10)
    if problem == 'no_scores':
        config = json.loads((PACKAGE / 'seven_diet_meals/sources.json').read_text())
        for relative in config.values():
            path = root / relative
            score = pd.read_csv(path, dtype={'participant_id': str})
            for col in score:
                if col.endswith('_z'):
                    score[col] = np.nan
            score.to_csv(path, index=False)
    elif problem == 'no_ready_meals':
        path = source / 'private/02_meals.csv'
        meals = pd.read_csv(path, dtype={'participant_id': str})
        meals['analysis_ready'] = False
        meals.to_csv(path, index=False)
        manifest = json.loads((source / 'manifest.json').read_text())
        manifest['status']['build']['meals_sha256'] = sha256(path)
        write_json(source / 'manifest.json', manifest)
    before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
    out = tmp_path / 'results'
    assert main(['--run-dir', str(source), '--output-root', str(out)]) == 0
    run = next(out.iterdir())
    state = json.loads((run / 'manifest.json').read_text())['status']['seven_diets']
    assert state['status'] == 'COMPLETE_WITH_LIMITATIONS' and state['estimated_models'] == 0
    result = pd.read_csv(run / 'tables/all_models.csv')
    assert len(result) == 126 and result.status.eq('NOT_ESTIMATED').all()
    assert result.q_family14.isna().all() and result.estimate.isna().all()
    for path in (run / 'tables').glob('*.csv'):
        assert 'participant_id' not in pd.read_csv(path).columns
    coefficients = pd.read_csv(run / 'tables/06_coefficients.csv')
    assert coefficients.empty and {'term', 'estimate', 'se', 'p'}.issubset(coefficients)
    desc = pd.read_csv(run / 'tables/05_person_response_descriptives.csv')
    assert desc.empty == (problem != 'few_people')
    if problem != 'few_people':
        assert pd.read_csv(run / 'private/02_model_membership.csv').empty
    assert before == {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}


def test_check_stage_does_not_run_models(tmp_path):
    root, source = make_source(tmp_path)
    out = tmp_path / 'check'
    assert main(['--run-dir', str(source), '--output-root', str(out), '--stage', 'check']) == 0
    run = next(out.iterdir())
    assert 'CHECK_PASS' in (run / 'SUMMARY.md').read_text()
    assert not (run / 'tables/all_models.csv').exists()


def test_outputs_inside_original_rejected(tmp_path):
    root, source = make_source(tmp_path)
    assert main(['--run-dir', str(source), '--output-root', str(source / 'bad')]) == 2
    assert not (source / 'bad').exists()


def test_show_missing_returns_clear_error(tmp_path, capsys):
    assert main(['--show', str(tmp_path / 'missing')]) == 2
    assert 'exact RUN_DIR' in capsys.readouterr().err


def test_current_bundle_is_complete():
    files = bundle()
    assert any(x.endswith('run_seven_diets.sh') for x in files)
