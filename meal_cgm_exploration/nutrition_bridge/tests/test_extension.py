"""Independent calculations and adversarial IO; all data are synthetic."""
import ast
import json
import os
import sqlite3
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from hppmeal.common import Run, sha256, write_json
from hppmeal.cgm import match_meal
from seven_diet_meals.tests.test_analysis import synthetic
from seven_diet_meals.models import features, prepare, attempt, aggregate_people, encode, extras, BASE, CONTEXT
from nutrition_bridge import inputs
from nutrition_bridge.analysis import (diagnostics, adjustment, positive, bridge_sample, bridge_fit,
    planned_fdr, run_models, STEPS)
from nutrition_bridge.run_analysis import main
from nutrition_bridge.reporting import show


def test_influence_matches_literal_leave_one_out():
    rng = np.random.RandomState(207)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=['score', 'a', 'b'])
    X['b'] += X.score * 2
    y = 2 + 1.4 * X.score - .3 * X.a + rng.normal(size=100)
    z = np.column_stack([np.ones(100), X])
    beta = np.linalg.lstsq(z, y, rcond=None)[0]
    differences = []
    for i in range(100):
        keep = np.arange(100) != i
        differences.append(np.linalg.lstsq(z[keep], y[keep], rcond=None)[0][1] - beta[1])
    got = diagnostics(y, X, 'score')
    assert got['max_abs_leave1_coefficient_change'] == pytest.approx(np.max(np.abs(differences)), rel=1e-9)
    residual = X.score - np.column_stack([np.ones(100), X[['a', 'b']]]).dot(
        np.linalg.lstsq(np.column_stack([np.ones(100), X[['a', 'b']]]), X.score, rcond=None)[0])
    assert got['score_vif'] == pytest.approx(np.var(X.score) / np.var(residual), rel=1e-9)
    assert 0 < got['score_residual_variance_fraction'] < 1


def test_adjustment_preserves_final_and_hc3_independent_reference():
    m, p = synthetic(110, 15)
    sample, _ = prepare(features(m), p, 'Carbohydrate_pct', 'primary_available', 'A')
    rows = [adjustment(sample, 'Carbohydrate_pct', 'source_primary', 'mean_increment', k, a)[0] for k, a in STEPS]
    old, _ = attempt(sample, 'Carbohydrate_pct', 'source_primary', 'A', 'mean_increment', 'meal_adjusted')
    assert rows[-1]['estimate'] == pytest.approx(old['estimate'], abs=1e-12)
    assert rows[-1]['se'] == pytest.approx(old['se'], abs=1e-12)
    assert len(set(r['meal_sha256'] for r in rows)) == 1
    pp = aggregate_people(sample)
    X = encode(pp, ['Carbohydrate_pct'] + BASE + extras('Carbohydrate_pct') + CONTEXT + ['carb_10', 'energy_100'])
    a = np.column_stack([np.ones(len(pp)), X])
    inv = np.linalg.inv(a.T.dot(a))
    b = inv.dot(a.T).dot(pp.mean_increment)
    residual = pp.mean_increment.to_numpy() - a.dot(b)
    h = np.sum(a.dot(inv) * a, axis=1)
    cov = inv.dot(a.T.dot(a * (residual / (1 - h))[:, None] ** 2)).dot(inv)
    assert rows[3]['estimate'] == pytest.approx(b[1], abs=1e-8)
    assert rows[3]['se'] == pytest.approx(np.sqrt(cov[1, 1]), abs=1e-8)


def test_positive_identical_to_net_and_missing_never_changes_members():
    m, p = synthetic(110, 15)
    m['positive_mean'] = m.mean_increment  # Identity case for model arithmetic.
    for branch in ['A', 'B']:
        sample, _ = prepare(features(m), p, 'AHEI', 'primary_available', branch)
        model = 'meal_adjusted' if branch == 'A' else 'carb_interaction'
        r, _ = positive(sample, 'AHEI', 'joint', branch, model)
        old, _ = attempt(sample, 'AHEI', 'joint', branch, 'mean_increment', model)
        assert r['estimate'] == pytest.approx(old['estimate'], abs=1e-12)
        assert r['se'] == pytest.approx(old['se'], abs=1e-12)
        sample.loc[sample.index[0], 'positive_mean'] = np.nan
        missing, coefs = positive(sample, 'AHEI', 'joint', branch, model)
        assert missing['status'] == 'NOT_ESTIMATED' and coefs == []
        assert missing['person_sha256'] == old['person_sha256'] and missing['meal_sha256'] == old['meal_sha256']
        assert np.isnan(missing['estimate']) and 'no deletion' in missing['reason']


@pytest.mark.parametrize('fault', ['connection', 'negative', 'less_than_net', 'malformed', 'wrong_person'])
def test_positive_and_connection_input_rejection(fault):
    m, _ = synthetic(3, 12)
    m['connection_id'] = '01'
    qm = m[['participant_id', 'meal_id', 'connection_id']].assign(joint=True)
    raw = qm.drop(columns='joint').assign(iauc_positive_0_120=np.maximum(m.iauc_net_0_120, 0) + 120)
    if fault == 'connection': raw.loc[0, 'connection_id'] = 'other'
    if fault == 'negative': raw.loc[0, 'iauc_positive_0_120'] = -120
    if fault == 'less_than_net':
        m.loc[0, 'mean_increment'] = 100
        raw.loc[0, 'iauc_positive_0_120'] = 1
    if fault == 'malformed': raw['iauc_positive_0_120'] = 'bad'
    if fault == 'wrong_person': qm.loc[0, 'participant_id'] = 'unmatched'
    with pytest.raises(ValueError): inputs.attach(m, raw, qm)


def test_bridge_matches_connection_and_needs_no_diet_score():
    m, p = synthetic(110, 15)
    m['positive_mean'] = np.maximum(m.mean_increment, 0) + 1
    m['joint'] = True
    m['connection_id'] = 'c'
    cg = pd.DataFrame({'participant_id': p.index, 'connection_id': 'c'})
    rng = np.random.RandomState(62)
    for _, f in inputs.CGM20: cg[f + '_z'] = rng.normal(size=len(cg))
    cg.loc[0, 'connection_id'] = 'other'
    p.loc[:, inputs.seven.DIETS] = np.nan
    d, flow, members = bridge_sample(features(m), p, cg, 'joint')
    assert len(d) == int(p.primary_eligible.sum()) - 1
    assert cg.participant_id.iloc[0] not in set(d.participant_id)
    r, _ = bridge_fit(d, 'mean_increment', 'Mean', 'cgm_mean', 'joint')
    assert r['status'] == 'ESTIMATED'
    assert r['people'] == len(d)
    d.loc[0, 'positive_mean'] = np.nan
    r2, _ = bridge_fit(d, 'positive_mean', 'Mean', 'cgm_mean', 'joint')
    assert r2['people'] == len(d) - 1
    d['cgm_mean_z'] = 0
    r3, _ = bridge_fit(d, 'mean_increment', 'Mean', 'cgm_mean', 'joint')
    assert r3['status'] == 'NOT_ESTIMATED'


def test_planned_fdr_and_python38_syntax():
    d = pd.DataFrame({'setting': ['source_primary'] * 60, 'p': [.001] + [np.nan] * 59})
    r = planned_fdr(d, 60)
    assert r.q.iloc[0] == pytest.approx(.06)
    assert r.q.iloc[1:].isna().all()
    for path in inputs.MODULE.rglob('*.py'):
        tree = ast.parse(path.read_text(), feature_version=(3, 8))
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                assert not isinstance(node.annotation, ast.Subscript)


def build_source(tmp_path, n=100):
    """Build frozen synthetic meal+seven+quality inputs with true positive areas."""
    import quality_timecourse.tests.test_extension as fixture
    original_seven = fixture.seven_main

    def add_positive_then_seven(args):
        source = Path(args[args.index('--run-dir') + 1])
        mf = source / 'manifest.json'
        manifest = json.loads(mf.read_text())
        meals = pd.read_csv(source / 'private/02_meals.csv', dtype={'participant_id': str})
        with sqlite3.connect(source / 'private/cache.sqlite') as con:
            for pid, group in meals.groupby('participant_id'):
                points = pd.read_sql_query('SELECT connection_id,t,glucose FROM cgm WHERE participant_id=? AND valid=1 ORDER BY t', con, params=[pid])
                connections = {k: (g.t.to_numpy(), g.glucose.to_numpy()) for k, g in points.groupby('connection_id')}
                for i, meal in group.iterrows():
                    response, _ = match_meal(connections, meal.start, manifest['config']['cgm'])
                    meals.loc[i, 'iauc_positive_0_120'] = response['iauc_positive_0_120']
        meals.to_csv(source / 'private/02_meals.csv', index=False)
        manifest['status']['build']['meals_sha256'] = sha256(source / 'private/02_meals.csv')
        write_json(mf, manifest)
        return original_seven(args)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fixture, 'seven_main', add_positive_then_seven)
        root, source, seven = fixture.make_curve_source(tmp_path, n=n)
    assert fixture.main(['--meal-run', str(source), '--seven-run', str(seven), '--data-root', str(root), '--output-root', str(tmp_path / 'quality')]) == 0
    quality = next((tmp_path / 'quality').glob('quality_timecourse_*'))
    m = pd.read_csv(source / 'private/02_meals.csv', dtype={'participant_id': str})
    cg = m.drop_duplicates('participant_id')[['participant_id', 'cohort', 'research_stage', 'connection_id']].copy()
    rng = np.random.RandomState(175)
    mean = m.groupby('participant_id').iauc_net_0_120.mean().reindex(cg.participant_id).to_numpy() / 120
    for _, f in inputs.CGM20:
        value = .35 * mean + rng.normal(0, 3, len(cg))
        cg[f + '_z'] = (value - value.mean()) / value.std(ddof=1)
    cgfile = root / inputs.DEFAULT_CGM
    cgfile.parent.mkdir(parents=True)
    cg.to_csv(cgfile, index=False)
    return root, source, seven, quality


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    return build_source(tmp_path_factory.mktemp('nutrition bridge synthetic'))


def test_cli_readonly_complete_and_all_pages(source, tmp_path, capsys):
    root, original, seven, quality = source
    pinned = {str(f): sha256(f) for folder in [root, seven, quality] for f in folder.rglob('*') if f.is_file()}
    assert main(['--quality-run', str(quality), '--data-root', str(root), '--output-root', str(tmp_path / 'results')]) == 0
    out = next((tmp_path / 'results').glob('nutrition_bridge_*'))
    m = json.loads((out / 'manifest.json').read_text())
    state = m['status']['nutrition_bridge']
    assert state['status'] == 'COMPLETE', (out / 'SUMMARY.md').read_text()
    assert state['reproduced'] == 84 and state['estimated_models'] == 302
    for name, n in [('04_adjustment_models.csv', 140), ('05_positive_models.csv', 42), ('06_bridge_models.csv', 120)]:
        table = pd.read_csv(out / 'tables' / name)
        assert len(table) == n and table.status.eq('ESTIMATED').all()
    for f in (out / 'tables').glob('*.csv'):
        assert 'participant_id' not in pd.read_csv(f, nrows=0).columns
    for p, digest in pinned.items(): assert sha256(p) == digest
    for page in range(1, 55):
        assert main(['--show', str(out), '--page', str(page)]) == 0
    assert main(['--show', 'latest', '--output-root', str(tmp_path / 'results')]) == 0
    assert 'SYNTHETIC' in (out / 'pages/01.txt').read_text()
    assert max(len(line) for line in (out / 'pages/07.txt').read_text().splitlines()) < 135
    # Altered screenshot output must never be silently shown as a verified run.
    with (out / 'pages/02.txt').open('a') as f: f.write('tampered')
    assert main(['--show', str(out), '--page', '2']) == 2


def test_scope_and_output_hash_rejected(source, tmp_path):
    root, original, seven, quality = source
    out = tmp_path / 'verify'; out.mkdir()
    r, c, paths, expected, _ = inputs.inspect(quality, root, out)
    cg = pd.read_csv(paths['whole_cgm'], dtype=str)
    cg['research_stage'] = 'wrong_visit'
    bad = tmp_path / 'bad_cgm.csv'; cg.to_csv(bad, index=False)
    paths['whole_cgm'] = bad
    with pytest.raises(ValueError, match='no rows'):
        inputs.load(Run(r, tmp_path / 'load', c), paths)
    # Copy only consumed quality files and change one; original source stays intact.
    import shutil
    copy = tmp_path / 'altered_quality'
    shutil.copytree(quality, copy)
    with (copy / 'private/03_meal_quality_timecourse.csv').open('a') as f: f.write('\n')
    with pytest.raises(ValueError, match='changed/missing'):
        inputs.inspect(copy, root, out)


@pytest.mark.parametrize('empty_joint', [False, True])
def test_no_support_keeps_planned_rows_and_bridge_independent_of_scores(empty_joint):
    from quality_timecourse.tests.test_extension import augmented_fixture
    from quality_timecourse.analysis import run_models as quality_models
    m, p, previous = augmented_fixture(65)
    m['positive_mean'] = np.maximum(m.mean_increment, 0) + 1
    m['connection_id'] = 'c'
    if empty_joint:
        m['joint'] = False
    qr = quality_models(m, p, previous, lambda _: None)
    # No positive outcomes: other branches should remain available; every positive specification retained.
    m['positive_mean'] = np.nan
    cg = p.reset_index()[['participant_id']].assign(connection_id='c')
    for _, f in inputs.CGM20: cg[f + '_z'] = 0.
    r = run_models(m, p, cg, qr['models'], qr['joint_membership'], lambda _: None)
    assert len(r['positive']) == 42 and r['positive'].status.eq('NOT_ESTIMATED').all()
    assert len(r['bridge']) == 120 and r['bridge'].status.eq('NOT_ESTIMATED').all()
    assert len(r['adjustment']) == 140


def test_launcher_from_arbitrary_working_directory(tmp_path):
    env = dict(os.environ, HPP_MEAL_PYTHON=os.sys.executable)
    result = subprocess.run(['bash', str(inputs.MODULE / 'run_nutrition_bridge.sh'), '--help'], cwd=str(tmp_path), env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0 and '--quality-run' in result.stdout
