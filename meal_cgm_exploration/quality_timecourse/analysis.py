"""Fixed model families; quality thresholds never selected using outcomes."""
import numpy as np
import pandas as pd
from hppmeal.common import require
from hppmeal.models import bh_planned
from seven_diet_meals.inputs import DIETS
from seven_diet_meals.models import features, prepare, attempt, OUTCOMES, COEFFICIENT_COLUMNS

SETTINGS = ['source_primary', 'aligned', 'cgm90', 'log80', 'joint', 'joint_people_all_meals']
TIMES = ['early_mean', 'late_mean', 'late_minus_early']
KEYS = ['diet', 'branch', 'outcome', 'model']


def fdr(table):
    table = table.copy()
    table['family'] = np.where(table.model.eq('background'), 'background_14',
        np.where(table.outcome.eq('mean_increment'), 'primary_14', 'peak_secondary_14'))
    for _, ix in table.groupby(['setting', 'family']).groups.items():
        require(len(ix) == 14, 'Planned family must contain 14 rows')
        table.loc[ix, 'q_family14'] = bh_planned(table.loc[ix, 'p'])
    table['planned_family_size'] = 14
    return table


def reproduce(current, previous):
    old = previous.loc[previous.setting.eq('primary_available')].copy()
    require(len(old) == 42 and len(current) == 42, 'Expected 42 original primary models')
    require(not old.duplicated(KEYS).any() and not current.duplicated(KEYS).any(), 'Duplicate model specification')
    comparison = current.merge(old, on=KEYS, suffixes=('_new', '_old'), validate='one_to_one')
    require(len(comparison) == 42, 'Original model keys differ')
    for col in ['status', 'people', 'meals', 'person_sha256', 'meal_sha256']:
        require(comparison[col + '_new'].eq(comparison[col + '_old']).all(), 'Original baseline mismatch: ' + col)
    for col in ['estimate', 'se', 'ci_low', 'ci_high', 'p', 'q_family14']:
        require(np.allclose(comparison[col + '_new'], comparison[col + '_old'], rtol=1e-6, atol=1e-9, equal_nan=True), 'Original coefficients do not reproduce: ' + col)
    return comparison[KEYS + ['people_new', 'meals_new']].assign(status='MATCHED')


def fit_sample(sample, diet, setting, branch):
    rows, coef = [], []
    for outcome in OUTCOMES:
        for model in (['background', 'meal_adjusted'] if branch == 'A' else ['carb_interaction']):
            row, coefficients = attempt(sample, diet, setting, branch, outcome, model)
            rows.append(row)
            coef.extend(coefficients)
    return rows, coef


def selection_balance(base, joint, diet, branch):
    rows = []
    p = base.groupby('participant_id').agg(age_years=('age_years', 'first'), bmi=('bmi', 'first'),
        score=(diet, 'first'), meals=('meal_id', 'size'), mean_carbs=('carbohydrate_g', 'mean'), mean_energy=('calories_kcal', 'mean'))
    static = base.drop_duplicates('participant_id').set_index('participant_id')
    for field in ['sex', 'cgm_device_type']:
        for level in sorted(static[field].dropna().unique()):
            p[field + '=' + str(level) + '_proportion'] = static[field].eq(level).astype(float)
    selected = p.index.isin(joint.participant_id)
    for label, flag in [('retained_people', selected), ('excluded_people', ~selected)]:
        for field in p.columns:
            x = p.loc[flag, field]
            rows.append(dict(diet=diet, branch=branch, group=label, field=field, people=len(x), mean=x.mean(), sd=x.std(), median=x.median()))
    return rows


def run_models(meals, people, previous, progress=print):
    # Calculate time origin and other features once, before removing any date/meal.
    data = features(meals)
    rows, flows, coefs, memberships, balances, overlaps = [], [], [], [], [], []
    # First reproduce the original results before running new specifications.
    for diet in DIETS:
        progress('REPRODUCE ' + diet)
        for branch in ['A', 'B']:
            sample, flow = prepare(data, people, diet, 'primary_available', branch)
            for item in flow:
                item['setting'] = 'source_primary'
            flows.extend(flow)
            person = sample.drop_duplicates('participant_id')
            overlaps.append(dict(diet=diet, branch=branch, people=len(person), meals=len(sample),
                people_absent_from_diary=int((~person.diary_person_present.eq(True)).sum()),
                meals_without_diary_date=int((~sample.diary_date_present.eq(True)).sum()),
                people_at_least_10_diary_dates=int(person.actual_diary_dates.ge(10).sum()),
                people_at_least_14_diary_dates=int(person.actual_diary_dates.ge(14).sum())))
            r, c = fit_sample(sample, diet, 'source_primary', branch)
            rows.extend(r)
            coefs.extend(c)
    check = reproduce(fdr(pd.DataFrame(rows)), previous)
    progress('REPRODUCTION=PASS: 42 original models, memberships and estimates matched')
    time_rows, time_coef = [], []
    # Keep only one diet/branch's samples in memory at a time.
    for diet in DIETS:
        for branch in ['A', 'B']:
            base, _ = prepare(data, people, diet, 'primary_available', branch)
            joint = None
            for setting in SETTINGS[1:]:
                progress('MODELS {} {} {}'.format(setting, diet, branch))
                if setting == 'joint_people_all_meals':
                    sample = base.loc[base.participant_id.isin(joint.participant_id)].copy().reset_index(drop=True)
                    require(set(sample.participant_id) == set(joint.participant_id), 'Matched-people comparator differs')
                    flow = [dict(stage='joint_people_original_meals', people=sample.participant_id.nunique(), meals=len(sample))]
                else:
                    pool = data.loc[data[setting].eq(True)]
                    sample, flow = prepare(pool, people, diet, 'primary_available', branch)
                for item in flow:
                    item.update(setting=setting, diet=diet, branch=branch)
                flows.extend(flow)
                if setting == 'joint':
                    joint = sample
                    memberships.append(sample[['participant_id', 'meal_id']].assign(diet=diet, branch=branch))
                    balances.extend(selection_balance(base, sample, diet, branch))
                r, c = fit_sample(sample, diet, setting, branch)
                rows.extend(r)
                coefs.extend(c)
            if branch == 'A':
                for setting, sample in [('source_primary', base), ('joint', joint)]:
                    require(sample[TIMES].notna().all().all(), 'Missing early/late measurements in analysis sample')
                    for outcome in TIMES:
                        # Reuse exactly the frozen A model: replace only its outcome vector.
                        d = sample.copy()
                        d['mean_increment'] = d[outcome]
                        row, coefficients = attempt(d, diet, setting, 'A', 'mean_increment', 'meal_adjusted')
                        row['outcome'] = outcome
                        row['family'] = 'timecourse_21'
                        row['planned_family_size'] = 21
                        time_rows.append(row)
                        for c in coefficients:
                            c['outcome'] = outcome
                            time_coef.append(c)
    table = fdr(pd.DataFrame(rows))
    time_table = pd.DataFrame(time_rows)
    for _, ix in time_table.groupby('setting').groups.items():
        require(len(ix) == 21, 'Timecourse family must contain 21 planned models')
        time_table.loc[ix, 'q_family21'] = bh_planned(time_table.loc[ix, 'p'])
    return dict(models=table, flow=pd.DataFrame(flows), coefficients=pd.DataFrame(coefs, columns=COEFFICIENT_COLUMNS),
        reproduction=check, joint_membership=pd.concat(memberships, ignore_index=True), balance=pd.DataFrame(balances), diary_overlap=pd.DataFrame(overlaps),
        timecourse=time_table, time_coefficients=pd.DataFrame(time_coef, columns=COEFFICIENT_COLUMNS))


def adjusted_comparison(previous):
    d = previous.loc[previous.branch.eq('A')]
    before = d.loc[d.model.eq('background')].drop(columns='model')
    after = d.loc[d.model.eq('meal_adjusted')].drop(columns='model')
    out = before.merge(after, on=['setting', 'diet', 'branch', 'outcome'], suffixes=('_background', '_meal_adjusted'), validate='one_to_one')
    require(out.person_sha256_background.eq(out.person_sha256_meal_adjusted).all() and out.meal_sha256_background.eq(out.meal_sha256_meal_adjusted).all(), 'Adjustment comparison samples differ')
    out['coefficient_change'] = out.estimate_meal_adjusted - out.estimate_background
    out['interpretation'] = 'Descriptive coefficient change in same sample; no mediation fraction or formal difference test'
    return out
