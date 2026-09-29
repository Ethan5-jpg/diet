"""Fixed-sample diagnostics, positive area sensitivity and whole-CGM linkage."""
import numpy as np
import pandas as pd
from scipy import stats
from hppmeal.common import require
from hppmeal.models import bh_planned
from seven_diet_meals.inputs import DIETS, BASE
from seven_diet_meals.models import (features, prepare, attempt, aggregate_people, encode, linear_fit,
    extras, sample_hash, OUTCOMES, CONTEXT, NUTRIENTS)
from quality_timecourse.analysis import fit_sample, fdr, reproduce
from .inputs import CGM20

SETTINGS = ['source_primary', 'joint']
STEPS = [('background', []), ('plus_energy', ['energy_100']), ('plus_carb', ['carb_10']),
         ('plus_both', ['carb_10', 'energy_100']), ('meal_adjusted', NUTRIENTS)]
MEASURES = ['mean_increment', 'peak_rise_0_120', 'positive_mean']


def planned_fdr(table, size, column='q'):
    table = table.copy()
    for _, ix in table.groupby('setting', sort=False).groups.items():
        require(len(ix) == size, 'Unexpected planned family size')
        table.loc[ix, column] = bh_planned(table.loc[ix, 'p'])
    table['planned_family_size'] = size
    return table


def standardized_design(X):
    a = X.copy().reset_index(drop=True).astype(float)
    a.insert(0, 'intercept', 1.)
    constant = a.std(ddof=0).le(1e-10)
    constant.loc['intercept'] = False
    a = a.loc[:, ~constant]
    scale = np.sqrt(np.mean(a.to_numpy() ** 2, axis=0))
    return a, a.to_numpy() / scale, scale


def diagnostics(y, X, focus):
    """OLS exact one-observation-deletion diagnostics; no automatic exclusion."""
    a, z, scale = standardized_design(X)
    y = np.asarray(y, float)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    inverse = (vt.T / s).dot(u.T)
    beta = inverse.dot(y)
    residual = y - z.dot(beta)
    leverage = np.sum(u ** 2, axis=1)
    j = a.columns.get_loc(focus)
    others = np.delete(z, j, axis=1)
    score = a[focus].to_numpy()
    score_residual = score - others.dot(np.linalg.lstsq(others, score, rcond=None)[0])
    fraction = np.sum(score_residual ** 2) / np.sum((score - score.mean()) ** 2)
    deletion_change = -inverse[j] * residual / (1 - leverage) / scale[j]
    mse = np.sum(residual ** 2) / (len(y) - z.shape[1])
    cook = residual ** 2 / (z.shape[1] * mse) * leverage / (1 - leverage) ** 2 if mse > 0 else np.full(len(y), np.nan)
    return dict(score_residual_variance_fraction=fraction, score_vif=1 / fraction,
        max_leverage=float(leverage.max()), max_cook=float(np.nanmax(cook)),
        max_abs_leave1_coefficient_change=float(np.abs(deletion_change).max()))


def safe_corr(x, y, method='spearman'):
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(stats.spearmanr(x, y)[0] if method == 'spearman' else np.corrcoef(x, y)[0, 1])


def correlations(p, diet, setting):
    rows = []
    _, background, _ = standardized_design(encode(p, BASE + extras(diet) + CONTEXT))
    score = p[diet].to_numpy(float)
    residual = score - background.dot(np.linalg.lstsq(background, score, rcond=None)[0])
    for f in NUTRIENTS:
        x = p[f].to_numpy(float)
        rx = x - background.dot(np.linalg.lstsq(background, x, rcond=None)[0])
        rows.append(dict(setting=setting, diet=diet, field=f, people=len(p),
            pearson=safe_corr(score, x, 'pearson'), spearman=safe_corr(score, x),
            background_residual_pearson=safe_corr(residual, rx, 'pearson')))
    return rows


def adjustment(sample, diet, setting, outcome, step, additional):
    row, _ = attempt(sample, diet, setting, 'A', outcome, 'background')
    row.update(model=step, status='NOT_ESTIMATED', estimate=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, reason='')
    coefs = []
    try:
        require(row['people'] >= 50, 'Fewer than 50 eligible participants')
        p = aggregate_people(sample)
        require(p[diet].nunique() >= 3, 'Fewer than three distinct dietary values')
        X = encode(p, [diet] + BASE + extras(diet) + CONTEXT + additional)
        coef, info = linear_fit(p[outcome], X, diet)
        row.update(next(r for r in coef if r['term'] == diet))
        row.update(info)
        row.update(diagnostics(p[outcome], X, diet))
        row['status'] = 'ESTIMATED'
        coefs = [dict(setting=setting, diet=diet, outcome=outcome, model=step, **r) for r in coef]
    except (ValueError, np.linalg.LinAlgError) as e:
        row['reason'] = str(e)
    return row, coefs


def positive(sample, diet, setting, branch, model):
    d = sample.copy()
    missing = int(d.positive_mean.isna().sum())
    # Frozen attempt reports the entire planned membership even when y is missing.
    d['mean_increment'] = d.positive_mean
    row, coef = attempt(d, diet, setting, branch, 'mean_increment', model)
    row['outcome'] = 'positive_mean'
    row['missing_positive_meals'] = missing
    if missing:
        row.update(status='NOT_ESTIMATED', reason='Positive area missing in {} fixed-sample meals; no deletion/imputation'.format(missing),
            estimate=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan)
        coef = []
    # aggregate_people.mean() would otherwise silently skip missing positive meals;
    # the explicit failure above prevents reporting that different estimand.
    for r in coef:
        r['outcome'] = 'positive_mean'
    if row['status'] == 'ESTIMATED' and branch == 'A':
        p = aggregate_people(d)
        X = encode(p, [diet] + BASE + extras(diet) + CONTEXT + (NUTRIENTS if model == 'meal_adjusted' else []))
        row.update(diagnostics(p.mean_increment, X, diet))
    return row, coef


def distribution(sample, setting, diet, branch):
    rows = []
    person = sample.groupby('participant_id')[MEASURES].mean()
    complete = sample.positive_mean.notna().groupby(sample.participant_id).all()
    person.loc[~complete, 'positive_mean'] = np.nan
    for level, d in [('meal', sample), ('person_equal_weight', person)]:
        for field in MEASURES + ['positive_minus_net']:
            x = d.positive_mean - d.mean_increment if field == 'positive_minus_net' else d[field]
            v = x.dropna()
            rows.append(dict(setting=setting, diet=diet, branch=branch, level=level, field=field,
                total=len(x), valid=len(v), missing=int(x.isna().sum()), zero_fraction=v.eq(0).mean(),
                mean=v.mean(), sd=v.std(), p10=v.quantile(.1), median=v.median(), p90=v.quantile(.9), p99=v.quantile(.99)))
    return rows


def bridge_sample(data, people, cgm, setting):
    pool = data if setting == 'source_primary' else data.loc[data.joint]
    cols = ['participant_id'] + BASE
    eligible = people.loc[people.primary_eligible & people[BASE].notna().all(axis=1)].reset_index()[cols]
    d = pool.merge(eligible, on='participant_id', how='inner', validate='many_to_one')
    flow = [dict(stage='primary_covariates_and_quality', people=d.participant_id.nunique(), meals=len(d))]
    d = d.merge(cgm[['participant_id', 'connection_id']].rename(columns={'connection_id': 'whole_connection'}),
                on='participant_id', how='inner', validate='many_to_one')
    flow.append(dict(stage='whole_CGM_participant_present', people=d.participant_id.nunique(), meals=len(d)))
    d = d.loc[d.connection_id.eq(d.whole_connection)].copy()
    flow.append(dict(stage='same_CGM_connection', people=d.participant_id.nunique(), meals=len(d)))
    d = d.dropna(subset=OUTCOMES + CONTEXT + NUTRIENTS)
    g = d.groupby('participant_id')
    enough = g.meal_id.transform('size').ge(5) & g.local_date.transform('nunique').ge(3)
    d = d.loc[enough].copy()
    flow.append(dict(stage='complete_core_fields_5meals_3days', people=d.participant_id.nunique(), meals=len(d)))
    p = d.groupby('participant_id')[MEASURES].mean()
    complete = d.positive_mean.notna().groupby(d.participant_id).all()
    p.loc[~complete, 'positive_mean'] = np.nan
    p['meals'] = d.groupby('participant_id').size()
    p['dates'] = d.groupby('participant_id').local_date.nunique()
    p = p.join(people[BASE]).join(cgm.set_index('participant_id')[[f + '_z' for _, f in CGM20]])
    membership = d[['participant_id', 'meal_id', 'connection_id']].assign(setting=setting)
    return p.reset_index(), [dict(setting=setting, **r) for r in flow], membership


def bridge_fit(p, measure, label, field, setting):
    paired = p.dropna(subset=[measure, field + '_z'] + BASE).copy()
    row = dict(setting=setting, meal_measure=measure, cgm=label, cgm_field=field, people=len(paired),
        meals=int(paired.meals.sum()), person_sha256=sample_hash(paired.participant_id), status='NOT_ESTIMATED', reason='',
        estimate=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan,
        whole_cgm_unique=int(paired[field + '_z'].nunique()), meal_measure_unique=int(paired[measure].nunique()),
        meal_sd=paired[measure].std(), spearman=safe_corr(paired[measure], paired[field + '_z']),
        unit='original whole-CGM Z per 1 paired-sample SD of personal meal measure')
    coefs = []
    try:
        require(len(paired) >= 50, 'Fewer than 50 paired participants')
        require(row['whole_cgm_unique'] >= 3 and row['meal_measure_unique'] >= 3, 'Fewer than three distinct values')
        require(np.isfinite(row['meal_sd']) and row['meal_sd'] > 1e-10, 'Degenerate meal feature SD')
        X = encode(paired, BASE)
        X['meal_z'] = (paired[measure].to_numpy() - paired[measure].mean()) / row['meal_sd']
        coef, info = linear_fit(paired[field + '_z'], X, 'meal_z')
        row.update(next(r for r in coef if r['term'] == 'meal_z'))
        row.update(info)
        row.update(diagnostics(paired[field + '_z'], X, 'meal_z'))
        row['status'] = 'ESTIMATED'
        coefs = [dict(setting=setting, meal_measure=measure, cgm=label, **r) for r in coef]
    except (ValueError, np.linalg.LinAlgError) as e:
        row['reason'] = str(e)
    return row, coefs


def run_models(meals, people, cgm, previous, previous_members, progress=print):
    data = features(meals)  # Time origin is fixed before any quality filtering.
    reused, checks, flows, members = [], [], [], []
    adjustment_rows, adjustment_coefs, corr_rows, positive_rows, positive_coefs, dist = [], [], [], [], [], []
    for setting in SETTINGS:
        pool = data if setting == 'source_primary' else data.loc[data.joint]
        current = []
        # Reproduce all 42 before using each setting for new specifications.
        for diet in DIETS:
            for branch in ['A', 'B']:
                sample, _ = prepare(pool, people, diet, 'primary_available', branch)
                r, _ = fit_sample(sample, diet, setting, branch)
                current.extend(r)
        current = fdr(pd.DataFrame(current))
        old = previous.loc[previous.setting.eq(setting)].copy()
        old['setting'] = 'primary_available'
        checks.append(reproduce(current, old).assign(setting=setting))
        reused.append(current)
        progress('REPRODUCTION {}: 42 models MATCHED'.format(setting))
        for diet in DIETS:
            progress('ADJUSTMENT/POSITIVE {} {}'.format(setting, diet))
            for branch in ['A', 'B']:
                sample, flow = prepare(pool, people, diet, 'primary_available', branch)
                flows.extend([dict(r, setting=setting) for r in flow])
                members.append(sample[['participant_id', 'meal_id']].assign(setting=setting, diet=diet, branch=branch))
                if setting == 'joint':
                    old_members = previous_members.loc[previous_members.diet.eq(diet) & previous_members.branch.eq(branch)]
                    require(set(zip(sample.participant_id, sample.meal_id)) == set(zip(old_members.participant_id, old_members.meal_id)),
                            'Joint participant-meal assignment differs from quality run')
                if branch == 'A':
                    p = aggregate_people(sample)
                    if len(p) >= 3:
                        corr_rows.extend(correlations(p, diet, setting))
                    for outcome in OUTCOMES:
                        for step, additional in STEPS:
                            row, coef = adjustment(sample, diet, setting, outcome, step, additional)
                            adjustment_rows.append(row)
                            adjustment_coefs.extend(coef)
                dist.extend(distribution(sample, setting, diet, branch))
                for model in (['background', 'meal_adjusted'] if branch == 'A' else ['carb_interaction']):
                    row, coef = positive(sample, diet, setting, branch, model)
                    positive_rows.append(row)
                    positive_coefs.extend(coef)
    bridge_rows, bridge_coefs, bridge_flow, bridge_members, bridge_people = [], [], [], [], []
    for setting in SETTINGS:
        progress('BRIDGE {}: all 20 whole-CGM phenotypes'.format(setting))
        p, flow, m = bridge_sample(data, people, cgm, setting)
        bridge_flow.extend(flow)
        bridge_members.append(m)
        bridge_people.append(p.assign(setting=setting))
        for measure in MEASURES:
            for label, field in CGM20:
                row, coef = bridge_fit(p, measure, label, field, setting)
                bridge_rows.append(row)
                bridge_coefs.extend(coef)
    result = dict(reproduction=pd.concat(checks, ignore_index=True), original=pd.concat(reused, ignore_index=True),
        adjustment=planned_fdr(pd.DataFrame(adjustment_rows), 70), positive=planned_fdr(pd.DataFrame(positive_rows), 21),
        bridge=planned_fdr(pd.DataFrame(bridge_rows), 60), flow=pd.DataFrame(flows),
        correlations=pd.DataFrame(corr_rows, columns=['setting', 'diet', 'field', 'people', 'pearson', 'spearman', 'background_residual_pearson']),
        distribution=pd.DataFrame(dist), members=pd.concat(members, ignore_index=True),
        bridge_flow=pd.DataFrame(bridge_flow), bridge_members=pd.concat(bridge_members, ignore_index=True),
        bridge_people=pd.concat(bridge_people, ignore_index=True))
    coefficient_columns = ['setting', 'diet', 'outcome', 'model', 'branch', 'meal_measure', 'cgm', 'term', 'estimate', 'se', 'ci_low', 'ci_high', 'p']
    result['coefficients'] = pd.DataFrame(adjustment_coefs + positive_coefs + bridge_coefs).reindex(columns=coefficient_columns)
    return result
