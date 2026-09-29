"""Strict lineage checks, including the completed feasibility run."""
import json
from pathlib import Path
import pandas as pd
from hppmeal.common import require, sha256, read_csv, norm, unique
from response_followup import inputs as previous
from response_followup.feasibility import bool_column
from . import VERSION

MODULE = Path(__file__).resolve().parent
PACKAGE = MODULE.parent


def bundle():
    path = MODULE / 'MANIFEST.json'; m = json.loads(path.read_text(encoding='utf-8'))
    require(m['version'] == VERSION, 'Date-split package version mismatch')
    required = {'__init__.py','inputs.py','metrics.py','build.py','models.py','reporting.py','run_analysis.py',
        'run_date_split_bridge.sh','CONTRACT.json','README.md','PLAN.md','REVIEW.md'}
    require(required.issubset(m['files']), 'Incomplete new upload manifest')
    expected = {str(path): sha256(path)}
    for name, digest in m['files'].items():
        p = (MODULE / name).resolve()
        require(MODULE in p.parents and p.is_file() and sha256(p) == digest, 'Changed/missing upload: ' + name)
        expected[str(p)] = digest
    expected.update(previous.bundle())
    return expected


def inspect(source, root_override, out):
    source = Path(source).expanduser().resolve()
    mf, sf = source/'manifest.json', source/'analysis_spec.json'
    m = json.loads(mf.read_text(encoding='utf-8')); s = json.loads(sf.read_text(encoding='utf-8'))
    state = m['status'].get('response_followup', {})
    require(state.get('status') in ['COMPLETE','COMPLETE_WITH_LIMITATIONS'], 'Follow-up source not complete')
    require(state.get('version') == s.get('version') == '2026.09.27.2' and s.get('stage') == 'all', 'Requires complete response_followup2026.09.27.2')
    require(m['data_kind'] == s['data_kind'] and m['data_kind'] in ['REAL','SYNTHETIC'], 'Source data label mismatch')
    require(state.get('membership_checks') == 14 and state.get('planned_contrasts') == 42, 'Incomplete source checks')
    root = Path(root_override or m['root']).expanduser().resolve(); oldroot = Path(m['root'])
    nutrition = previous.relocate(s['nutrition_run'], oldroot, root)
    root, c, paths, expected, audit = previous.inspect(nutrition, root, out)
    require(c == m['config'], 'Source configuration mismatch')
    expected.update(previous.verify_outputs(source, state))
    expected.update({str(mf):sha256(mf), str(sf):sha256(sf)})
    for item in m['inputs']:
        p = previous.relocate(item['path'], oldroot, root).resolve()
        require(p.is_file() and sha256(p) == item['sha256'], 'Follow-up input changed/missing: ' + str(p))
        if str(p) in expected: require(expected[str(p)] == item['sha256'], 'Conflicting lineage hashes')
        expected[str(p)] = item['sha256']
    for name, rel in [('windows','private/person_windows.csv'),('first_members','private/first_window_members.csv'),
                      ('flow','tables/feasibility.csv')]:
        p = source/rel
        require(str(p) in expected, 'Missing source output contract: ' + rel)
        paths['followup_'+name] = p
    paths['followup_run'] = source
    return root, c, paths, expected, audit


def load(run, paths):
    meals, people, cgm, tables = previous.load(run, paths)
    w = read_csv(paths['followup_windows']); members = read_csv(paths['followup_first_members'])
    for d in [w, members]:
        for k in ['participant_id','connection_id']: d[k] = norm(d[k])
    members['meal_id'] = norm(members.meal_id)
    unique(w, ['setting','participant_id'], 'Prior windows')
    unique(members, ['setting','meal_id'], 'Prior first-window membership')
    bools = ['first_positive_complete','first_meals_ok','second_days_ok','candidate_basic','candidate_MODD_proxy','no_overlap']
    for k in bools: w[k] = bool_column(w[k], k)
    for k in w.columns:
        if k not in bools + ['participant_id','connection_id','setting','split_local_date']:
            w[k] = pd.to_numeric(w[k], errors='raise')
    flow = read_csv(paths['followup_flow']); flow['people'] = pd.to_numeric(flow.people, errors='raise')
    tables.update(prior_windows=w, prior_first_members=members, prior_flow=flow)
    return meals, people, cgm, tables
