"""Verify frozen provenance and reuse the original data loaders, read-only."""
import json
from pathlib import Path
import pandas as pd
from hppmeal.common import require, sha256, read_csv, norm
from nutrition_bridge import inputs as frozen
from . import VERSION

MODULE = Path(__file__).resolve().parent
PACKAGE = MODULE.parent


def bundle():
    p=MODULE/'MANIFEST.json'
    m=json.loads(p.read_text(encoding='utf-8'))
    require(m['version']==VERSION, 'Follow-up version mismatch')
    required={'__init__.py','inputs.py','support.py','feasibility.py','reporting.py','run_analysis.py','run_response_followup.sh','README.md','PLAN.md'}
    require(required.issubset(m['files']), 'Incomplete follow-up manifest')
    expected={str(p):sha256(p)}
    for name,digest in m['files'].items():
        f=(MODULE/name).resolve()
        require(MODULE in f.parents and f.is_file() and sha256(f)==digest,'Incomplete/changed follow-up upload: '+name)
        expected[str(f)]=digest
    expected.update(frozen.bundle())
    return expected


def relocate(value, old_root, new_root):
    p=Path(value)
    try:return new_root/p.relative_to(old_root)
    except ValueError:return p


def verify_outputs(source, state):
    outputs=state.get('result_sha256',{})
    require(bool(outputs),'Frozen result hashes missing')
    expected={}
    for rel,digest in outputs.items():
        p=(source/rel).resolve()
        require(source in p.parents and p.is_file() and sha256(p)==digest,'Frozen result changed/missing: '+rel)
        expected[str(p)]=digest
    return expected


def inspect(source, root_override, out):
    source=Path(source).expanduser().resolve()
    mf,sf=source/'manifest.json',source/'analysis_spec.json'
    prior=json.loads(mf.read_text(encoding='utf-8'));spec=json.loads(sf.read_text(encoding='utf-8'))
    state=prior['status'].get('nutrition_bridge',{})
    require(state.get('status') in ['COMPLETE','COMPLETE_WITH_LIMITATIONS'],'Nutrition run must be complete')
    require(spec['version']=='2026.09.26.1' and spec.get('stage')=='all','Requires frozen nutrition_bridge 2026.09.26.1 all run')
    require(spec['data_kind']==prior['data_kind'],'Data label mismatch')
    require(not any(x.get('status') in ['FAILED','RUNNING','INTERRUPTED'] for x in prior['status'].values()),'Frozen run incomplete')
    expected=verify_outputs(source,state)
    expected.update({str(mf):sha256(mf),str(sf):sha256(sf)})
    old_root=Path(prior['root']);root=Path(root_override or old_root).expanduser().resolve()
    quality=relocate(spec['quality_run'],old_root,root)
    whole=relocate(spec['whole_cgm'],old_root,root)
    root,c,paths,checked,audit=frozen.inspect(quality,root,out,whole)
    require(c==prior['config'],'Frozen configuration differs')
    expected.update(checked)
    # Every prior input, including frozen code and CGM export, must still match.
    for item in prior['inputs']:
        p=relocate(item['path'],old_root,root).resolve()
        require(p.is_file() and sha256(p)==item['sha256'],'Frozen nutrition input changed/missing: '+str(p))
        if str(p) in expected:require(expected[str(p)]==item['sha256'],'Inconsistent historical input hashes')
        expected[str(p)]=item['sha256']
    needed={'original':'tables/02_original_models.csv','positive':'tables/05_positive_models.csv',
            'members':'private/members.csv','bridge_members':'private/bridge_members.csv',
            'bridge_people':'private/bridge_people.csv','coefficients':'tables/07_coefficients.csv'}
    for key,rel in needed.items():
        p=source/rel
        require(str(p) in expected,'Missing frozen output contract: '+rel)
        paths['nutrition_'+key]=p
    qm=json.loads((quality/'manifest.json').read_text(encoding='utf-8'))
    dayfile=quality/'private/02_daily_coverage.csv'
    digest=qm['status']['quality_timecourse']['result_sha256'].get('private/02_daily_coverage.csv')
    require(digest and dayfile.is_file() and sha256(dayfile)==digest,'Frozen daily coverage missing/changed')
    expected[str(dayfile)]=digest;paths['quality_days']=dayfile
    paths['nutrition_run']=source;paths['quality_run']=quality
    return root,c,paths,expected,audit


def load(run,paths):
    meals,people,cgm,_,_=frozen.load(run,paths)
    tables={k:read_csv(paths['nutrition_'+k]) for k in ['original','positive','members','bridge_members','bridge_people','coefficients']}
    for key in ['original','positive','coefficients']:
        for col in ['people','meals','estimate','se','ci_low','ci_high','p','q','q_family14']:
            if col in tables[key]:tables[key][col]=pd.to_numeric(tables[key][col],errors='raise')
    for key in ['members','bridge_members','bridge_people']:
        for col in ['participant_id','meal_id','connection_id']:
            if col in tables[key]:tables[key][col]=norm(tables[key][col])
    return meals,people,cgm,tables
