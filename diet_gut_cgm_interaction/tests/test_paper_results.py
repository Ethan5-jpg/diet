import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

from paper_results.paperlib.data import load_preview,load_server,sha,KEY,MODELS
from paper_results.paperlib.report import prepare_tables

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'paper_results'
FIXTURE=ROOT/'tests/_artifacts/extension_sensitivity/final_demo'


def test_preview_known_coverage_and_corrections():
    b=load_preview(PACKAGE)
    assert len(b['primary'])==43 and len(b['slopes'])==44 and len(b['stability'])==19
    assert b['extension'].empty and b['all_followup'].empty
    p=b['primary'].set_index(KEY)
    assert p.loc[('rEDIH','T096','CV'),'q_all84']==.04872153914
    assert p.loc[('rEDIH','T096','CV'),'interaction_p']==.01218038478
    r=b['original']
    target=r[(r.exposure=='NOVA4')&(r.taxon=='T068')&(r.outcome=='CV')&(r.model=='protocol_adjusted')]
    assert target.interaction_ci_low_native.iloc[0]==-.3059562771
    t=prepare_tables(b)
    assert len(t['Table_1_focus_primary'])==24
    assert t['Table_S1_primary_all_available'].interaction_p.notna().sum()==1
    expected={('AMED','T096','CV'),('AMED','T096','SDw'),('rEDIH','T068','CV'),('rEDIH','T068','Mean'),('rEDIH','T068','SDw'),('rEDIH','T096','Mean'),('rEDIH','T096','SDw')}
    assert set(t['Table_4_targets_retained_all_six'][KEY].itertuples(index=False,name=None))==expected
    assert sum((b['stability'][[m+'__q_all84' for m in MODELS]]<.05).all(axis=1))==7


def test_synthetic_cannot_be_exported_as_real():
    with pytest.raises(ValueError,match='Data kind mismatch'):
        load_server(FIXTURE/'reference_demo',FIXTURE)


def copy_runs(tmp_path):
    module=tmp_path/'Different HPP root'/'Data'/'diet_gut_cgm_interaction'
    old=module/'outputs/run_20261010T085601_096554Z_4d43e0'
    new=module/'extension_sensitivity/outputs_extension_sensitivity/extension_20261010T105930_496220Z_ff8085'
    shutil.copytree(FIXTURE/'reference_demo',old)
    (new/'tables').mkdir(parents=True)
    for file in (FIXTURE/'tables').glob('*.csv'):shutil.copy2(file,new/'tables'/file.name)
    meta=json.loads((FIXTURE/'manifest.json').read_text());meta['reference_run']=str(old)
    (new/'manifest.json').write_text(json.dumps(meta))
    return module,old,new


def test_tampered_saved_table_rejected(tmp_path):
    _,old,new=copy_runs(tmp_path)
    file=new/'tables/all_interactions.csv';file.write_bytes(file.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Saved result changed'):load_server(old,new,True)


def test_reference_identity_rejected(tmp_path):
    _,old,new=copy_runs(tmp_path)
    meta=json.loads((new/'manifest.json').read_text());meta['reference_run']=str(tmp_path/'wrong')
    (new/'manifest.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='another original run'):load_server(old,new,True)


def test_full_table_scope_and_no_participant_columns():
    b=load_server(FIXTURE/'reference_demo',FIXTURE,True)
    tables=prepare_tables(b)
    assert len(tables['Table_S1_primary_all_available'])==84
    assert len(tables['Table_S2_original_models'])==228
    assert len(tables['Table_S3_followup_all_models'])==588
    assert len(tables['Table_S4_conditional_slopes'])==1764
    for table in tables.values():
        assert not set(table.columns)&{'participant_id','sample_sha256','IID','FID'}
    p=b['primary'].set_index(KEY)
    slopes=tables['Table_S4_conditional_slopes']
    for _,row in slopes[slopes.model.eq('primary')].iterrows():assert row.n==p.loc[tuple(row[k] for k in KEY),'n']


def test_standalone_relocation_full_export_read_only(tmp_path):
    module,old,new=copy_runs(tmp_path)
    package=module/'paper_results'
    shutil.copytree(PACKAGE,package,ignore=shutil.ignore_patterns('preview_from_screenshots','outputs*','LATEST*','__pycache__'))
    assert not (module/'interaction').exists()
    before={str(p):sha(p) for run in [old,new] for p in run.rglob('*') if p.is_file()}
    env=dict(os.environ,PYTHON=sys.executable,PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    command=['bash',str(package/'run_paper.sh'),'--allow-synthetic']
    process=subprocess.run(command,cwd=str(tmp_path),env=env,text=True,capture_output=True)
    assert process.returncode==0,process.stdout+process.stderr
    out=Path((package/'LATEST_TEST.txt').read_text().strip())
    assert out.parent==package/'outputs_test'
    manifest=json.loads((out/'export_manifest.json').read_text())
    assert manifest['data_mode']=='SYNTHETIC_TEST' and manifest['status']=='EXPORT_COMPLETE'
    assert len(manifest['figures'])==10 and manifest['reference_reproduced']==204
    assert manifest['table_rows']['Table_S3_followup_all_models']==588
    assert all(sha(out/name)==digest for name,digest in manifest['output_sha256'].items())
    assert all(sha(Path(path))==digest for path,digest in before.items())
    for file in (out/'source_data').glob('*.csv'):
        assert not {'participant_id','sample_sha256'}&set(pd.read_csv(file).columns)
    assert 'SYNTHETIC SOFTWARE TEST' in (out/'figures/Figure_I1_focus_interactions.svg').read_text()
    assert '<text' in (out/'figures/Figure_I1_focus_interactions.svg').read_text()
    assert not list(out.rglob('*.zip'))
    assert 'PAPER_ZIP=' not in process.stdout
    saved=sha(out/'export_manifest.json')
    repeat=subprocess.run(command+['--output-dir',str(out)],cwd=str(tmp_path),env=env,text=True,capture_output=True)
    assert repeat.returncode==2 and 'already exists' in repeat.stderr
    assert sha(out/'export_manifest.json')==saved


def test_python38_and_shell_syntax():
    for path in PACKAGE.rglob('*.py'):ast.parse(path.read_text(),feature_version=(3,8))
    result=subprocess.run(['bash','-n',str(PACKAGE/'run_paper.sh')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
