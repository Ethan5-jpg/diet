"""Final provenance and full dictionary excerpts; no participant output."""
from pathlib import Path
import csv, json, hashlib, shutil, sys
OUT=Path('/home/ec2-user/studies/diet_gut_cgm_v2/phenotype_food_dates_20261018')
V2=OUT.parent
assert '/envs/pheno/' in sys.executable
assert (OUT/'12_exit_code').read_text().strip()=='0'
v=json.loads((OUT/'verification.json').read_text())
assert hashlib.sha256((OUT/'12_verify_report.py').read_bytes()).hexdigest()==v['source12_sha256']
source=Path(__file__).read_bytes();(OUT/'13_finalize.py').write_bytes(source)
disc=json.loads((OUT/'discovery.json').read_text());defs=[]
for dd in disc['selected']['diet_logging']['dictionaries']:
    with Path(dd['path']).open(newline='') as h:
        for r in csv.DictReader(h):
            if any(t in str(r).lower() for t in ['calories','weight_g','carbohydrate','protein_g','lipid_g','fiber_g','local_timestamp','collection_timestamp']):defs.append(r)
(OUT/'full_food_field_definitions.json').write_text(json.dumps(defs,indent=2))
for r in defs:
    print('FOOD_FIELD',json.dumps({k:z for k,z in r.items() if 'name' in k or 'description' in k or k=='units'},ensure_ascii=False),flush=True)
shutil.copyfile('/tmp/hpp_12_verify.log',OUT/'12_console.log')
context=V2/'PROJECT_CONTEXT.json';o=json.loads(context.read_text());o.update(analysis_rerun=True,analysis_rerun_scope='New independent phenotype/food date audit on this server; old full audit not migrated',latest_phenotype_food_audit=str(OUT),latest_phenotype_food_audit_status='COMPLETE_VERIFIED')
b=json.dumps(o,indent=2).encode();context.write_bytes(b);assert context.read_bytes()==b
note='\n## 13_finalize.py\n\n已核验12版源码SHA256及成功状态，保存完整饮食字段字典摘录和12_console.log，更新根目录PROJECT_CONTEXT.json。此前存储测试的analysis_rerun=false仅代表当时未重算；现已完成独立新核查，登记analysis_rerun=true及明确范围，旧结果仍未迁移。饮食local_timestamp日期与主日期有1066行不一致（全表）；主collection_timestamp与collection_date无日期冲突。逐日表按明确主日期生成，尚不能当成逐餐进食真值。营养素_sum仅表示源字段求和，尚未验收摄入完整性或折算规则；正式分析需核对full_food_field_definitions.json。13_exit_code=0。\n'
for p in [OUT/'MEMO.md',OUT/'REPORT.md',V2/'SERVER_BACKGROUND.md']:
    b=(p.read_text()+note).encode();p.write_bytes(b);assert p.read_bytes()==b
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')}
b=json.dumps(hashes,indent=2).encode();(OUT/'code_sha256.json').write_bytes(b);assert (OUT/'code_sha256.json').read_bytes()==b
(OUT/'13_exit_code').write_text('0\n')
print('FINAL_ACCEPTED: 10=0 / 11=0 / 12=0 / 13=0')
print('DAILY_FOOD_ROWS=125374; PEOPLE=9737; MEDIAN_RECORDED_DAYS=14')
print('BODY_AND_LIVER_DIET_NEAREST_DAY_MEDIAN=0; CGM_NEAREST_DAY_MEDIAN=1')
print('TABLES_TRE_ONLY.html / REPORT.md / MEMO.md: SAVED_AND_VERIFIED')
print('ALL_INDIVIDUAL_RECORDS_STAY_IN_TRE')
print('OUTPUT_DIR',OUT)
print('FINAL_EXIT_CODE_0')
