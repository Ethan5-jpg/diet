"""Verify the completed date audit, locate file-pointer sources, retry cloud save."""
from pathlib import Path
import os, json, sys, hashlib, shutil, datetime
import pyarrow.parquet as pq

T=Path('/home/ec2-user/hpp_time_alignment_audit_20261018T073819Z')
C=Path('/home/ec2-user/studies/diet_gut_cgm/time_alignment_audit_20261018')
R=Path('/home/ec2-user/studies/hpp_datasets')
assert '/envs/pheno/' in sys.executable
assert (T/'03_exit_code').read_text().strip()=='0'
m=json.loads((T/'source_manifest.json').read_text())
s=json.loads((T/'aggregate_summary.json').read_text())
assert len(m['sources'])==5
assert all(x['rows_scanned']==x['rows_metadata'] for x in m['sources'])
assert hashlib.sha256((T/'03_time_alignment.py').read_bytes()).hexdigest()==m['source_sha256']
info=[]
for module,file in [('cgm','cgm.parquet'),('oral_microbiome','oral_microbiome.parquet'),('gut_microbiome','gut_microbiome.parquet')]:
    p=R/module/file
    f=pq.ParquetFile(p,pre_buffer=False)
    names=f.schema_arrow.names
    fields=[x.name for x in f.schema_arrow if str(x.type)=='string' and
            any(t in x.name.lower() for t in ['parquet','arrow','time_series','timeseries'])]
    print('POINTER_FIELDS',module,fields,flush=True)
    if module=='cgm':
        print('CGM_FULL_SCHEMA',[(x.name,str(x.type)) for x in f.schema_arrow],flush=True)
    pointers={}
    if fields:
        for batch in f.iter_batches(batch_size=512,columns=fields,use_threads=False):
            for field in fields:
                vals=[x for x in batch.column(batch.schema.get_field_index(field)).to_pylist() if x]
                if vals and field not in pointers:
                    pointers[field]=vals[0]
            if len(pointers)==len(fields): break
    rec=dict(table=str(p),pointer_fields=fields,representative_pointers=pointers)
    info.append(rec)
    for field,value in pointers.items():
        # File-reference metadata only; no patient identifiers or phenotype rows.
        print('POINTER_LOCATION',module,field,value,flush=True)

# Discovery remains bounded; this is not a full raw-timestamp scan.
raw=R/'cgm'/'timeseries'
entry_counts={}; examples=[]
for i,p in enumerate(raw.iterdir()):
    if i>=10000: break
    kind='directory' if p.is_dir() else p.suffix
    entry_counts[kind]=entry_counts.get(kind,0)+1
    if len(examples)<3: examples.append(dict(path=str(p),is_dir=p.is_dir(),bytes=p.stat().st_size))
print('RAW_CGM_TOPLEVEL_TYPES_BOUNDED_10000',entry_counts,flush=True)
info.append(dict(raw_cgm_dir=str(raw),bounded_toplevel_types=entry_counts,examples=examples))
(T/'file_pointer_locations.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))

files=['01_discover.py','discovery.json','02_time_alignment.py','02_console.log','02_failure.log','02_exit_code','04_verify_and_locate.py',
       '03_time_alignment.py','03_console.log','03_exit_code','source_manifest.json','aggregate_summary.json',
       'person_visit_alignment_TRE_ONLY.csv','four_source_dates_TRE_ONLY.csv','date_field_dictionary_excerpts.json',
       'file_pointer_locations.json','MEMO.md','audit.log']
assert all((T/f).is_file() and (T/f).stat().st_size>0 for f in files)
cloud_error=None
try:
    C.mkdir(exist_ok=True)
    with (C/'WRITE_VERIFICATION.md').open('w') as h:
        h.write('Actual file write succeeded: '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
    for name in files:
        shutil.copy2(T/name,C/name)
        assert hashlib.sha256((T/name).read_bytes()).digest()==hashlib.sha256((C/name).read_bytes()).digest()
except OSError as exc:
    cloud_error=repr(exc)

memo=(T/'MEMO.md').read_text()
memo+='''\n## 运行版本和核验记录\n\n01_discover.py完成有界元数据盘点。02_time_alignment.py首次扫描在CGM连接汇总表缺乏物理时区信息时停止，02_console.log/02_failure.log/02_exit_code=1保留。03_time_alignment.py完整扫描日期所需的五张表，CGM连接汇总表只读取schema；真实共同日期依据iglu_daily.collection_date，不推定连接汇总的无时区时间戳。03_exit_code=0，全部五源扫描行数与元数据一致，源代码哈希一致。04_verify_and_locate.py核验完成状态、定位丰度/时间序列文件指针、重试云盘写入和校验复制。\n\nCGM日表最多13个观测日，这只描述当前日表，不能据此宣称原始传感器只采集13天或统一修改分母；15分钟原始数据未全量读取。口腔日期按源字典定义为sample-given date。口腔与肠道日期相隔较长，不能当作同步采样或暴露前共同基线。414行口腔无有效关联键，未编造访视。\n\n'''
memo+='最终云盘写入错误：'+str(cloud_error)+'\n\n实际完整产物目录：'+str(T)+'\n'
(T/'MEMO.md').write_text(memo)
if cloud_error is None:
    shutil.copy2(T/'MEMO.md',C/'MEMO.md')
verification=dict(status='AUDIT_VERIFIED_CLOUD_SAVED' if cloud_error is None else 'AUDIT_VERIFIED_CLOUD_WRITE_BLOCKED',
                  actual_dir=str(C if cloud_error is None else T),cloud_error=cloud_error,
                  source_sha256=m['source_sha256'],files={f:(T/f).stat().st_size for f in files})
(T/'verification.json').write_text(json.dumps(verification,indent=2))
if cloud_error is None: shutil.copy2(T/'verification.json',C/'verification.json')
print('VERIFICATION',verification['status'],'ACTUAL_DIR',verification['actual_dir'],'CLOUD_ERROR',cloud_error,flush=True)
print('INPUT_ROWS',[(x['role'],x['rows_scanned'],x['invalid_key_rows'],x['missing_date_rows']) for x in m['sources']],flush=True)
print('SOURCE_HASH',m['source_sha256'],'CODE_EXIT', (T/'03_exit_code').read_text().strip(),flush=True)
print('FOUR_SOURCE_COMMON_DAYS',s['four_source'],flush=True)
print('OUTPUT_FILES_VERIFIED',len(files),flush=True)
(T/'04_exit_code').write_text('0\n')
if cloud_error is None: shutil.copy2(T/'04_exit_code',C/'04_exit_code')
