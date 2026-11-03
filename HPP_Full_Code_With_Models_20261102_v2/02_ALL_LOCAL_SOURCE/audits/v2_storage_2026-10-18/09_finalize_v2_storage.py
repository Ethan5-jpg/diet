"""Finalize V2 notes using complete-file writes; preserve unsupported-append evidence."""
from pathlib import Path
import json, hashlib

V2=Path('/home/ec2-user/studies/diet_gut_cgm_v2')
OUT=V2/'workspace_setup_20261018'
ctx=json.loads((V2/'PROJECT_CONTEXT.json').read_text())
assert ctx['active_project_root']==str(V2)
assert json.loads((OUT/'storage_verification.json').read_text())['status']=='V2_WRITE_CLOSE_READBACK_PASS'
note='''\n## V2写入方式核验与后续约定\n\nV2已通过创建目录、创建文件、关闭重开读回核验。08版登记程序已建立根目录文档和上下文，但在MEMO.md追加模式open("a")遇到OSError errno 95 Operation not supported而停止。保留08版源码和08_failure.log，不将它说成成功。\n\n09_finalize_v2_storage.py改用读取现有文本、合并新增内容后完整写回，关闭重开校验每个文件SHA256。后续备忘录使用完整写回；运行日志优先在服务器本地运行目录生成，再完整同步到V2，不直接在云盘追加。该规则来自实际操作结果。\n\n07版功能：存储核验及新旧目录可见性检查。08版功能：pheno导入检查、根目录README.md／SERVER_BACKGROUND.md／PROJECT_CONTEXT.json登记。09版功能：追加模式兼容修复、完整写回验收、保留失败记录并登记最终状态。修改入口为V2、OUT。\n\n当前项目代码、配置、文档和完成产物优先保存在V2；旧完整日期核查尚未迁移，仍以旧服务器来源为准。本轮没有重新运行日期分析，也没有读取参与者数据。\n'''
def full_write_verify(p,text):
    b=text.encode('utf-8')
    with p.open('wb') as h: h.write(b)
    with p.open('rb') as h: actual=h.read()
    assert actual==b and hashlib.sha256(actual).digest()==hashlib.sha256(b).digest()
    return hashlib.sha256(actual).hexdigest()

(OUT/'08_failure.log').write_bytes(Path('/tmp/hpp_v2_setup.log').read_bytes())
(OUT/'08_exit_code').write_text('1\n')
memo=(OUT/'MEMO.md').read_text()+note
memo+='\npheno导入检查：\n\n```text\n'+ctx['pheno_smoke_test']['stdout']+'\n'+ctx['pheno_smoke_test']['stderr']+'\n```\n'
hashes={}
hashes['MEMO.md']=full_write_verify(OUT/'MEMO.md',memo)
for name in ['README.md','SERVER_BACKGROUND.md']:
    hashes[name]=full_write_verify(V2/name,(V2/name).read_text()+note)
ctx.update(status='V2_PROJECT_READY_FULL_WRITE_VERIFIED',append_supported=False,full_file_overwrite_verified=True,
           file_update_mode='read_merge_full_write_close_readback',log_workflow='generate locally on server, then copy completed snapshots to V2',
           setup_exit_codes={'07':0,'08':1,'09':0},document_sha256=hashes)
full_write_verify(V2/'PROJECT_CONTEXT.json',json.dumps(ctx,indent=2,ensure_ascii=False))
(OUT/'09_finalize_v2_storage.py').write_bytes(Path(__file__).read_bytes())
(OUT/'09_exit_code').write_text('0\n')
lines=['V2_PROJECT_READY_FULL_WRITE_VERIFIED','PROJECT_ROOT '+str(V2),
       'DIRECTORY_CREATE / FILE_WRITE / CLOSE_READBACK / FULL_OVERWRITE: PASS',
       'APPEND: UNSUPPORTED (errno 95); full-file update verified',
       'ENTRY_DOCS: README.md / SERVER_BACKGROUND.md / PROJECT_CONTEXT.json',
       'MEMO: workspace_setup_20261018/MEMO.md',
       'PHENO_IMPORT_CHECK_EXIT_CODE '+str(ctx['pheno_smoke_test']['exit_code']),
       'OLD_RESULTS_MIGRATED: False; ANALYSIS_RERUN: False','SETUP_EXIT_CODE_0']
(OUT/'final_status.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines),flush=True)
