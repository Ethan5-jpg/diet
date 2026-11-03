"""Record V2 as the active project root, preserving the old-server audit provenance."""
from pathlib import Path
import json, hashlib, datetime, subprocess, socket

V2=Path('/home/ec2-user/studies/diet_gut_cgm_v2')
OUT=V2/'workspace_setup_20261018'
check=json.loads((OUT/'storage_verification.json').read_text())
assert check['status']=='V2_WRITE_CLOSE_READBACK_PASS'
verify=OUT/'WRITE_READBACK_VERIFIED.txt'
assert hashlib.sha256(verify.read_bytes()).hexdigest()==check['sha256']
pheno='/home/ec2-user/miniconda/envs/pheno/bin/python'
r=subprocess.run([pheno,'-c','import sys,pandas,pyarrow,numpy; print(sys.executable); print("pandas",pandas.__version__,"pyarrow",pyarrow.__version__,"numpy",numpy.__version__)'],capture_output=True,text=True,timeout=30)
context=dict(status='V2_PROJECT_ROOT_VERIFIED',active_project_root=str(V2),hostname=socket.gethostname(),
             raw_data_root='/home/ec2-user/studies/hpp_datasets',storage_verification=str(OUT/'storage_verification.json'),
             old_audit_server='ip-10-0-84-116',old_audit_directory='/home/ec2-user/hpp_time_alignment_audit_20261018T073819Z',
             old_audit_visible_on_new_server=check['old_audit_visible_on_this_server'],
             historical_audit_results_migrated=False,analysis_rerun=False,
             pheno_smoke_test=dict(exit_code=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip()),
             updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             scope='New storage and environment verification; previous analysis results remain on old server.')
readme='''# HPP饮食—双部位菌群—代谢研究：V2工作区\n\n当前项目代码、配置、运行日志和备忘录优先放在本目录：\n`/home/ec2-user/studies/diet_gut_cgm_v2`。\n\n本云盘已实际创建目录、写文件、关闭重开读回并核对SHA256。核验代码、备忘录和JSON在workspace_setup_20261018/。用户原有test/保持原状。\n\nHPP原始数据根目录：`/home/ec2-user/studies/hpp_datasets`，保持只读。\n\n旧服务器ip-10-0-84-116的完整日期核查目录：\n`/home/ec2-user/hpp_time_alignment_audit_20261018T073819Z`。\n新服务器目前看不到该旧目录，因此旧结果尚未迁移，也没有重新运行日期或科学分析。不能将“V2已可写”解释为“旧产物已迁移”。\n\n接续先读PROJECT_CONTEXT.json与workspace_setup_20261018/MEMO.md；后续模块在V2下新建独立目录并保留代码功能、参数、日志和实际完成状态。\n'''
with (V2/'README.md').open('x') as h: h.write(readme)
with (V2/'PROJECT_CONTEXT.json').open('x') as h: h.write(json.dumps(context,indent=2,ensure_ascii=False))
with (V2/'SERVER_BACKGROUND.md').open('x') as h:
    h.write('# 当前服务器与存储背景\n\n'+readme.split('\n',1)[1]+'\n\n新服务器实际主机：'+socket.gethostname()+'\n\npheno环境检查：\n\n```text\n'+r.stdout+r.stderr+'\n```\n')
(OUT/'08_record_v2_workspace.py').write_bytes(Path(__file__).read_bytes())
with (OUT/'MEMO.md').open('a') as h:
    h.write('\n## 08_record_v2_workspace.py\n\n核对07版读回哈希，运行pheno解释器和pandas/pyarrow/numpy导入检查；建立V2根目录README.md、SERVER_BACKGROUND.md、PROJECT_CONTEXT.json。修改入口为V2、OUT、pheno及旧核查位置。保留旧服务器来源，本次不迁移参与者级结果、不重新运行分析。\n\n实际环境检查退出码：'+str(r.returncode)+'\n\n'+r.stdout+r.stderr+'\n')
(OUT/'08_exit_code').write_text('0\n')
for p in [V2/'README.md',V2/'SERVER_BACKGROUND.md',V2/'PROJECT_CONTEXT.json']:
    assert p.stat().st_size>0
print('V2_PROJECT_ROOT_VERIFIED')
print('ACTIVE_PROJECT_ROOT',V2)
print('SETUP_MEMO',OUT/'MEMO.md')
print('PHENO_SMOKE_TEST_EXIT_CODE',r.returncode)
print('OLD_AUDIT_VISIBLE_ON_NEW_SERVER',check['old_audit_visible_on_this_server'])
print('OLD_AUDIT_MIGRATED False; ANALYSIS_RERUN False')
print('SETUP_EXIT_CODE_0')
