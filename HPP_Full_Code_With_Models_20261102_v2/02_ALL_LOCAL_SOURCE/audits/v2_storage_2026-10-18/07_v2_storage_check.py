"""Verify V2 storage and save continuation notes; never read participant data."""
from pathlib import Path
import os, sys, json, hashlib, datetime, socket

V2=Path('/home/ec2-user/studies/diet_gut_cgm_v2')
OLD=Path('/home/ec2-user/hpp_time_alignment_audit_20261018T073819Z')
RAW=Path('/home/ec2-user/studies/hpp_datasets')
assert V2.is_dir(), V2
OUT=V2/'workspace_setup_20261018'
OUT.mkdir(exist_ok=False)
payload=b'HPP V2 storage verification: create, close, reopen, compare SHA256.\n'
p=OUT/'WRITE_READBACK_VERIFIED.txt'
with p.open('xb') as h: h.write(payload)
with p.open('rb') as h: reread=h.read()
assert reread==payload
assert hashlib.sha256(reread).hexdigest()==hashlib.sha256(payload).hexdigest()
manifest=dict(status='V2_WRITE_CLOSE_READBACK_PASS',v2_root=str(V2),setup_directory=str(OUT),
              hostname=socket.gethostname(),python=sys.executable,
              pheno_python_exists=Path('/home/ec2-user/miniconda/envs/pheno/bin/python').is_file(),
              raw_root_exists=RAW.is_dir(),old_audit_visible_on_this_server=OLD.is_dir(),
              old_audit_directory=str(OLD),sha256=hashlib.sha256(reread).hexdigest(),
              utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              scope='Storage verification only; no raw participant data read or copied.')
(OUT/'storage_verification.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
(OUT/'07_v2_storage_check.py').write_bytes(Path(__file__).read_bytes())
(OUT/'MEMO.md').write_text('''# V2存储核验备忘录\n\n07_v2_storage_check.py：确认云盘目录存在，创建独立文件夹，写入固定验证文本，关闭文件后重开读回，逐字节及SHA256校验，并检查原始仓库、pheno解释器和旧核查目录的可见性。\n\n修改入口：V2为新项目云盘根目录，OUT为本次核验目录，OLD为旧服务器独立核查目录。脚本不读取参与者数据，不复制原始仓库，不覆盖用户test文件夹。\n\n产物：WRITE_READBACK_VERIFIED.txt、storage_verification.json、本源码和本备忘录。实际状态以JSON为准；文件读回成功不等于已验证重启或跨服务器持久性。\n''')
(OUT/'exit_code').write_text('0\n')
print('V2_WRITE_CLOSE_READBACK_PASS',flush=True)
print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
