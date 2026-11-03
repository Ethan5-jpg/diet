from pathlib import Path
import shlex,ast
S=Path(__file__).parent;remote='/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/scripts/w20_targeted_20261101_v3';runtime='/home/ec2-user/multisystem_runtime_20261021/w20_targeted_20261101_v3'
for stage,script in [('49b','49b_models.py'),('50b','50b_daily.py')]:
 runner=f'''from pathlib import Path
import os,subprocess,sys,json,datetime
l=Path({runtime!r});s=Path({remote!r});env=os.environ.copy();env['PYTHONPATH']=str(s)+':'+str(s.parent/'w20_simplified_20261101_v2')+':'+str(s.parent/'preprocessing_20261030_v1')
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[k]='1'
(l/'{stage}_runner_pid').write_text(str(os.getpid()))
with (l/'{stage}.log').open('x') as log:
 p=subprocess.Popen([sys.executable,'-u',str(s/{script!r})],env=env,stdout=log,stderr=subprocess.STDOUT)
 (l/'{stage}_worker_pid').write_text(str(p.pid));rc=p.wait()
(l/'{stage}_actual_exit_code').write_text(str(rc));(l/'{stage}_finished_UTC').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat());sys.exit(rc)
'''
 wrapper=f'''from pathlib import Path
import subprocess,sys,hashlib,json
s=Path({remote!r});l=Path({runtime!r});p=s/'{stage}_runner.py'
assert not (l/'{stage}_worker_pid').exists() and not (l/'{stage}.log').exists()
with p.open('x') as f:f.write({runner!r})
compile(p.read_text(),str(p),'exec')
j=subprocess.Popen([sys.executable,'-u',str(p)],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
(l/'{stage}_launch_pid').write_text(str(j.pid));print('STARTED_STAGE {stage} runner',j.pid,'worker_pid_file',str(l/'{stage}_worker_pid'))
'''
 ast.parse(runner);ast.parse(wrapper);(S/(stage+'_launch_command.txt')).write_text('/home/ec2-user/miniconda/envs/pheno/bin/python -c '+shlex.quote('exec('+repr(wrapper)+')'))
