"""Run authorized transforms on synthetic inputs in the actual pheno environment."""
from pathlib import Path
import hashlib, json, subprocess, sys

P = Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
S = Path(__file__).parent
Q = P/'data_qc/preprocessing_20261030_v1'
L = Path('/home/ec2-user/multisystem_runtime_20261021/preprocessing_20261030_v1')
assert (L/'01_actual_exit_code').read_text().strip() == '0'
pre = json.loads((Q/'preflight_TRE_ONLY.json').read_text())
assert pre['new_scientific_Y_reads'] == 0
with (L/'02_synthetic_tests.log').open('x') as log:
    p = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(S),
                        '-p', 'test_preprocessing_v1_20261030.py', '-v'], stdout=log, stderr=subprocess.STDOUT)
(L/'02_synthetic_tests_actual_exit_code').write_text(str(p.returncode)+'\n')
assert p.returncode == 0, 'pheno synthetic checks failed; inspect preserved log'
config = json.loads((S/'preprocessing_v1_20261030.json').read_text())
assert config['microbiome']['prevalence_min'] == 0.05
assert config['execution']['real_Y_reads'].startswith('none')
files = [{'name': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
         for p in sorted(S.iterdir()) if p.is_file()]
receipt = {'status':'pheno_synthetic_checks_passed', 'test_count':11, 'test_actual_exit':p.returncode,
           'new_scientific_Y_reads':0, 'real_fit':False, 'source_files':files,
           'role_status':pre['final_roles'], 'formal_fit_requires_W18_roles':True}
(Q/'server_synthetic_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('SERVER_SYNTHETIC_PASS', json.dumps({k:v for k,v in receipt.items() if k!='source_files'}), flush=True)
