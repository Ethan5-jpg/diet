"""Build a reviewable, guarded source-only repair; does not access native data."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
OLD=json.loads((HERE/'previous_release/manifest.json').read_text())
NEW=json.loads((ROOT/'manifest.json').read_text())
OLD_MANIFEST_SHA=hashlib.sha256((HERE/'previous_release/manifest.json').read_bytes()).hexdigest()
NEW_MANIFEST_SHA=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
CHANGED=['pilot.py','check_synthetic.py','README.md']
assert {name for name in OLD['files'] if OLD['files'][name]!=NEW['files'][name]}==set(CHANGED)

header='''#!/usr/bin/env bash
bash <<'HPP_FUNCTION_IO_REPAIR1'
set -euo pipefail
hpp_io_root="${HPP_FUNCTION_IO_ROOT:-/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021/collaborator_handoff_20261102_v1/experiments/diet_effect_modification_20261109_v12/functional_followup_20261110_v1}"
cd "$hpp_io_root"
hpp_io_python=python
if [ -s ../.python_path ]; then
    IFS= read -r hpp_io_python < ../.python_path
fi
"$hpp_io_python" -B - <<'HPP_IO_PREFLIGHT'
import hashlib,json,os,socket
from pathlib import Path
root=Path.cwd()
'''
preflight='''
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def stop(message):
    raise SystemExit('IO_REPAIR_STOP: '+message)
if root.name!='functional_followup_20261110_v1':
    stop('unexpected module directory')
for name in OLD['files']:
    path=root/name
    if path.is_symlink() or not path.is_file() or digest(path) not in (OLD['files'][name],NEW['files'][name]):
        stop('unknown source bytes: '+name)
manifest=root/'manifest.json'
if manifest.is_symlink() or digest(manifest) not in (OLD_MANIFEST_SHA,NEW_MANIFEST_SHA):
    stop('unknown source manifest')
oldrun=root/'runs/function_20261110_v1'
newrun=root/'runs/function_20261110_v1_iofix1'
for run in (oldrun,newrun):
    lock=run/'RUNNING.lock'
    if lock.exists():
        value=json.loads(lock.read_text())
        if value.get('host')!=socket.gethostname():
            stop('cannot verify a lock from another host')
        pid=value.get('pid')
        if not isinstance(pid,int) or pid<=0:
            stop('invalid lock owner')
        try:
            os.kill(pid,0)
        except ProcessLookupError:
            pass
        except PermissionError:
            stop('cannot verify lock owner')
        else:
            stop('a run process is still active; no sources changed')
if (oldrun/'COMPLETE.json').exists() or (oldrun/'summary.json').exists() or list((oldrun/'TRE_ONLY').glob('r*_f*_COMPLETE.json')):
    stop('old run already has completed scientific work; do not replace its source identity')
registration=oldrun/'registration.json'
if registration.exists():
    value=json.loads(registration.read_text())
    if value.get('config')!=json.loads((root/'config.json').read_text()) or value.get('synthetic_only') is not False:
        stop('old run configuration/scope differs')
archive=root/'research_records/2026-11-10_iofix1/server_previous_sources'
if archive.is_symlink():
    stop('source backup directory is a symlink')
archive.mkdir(parents=True,exist_ok=True)
for name in CHANGED+['manifest.json']:
    path=root/name
    expected=OLD_MANIFEST_SHA if name=='manifest.json' else OLD['files'][name]
    backup=archive/name
    if backup.is_symlink() or backup.exists() and digest(backup)!=expected:
        stop('unexpected previous-source backup: '+name)
    if digest(path)==expected and not backup.exists():
        with backup.open('xb') as stream:
            stream.write(path.read_bytes())
        if digest(backup)!=expected:
            stop('previous-source backup verification failed')
state=dict(old_run_id='function_20261110_v1',new_run_id='function_20261110_v1_iofix1',
    old_registration_sha256=digest(registration) if registration.exists() else None,
    old_lock_sha256=digest(oldrun/'RUNNING.lock') if (oldrun/'RUNNING.lock').exists() else None,
    statistical_config_sha256=digest(root/'config.json'),old_completed_fold_count=0,old_summary_present=False)
state_path=archive.parent/'original_failed_run_state.json'
if state_path.exists():
    if json.loads(state_path.read_text())!=state:
        stop('failed-run state changed since initial repair guard')
else:
    with state_path.open('x') as stream:
        stream.write(json.dumps(state,sort_keys=True,indent=2)+'\\n')
print('IO_REPAIR_PREFLIGHT_OK: known sources, no completed old folds, no active owner; failed run retained.')
HPP_IO_PREFLIGHT
hpp_io_write() {
    local hpp_name="$1"
    local hpp_old="$2"
    local hpp_new="$3"
    local hpp_temporary
    hpp_temporary="$(mktemp "$PWD/.iofix-source.XXXXXXXX")"
    cat > "$hpp_temporary"
    "$hpp_io_python" -B - "$hpp_name" "$hpp_old" "$hpp_new" "$hpp_temporary" <<'HPP_IO_SOURCE'
import hashlib,os,sys
from pathlib import Path
name,old,new,incoming=sys.argv[1],sys.argv[2],sys.argv[3],Path(sys.argv[4])
try:
    target=Path.cwd()/name
    if Path(name).name!=name or target.is_symlink():
        raise SystemExit('IO_REPAIR_STOP: invalid source target')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(incoming)!=new:
        raise SystemExit('IO_REPAIR_STOP: incoming checksum mismatch')
    current=digest(target)
    if current==new:
        pass
    elif current==old:
        backup=Path.cwd()/'research_records/2026-11-10_iofix1/server_previous_sources'/name
        if not backup.is_file() or digest(backup)!=old:
            raise SystemExit('IO_REPAIR_STOP: old source backup missing')
        os.replace(str(incoming),str(target))
        if digest(target)!=new:
            raise SystemExit('IO_REPAIR_STOP: source readback mismatch')
    else:
        raise SystemExit('IO_REPAIR_STOP: source changed after preflight')
finally:
    if incoming.exists():
        incoming.unlink()
HPP_IO_SOURCE
}
'''
constants='OLD=json.loads(r\'\'\'%s\'\'\')\nNEW=json.loads(r\'\'\'%s\'\'\')\nOLD_MANIFEST_SHA=%r\nNEW_MANIFEST_SHA=%r\nCHANGED=%r\n' % (
    json.dumps(OLD,sort_keys=True),json.dumps(NEW,sort_keys=True),OLD_MANIFEST_SHA,NEW_MANIFEST_SHA,CHANGED)
parts=[header,constants,preflight]
for i,name in enumerate(CHANGED+['manifest.json']):
    content=(ROOT/name).read_text()
    token='HPP_IO_SOURCE_%02d_END' % i
    assert token not in content and 'HPP_FUNCTION_IO_REPAIR1\n' not in content
    old=OLD_MANIFEST_SHA if name=='manifest.json' else OLD['files'][name]
    new=NEW_MANIFEST_SHA if name=='manifest.json' else NEW['files'][name]
    parts.append("hpp_io_write '%s' '%s' '%s' <<'%s'\n%s%s\n" % (name,old,new,token,content,token))
parts.append('''"$hpp_io_python" -B - <<'HPP_IO_FINAL'
import hashlib,json
from pathlib import Path
root=Path.cwd()
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'manifest.json').read_text())
for name,expected in manifest['files'].items():
    if Path(name).name!=name or digest(root/name)!=expected:
        raise SystemExit('IO_REPAIR_STOP: repaired source manifest mismatch')
archive=root/'research_records/2026-11-10_iofix1'
state=json.loads((archive/'original_failed_run_state.json').read_text())
oldrun=root/'runs/function_20261110_v1'
for name,key in [('registration.json','old_registration_sha256'),('RUNNING.lock','old_lock_sha256')]:
    actual=digest(oldrun/name) if (oldrun/name).exists() else None
    if actual!=state[key]:
        raise SystemExit('IO_REPAIR_STOP: old failed run changed')
if digest(root/'config.json')!=state['statistical_config_sha256']:
    raise SystemExit('IO_REPAIR_STOP: statistical configuration changed')
receipt=dict(status='STORAGE_PATCH_INSTALLED',analysis_configuration_unchanged=True,old_failed_run_preserved=True,
    new_run_id='function_20261110_v1_iofix1',manifest_sha256=digest(root/'manifest.json'),native_storage_success_not_yet_verified=True)
with (archive/'repair_installed.json').open('w') as stream:
    stream.write(json.dumps(receipt,sort_keys=True,indent=2)+'\\n')
print('IO_REPAIR_INSTALLED: memory ZIP plus sequential writes; statistical plan unchanged; old run preserved.')
HPP_IO_FINAL
if [ "${HPP_FUNCTION_IO_REPAIR_ONLY:-0}" = 1 ]; then
    echo 'IO_REPAIR_ONLY_DONE; no models started.'
    exit 0
fi
if [ -f runs/function_20261110_v1_iofix1/COMPLETE.json ]; then
    bash server_run.sh --threads 2 --run-id function_20261110_v1_iofix1 --report
elif [ -f runs/function_20261110_v1_iofix1/registration.json ]; then
    bash server_run.sh --threads 2 --run-id function_20261110_v1_iofix1 --resume
else
    bash server_run.sh --threads 2 --run-id function_20261110_v1_iofix1
fi
HPP_FUNCTION_IO_REPAIR1
''')
(HERE/'repair_and_start.sh').write_text(''.join(parts))
print('Built guarded IO repair and restart command.')
