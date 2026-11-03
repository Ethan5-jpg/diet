from pathlib import Path
import re,zlib,json,hashlib
A=Path(__file__).resolve().parent
cmd=(A/'common_command.txt').read_text();outer=re.search(r'fromhex\("([0-9a-f]+)"\)',cmd).group(1);boot=zlib.decompress(bytes.fromhex(outer)).decode();inner=re.search(r"fromhex\('([0-9a-f]+)'\)",boot).group(1);previous=zlib.decompress(bytes.fromhex(inner))
items={p.name:{'source':p.read_text(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [A/'05_build_w08_common.py',A/'06_independent_verify_common.py',A/'07_finalize_and_memo.py',A/'08_display_aggregate.py']}
for n,v in items.items():compile(v['source'],n,'exec')
new=json.dumps(items,separators=(',',':')).encode();boot=boot.replace(inner,zlib.compress(new).hex()).replace(hashlib.sha256(previous).hexdigest(),hashlib.sha256(new).hexdigest());compile(boot,'bootstrap','exec')
(A/'common_command.txt').write_text("/home/ec2-user/miniconda/envs/pheno/bin/python -c 'import zlib;exec(zlib.decompress(bytes.fromhex(\""+zlib.compress(boot.encode()).hex()+"\")))'")
(A/'source_manifest_05_08.json').write_text(json.dumps({n:{'sha256':v['sha256'],'bytes':len(v['source'].encode())} for n,v in items.items()},indent=2));print('refreshed common source package',len((A/'common_command.txt').read_text()))
