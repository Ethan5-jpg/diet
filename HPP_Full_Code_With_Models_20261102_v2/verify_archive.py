from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
items=json.loads((root/'05_MANIFEST/files.json').read_text())
for item in items:
    p=root/item['archive_path'];b=p.read_bytes()
    assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'],item['archive_path']
print('PASS',len(items),'files verified')
