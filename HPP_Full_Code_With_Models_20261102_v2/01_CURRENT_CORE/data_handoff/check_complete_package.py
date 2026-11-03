"""Verify immutable package payload hashes without exposing any participant values."""
from pathlib import Path
import hashlib, json
ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
def check():
    entries = json.loads((ROOT / 'manifests' / 'payload_sha256_final.json').read_text())
    for item in entries:
        p = ROOT / item['relative_path']
        assert p.is_file() and p.stat().st_size == item['bytes'], item['relative_path']
        assert sha(p) == item['sha256'], item['relative_path']
    return dict(status='PASS', checked_files=len(entries), checked_bytes=sum(i['bytes'] for i in entries))
if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
