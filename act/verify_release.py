"""Verify immutable release assets. Reports and Python caches are outputs."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent


def verify():
    rows=json.loads((ROOT/'release_manifest.json').read_text())['files']
    for row in rows:
        p=ROOT/row['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Release hash mismatch: '+row['path'])
    return len(rows)


if __name__=='__main__':print('Immutable release files verified:',verify())
