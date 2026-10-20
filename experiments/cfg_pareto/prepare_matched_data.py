"""Create a lossless APPLE input view of the exact prior ALADDIN sweep pairs."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--aladdin-config',type=Path,required=True)
p.add_argument('--canonical-eval', type=Path, help='Verify evaluator code and weights against prior experiment')
p.add_argument('--destination',type=Path,required=True)
a=p.parse_args()
d=json.loads(a.aladdin_config.read_text())
if a.canonical_eval is not None:
    for name in ['score_id.py', 'score_pose_expr.py']:
        if hashlib.sha256((a.canonical_eval/name).read_bytes()).hexdigest() != d['evaluator_code_sha256'][name]:
            raise SystemExit(f'Canonical evaluator changed since ALADDIN experiment: {name}')
    for name in ['webface', 'hopenet', 'deep3d']:
        spec = d['evaluator_weights'][name]
        if hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest() != spec['sha256']:
            raise SystemExit(f'Canonical evaluator weights changed: {name}')
    print('Canonical evaluator code and weight checksums match prior ALADDIN experiment')
checked=set()
for pair in d['pairs']:
    for key in ['src','tgt','src_mask','tgt_mask']:
        path=Path(pair[key])
        if path in checked:
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pair[key+'_sha256']:
            raise SystemExit(f'Changed ALADDIN input: {path}')
        with Image.open(path) as im:
            if im.size!=(512,512):
                raise SystemExit(f'Expected 512x512: {path}')
            im.verify()
        checked.add(path)
    for folder,key in [('src','src'),('trg','tgt')]:
        dest=a.destination/folder/f"{pair['pair_id']:06d}.jpg"
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.is_symlink() and dest.resolve()==Path(pair[key]).resolve():
            continue
        if dest.exists() or dest.is_symlink():
            raise SystemExit(f'Conflicting file: {dest}')
        # APPLE discovers .jpg filenames; PIL decodes the original PNG content.
        dest.symlink_to(Path(pair[key]).resolve())
for e in d['gallery']:
    for key,hashkey in [('path','sha256'),('mask','mask_sha256')]:
        if hashlib.sha256(Path(e[key]).read_bytes()).hexdigest()!=e[hashkey]:
            raise SystemExit(f'Changed gallery input: {e[key]}')
payload=json.dumps({k:d[k] for k in ['pairs','gallery']},indent=2)+'\n'
manifest=a.destination/'pairs.json'
if manifest.exists() and manifest.read_text()!=payload:
    raise SystemExit('Existing pair manifest differs')
manifest.write_text(payload)
print(f"Verified {len(d['pairs'])} pairs, {len(d['gallery'])} gallery entries; view: {a.destination}")
