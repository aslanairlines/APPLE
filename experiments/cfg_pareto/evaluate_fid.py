"""Score saved APPLE outputs with ALADDIN's exact CLIP-FID implementation."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--ffhq-base', type=Path, required=True)
    p.add_argument('--canonical-eval', type=Path, required=True)
    p.add_argument('--aladdin-config', type=Path, required=True)
    p.add_argument('--real-features', type=Path, required=True)
    p.add_argument('--run-names', nargs='+', help='Score only this requested cohort')
    a = p.parse_args()
    sys.path.insert(0, str(a.canonical_eval.resolve()))
    import config
    import fid_clip
    from evaluate import collect_runs
    prior = json.loads(a.aladdin_config.read_text())
    for name in ['fid_clip.py', 'score_fid.py']:
        if sha(a.canonical_eval/name) != prior['evaluator_code_sha256'][name]:
            raise SystemExit(f'ALADDIN evaluator fingerprint differs: {name}')
    if sha(config.CLIP_W) != prior['evaluator_weights']['clip']['sha256']:
        raise SystemExit('CLIP weights differ from ALADDIN')
    runs = sorted(collect_runs(a.output_root, a.ffhq_base, a.run_names), key=lambda r: (r[1]["stage"], r[1]["id_guidance_scale"], r[1]["id_weight"]))
    real = np.load(a.real_features, allow_pickle=False)
    if real.shape != (70000,512) or real.dtype != np.float32 or not np.isfinite(real).all():
        raise SystemExit('Expected complete finite FFHQ-70k CLIP feature cache')
    mean, cov = real.mean(0), np.cov(real, rowvar=False)
    fingerprint = dict(real_features_sha256=sha(a.real_features), clip_weights_sha256=sha(config.CLIP_W),
                       evaluator_sha256=sha(a.canonical_eval/'fid_clip.py'), precision='fp16 on CUDA',
                       reference='FFHQ-70k', feature_extractor='OpenAI CLIP ViT-B/32 quickgelu')
    cache = a.output_root/'fid_cache'
    cache.mkdir(exist_ok=True)
    rows=[]
    for manifest, meta, files in runs:
        if len(files)<1000:
            raise SystemExit('CLIP-FID requires at least 1000 generated images per setting')
        outputs_sha = hashlib.sha256(json.dumps([(f.name,sha(f)) for f in files]).encode()).hexdigest()
        stamp = dict(fingerprint, generated_sha256=outputs_sha)
        feature_path=cache/(manifest.parent.name+'.npy')
        stamp_path=cache/(manifest.parent.name+'.json')
        if feature_path.exists() and stamp_path.exists() and json.loads(stamp_path.read_text())==stamp:
            features=np.load(feature_path,allow_pickle=False)
        else:
            print(f'CLIP features: CFG {meta["id_guidance_scale"]:g}, id_weight {meta["id_weight"]:g}, {len(files)} saved images',flush=True)
            features=fid_clip.features(files, batch_size=256, num_workers=8)
            np.save(feature_path,features,allow_pickle=False)
            stamp_path.write_text(json.dumps(stamp,indent=2))
        if features.shape!=(len(files),512) or not np.isfinite(features).all():
            raise SystemExit('Invalid generated CLIP feature cache')
        value=fid_clip.frechet_distance(features.mean(0),np.cov(features,rowvar=False),mean,cov)
        if not np.isfinite(value):
            raise SystemExit('Nonfinite FID')
        rows.append(dict(stage=meta['stage'],id_guidance_scale=meta['id_guidance_scale'],
                         id_weight=meta['id_weight'],fid=value,fid_n_generated=len(files),fid_n_real=70000))
        print(f'CFG {meta["id_guidance_scale"]:g}, id_weight {meta["id_weight"]:g}: CLIP-FID = {value:.6f}',flush=True)
        with (a.output_root/'fid.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.output_root/'fid.protocol.json').write_text(json.dumps(fingerprint,indent=2))

if __name__=='__main__':
    main()
