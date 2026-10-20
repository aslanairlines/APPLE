"""Score APPLE using the existing ALADDIN metric implementations and fixed gallery."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def score_runs(args, runs):
    import torch
    sys.path.insert(0, str(args.canonical_eval.resolve()))
    import config
    import score_id
    import score_pose_expr as pe
    config.DEVICE = torch.device(args.device)
    data = json.loads((args.ffhq_base / 'pairs.json').read_text())
    pairs = {f"{p['pair_id']:06d}": p for p in data['pairs']}
    gallery = data['gallery']
    checked = set()
    for pair in pairs.values():
        for key in ['src', 'tgt', 'src_mask', 'tgt_mask']:
            path = Path(pair[key])
            if path not in checked:
                if hashlib.sha256(path.read_bytes()).hexdigest() != pair[key + '_sha256']:
                    raise ValueError(f'ALADDIN input changed before evaluation: {path}')
                checked.add(path)
    for entry in gallery:
        for key, hashkey in [('path', 'sha256'), ('mask', 'mask_sha256')]:
            path = Path(entry[key])
            if path not in checked:
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry[hashkey]:
                    raise ValueError(f'Gallery input changed: {path}')
                checked.add(path)
    for pair in pairs.values():
        entry = gallery[pair['gallery_index']]
        if (entry['path'], entry['mask'], entry['source_id']) != (pair['src'], pair['src_mask'], pair['source_id']):
            raise ValueError('Pair source disagrees with fixed gallery')
    encoder = 'webface_r50'
    model = score_id._build_model(encoder, config.DEVICE)
    def embed(paths, masks):
        return score_id._embed_masked(model, encoder, list(map(Path, paths)),
                                     list(map(Path, masks)), batch_size=32, device=config.DEVICE)
    print(f'Extracting fixed gallery: {len(gallery)} source images', flush=True)
    g = embed([e['path'] for e in gallery], [e['mask'] for e in gallery])
    labels = [e['source_id'] for e in gallery]
    if len(set(labels)) != len(labels):
        raise ValueError('This benchmark requires one gallery image per source identity')
    targets, rows = {}, []
    for manifest, meta, images in runs:
        print(f'Canonical scoring {manifest.parent.name}: {len(images)} pairs', flush=True)
        for image in images:
            p = pairs[image.stem]
            row = dict(stage=meta['stage'], id_guidance_scale=meta['id_guidance_scale'],
                       id_weight=meta['id_weight'], pair_id=image.stem, success=False, error='',
                       id_similarity='', pose='', exp='', top1='', top5='')
            try:
                if image.stem not in targets:
                    targets[image.stem] = pe.pose_deg(p['tgt']), pe.exp_coeff(p['tgt'])
                q = embed([image], [p['tgt_mask']])
                rank = (q @ g.T)[0].topk(min(5, len(labels))).indices.tolist()
                identity = p['source_id']
                row.update(id_similarity=float((q @ g[p['gallery_index']:p['gallery_index']+1].T)[0, 0]),
                           pose=float(np.linalg.norm(pe.pose_deg(image) - targets[image.stem][0])),
                           exp=float(np.linalg.norm(pe.exp_coeff(image) - targets[image.stem][1])),
                           top1=float(labels[rank[0]] == identity),
                           top5=float(identity in [labels[i] for i in rank]))
                row['success'] = all(np.isfinite(row[k]) for k in ['id_similarity','pose','exp','top1','top5'])
            except (OSError, ValueError, RuntimeError) as exc:
                row['error'] = str(exc)
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f'Scored {len(rows)} observations', flush=True)
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    with args.scores.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(protocol='ALADDIN masked WebFace600K R50 + whole-frame HopeNet/Deep3D',
                      adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      gallery_size=len(gallery), retrieval_units='fraction; multiply by 100 for percent',
                      pairs_sha256=hashlib.sha256((args.ffhq_base/'pairs.json').read_bytes()).hexdigest(),
                      evaluator_sha256={name: hashlib.sha256((args.canonical_eval/name).read_bytes()).hexdigest()
                                        for name in ['score_id.py','score_pose_expr.py']},
                      attribute_weights_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                                for path in [config.HOPENET_W, config.DEEP3D_W]},
                      recognizer_sha256=score_id._verified_weights(encoder)[1])
    args.scores.with_suffix('.protocol.json').write_text(json.dumps(provenance, indent=2))
    print(f'Wrote {len(rows)} canonical observations: {args.scores}', flush=True)
    if not any(r['success'] for r in rows):
        raise SystemExit('All evaluations failed; inspect the CSV error column')
