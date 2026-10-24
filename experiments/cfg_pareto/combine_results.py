"""Combine comparable saved CFG sweeps without rerunning inference or evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.resolve() in [p.resolve() for p in args.inputs]:
        raise SystemExit('Use a separate combined output directory')
    reference_protocol = reference_fid = reference_context = None
    observations, distributional, provenance = [], [], []
    context_keys = ['checkpoint', 'checkpoint_size', 'checkpoint_mtime_ns', 'model_id',
                    'num_steps', 'seed', 'condition_type', 'inference_sha256', 'ffhq_base', 'pair_ids']
    for root in args.inputs:
        protocol = json.loads((root/'scores.protocol.json').read_text())
        fid_protocol = json.loads((root/'fid.protocol.json').read_text())
        if reference_protocol is not None and protocol != reference_protocol:
            raise SystemExit(f'Canonical scoring protocols differ: {root}')
        if reference_fid is not None and fid_protocol != reference_fid:
            raise SystemExit(f'FID protocols differ: {root}')
        reference_protocol, reference_fid = protocol, fid_protocol
        data = pd.read_csv(root/'scores.csv', dtype={'pair_id': str})
        fid = pd.read_csv(root/'fid.csv')
        for _, setting in fid.iterrows():
            matches = []
            for path in root.glob('*/manifest.json'):
                meta = json.loads(path.read_text())
                if (meta['stage'], meta['id_guidance_scale'], meta['id_weight']) == (
                        setting.stage, setting.id_guidance_scale, setting.id_weight):
                    matches.append((path, meta))
            if len(matches) != 1:
                raise SystemExit(f'Expected exactly one manifest for setting: {setting.to_dict()}')
            path, meta = matches[0]
            if meta['status'] != 'complete':
                raise SystemExit(f'Incomplete run: {path}')
            context = {key: meta.get(key) for key in context_keys}
            if reference_context is not None and context != reference_context:
                changed = [key for key in context if context[key] != reference_context[key]]
                raise SystemExit(f'Inference settings differ ({changed}): {path}')
            reference_context = context
            part = data[(data.stage == setting.stage)
                        & (data.id_guidance_scale == setting.id_guidance_scale)
                        & (data.id_weight == setting.id_weight)]
            if set(part.pair_id) != set(meta['pair_ids']) or len(part) != len(meta['pair_ids']):
                raise SystemExit(f'Pair observations do not match manifest: {path}')
            if len(part) != int(setting.fid_n_generated) or int(setting.fid_n_real) != 70000:
                raise SystemExit(f'FID sample count mismatch: {root}')
        observations.append(data)
        distributional.append(fid)
        provenance.append(dict(path=str(root.resolve()), scores_sha256=hashlib.sha256((root/'scores.csv').read_bytes()).hexdigest(),
                               fid_sha256=hashlib.sha256((root/'fid.csv').read_bytes()).hexdigest()))
    data = pd.concat(observations, ignore_index=True)
    fid = pd.concat(distributional, ignore_index=True)
    keys = ['stage', 'id_guidance_scale', 'id_weight']
    metrics = ['id_similarity', 'top1', 'top5', 'pose', 'exp']
    if data.duplicated(keys+['pair_id']).any() or fid.duplicated(keys).any():
        raise SystemExit('Overlapping settings in input experiments')
    if not data.success.astype(str).str.lower().eq('true').all() or not np.isfinite(data[metrics]).all().all():
        raise SystemExit('This merge requires successful finite scores for every pair')
    summary = data.groupby(keys)[metrics].mean().reset_index().merge(fid, on=keys, validate='one_to_one')
    summary['n_common'] = data.groupby(keys).size().to_numpy()
    summary = summary.sort_values('id_guidance_scale').reset_index(drop=True)
    for metric in ['fid', 'pose', 'exp']:
        x, y = summary.id_similarity.to_numpy(), summary[metric].to_numpy()
        summary[f'pareto_{metric}'] = [not np.any((x >= x[i]) & (y <= y[i]) & ((x > x[i]) | (y < y[i])))
                                       for i in range(len(summary))]
    out = args.output_root
    figures = out/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    data.sort_values(keys+['pair_id']).to_csv(out/'scores.csv', index=False)
    fid.sort_values(keys).to_csv(out/'fid.csv', index=False)
    summary.to_csv(figures/'summary.csv', index=False)
    table = summary.rename(columns={'id_guidance_scale':'w', 'top1':'top1_percent', 'top5':'top5_percent',
                                    'pose':'pose_l2', 'exp':'expression_l2'})
    table.top1_percent *= 100
    table.top5_percent *= 100
    table = table[['w','id_similarity','top1_percent','top5_percent','pose_l2','expression_l2','fid','n_common']]
    table.to_csv(figures/'combined_metrics.csv', index=False)
    (out/'combined.protocol.json').write_text(json.dumps(dict(inputs=provenance, canonical=reference_protocol,
                                                            fid=reference_fid, inference=reference_context), indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
    for ax, metric, title, ylabel in zip(axes, ['fid','pose','exp'], ['Identity vs CLIP-FID','Identity vs pose','Identity vs expression'],
                                       ['CLIP-FID vs FFHQ-70k ↓','HopeNet pose L2, degrees ↓','Deep3D expression L2 ↓']):
        ax.plot(summary.id_similarity, summary[metric], 'o-', color='#2563a6', linewidth=1.8, markersize=6, label='Combined CFG sweep')
        front = summary[summary[f'pareto_{metric}']]
        ax.scatter(front.id_similarity, front[metric], s=120, facecolors='none', edgecolors='#333333', linewidths=1, zorder=3)
        new = summary[np.isclose(summary.id_guidance_scale, 1.2)]
        ax.scatter(new.id_similarity, new[metric], s=200, marker='*', color='#ea780a', edgecolors='black', linewidths=.5, zorder=5, label='Added w = 1.2')
        for _, row in summary.iterrows():
            ax.annotate(f'{row.id_guidance_scale:g}', (row.id_similarity, row[metric]), xytext=(7, 5), textcoords='offset points',
                        fontsize=10, color='#b55300' if np.isclose(row.id_guidance_scale, 1.2) else '#222222')
        ax.set(title=title, xlabel='Masked WebFace600K R50 identity cosine ↑', ylabel=ylabel)
        ax.grid(alpha=.22)
        ax.margins(x=.12, y=.12)
    axes[0].legend(loc='upper left', fontsize=9)
    fig.suptitle(f'APPLE student: combined identity CFG sweep — {int(summary.n_common.min()):,} matched FFHQ pairs per setting\n'
                 'Labels = identity CFG w; right is better identity, down is lower error / FID', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .89))
    fig.savefig(figures/'combined_tradeoffs.png', dpi=220, bbox_inches='tight')
    fig.savefig(figures/'combined_tradeoffs.svg', bbox_inches='tight')
    plt.close(fig)
    print(table.to_string(index=False))
    print(f'Saved combined metrics and three-panel plot in {figures.resolve()}')

if __name__ == '__main__':
    main()
