"""Paired canonical identity/pose/expression Pareto plots."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Y_LABELS = {'pose': 'HopeNet pose L2, degrees', 'exp': 'Deep3D expression L2',
            'fid': 'CLIP-FID vs FFHQ-70k (lower is better)'}


def draw_tradeoff(ax, summary, metric):
    for stage, part in summary.groupby('stage'):
        control = 'id_guidance_scale' if stage == 'cfg' else 'id_weight'
        part = part.sort_values(control)
        label = 'Identity CFG' if stage == 'cfg' else 'PuLID id_weight'
        ax.plot(part.id_similarity, part[metric], 'o-', label=label)
        for _, row in part.iterrows():
            ax.annotate(f'{row[control]:g}', (row.id_similarity, row[metric]),
                        xytext=(5, 4), textcoords='offset points', fontsize=9)
    front = summary[summary[f'pareto_{metric}']]
    ax.scatter(front.id_similarity, front[metric], s=130, facecolors='none',
               edgecolors='black', label='non-dominated')
    ax.set(xlabel='Masked WebFace600K R50 cosine (higher is better)', ylabel=Y_LABELS[metric])
    ax.grid(alpha=.3)
    ax.margins(x=.12, y=.12)
    ax.legend(fontsize=8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--combined', action='store_true', help='Also save all trade-offs side by side')
    a = p.parse_args()
    d = pd.read_csv(a.scores, dtype={'pair_id': str})
    keys = ['stage','id_guidance_scale','id_weight']
    metrics = ['id_similarity','pose','exp','top1','top5']
    if d.duplicated(keys+['pair_id']).any():
        raise SystemExit('Duplicate setting/pair')
    sets = list(d.groupby(keys).pair_id.agg(set))
    if not sets or any(s != sets[0] for s in sets):
        raise SystemExit('Unequal pair sets')
    valid = d[d.success.astype(str).str.lower().eq('true') & np.isfinite(d[metrics]).all(axis=1)]
    good = {k:set(v.pair_id) for k,v in valid.groupby(keys)}
    common = set.intersection(*(good.get(k,set()) for k in d.groupby(keys).groups))
    if not common:
        raise SystemExit('No common successful pairs')
    summary = valid[valid.pair_id.isin(common)].groupby(keys)[metrics].mean()
    summary['n_common'] = len(common)
    summary['n_generated'] = d.groupby(keys).size()
    summary['n_success'] = valid.groupby(keys).size()
    summary = summary.reset_index()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    fid_path = a.scores.parent / 'fid.csv'
    plot_metrics = ['pose', 'exp']
    if fid_path.exists():
        fid = pd.read_csv(fid_path)
        summary = summary.merge(fid, on=keys, how='left', validate='one_to_one')
        if summary.fid.isna().any() or not np.isfinite(summary.fid).all():
            raise SystemExit('Missing or invalid FID settings')
        if not summary.n_common.eq(summary.fid_n_generated).all():
            raise SystemExit('FID and paired metrics use different sample counts')
        plot_metrics.append('fid')
    for metric in plot_metrics:
        x, y = summary.id_similarity.to_numpy(), summary[metric].to_numpy()
        front = [not np.any((x>=x[i]) & (y<=y[i]) & ((x>x[i]) | (y<y[i]))) for i in range(len(x))]
        summary[f'pareto_{metric}'] = front
        fig, ax = plt.subplots(figsize=(8,5))
        draw_tradeoff(ax, summary, metric)
        ax.set_title(f'APPLE student: {len(common)} common FFHQ pairs')
        fig.tight_layout()
        fig.savefig(a.out_dir/f'tradeoff_{metric}.png',dpi=180)
        fig.savefig(a.out_dir/f'tradeoff_{metric}.svg')
        plt.close(fig)
    if a.combined:
        ordered = [m for m in ['fid', 'pose', 'exp'] if m in plot_metrics]
        fig, axes = plt.subplots(1, len(ordered), figsize=(7 * len(ordered), 5.5), squeeze=False)
        for ax, metric in zip(axes[0], ordered):
            draw_tradeoff(ax, summary, metric)
            ax.set_title({'fid': 'Identity vs CLIP-FID', 'pose': 'Identity vs pose',
                          'exp': 'Identity vs expression'}[metric])
        if summary.stage.eq('pulid').all() and summary.id_guidance_scale.nunique() == 1:
            detail = f'identity CFG fixed at {summary.id_guidance_scale.iloc[0]:g}; labels = PuLID id_weight'
        else:
            detail = 'labels = identity CFG for cfg runs, PuLID id_weight for pulid runs'
        fig.suptitle(f'APPLE student: {len(common):,} matched FFHQ pairs — {detail}', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(a.out_dir/'combined_tradeoffs.png', dpi=220, bbox_inches='tight')
        fig.savefig(a.out_dir/'combined_tradeoffs.svg', bbox_inches='tight')
        plt.close(fig)
    summary.to_csv(a.out_dir/'summary.csv',index=False)
    table = summary.rename(columns={'id_guidance_scale':'w', 'top1':'top1_percent', 'top5':'top5_percent', 'pose':'pose_l2', 'exp':'expression_l2'})
    table['top1_percent'] *= 100
    table['top5_percent'] *= 100
    columns = ['stage','w','id_weight','id_similarity','top1_percent','top5_percent','pose_l2','expression_l2']
    if 'fid' in table:
        columns.append('fid')
    table[columns + ['n_common']].to_csv(a.out_dir/'metrics_table.csv',index=False)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
