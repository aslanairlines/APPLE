"""Paired canonical identity/pose/expression Pareto plots."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
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
        for stage, part in summary.groupby('stage'):
            part = part.sort_values('id_guidance_scale' if stage=='cfg' else 'id_weight')
            ax.plot(part.id_similarity, part[metric], 'o-', label=stage)
            for _, r in part.iterrows():
                ax.annotate(f"{r.id_guidance_scale:g}" if stage=='cfg' else f"{r.id_weight:g}",
                            (r.id_similarity,r[metric]), xytext=(5,5), textcoords='offset points')
        ax.scatter(x[front],y[front],s=130,facecolors='none',edgecolors='black',label='non-dominated')
        ax.set(xlabel='Masked WebFace600K R50 cosine (higher is better)',
               ylabel={'pose':'HopeNet pose L2, degrees', 'exp':'Deep3D expression L2', 'fid':'CLIP-FID vs FFHQ-70k (lower is better)'}[metric],
               title=f'APPLE student: {len(common)} common FFHQ pairs')
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.out_dir/f'tradeoff_{metric}.png',dpi=180)
        fig.savefig(a.out_dir/f'tradeoff_{metric}.svg')
        plt.close(fig)
    summary.to_csv(a.out_dir/'summary.csv',index=False)
    table = summary.rename(columns={'id_guidance_scale':'w', 'top1':'top1_percent', 'top5':'top5_percent', 'pose':'pose_l2', 'exp':'expression_l2'})
    table['top1_percent'] *= 100
    table['top5_percent'] *= 100
    columns = ['w','id_similarity','top1_percent','top5_percent','pose_l2','expression_l2']
    if 'fid' in table:
        columns.append('fid')
    table[columns + ['n_common']].to_csv(a.out_dir/'metrics_table.csv',index=False)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
