"""Matched student PuLID injection sweep at fixed identity CFG; dry-run by flag."""
import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
ASSETS = PROJECT / 'data/aladdin'
WEIGHTS = (0.65, 0.75, 0.90, 1.0, 1.10, 1.25)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weight-scales', type=float, nargs='+', default=list(WEIGHTS))
    p.add_argument('--fixed-cfg', type=float, default=1.2)
    p.add_argument('--output-root', type=Path, default=HERE/'outputs/student_pulid_cfg1p2_full1000')
    p.add_argument('--dry-run', action='store_true', help='Print all commands; perform no computation or writes')
    a = p.parse_args(argv)
    if not math.isfinite(a.fixed_cfg) or a.fixed_cfg < 1:
        p.error('Identity CFG must be finite and >= 1')
    if any(not math.isfinite(w) or w < 0 for w in a.weight_scales):
        p.error('PuLID weights must be finite and >= 0')
    if len(set(a.weight_scales)) != len(a.weight_scales):
        p.error('Duplicate PuLID weights')
    # Keep output names unambiguous and prevent colliding with earlier experiments.
    if len({f'{w:g}' for w in a.weight_scales}) != len(a.weight_scales):
        p.error('PuLID weights produce duplicate run names')
    a.output_root = a.output_root.resolve()
    protected = ['student_matched_full1000', 'student_matched_cfg1p2', 'student_matched_combined']
    if any(a.output_root == (HERE/'outputs'/name).resolve() for name in protected):
        p.error('Use a dedicated PuLID output root, not an existing CFG experiment')
    return a


def build_commands(a):
    inference_python = str(ROOT/'.venv/bin/python')
    eval_python = str(PROJECT/'.venv/bin/python')
    data = str(HERE/'data/aladdin_full_1000')
    canonical = str(PROJECT/'eval')
    prior = str(PROJECT/'outputs/pulid_scale_sweep/full_1000/config.json')
    out = str(a.output_root)
    model = str(ASSETS/'cache/huggingface/hub/models--black-forest-labs--FLUX.1-Krea-dev/snapshots/8162a9c7b05a641be098422bf2fcf335615c2f28')
    names = [f'pulid_cfg{a.fixed_cfg:g}_w{w:g}_steps28_seed42' for w in a.weight_scales]
    return [
        ('Verify matched inputs and evaluator fingerprints', [inference_python, str(HERE/'prepare_matched_data.py'),
            '--aladdin-config', prior, '--destination', data, '--canonical-eval', canonical]),
        ('Student inference: fixed identity CFG, varying PuLID weight', [inference_python, '-u', str(HERE/'run_sweep.py'),
            '--stage', 'pulid', '--fixed-cfg', str(a.fixed_cfg), '--weight-scales', *map(str, a.weight_scales),
            '--ffhq-base', data, '--checkpoint', str(ROOT/'checkpoints/student'), '--condition-type', 'raw',
            '--model-id', model, '--max-pairs', '0', '--num-steps', '28', '--seed', '42', '--nproc', '1',
            '--output-root', out]),
        ('Canonical identity, retrieval, pose and expression', [eval_python, '-u', str(HERE/'evaluate.py'),
            '--protocol', 'canonical', '--canonical-eval', canonical, '--ffhq-base', data,
            '--output-root', out, '--scores', str(a.output_root/'scores.csv'), '--device', 'cuda',
            '--run-names', *names]),
        ('CLIP-FID against the same FFHQ-70k reference', [eval_python, '-u', str(HERE/'evaluate_fid.py'),
            '--canonical-eval', canonical, '--ffhq-base', data, '--output-root', out,
            '--aladdin-config', prior, '--run-names', *names,
            '--real-features', str(ASSETS/'data/eval/fid/_feats_clip/ffhq_images512_clip224_n70000.npy')]),
        ('Summary and all three trade-off panels', [eval_python, str(HERE/'plot_canonical.py'),
            '--scores', str(a.output_root/'scores.csv'), '--out-dir', str(a.output_root/'figures'), '--combined']),
    ]


def main(argv=None):
    a = parse_args(argv)
    print(f'Student only; identity CFG={a.fixed_cfg:g}; PuLID weights={a.weight_scales}', flush=True)
    print(f'1,000 matched pairs per setting; output: {a.output_root}', flush=True)
    env = dict(os.environ, HF_HOME=str(ASSETS/'cache/huggingface'), HF_HUB_OFFLINE='1',
               EVAL_FFHQ_DIR=str(ASSETS/'data/FFHQ'), EVAL_WEIGHTS_DIR=str(ASSETS/'eval_weights'),
               OMP_NUM_THREADS='8')
    for label, command in build_commands(a):
        print(f'\n{label}\n{shlex.join(command)}', flush=True)
        if not a.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    if a.dry_run:
        print('\nDRY RUN ONLY: no data writes, model loading, inference or evaluation.', flush=True)


if __name__ == '__main__':
    main()
