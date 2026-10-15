"""Run APPLE CFG inference sweep, evaluation and Pareto plot in sequence.

Run this inside tmux only AFTER GPU smoke test succeeds.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ffhq-base", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-id", default="black-forest-labs/FLUX.1-Krea-dev")
    p.add_argument("--condition-type", default="blur_landmark_iris")
    p.add_argument("--cfg-scales", type=float, nargs="+", default=[1, 1.5, 2, 2.5, 3, 4, 5])
    p.add_argument("--fixed-weight", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=28)
    p.add_argument("--max-pairs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--eval-device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--output-root", type=Path, default=HERE / "outputs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    data = args.ffhq_base.resolve()
    ckpt = args.checkpoint.resolve()
    out_root = args.output_root.resolve()
    commands = [
        ("inference", [
            sys.executable, str(HERE / "run_sweep.py"),
            "--stage", "cfg", "--ffhq-base", str(data), "--checkpoint", str(ckpt),
            "--model-id", args.model_id, "--condition-type", args.condition_type,
            "--cfg-scales", *(str(x) for x in args.cfg_scales),
            "--fixed-weight", str(args.fixed_weight),
            "--num-steps", str(args.num_steps), "--max-pairs", str(args.max_pairs),
            "--seed", str(args.seed), "--nproc", str(args.nproc),
            "--output-root", str(out_root),
        ]),
        ("evaluate", [
            sys.executable, str(HERE / "evaluate.py"), "--ffhq-base", str(data),
            "--output-root", str(out_root), "--scores", str(HERE / "scores.csv"),
            "--device", args.eval_device,
        ]),
        ("plot", [
            sys.executable, str(HERE / "plot_pareto.py"),
            "--scores", str(HERE / "scores.csv"), "--out-dir", str(HERE / "figures"),
        ]),
    ]
    for label, command in commands:
        if args.dry_run and label == "inference":
            command = command + ["--dry-run"]
        print(f"\n=== {label.upper()} ===\n{shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        print(f"\nExperiment completed. Summary: {HERE / 'figures/summary.csv'}", flush=True)
        print(f"Trade-off plot: {HERE / 'figures/tradeoff.png'}", flush=True)


if __name__ == "__main__":
    main()
