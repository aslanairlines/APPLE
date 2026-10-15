"""Inference-only CFG or PuLID scale sweep using official APPLE FFHQ code."""
import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "pulid_omini_inference_ffhq_args_multigpu.py"
CONDITION_DIRS = {
    "blur_landmark": "condition_blended_image_blurdownsample8_segGlass_landmark",
    "blur_landmark_iris": "condition_blended_image_blurdownsample8_segGlass_landmark_iris",
    "mask_landmark_iris": "condition_blended_image_mask_segGlass_landmark_iris",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["cfg", "pulid"], default="cfg")
    parser.add_argument("--ffhq-base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments/cfg_pareto/outputs")
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-Krea-dev")
    parser.add_argument("--condition-type", choices=CONDITION_DIRS, default="blur_landmark_iris")
    parser.add_argument("--cfg-scales", type=float, nargs="+", default=[1, 1.5, 2, 2.5, 3, 4, 5])
    parser.add_argument("--weight-scales", type=float, nargs="+", default=[0.5, 0.75, 1, 1.25, 1.5, 2])
    parser.add_argument("--fixed-cfg", type=float, default=1.0)
    parser.add_argument("--fixed-weight", type=float, default=1.0)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all; otherwise first N sorted pair IDs")
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nproc", type=int, default=1, help="Number of GPUs available to torchrun")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not SCRIPT.is_file():
        raise SystemExit(f"Missing script: {SCRIPT}")
    if not args.dry_run:
        script_text = SCRIPT.read_text(encoding="utf-8")
        if "CFG_PARETO_PATCH_V1" not in script_text:
            raise SystemExit("Run `python experiments/cfg_pareto/patch_apple.py` first")
        if not args.checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
        ckpt_file = args.checkpoint / "default.safetensors" if args.checkpoint.is_dir() else args.checkpoint
        if not ckpt_file.is_file():
            raise SystemExit(f"Cannot find checkpoint weights: {ckpt_file}")
        with ckpt_file.open("rb") as fh:
            if fh.read(48).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise SystemExit(f"Checkpoint is a Git LFS pointer, not weights: {ckpt_file}. Run git lfs pull")
        if not args.ffhq_base.is_dir():
            raise SystemExit(f"Dataset not found: {args.ffhq_base}")
        if not (args.ffhq_base / "src").is_dir() or not (args.ffhq_base / "trg").is_dir():
            raise SystemExit("Dataset must have src/ and trg/ folders")
        condition_path = args.ffhq_base / "trg" / CONDITION_DIRS[args.condition_type]
        if not condition_path.is_dir():
            raise SystemExit(f"Preprocessed target conditions missing: {condition_path}")
        source_images = sorted((args.ffhq_base / "src").glob("*.jpg"))
        if args.max_pairs > 0 and len(source_images) < args.max_pairs:
            raise SystemExit(f"Requested {args.max_pairs} pairs but only {len(source_images)} source JPGs exist")
        if args.max_pairs > 0:
            source_images = source_images[:args.max_pairs]
        if not source_images:
            raise SystemExit("No source JPG images found. APPLE expects src/*.jpg")
        expected_ids = {p.stem for p in source_images}
        missing = []
        for src_path in source_images:
            pair_id = src_path.stem
            if not (args.ffhq_base / "trg" / f"{pair_id}.jpg").is_file():
                missing.append(f"missing target: {pair_id}.jpg")
            if not any((condition_path / f"{pair_id}{ext}").is_file() for ext in (".png", ".jpg")):
                missing.append(f"missing condition: {pair_id}.png/.jpg")
        if missing:
            raise SystemExit(f"Invalid FFHQ pairs (showing 10/{len(missing)}): " + "; ".join(missing[:10]))
        print(f"Preflight OK: {len(expected_ids)} paired images with target conditions", flush=True)
    if args.num_steps <= 0 or args.nproc <= 0 or args.max_pairs < 0:
        raise SystemExit("num-steps and nproc must be positive; max-pairs must be nonnegative")
    if args.stage == "cfg":
        grid = [(scale, args.fixed_weight) for scale in args.cfg_scales]
    else:
        grid = [(args.fixed_cfg, weight) for weight in args.weight_scales]

    for cfg, weight in grid:
        if cfg < 1 or weight < 0:
            raise SystemExit("Use id_guidance_scale >= 1 and id_weight >= 0")
        run_name = f"{args.stage}_cfg{cfg:g}_w{weight:g}_steps{args.num_steps}_seed{args.seed}"
        run_dir = args.output_root.resolve() / run_name
        images_dir = run_dir / "images"
        command = [
            "torchrun", "--standalone", f"--nproc_per_node={args.nproc}", str(SCRIPT),
            "--base_path", str(ROOT),
            "--ffhq_base_path", str(args.ffhq_base.resolve()),
            "--checkpoint_path", str(args.checkpoint.resolve()),
            "--model_id", args.model_id,
            "--guidance_scale", "1.0",
            "--image_guidance_scale", "1.0",
            "--id_guidance_scale", str(cfg),
            "--id_weight", str(weight),
            "--condition_type", args.condition_type,
            "--num_inference_steps", str(args.num_steps),
            "--max_pairs", str(args.max_pairs),
            "--seed", str(args.seed),
            "--output_dir", str(images_dir),
        ]
        print("\n" + shlex.join(command), flush=True)
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "stage": args.stage,
            "id_guidance_scale": cfg,
            "id_weight": weight,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "max_pairs": args.max_pairs,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_size": (args.checkpoint / "default.safetensors" if args.checkpoint.is_dir() else args.checkpoint).stat().st_size,
            "checkpoint_mtime_ns": (args.checkpoint / "default.safetensors" if args.checkpoint.is_dir() else args.checkpoint).stat().st_mtime_ns,
            "model_id": args.model_id,
            "ffhq_base": str(args.ffhq_base.resolve()),
            "condition_type": args.condition_type,
            "command": command,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        manifest = run_dir / "manifest.json"
        if manifest.is_file():
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            comparable = ["stage", "id_guidance_scale", "id_weight", "num_steps", "seed",
                          "checkpoint", "checkpoint_size", "checkpoint_mtime_ns", "model_id",
                          "ffhq_base", "condition_type"]
            mismatched = [key for key in comparable if previous.get(key) != metadata.get(key)]
            if mismatched:
                raise SystemExit(f"Unsafe resume: existing {manifest} has different settings: {mismatched}")
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with (run_dir / "run.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        metadata["status"] = "complete" if result.returncode == 0 else "failed"
        metadata["returncode"] = result.returncode
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if result.returncode != 0:
            raise SystemExit(f"FAILED {run_name}. Inspect {run_dir / 'run.log'}")
        produced_ids = {p.stem for p in images_dir.glob("*.png")}
        absent = sorted(expected_ids - produced_ids)
        extra = sorted(produced_ids - expected_ids)
        if absent or extra:
            metadata["status"] = "failed_incomplete"
            metadata["missing_count"] = len(absent)
            metadata["extra_count"] = len(extra)
            manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            raise SystemExit(
                f"Incomplete/contaminated outputs for {run_name}: "
                f"missing={len(absent)} {absent[:5]}, extra={len(extra)} {extra[:5]}. "
                "Do not evaluate mixed pair sets."
            )
        print(f"Finished {run_name}: {len(produced_ids)} verified outputs", flush=True)


if __name__ == "__main__":
    main()
