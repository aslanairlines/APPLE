"""Idempotently expose existing PuLID controls in APPLE's FFHQ inference CLI.
Run from any directory: python experiments/cfg_pareto/patch_apple.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "pulid_omini_inference_ffhq_args_multigpu.py"
MARKER = "# CFG_PARETO_PATCH_V2"
LEGACY_MARKER = "# CFG_PARETO_PATCH_V1"


def replace_once(content: str, old: str, new: str) -> str:
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly 1 match, found {count}: {old[:100]!r}")
    return content.replace(old, new, 1)


def patch_controls(text: str) -> str:
    text = replace_once(text, "import os\n", f"{MARKER}\nimport os\nimport zlib\n")
    text = replace_once(
        text,
        "    output_dir = f'{base_path}/results/{run_name}_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval/base'",
        "    output_dir = args.output_dir or f'{base_path}/results/{run_name}_ckpt{ckpt}_gs{guidance_scale}_imgGS{image_guidance_scale}_idGS{id_guidance_scale}/ffhq_eval/base'",
    )
    text = replace_once(
        text,
        "    src_img_path_list = src_img_path_list[rank::world_size]",
        "    if args.max_pairs > 0:\n"
        "        src_img_path_list = src_img_path_list[:args.max_pairs]\n"
        "    src_img_path_list = src_img_path_list[rank::world_size]",
    )
    text = replace_once(
        text,
        "        src_num = os.path.basename(src_img_path).split('.')[0]",
        "        src_num = os.path.basename(src_img_path).split('.')[0]\n"
        "        pair_seed = (args.seed + zlib.crc32(src_num.encode('utf-8'))) % (2**32)",
    )
    text = replace_once(
        text,
        "                generator=torch.Generator('cpu').manual_seed(0),",
        "                generator=torch.Generator('cpu').manual_seed(pair_seed),",
    )
    text = replace_once(
        text,
        "                height=512,\n                width=512,",
        "                height=512,\n                width=512,\n                num_inference_steps=args.num_inference_steps,",
    )
    text = replace_once(
        text,
        "                id_guidance_scale=id_guidance_scale,",
        "                id_guidance_scale=id_guidance_scale,\n"
        "                id_weight=args.id_weight,\n"
        "                uncond_id_weight=args.id_weight,",
    )
    text = replace_once(
        text,
        '    parser.add_argument("--id_guidance_scale", type=float, default=1.0)',
        '    parser.add_argument("--id_guidance_scale", type=float, default=1.0)\n'
        '    parser.add_argument("--id_weight", type=float, default=1.0)\n'
        '    parser.add_argument("--num_inference_steps", type=int, default=28)\n'
        '    parser.add_argument("--max_pairs", type=int, default=0, help="0 = all pairs")\n'
        '    parser.add_argument("--seed", type=int, default=42)\n'
        '    parser.add_argument("--output_dir", type=str, default=None)',
    )
    return text


def main() -> None:
    if not SCRIPT.is_file():
        raise FileNotFoundError(f"Cannot find APPLE inference script: {SCRIPT}")
    text = SCRIPT.read_text(encoding="utf-8")
    if MARKER in text:
        print("Patch already installed:", SCRIPT)
        return
    if LEGACY_MARKER in text:
        text = replace_once(text, LEGACY_MARKER, MARKER)
    else:
        text = patch_controls(text)
    text = replace_once(text, "import zlib\n", "import zlib\nimport random\nimport numpy as cfg_numpy\n")
    text = replace_once(
        text, "    torch.cuda.set_device(rank)\n",
        "    torch.cuda.set_device(rank)\n"
        "    random.seed(args.seed)\n"
        "    cfg_numpy.random.seed(args.seed % (2**32))\n"
        "    torch.manual_seed(args.seed)\n",
    )
    seed_line = "        pair_seed = (args.seed + zlib.crc32(src_num.encode('utf-8'))) % (2**32)"
    text = replace_once(
        text, seed_line,
        seed_line + "\n"
        "        # Also control VAE condition sampling, which uses the global RNG.\n"
        "        random.seed(pair_seed)\n"
        "        cfg_numpy.random.seed(pair_seed)\n"
        "        torch.manual_seed(pair_seed)",
    )
    compile(text, str(SCRIPT), "exec")

    backup = ROOT / ".git" / "apple_inference_before_cfg_pareto.py"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    SCRIPT.write_text(text, encoding="utf-8")
    print(f"Patched: {SCRIPT}\nBackup: {backup}")


if __name__ == "__main__":
    main()
