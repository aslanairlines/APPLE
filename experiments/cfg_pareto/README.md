# APPLE identity CFG / PuLID weight inference sweep

This is an **inference-only** experiment. It uses the official APPLE FFHQ inference script and fixed source/target pair filenames. No training is performed.

## Setup

Copy `experiments/cfg_pareto/` into the root of your APPLE fork, preserving the folder structure. Then, from APPLE root:

```bash
python experiments/cfg_pareto/patch_apple.py
python -m compileall -q experiments/cfg_pareto pulid_omini_inference_ffhq_args_multigpu.py
```

`patch_apple.py` changes `pulid_omini_inference_ffhq_args_multigpu.py` only. It checks exact source strings and fails rather than silently patching the wrong version. It writes an original backup; do not commit that backup. This patch is idempotent.

Verify the FFHQ dataset includes `src/*.jpg`, `trg/*.jpg`, **and** `trg/condition_blended_image_blurdownsample8_segGlass_landmark_iris/<pairid>.png` (or `.jpg`). Use the matching checkpoint in the APPLE repo (usually `checkpoints/student/default.safetensors`), plus the required FLUX/PuLID and preprocessing dependencies; merely having a dataset of JPEGs is insufficient.

## Run dry-run on your local laptop (no checkpoints needed)

```bash
python experiments/cfg_pareto/run_sweep.py --dry-run \
    --ffhq-base /PATH/TO/FFHQ --checkpoint checkpoints/student --max-pairs 20
```

## Smoke test on GPU

```bash
python experiments/cfg_pareto/run_sweep.py \
    --ffhq-base /PATH/TO/FFHQ --checkpoint checkpoints/student \
    --stage cfg --cfg-scales 1 2 --max-pairs 10 --nproc 1
```

## CFG sweep

```bash
python experiments/cfg_pareto/run_sweep.py \
    --ffhq-base /PATH/TO/FFHQ --checkpoint checkpoints/student \
    --stage cfg --cfg-scales 1 1.5 2 2.5 3 4 5 \
    --fixed-weight 1 --max-pairs 200 --nproc 1
```

`--max-pairs 0` uses all pairs. When increasing it, existing images are skipped by the original APPLE script. Use the same seed, steps, checkpoint, dataset and preprocessing across all settings. Inversion is *off* for this experiment.

## Evaluate and plot (quick proxy)

```bash
python experiments/cfg_pareto/evaluate.py --ffhq-base /PATH/TO/FFHQ
python experiments/cfg_pareto/plot_pareto.py
```

The evaluator requires `insightface`, `onnxruntime-gpu` (if CUDA), `face_alignment`, `opencv-python`, `numpy`; the plotter requires `pandas` and `matplotlib`. The first evaluation run may download external models (`buffalo_l` and 2D landmark backbone). `buffalo_l` is a different pretrained recognition model/checkpoint from PuLID's `antelopev2`, but shares underlying ArcFace ideas; it is *not fully architecture-independent*. The landmark metric is just a **pose/expression geometry proxy** and is affected by identity-specific shape. It is **not** a complete target-preservation score. For comparisons with ALADDIN, replace `evaluate.py` with your laboratory's evaluator using the same metrics and preprocessing on both models; keep the `scores.csv` schema so `plot_pareto.py` can consume it, if that schema corresponds to the same metric interpretation.

## PuLID sweep once a CFG point is selected

```bash
python experiments/cfg_pareto/run_sweep.py \
    --ffhq-base /PATH/TO/FFHQ --checkpoint checkpoints/student \
    --stage pulid --fixed-cfg 2.0 \
    --weight-scales 0.5 0.75 1 1.25 1.5 2 --max-pairs 200
```

## Notes

- Baseline `id_guidance_scale=1` disables the extra unconditional forward. For scales >1 the sampling loop runs extra model evaluations, so measure runtime too.
- `image_guidance_scale` is fixed at 1.0, as identity CFG and image CFG are mutually exclusive in APPLE's `generate_ca` implementation.
- PuLID `id_weight` is matched between conditional and unconditional branches when sweeping CFG/weight.
- Repeated runs will overwrite the corresponding `run.log` by appending; images that exist are skipped. The deterministic noise seed for each pair is `seed + CRC32(pair_id)`, shared across scales and ranks.
- Outputs go under `experiments/cfg_pareto/outputs/` and should be **ignored in Git**.
- Do not treat this evaluator as the old ALADDIN evaluator. For rigorous Pareto comparisons use the exact same metric implementation, dataset/pair split, and detection-failure treatment for both models.

## Overnight orchestration (updated package)

`run_overnight.py` sequentially runs **CFG inference → proxy evaluation → Pareto plot**.
It deliberately does **not** launch a PuLID sweep: select CFG from results first.
It aborts on errors, never silently marks a partial/contaminated pair set as complete.

After successful GPU smoke testing:

```bash
python -u experiments/cfg_pareto/run_overnight.py \
    --ffhq-base /PATH/TO/PREPROCESSED/FFHQ \
    --checkpoint checkpoints/student \
    --cfg-scales 1 1.5 2 2.5 3 4 5 \
    --max-pairs 1000 --nproc 1
```

**Important:** `evaluate.py` uses independent recognition weights and a **landmark-geometry proxy**, NOT the full ALADDIN evaluation protocol. Run in an environment with `insightface`, `face_alignment`, `pandas`, `matplotlib`, `onnxruntime-gpu`, etc. The run cannot be guaranteed before at least one inference and evaluation smoke test on the GPU. The script never estimates completion time.

The new ZIP archive contains only `experiments/cfg_pareto/` at its top level: extract directly into the APPLE repository root; all other APPLE files remain untouched until `patch_apple.py` is explicitly run.
