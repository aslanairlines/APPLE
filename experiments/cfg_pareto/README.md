# APPLE identity CFG / PuLID weight inference sweep

For this machine, use [the experiment matched to ALADDIN](MATCHED_EXPERIMENT.md),
which verifies the prior 1,000-pair dataset and runs canonical masked identity,
HopeNet pose, and Deep3D expression scoring. The generic instructions below use
the historical proxy evaluator.

This is an **inference-only** experiment. It uses the official APPLE FFHQ inference script and fixed source/target pair filenames. No training is performed.

It sweeps **`id_guidance_scale`**, keeping FLUX `guidance_scale=1`, image guidance at 1, and PuLID weight fixed. This is an additional experiment, not a reproduction of a CFG sweep reported in the paper. The default is **student inference with the full, sharp target image** (`--condition-type raw`).

## Setup

Copy `experiments/cfg_pareto/` into the root of your APPLE fork, preserving the folder structure. Then, from APPLE root:

```bash
python experiments/cfg_pareto/patch_apple.py
python -m compileall -q experiments/cfg_pareto pulid_omini_inference_ffhq_args_multigpu.py
```

`patch_apple.py` changes `pulid_omini_inference_ffhq_args_multigpu.py` only. It checks exact source strings and fails rather than silently patching the wrong version. It writes an original backup; do not commit that backup. This patch is idempotent.

For the student, FFHQ must contain `src/<pairid>.jpg` and `trg/<pairid>.jpg` with matching filenames. No blurred conditions or landmark overlays are required. Use `checkpoints/student/default.safetensors` with the required FLUX/PuLID inference dependencies and model weights. A 134-byte Git LFS pointer is not a usable checkpoint; download the weights with `git lfs pull --include='checkpoints/student/default.safetensors'`.

For a **teacher** experiment, explicitly use the teacher checkpoint and `--condition-type blur_landmark_iris`; this additionally requires `trg/condition_blended_image_blurdownsample8_segGlass_landmark_iris/<pairid>.png` (or `.jpg`). Keep teacher and student experiments in separate output roots.

Run scripts from the Python environment containing PyTorch and APPLE's dependencies. The sweep launches distributed inference with that same Python interpreter, avoiding a different `torchrun` on `PATH`.

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

`--max-pairs 0` uses all pairs. When increasing it, existing images are skipped by the original APPLE script. Use the same seed, steps, checkpoint, dataset and conditioning across all settings. Inversion is *off* for this experiment. The manifest records the selected pair IDs and inference-script hash; changed settings or outputs without a manifest are rejected on resume. Old manifests without the script hash require a new output root.

## Evaluate and plot (quick proxy)

```bash
python experiments/cfg_pareto/evaluate.py --ffhq-base /PATH/TO/FFHQ
python experiments/cfg_pareto/plot_pareto.py
```

Evaluation checks that completed runs have the same dataset, seed, checkpoint, steps, conditioning, and generated pair set. Use `evaluate.py --run-names <run-directory-name> ...` to select one comparable cohort if the output root contains other experiments. Missing images in a completed run are an error.

Both output and target are resized to the inference resolution (512×512) **before landmark detection**, so 1024×1024 FFHQ originals do not introduce a coordinate-scale error. The plot averages only the **common successful pairs across all settings**, while reporting each setting's detection success rate separately. `n_valid` is the common paired sample count; `n_detected` is the setting's total successful count. Inspect both counts: common-pair filtering can still exclude difficult faces and bias the comparison.

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
- Repeated runs append to the corresponding `run.log`; images that exist are skipped. The deterministic noise seed for each pair is `(seed + CRC32(pair_id)) % 2**32`, shared across scales and ranks.
- Patch V2 also seeds Python, NumPy, and PyTorch before model loading and for every pair. This controls VAE condition sampling as well as the explicit noise generator. Rerun the patch script to upgrade V1; use a fresh output root for new results. CUDA kernels can still introduce numerical nondeterminism.
- Outputs go under `experiments/cfg_pareto/outputs/` and should be **ignored in Git**.
- Do not treat this evaluator as the old ALADDIN evaluator. For rigorous Pareto comparisons use the exact same metric implementation, dataset/pair split, and detection-failure treatment for both models.

## Overnight orchestration (updated package)

`run_overnight.py` sequentially runs **CFG inference → proxy evaluation → Pareto plot**.
It deliberately does **not** launch a PuLID sweep: select CFG from results first.
It aborts on errors, never silently marks a partial/contaminated pair set as complete.

After successful GPU smoke testing:

```bash
python -u experiments/cfg_pareto/run_overnight.py \
    --ffhq-base /PATH/TO/FFHQ \
    --checkpoint checkpoints/student \
    --cfg-scales 1 1.5 2 2.5 3 4 5 \
    --max-pairs 1000 --nproc 1
```

The overnight command evaluates only its requested CFG settings. It saves `scores.csv`, `figures/summary.csv`, and `figures/tradeoff.png` inside `--output-root` (by default, `experiments/cfg_pareto/outputs/`).

**Important:** `evaluate.py` uses independent recognition weights and a **landmark-geometry proxy**, NOT the full ALADDIN evaluation protocol. Run in an environment with `insightface`, `face_alignment`, `pandas`, `matplotlib`, `onnxruntime-gpu`, etc. The run cannot be guaranteed before at least one inference and evaluation smoke test on the GPU. The script never estimates completion time.

## CPU regression checks

With NumPy, pandas, Matplotlib, and Pillow installed:

```bash
python experiments/cfg_pareto/test_experiment.py -v
```

These checks exercise student command construction, checkpoint/pair validation, patching, completed-output checks, common-resolution landmarks, and fair paired aggregation. They do not execute FLUX or replace the GPU smoke test.
