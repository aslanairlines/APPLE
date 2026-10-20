# Student experiment matched to the existing ALADDIN sweep

Dataset: `/home/coder/project/data/aladdin/data/FFHQ/images512`, masks in the adjacent `seg12_mask` directory. The prior experiment is `/home/coder/project/outputs/pulid_scale_sweep/full_1000/config.json`. It records 1,000 benchmark pairs (69000+i → 68000+i), in a shuffled order, and a fixed 1,000-source retrieval gallery. “Entire dataset” here means all 1,000 inference pairs used by that experiment, not all possible pairings of FFHQ's 70,000 images.

`prepare_matched_data.py` verifies image/mask SHA-256 hashes, dimensions and gallery hashes against that experiment, then creates `data/aladdin_full_1000/{src,trg}` symlinks with APPLE pair filenames. The `.jpg` names point to the original PNG bytes: Pillow reads them without JPEG conversion. `pairs.json` retains original filenames, masks, identities and gallery indices.

The student uses `checkpoints/student/default.safetensors` and the sharp full target (`condition-type raw`), with no inversion or gaze conditioning. FLUX is loaded from the existing local Krea snapshot. The pipeline is constructed from its already loaded components and cached scheduler, avoiding the absent `model_index.json` in this component-only snapshot.

Identity CFG is `unconditional + scale * (conditional - unconditional)` in `omini/pipeline/flux_omini.py`. At 1 the additional unconditional pass is disabled. The current runner accepts finite CFG >=1. Identity CFG and image CFG cannot both be active; image CFG stays 1. The selected grid is 1, 1.5, 2, 2.5, 3, 4, 5; PuLID weight stays 1, FLUX guidance stays 1, steps=28, seed=42. This is an exploratory inference sweep, not a claimed paper reproduction. ALADDIN's alpha and APPLE's identity CFG are different controls.

Run from APPLE root:

```bash
bash experiments/cfg_pareto/run_matched.sh
# Optional custom identity CFG grid:
bash experiments/cfg_pareto/run_matched.sh 1 2 3
```

The script uses APPLE's `.venv` for inference and `/home/coder/project/.venv` for canonical evaluation/plotting. It sets the shared model cache, offline mode and evaluator weights explicitly. Each run writes `manifest.json` and `run.log`; incompatible resumes are rejected. Existing generated images are skipped.

Evaluation calls `evaluate.py --protocol canonical --canonical-eval /home/coder/project/eval`. It directly reuses ALADDIN's `score_id._embed_masked`, fixed-gallery cosine/Top-1/5 retrieval, and `score_pose_expr.pose_deg` / `exp_coeff`. Source identity uses source masks; generated identity uses target masks. Pose is whole-frame HopeNet yaw/pitch/roll L2 in degrees. Expression is whole-frame Deep3D coefficient [80:144] L2. There is no detection-based filtering in these canonical metrics. Retrieval values in `scores.csv` and `figures/summary.csv` are fractions, while ALADDIN summary retrieval values may be percentages. Multiply by 100 for that display convention.

The original proxy evaluator remains available via `--protocol proxy` (the compatibility default). Its buffalo_l detected-face cosine and normalized landmark distance are different metrics and must not be compared numerically with canonical results. The matched launcher always explicitly selects canonical scoring. Do not use `run_overnight.py` for this matched experiment: it still runs the proxy protocol.

Outputs: `experiments/cfg_pareto/outputs/student_matched_full1000/`:

- Per-pair `scores.csv` and `scores.protocol.json` with metric-code hashes and recognizer checksum.
- `figures/summary.csv` with common-pair means, generated/success counts and two Pareto flags.
- `fid.csv`, `fid.protocol.json`, and fingerprinted generated-feature caches in `fid_cache/`.
- `figures/tradeoff_pose.{png,svg}`, `figures/tradeoff_exp.{png,svg}`, and `figures/tradeoff_fid.{png,svg}`.
- `figures/metrics_table.csv` includes CLIP-FID after plotting.

Only successful pairs common to all selected settings enter the plotted means. Failures remain explicit in the CSV. These plots cover canonical identity/retrieval, pose and expression. CLIP-FID is now included, using ALADDIN's exact `fid_clip.py` implementation, local OpenAI CLIP ViT-B/32 weights, and cached FFHQ-70k real features. Background L1 and optional reconstruction LPIPS are not calculated by this launcher. Confidence intervals are not currently plotted. Generator randomness is shared across APPLE coefficients; it does not reproduce ALADDIN's latent draws across architectures.

## Validation on this machine

- Student safetensors: 463,703,368 bytes, 686 tensors; SHA-256 `7329225d329bb1b403f07be00a7bd1739d0e305ab719621cc2c1ecda950d8565`, matching the repository's LFS record.
- Both CFG 1 and CFG 2 passed 28-step GPU inference, canonical evaluation and plotting on the first matched pair. Evidence is under `outputs/smoke_verified/`. This is a smoke test, not a dataset-level result.
- Rescoring the first saved ALADDIN output gave identical pose, expression, Top-1 and Top-5 values. Identity cosine differed by `8.67992639541626e-07` (different embedding batch sizes can introduce numerical differences). Evidence is `outputs/parity_check/comparison.json`; all five values matched within `1e-4`.
- All 12 existing CPU regression checks passed using the parent project's Python environment. APPLE's inference environment does not contain pandas; the launcher explicitly uses the parent environment for evaluation and plotting.
- The full launcher also verifies the two canonical metric source files and three evaluator weight files against the fingerprints in the prior ALADDIN config.
- The full 7 × 1,000 sweep has completed. All 7,000 samples passed canonical evaluation. CLIP-FID was subsequently added using the saved generated images; generation was not repeated. The cached real-feature mean and covariance matched ALADDIN's saved `fid_real_statistics.npz` exactly.
