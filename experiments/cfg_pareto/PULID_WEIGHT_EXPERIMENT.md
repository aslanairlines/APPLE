# APPLE student: PuLID injection weight at fixed identity CFG

This experiment keeps identity CFG **w = 1.2** and varies **PuLID `id_weight`** over **0.65, 0.75, 0.90, 1.0, 1.10, 1.25**. It uses the same student, exact 1,000 FFHQ pairs, source gallery, masks, seed and evaluator implementations as the earlier matched CFG experiment. No teacher is loaded and no training is performed.

## What changes

PuLID's identity cross-attention contribution is added to transformer hidden states with a multiplier:

```text
hidden_states = hidden_states + id_weight * identity_attention_output
```

The inference code passes the selected `id_weight` to **both the identity-conditioned and identity-free CFG branches**. Identity CFG is then applied at the fixed value 1.2:

```text
v_guided = v_without_source_identity
           + 1.2 * (v_with_source_identity - v_without_source_identity)
```

Both predictions keep the target image condition. Identity-free PuLID features are obtained by passing zero features through the encoder; this branch need not have zero identity-attention output. Matching weights across the two branches keeps the experiment from inadvertently changing only one side of the CFG calculation. This is distinct from the previous experiment, which swept identity CFG while holding `id_weight = 1`.

| Parameter | Setting |
|---|---|
| Model | APPLE student, `checkpoints/student/default.safetensors` |
| Backbone / main precision | Local FLUX.1-Krea-dev / BF16 | and the VAE is also in BF16
| Identity CFG (`id_guidance_scale`, w) | **1.2, fixed** |
| PuLID `id_weight` | **0.65, 0.75, 0.90, 1.0, 1.10, 1.25** |
| Target condition | Sharp full target image (`raw`) |
| Image guidance / FLUX guidance | 1 / 1 |
| Sampling steps / seed | 28 / 42, with the same pair-specific seeds as before |
| Pair selection | All 1,000 pairs from the previous ALADDIN matched experiment |
| Inversion / gaze conditioning | Off |
| Expected generated images | 6 settings × 1,000 pairs = 6,000 images |
| Dedicated output root | `experiments/cfg_pareto/outputs/student_pulid_cfg1p2_full1000/` |

## Review without running the experiment

From APPLE root:

```bash
bash experiments/cfg_pareto/run_pulid_matched.sh --dry-run
```

This prints the full inference/evaluation/plotting commands. It performs no inference, evaluation, model loading, data preparation, or output writes. The coefficients are defined in `run_pulid_matched.py`; the existing CFG launcher's edited defaults are not used because this launcher explicitly passes every coefficient and selects `--stage pulid --fixed-cfg 1.2`.

For review, the expanded six distributed-inference commands are also saved in [PULID_WEIGHT_PREVIEW.txt](PULID_WEIGHT_PREVIEW.txt). `torch.distributed.run --nproc_per_node=1` starts one inference process on one GPU; it is not a separate coefficient.

## Run after reviewing

```bash
bash experiments/cfg_pareto/run_pulid_matched.sh
```

The implementation has been prepared and CPU-tested; the GPU experiment has **not** been launched as part of its implementation. The command above is the actual full run. A different grid can be supplied explicitly:

```bash
bash experiments/cfg_pareto/run_pulid_matched.sh --weight-scales 0.65 0.75 0.90 1.0 1.10 1.25
```

Run directory names encode both controls, for example `pulid_cfg1.2_w0.65_steps28_seed42`. Here `cfg1.2` is identity CFG and `_w0.65` is the PuLID injection weight. The CSV columns are clearer: `id_guidance_scale` and `id_weight`. The final metrics table retains `w = 1.2` and includes a separate `id_weight` column.

## Execution and metrics

The launcher sequentially:

1. Verifies matched source/target image and mask checksums, gallery inputs, and canonical evaluator fingerprints.
2. Runs student inference at the six specified weights, with a manifest and log for each setting. Incompatible resumes and incomplete image sets are rejected by the existing sweep runner.
3. Computes masked WebFace600K-R50 source identity cosine and fixed-gallery Top-1/Top-5 retrieval, whole-frame HopeNet target pose L2, and whole-frame Deep3D target expression L2.
4. Computes **CLIP-FID** with ALADDIN's exact implementation, local OpenAI CLIP ViT-B/32 quickgelu weights, and the same cached real FFHQ-70k reference. FID is a distribution-level metric, not a per-pair mean.
5. Saves the metrics table, individual plots and a **single PNG containing identity-vs-FID, identity-vs-pose and identity-vs-expression side by side**. Plot point labels are **PuLID id_weight**, and the combined figure title states that identity CFG is fixed at 1.2.

Inference uses APPLE's `.venv`; evaluation and plotting use the parent project's `.venv`. The launcher explicitly configures the local model cache and evaluation weights, with Hub offline mode enabled. It never downloads a teacher checkpoint. Both scoring stages select only the requested six run names.

Expected artifacts under the dedicated output root:

- Per-setting `images/`, `manifest.json`, and `run.log`.
- `scores.csv` and `scores.protocol.json` for per-pair canonical measurements.
- `fid.csv`, `fid.protocol.json`, and fingerprinted generated-feature caches.
- `figures/summary.csv` and `figures/metrics_table.csv`, including **both identity CFG and PuLID id_weight**.
- `figures/tradeoff_fid.png`, `tradeoff_pose.png`, and `tradeoff_exp.png`, plus SVG versions.
- `figures/combined_tradeoffs.png` and `.svg` for the three-panel comparison.

The original CFG experiment directories are protected from use as this launcher's output root. The original results and the previously combined CFG results remain separate.

## How to interpret the eventual results

This tests whether scaling PuLID injection at a selected fixed identity CFG offers a useful identity–attribute balance. Lower weights may weaken source identity transfer; higher weights may strengthen it while affecting target expression, pose or visual distribution. These are hypotheses, not measured results for this experiment yet; monotonic improvement is not guaranteed.

The `id_weight = 1` point is a baseline at the same identity CFG as the completed 1.2 experiment and can be checked against those saved results after completion. It is generated afresh in this experiment rather than silently substituted from earlier outputs.

Metrics and input pairs match the ALADDIN experiment, enabling a shared metric-space comparison. Equal injection weight values across different model architectures do not imply equal effective identity influence. Final selection should use the measured Pareto fronts and matched visual examples. Pose and expression do not measure all attributes, and the summary does not add bootstrap confidence intervals.

## CPU validation

```bash
/home/coder/project/.venv/bin/python experiments/cfg_pareto/test_pulid_matched.py -v
/home/coder/project/.venv/bin/python experiments/cfg_pareto/test_experiment.py -v
```

The new checks verify all six actual inference commands, fixed CFG, student selection, all-pair selection, matching evaluator cohorts, output protection, dry-run behavior, the weight column in the table, and the three-panel plot using temporary synthetic score fixtures. They do not load a generator or provide experiment results.
