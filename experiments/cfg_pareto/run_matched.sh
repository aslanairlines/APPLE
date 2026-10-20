#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HOME=/home/coder/project/data/aladdin/cache/huggingface
export HF_HUB_OFFLINE=1
export EVAL_FFHQ_DIR=/home/coder/project/data/aladdin/data/FFHQ
export EVAL_WEIGHTS_DIR=/home/coder/project/data/aladdin/eval_weights
export OMP_NUM_THREADS=8
DATA=experiments/cfg_pareto/data/aladdin_full_1000
OUT=experiments/cfg_pareto/outputs/student_matched_full1000
EVAL_PY=/home/coder/project/.venv/bin/python
# Pass coefficient values as positional arguments, e.g. bash run_matched.sh 1 2 3.
if [ "$#" -eq 0 ]; then set -- 1 1.5 2 2.5 3 4 5; fi
.venv/bin/python experiments/cfg_pareto/prepare_matched_data.py \
  --aladdin-config /home/coder/project/outputs/pulid_scale_sweep/full_1000/config.json --destination "$DATA" --canonical-eval /home/coder/project/eval
.venv/bin/python -u experiments/cfg_pareto/run_sweep.py \
  --ffhq-base "$DATA" --checkpoint checkpoints/student --condition-type raw \
  --model-id /home/coder/project/data/aladdin/cache/huggingface/hub/models--black-forest-labs--FLUX.1-Krea-dev/snapshots/8162a9c7b05a641be098422bf2fcf335615c2f28 \
  --cfg-scales "$@" --fixed-weight 1 --max-pairs 0 --num-steps 28 --seed 42 --nproc 1 --output-root "$OUT"
RUNS=()
for scale in "$@"; do
  formatted=$(.venv/bin/python -c 'import sys; print(f"{float(sys.argv[1]):g}")' "$scale")
  RUNS+=("cfg_cfg${formatted}_w1_steps28_seed42")
done
"$EVAL_PY" -u experiments/cfg_pareto/evaluate.py --protocol canonical \
  --canonical-eval /home/coder/project/eval --ffhq-base "$DATA" \
  --output-root "$OUT" --scores "$OUT/scores.csv" --device cuda --run-names "${RUNS[@]}"
"$EVAL_PY" -u experiments/cfg_pareto/evaluate_fid.py \
  --canonical-eval /home/coder/project/eval --ffhq-base "$DATA" --output-root "$OUT" \
  --aladdin-config /home/coder/project/outputs/pulid_scale_sweep/full_1000/config.json \
  --real-features /home/coder/project/data/aladdin/data/eval/fid/_feats_clip/ffhq_images512_clip224_n70000.npy
"$EVAL_PY" experiments/cfg_pareto/plot_canonical.py --scores "$OUT/scores.csv" --out-dir "$OUT/figures"
