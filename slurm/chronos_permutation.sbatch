#!/usr/bin/env bash
#SBATCH --job-name=qqq-permutation
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=studentkillable
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:1
#SBATCH --array=0-11%6
#SBATCH --output=slurm-perm-%A_%a.out

# Permutation importance on fine-tuned walk-forward checkpoints and on the pretrained model
# with the same features and windows. Cells are <rung>-fold<year>-seed<S> directories under
# $WALK_FORWARD_ROOT (default models/walk-forward-896510). Task index i selects
# CELLS[i // 2]; even i = fine-tuned checkpoint, odd i = pretrained baseline.
#
#   WALK_FORWARD_ROOT=models/walk-forward-896510 sbatch slurm/chronos_permutation.sbatch

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch as an array job}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

TASK_PYTHON="${CHRONOS_PYTHON:-}"
if [[ -z "$TASK_PYTHON" ]]; then
  if [[ -x .venv/bin/python && -x .venv/bin/pip ]]; then TASK_PYTHON=.venv/bin/python; else TASK_PYTHON=python3; fi
fi
if [[ -d .pythonlibs ]]; then export PYTHONPATH="$PWD/.pythonlibs${PYTHONPATH:+:$PYTHONPATH}"; fi
if [[ -d .torchlibs ]]; then export PYTHONPATH="$PWD/.torchlibs${PYTHONPATH:+:$PYTHONPATH}"; fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/models/huggingface}"

WALK_FORWARD_ROOT="${WALK_FORWARD_ROOT:-models/walk-forward-896510}"
OUTPUT_ROOT="${PERMUTATION_OUTPUT_ROOT:-models/permutation-${SLURM_ARRAY_JOB_ID}}"
MODEL_REVISION="${MODEL_REVISION:-29ec3766d36d6f73f0696f85560a422f50e8498c}"
REPEATS="${PERMUTATION_REPEATS:-3}"
CELLS=(
  r3_qqq_calendar_market-fold2025-seed42
  r5_plus_fed-fold2025-seed42
  r7_all_external-fold2025-seed42-bw4
  r3_qqq_calendar_market-fold2022-seed42
  r5_plus_fed-fold2022-seed42
  r7_all_external-fold2022-seed42-bw4
)
CELL="${CELLS[$((SLURM_ARRAY_TASK_ID / 2))]}"
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then ARM=finetuned; else ARM=pretrained; fi
CELL_DIR="$WALK_FORWARD_ROOT/$CELL"
[[ -f "$CELL_DIR/comparison_metadata.json" ]] || { echo "Missing $CELL_DIR/comparison_metadata.json"; exit 1; }
mkdir -p "$OUTPUT_ROOT"

# Pull the exact feature list and validation size from the cell's metadata; sample bounds from the fold config.
read -r FEATURES VALIDATION_ROWS START_DATE END_DATE < <("$TASK_PYTHON" - "$CELL_DIR" "$CELL" <<'PY'
import json, re, sys
from pathlib import Path
cell_dir, cell = Path(sys.argv[1]), sys.argv[2]
meta = json.loads((cell_dir / "comparison_metadata.json").read_text())
plan = json.loads(Path("configs/walk_forward.json").read_text())
fold = re.search(r"-fold([^-]+)-", cell).group(1)
features = ",".join(meta["feature_columns"]) or "none"
print(features, meta["validation_rows"], plan["sample_start"], plan["folds"][fold]["validation_end"])
PY
)

"$TASK_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible in the Slurm allocation")
print("GPU:", torch.cuda.get_device_name(0))
PY

ARGS=(scripts/permutation_importance.py
  --data data/processed/daily_feature_table.csv
  --feature-columns "$FEATURES"
  --model-id amazon/chronos-2 --model-revision "$MODEL_REVISION"
  --start-date "$START_DATE" --end-date "$END_DATE" --validation-rows "$VALIDATION_ROWS"
  --stride 5 --prediction-length 5 --context-length 512
  --repeats "$REPEATS" --seed 42 --device-map cuda
  --output-dir "$OUTPUT_ROOT/${CELL}-${ARM}")
if [[ "$ARM" == finetuned ]]; then
  CKPT="$CELL_DIR/finetuned/finetuned-ckpt"
  [[ -d "$CKPT" ]] || CKPT="$CELL_DIR/finetuned"
  ARGS+=(--checkpoint "$CKPT")
fi
echo "cell=$CELL arm=$ARM features=$(tr ',' '\n' <<<"$FEATURES" | grep -c .) validation_rows=$VALIDATION_ROWS sample=$START_DATE..$END_DATE"
"$TASK_PYTHON" -u "${ARGS[@]}"
