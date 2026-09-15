#!/usr/bin/env bash
#SBATCH --job-name=qqq-walk-forward
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=studentkillable
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:1
#SBATCH --array=0-95%4
#SBATCH --output=slurm-wf-%A_%a.out

# Walk-forward covariate ladder: 8 rungs x 4 folds x 3 seeds = 96 LoRA fits (configs/walk_forward.json).
# Task index -> (rung, fold, seed) is enumerated by scripts/run_walk_forward.py --list-tasks.

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

OUTPUT_ROOT="${WALK_FORWARD_OUTPUT_ROOT:-models/walk-forward-${SLURM_ARRAY_JOB_ID}}"
AS_OF="${WALK_FORWARD_AS_OF:-2026-09-04T23:59:59Z}"
mkdir -p "$OUTPUT_ROOT"

"$TASK_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible in the Slurm allocation")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
PY
"$TASK_PYTHON" -m pip freeze > "$OUTPUT_ROOT/environment-task${SLURM_ARRAY_TASK_ID}.txt"
# Validate the frozen feature snapshot at its build cutoff, not the wall-clock date.
"$TASK_PYTHON" scripts/check_data_readiness.py --allow-stale-news \
  --as-of "$AS_OF" \
  --output "$OUTPUT_ROOT/data_readiness-task${SLURM_ARRAY_TASK_ID}.json"
"$TASK_PYTHON" -u scripts/run_walk_forward.py \
  --task-index "$SLURM_ARRAY_TASK_ID" --output-root "$OUTPUT_ROOT" --device-map cuda ${WALK_FORWARD_EXTRA_ARGS:-}
