#!/usr/bin/env bash
#SBATCH --job-name=qqq-news-robustness
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --partition=studentkillable
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%1
#SBATCH --output=slurm-news-robustness-%A_%a.out

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

SETUPS=(control uncertainty all_external control uncertainty all_external)
SEEDS=(43 43 43 44 44 44)
SETUP="${SETUPS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_ROOT="${NEWS_ABLATION_OUTPUT_ROOT:-models/news-robustness-${SLURM_ARRAY_JOB_ID}}"
mkdir -p "$OUTPUT_ROOT"

"$TASK_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible in the Slurm allocation")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
PY
"$TASK_PYTHON" -m pip freeze > "$OUTPUT_ROOT/environment-${SETUP}-seed${SEED}.txt"
"$TASK_PYTHON" scripts/check_data_readiness.py --allow-stale-news \
  --as-of 2026-09-04T23:59:59Z \
  --output "$OUTPUT_ROOT/data_readiness-${SETUP}-seed${SEED}.json"
"$TASK_PYTHON" -u scripts/run_news_ablation.py \
  --config-name "$SETUP" --seed "$SEED" --output-root "$OUTPUT_ROOT" --device-map cuda
