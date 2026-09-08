#!/usr/bin/env bash
#SBATCH --job-name=qqq-chronos-validation
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=studentkillable
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
TASK_PYTHON="${CHRONOS_PYTHON:-}"
if [[ -z "$TASK_PYTHON" ]]; then
  if [[ -x .venv/bin/python && -x .venv/bin/pip ]]; then TASK_PYTHON=.venv/bin/python; else TASK_PYTHON=python3; fi
fi
if [[ -d .pythonlibs ]]; then export PYTHONPATH="$PWD/.pythonlibs${PYTHONPATH:+:$PYTHONPATH}"; fi
if [[ -d .torchlibs ]]; then export PYTHONPATH="$PWD/.torchlibs${PYTHONPATH:+:$PYTHONPATH}"; fi
TASK_OUTPUT="models/slurm-validation-${SLURM_JOB_ID}"
mkdir -p "$TASK_OUTPUT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/models/huggingface}"

"$TASK_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('No CUDA GPU is visible in the Slurm allocation')
print('GPU:', torch.cuda.get_device_name(0))
PY
"$TASK_PYTHON" -m pip freeze > "$TASK_OUTPUT/environment.txt"
"$TASK_PYTHON" scripts/check_data_readiness.py --output "$TASK_OUTPUT/data_readiness.json"
"$TASK_PYTHON" -u scripts/compare_chronos2_validation.py \
  --feature-profile target --base-only --device-map cuda --model-revision 29ec3766d36d6f73f0696f85560a422f50e8498c --seed 42 \
  --output-dir "$TASK_OUTPUT/target-only"
"$TASK_PYTHON" -u scripts/compare_chronos2_validation.py \
  --feature-profile core --device-map cuda --model-revision 29ec3766d36d6f73f0696f85560a422f50e8498c --seed 42 --output-dir "$TASK_OUTPUT/core-lora"
"$TASK_PYTHON" scripts/summarize_chronos2_comparison.py --comparison-dir "$TASK_OUTPUT/core-lora"
