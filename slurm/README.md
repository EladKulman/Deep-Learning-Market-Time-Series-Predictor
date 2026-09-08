# First Slurm experiments

The TAU Slurm cluster was exercised on 2026-09-08. The working allocation is account
`gpu-students`, partition `studentkillable`, feature `geforce_rtx_2080`, with the project
staged on shared NFS under
`/specific/scratches/scratch/eladkulman/qqq-chronos-smoke-codex-20260908`. Login-node home
directories are local to each load-balanced login host and are not visible from compute
nodes, so jobs must run from shared scratch.

## Stage the repository and environment

Transfer this working tree, including `scripts`, `tests`, `data`, `requirements.txt`, and
`slurm`. Do not transfer the macOS `.venv`, `.env`, `oath/`, or old model adapters. The existing
checkpoints predate the repaired feature schema.

On the cluster, use its supported Python (3.11 or 3.12) and CUDA/PyTorch setup. From the
repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/download_chronos.py
.venv/bin/python scripts/compare_chronos2_validation.py --smoke-test --prepare-only
```

If `python3 -m venv` reports that `ensurepip` is unavailable and you cannot install
`python3-venv`, install into a project-local directory instead:

```bash
python3 -m pip install --target .pythonlibs -r requirements.txt
PYTHONPATH="$PWD/.pythonlibs" python3 -m unittest discover -s tests -v
```

The batch files detect `.pythonlibs`, add it to `PYTHONPATH`, and fall back to `python3`.
They select `.venv/bin/python` only when both Python and pip exist, so a partial failed
virtual environment does not mask the system interpreter.

The cluster's default PyPI resolution originally selected PyTorch 2.14 with CUDA 13, which
the installed NVIDIA driver rejected. `requirements.txt` now excludes that untested range.
The successful run used the official CUDA 12.8 wheel in an isolated overlay:

```bash
python3 -m pip install --target .torchlibs \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
```

When present, `.torchlibs` is prepended to `.pythonlibs`. The verified runtime was Python
3.12.3, PyTorch 2.11.0+cu128, CUDA 12.8, Chronos Forecasting 2.3.2, Transformers 5.16.1,
PEFT 0.20.0, and Accelerate 1.14.0 on an NVIDIA GeForce RTX 2080 Ti.

`download_chronos.py` stages the tested model revision
`29ec3766d36d6f73f0696f85560a422f50e8498c` under `models/huggingface` while network access is
available. Compute jobs run offline and receive that exact revision explicitly. If login
nodes cannot download, stage the same Hugging Face cache on an allowed transfer node.
If the cluster supplies an existing environment, set `CHRONOS_PYTHON` to its Python path.
Do not run training on login nodes.

Local integration testing used Python 3.12.10, chronos-forecasting 2.3.1, torch 2.12.0,
transformers 5.13.0, peft 0.19.1, pandas 2.3.0 and pandas_market_calendars 5.4.0.
The cluster's GPU driver must support its selected PyTorch CUDA wheel; local CPU success
does not verify that compatibility. Each job saves its actual package versions. A local offline reload was tested with only the pinned base snapshot present and no `main` reference; the adapter config now pins that base revision too.

## Submit the infrastructure smoke test

From the shared repository root:

```bash
sbatch --account=gpu-students slurm/chronos_smoke.sbatch
```

The batch file supplies the TAU partition and RTX 2080 constraint. It requests one GPU,
four CPUs, 16 GB host memory, and 30 minutes. It compares pretrained and five-step LoRA models
on five three-session windows using six features, checks checkpoint reload and holdout
prediction, and writes `models/slurm-smoke-JOBID/`. These results verify infrastructure;
they are too small for research conclusions. The smoke feature set excludes GDELT and
explicitly records an allowed stale-news exception in `data_readiness.json`.

Inspect `squeue -j JOBID`, `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS`, and
`slurm-JOBID.out`. A successful job must also contain the checkpoint, comparison metadata,
validation predictions/metrics, holdout forecast, and summary report.

Smoke job `869927` completed on `s-005` in 4:00 with exit code 0. The adapter reload and
holdout forecast passed. On the intentionally tiny 15-point sample, LoRA improved weighted
quantile loss by 3.31%, MAE by 1.84%, and RMSE by 2.08% over the pretrained run; the Gaussian
baseline still had the lowest loss and error. These figures verify execution only. Jobs
`869909` (CUDA 13/driver mismatch) and `869924` (SIGBUS while validating the newly installed
overlay) were failed/cancelled setup attempts and are retained in the engineering record.

## Run the first full comparison

```bash
.venv/bin/python scripts/check_data_readiness.py
sbatch --account=gpu-students slurm/chronos_validation.sbatch
```

This requires all selected sources to be fresh. It runs target-only pretrained Chronos-2
and a core-feature pretrained-versus-LoRA comparison with the same final 252-session
validation split, five-session horizon, context length 512, and 200 optimizer steps.
With stride five, 50 complete windows evaluate 250 sessions; two tail sessions remain
unused. LoRA uses only the training split, with no checkpoint selection on those reported
validation outcomes. Treat this as development validation; reserve a separate untouched
test interval before using these results for final model selection claims.

The Gaussian baseline forecasts zero mean and estimates daily-return volatility from the
last 60 observed returns at each origin. Weighted quantile loss is twice mean pinball loss
over the 21 requested quantiles, divided by mean absolute realized return. Coverage is
reported for the 10/90 and 1/99 intervals. Interval width alone has no preferred direction.

Record the job ID, allocation, data hashes, model revision, package versions, seed, and
outcome in `docs/REPORT_DECISIONS.md` after a real cluster run.
