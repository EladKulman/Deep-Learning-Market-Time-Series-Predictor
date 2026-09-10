# QQQ forecasting with Chronos-2

Workshop project by Elad Kulman and Tom Weitman. The current experiment forecasts QQQ
one-day log returns at horizons of one to five NYSE sessions, using pretrained Chronos-2,
LoRA fine-tuning, and feature-group ablations. Each forecast origin is the NYSE close.

The immediate priorities are completing the four remaining rate-limited GDELT tails and
testing which source family drives the winning combined-source development result.
The original BTC/TFT proposal is historical context; see `docs/REPORT_DECISIONS.md` for
the current research choices and `docs/DATA_NOTES.md` for source timing and limitations.

## Current status (2026-09-10)

- Repaired SEC timestamp/early-close alignment, CPI revision handling, two missing Fed
  statements, and the scheduled-event calendar. Fed inventory and tone cover 170 statements.
- Training, comparison, and prediction share one feature contract. Missing past values stay
  masked; only declared calendar features can enter the forecast horizon.
- Local CPU tests trained and reloaded LoRA adapters, exercised target-only and complete
  masked-covariate inputs, and verified the pinned model cache offline.
- The AI and semiconductor GDELT topics are refreshed through September 7; four topic caches still stop at June 30, 2026. The incremental refresh is resumable;
  run the readiness check below for the authoritative current status.
- TAU Slurm job `869927` completed on an RTX 2080 Ti with PyTorch 2.11.0+cu128: pinned
  pretrained inference, five LoRA steps, validation, adapter reload, and holdout prediction
  all passed. This tiny run checks execution, not forecasting skill. See `slurm/README.md`
  and ignored local artifacts under `models/slurm-smoke-869927/`.
- Slurm arrays `869989` and `871482` completed 12 full source-screen fits. Across three
  seeds, the combined `all_external` setup led control by 0.60% mean WQL and had better
  calibration, but its paired-window interval still crossed zero. See
  `docs/NEWS_ABLATION_RESULTS.md`.
- Slurm array `874192` completed the six matched single-topic GDELT fits. Fed news had the
  best seed-42 WQL (`0.686526`, 1.28% below control), with recession nearly tied at
  `0.686796`. See `docs/GDELT_TOPIC_RESULTS.md`.

## Setup

Use Python 3.11 or 3.12. Create a local environment, then install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

FRED downloads require `FRED_API_KEY`; SEC downloads require a real contact address in
`SEC_USER_AGENT_EMAIL`. The saved FRED/SEC snapshots support `--cached` replay without keys.
Keep `.env`, local connection notes in `oath/`, and model files out of version control.

## Refresh the data

Run from the repository root using the environment's Python:

```bash
# 2005 price history warms up the rolling features; the modeling sample begins in 2006.
python scripts/fetch_qqq_prices.py --start 2005-01-01
python scripts/fetch_cross_asset_prices.py
python scripts/fetch_cboe_vol_indices.py
python scripts/fetch_epu.py
python scripts/fetch_gdelt_news.py --start 2017-01-01
python scripts/fetch_fomc_statements.py --start 2006-01-01
python scripts/score_fomc_tone.py
python scripts/fetch_fred_macro.py --start 2006-01-01
python scripts/fetch_sec_filings.py --start 2006-01-01
python scripts/build_event_calendar.py
python scripts/build_daily_feature_table.py
python scripts/check_data_readiness.py --output data/processed/data_readiness.json
python scripts/validate_feature_table.py
python scripts/audit_processed_data.py --leakage
```

GDELT extends each topic's cache with an overlapping tail and saves every successful API
response. HTTP 429 responses trigger backoff; refreshes can take much longer than half an
hour. Use `--refresh --start YYYY-MM-DD --end YYYY-MM-DD` to revisit a historical interval.
Use `--topics fed inflation` to target retries, or `--cached` to rebuild from completed topic caches. An API outage remains missing, never a fabricated zero. Do not delete the raw caches.

For an offline replay of the repaired Fed/FRED/SEC/calendar sources:

```bash
python scripts/fetch_fomc_statements.py --cached --end 2026-09-07
python scripts/fetch_fred_macro.py --cached --end 2026-09-07
python scripts/fetch_sec_filings.py --cached --end 2026-09-07
python scripts/build_event_calendar.py --cached --end 2027-01-05
python scripts/build_daily_feature_table.py
```

Tone scoring needs the cached classifier or a first model download. It pins the tested
classifier revision and writes source/model hashes to `fomc_tone_daily.metadata.json`.
`fetch_vix_cboe.py` and `vix_daily.csv` are legacy; the builder uses `cboe_vol_daily.csv`.

## Feature and time contract

The builder writes `data/processed/daily_feature_table.csv` and its matching
`daily_feature_table_groups.json`. A build metadata file hashes every source so a source
refresh cannot silently leave an outdated modeling table in use. The full schema has one target, six raw OHLCV columns,
55 past covariates, and 14 known-future calendar covariates, plus `date` (77 columns total).
The core model selects 21 past and six known-future covariates. Raw OHLCV is never selected
as model covariates. SEC earnings counts are past observations, not scheduled earnings dates.

Use `--feature-profile core`, `full`, or `target` in the modeling scripts. Combine
`--feature-profile full --feature-groups qqq,calendar,market` for a group ablation, or pass
`--feature-columns` explicitly. The roles file validates selection and future inputs.
`--profile core` on the builder can optionally write a smaller table.

No modeling script drops sessions because a covariate is NaN. Invalid dates, missing
returns, and incomplete known-future inputs fail clearly. Forecast dates use the NYSE
calendar, including holidays. Unknown schedules beyond their published horizon remain NaN.
The readiness gate checks source tails and the exact next five exchange sessions; it
cannot prove every historical observation was available at the forecast origin.

## Pretrained baseline and LoRA

First inspect the split and features without loading a model:

```bash
python scripts/compare_chronos2_validation.py --smoke-test --prepare-only
python scripts/compare_chronos2_validation.py --feature-profile core --prepare-only
```

Run a small local comparison and reload its adapter:

```bash
python scripts/compare_chronos2_validation.py --smoke-test --device-map cpu
python scripts/summarize_chronos2_comparison.py
python scripts/predict_chronos2.py \
  --checkpoint models/chronos2-validation-comparison/finetuned/finetuned-ckpt \
  --holdout-rows 5 --device-map cpu
```

Comparison smoke defaults: 512 total rows, final 60 for validation, five three-session
windows, 64-session model context, five optimizer steps, six covariates. The separate
`fine_tune_chronos2.py --smoke-test` command has 384-row defaults and saves under
`models/chronos2-qqq-smoke/`.

A pretrained target-only arm uses `--feature-profile target --base-only`. Live pretrained
prediction uses `predict_chronos2.py --pretrained --prediction-length 5`. For the actual
GPU smoke test and first full comparison, follow `slurm/README.md`. The controlled
news-source screen is specified in `docs/NEWS_ABLATION_PLAN.md` and
`configs/news_ablation.json`; completed results are in `docs/NEWS_ABLATION_RESULTS.md`.

Full comparison defaults reserve the final 252 sessions, train LoRA on earlier rows only,
and evaluate pretrained, fine-tuned, and zero-mean Gaussian forecasts on identical windows.
With stride five, 50 complete windows score 250 sessions. Primary score: weighted quantile
loss over 21 quantiles. Also report MAE, RMSE, direction, and 80%/98% interval coverage;
coverage should approach its nominal level. Narrower intervals alone are not better.

Outputs include predictions, metrics, checkpoint metadata, data/role/script hashes,
model revision, seed, and package versions. Use development validation to choose settings;
reserve an untouched test interval before making final model-selection claims. Adapters
from the old feature schema must be retrained.

## Checks and directories

```bash
python -m unittest discover -s tests -v
bash -n slurm/chronos_smoke.sbatch
bash -n slurm/chronos_validation.sbatch
bash -n slurm/chronos_news_ablation.sbatch
bash -n slurm/chronos_news_robustness.sbatch
bash -n slurm/chronos_gdelt_topic_ablation.sbatch
```

- `data/raw/`: downloaded observations, source schedules, statement text, and labels.
- `data/processed/`: source tables, event calendar, modeling table, and readiness report.
- `scripts/`: fetchers, transformations, validation, training, comparison, and prediction.
- `tests/`: time alignment, missingness, and future-input isolation regression checks.
- `slurm/`: single-GPU jobs and cluster setup instructions.
- `models/`: ignored model caches, adapters, predictions, and run reports.
