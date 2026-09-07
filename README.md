# Deep Learning Time-Series Forecasting Project

This project builds a data ingestion layer to forecast short-term Nasdaq-100 / QQQ returns using various features including market price data, macro/market indicators, and alternative data.

## Directory Structure

- `data/raw/`: Contains raw downloaded datasets from various sources.
- `data/processed/`: Contains cleaned and normalized CSV files, combined by daily frequency.
- `data/logs/`: Logs for data ingestion scripts.
- `scripts/`: Contains Python scripts for fetching and processing data.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy the `.env.example` file to `.env` and fill in your API keys (e.g., FRED API key).
   ```bash
   cp .env.example .env
   ```

## Ingestion Pipeline

Run the fetchers in this order (each writes one file to `data/processed/`), then build the
table. Every step is idempotent and re-downloads from the source; `docs/DATA_NOTES.md`
explains the availability lags and the no-zero-fill rule that all files follow.

```bash
# 1. Prices and market indices (no keys needed)
python scripts/fetch_qqq_prices.py --start 2006-01-01          # qqq_ohlcv_daily.csv
python scripts/fetch_cross_asset_prices.py                     # cross_asset_daily.csv  (TLT, HYG, SPY, IWM, SMH, GLD, oil, DXY)
python scripts/fetch_cboe_vol_indices.py                       # cboe_vol_daily.csv     (VIX, VXN, VIX3M, VIX9D, VVIX + term-structure ratios)

# 2. Uncertainty and news (no keys needed)
python scripts/fetch_epu.py                                    # epu_daily.csv          (daily EPU and Equity Market Uncertainty, 1985+)
python scripts/fetch_gdelt_news.py --start 2017-01-01          # gdelt_topic_daily.csv  (cached per topic; delete data/raw/gdelt_timeline/*.csv to refetch)

# 3. Fed communication (no keys needed; the tone model downloads ~1.4 GB once)
python scripts/fetch_fomc_statements.py --start 2006-01-01     # fomc_events_daily.csv + raw statement text
python scripts/score_fomc_tone.py                              # fomc_tone_daily.csv    (hawkish/dovish sentence classifier)

# 4. Sources that need .env keys
python scripts/fetch_fred_macro.py --start 2006-01-01          # fred_macro_daily.csv   (FRED_API_KEY; CPI via ALFRED vintages)
python scripts/fetch_sec_filings.py --start 2006-01-01         # sec_filings_daily.csv  (SEC_USER_AGENT_EMAIL; full history + 8-K earnings items)

# 5. Known-future calendar and the join
python scripts/build_event_calendar.py                         # event_calendar_daily.csv (FOMC/CPI/NFP days, month-end window, opex; runs 120 days ahead)
python scripts/build_daily_feature_table.py                    # daily_feature_table.csv + daily_feature_table_groups.json
python scripts/audit_processed_data.py --leakage               # coverage, dead columns, same-day leakage check
python scripts/validate_feature_table.py                       # 47 checks against known market history (crashes, inversions, release days, event-day |returns|)
```

Notes:

- GDELT's DOC API is rate limited (one request every few minutes). The timeline fetcher caches each topic, so a full refetch takes about half an hour.
- `fetch_vix_cboe.py` still works but is superseded by `fetch_cboe_vol_indices.py`; the feature table no longer reads `vix_daily.csv`.
- The CPI columns are built from ALFRED release vintages, so the value on any day is the one that had actually been published by then. The old forward-fill by reference month leaked about six weeks.

## Feature Table

`build_daily_feature_table.py` joins everything onto QQQ trading days with publication-aware
alignment and writes two files:

- `data/processed/daily_feature_table.csv`: the target `log_return_1d`, raw OHLCV (not features), and every covariate already transformed to a stationary form (log changes, basis-point differences, ratios, trailing z-scores).
- `data/processed/daily_feature_table_groups.json`: maps each column to a `group` (`qqq`, `market`, `rates`, `uncertainty`, `news`, `fed`, `sec`, `calendar`) and a `role` (`target`, `past`, `known_future`, `raw`). Use it to select feature sets for the ablation ladder and to pass the `known_future` columns as `known_covariates_names`.

`--profile core` writes the compact set from the covariate brief (about 20 past-only columns plus the calendar flags); the default `full` profile writes all groups. Future calendar values for the forecast horizon live in `event_calendar_daily.csv`.

## Chronos-2 Fine-Tuning

Install the project dependencies, build the feature table, then start with LoRA fine-tuning:

```bash
pip install -r requirements.txt
python scripts/build_daily_feature_table.py
python scripts/fine_tune_chronos2.py \
  --prediction-length 5 \
  --context-length 512 \
  --num-steps 200 \
  --batch-size 64 \
  --finetune-mode lora
```

The script loads `amazon/chronos-2`, uses `log_return_1d` as the default target, keeps covariates as past-only features, and saves the fine-tuned checkpoint under `models/chronos2-qqq/finetuned-ckpt`.

### Local Smoke Test

On a local Apple Silicon machine, first validate the data split and selected features without loading the model:

```bash
python scripts/fine_tune_chronos2.py --smoke-test --prepare-only
```

Then run a tiny LoRA fine-tuning job:

```bash
python scripts/fine_tune_chronos2.py --smoke-test --device-map auto
```

Smoke-test mode uses the most recent 384 rows, a 3-day prediction horizon, 64 rows of context, 5 training steps, batch size 8, and a smaller covariate set. If `auto` fails on MPS, retry with `--device-map cpu`; it is slower but usually more predictable.

### Chronos-2 Prediction Test

After fine-tuning, generate a 3-step forecast from the smoke checkpoint:

```bash
python scripts/predict_chronos2.py \
  --checkpoint models/chronos2-qqq-smoke/finetuned-ckpt \
  --device-map auto
```

This writes `data/processed/chronos2_smoke_forecast.csv` with forecast dates, the median prediction, and quantile columns such as `q0.1`, `q0.5`, and `q0.9`.

To test the checkpoint against known data, hold back the last 3 rows and compare predictions to actual values:

```bash
python scripts/predict_chronos2.py \
  --checkpoint models/chronos2-qqq-smoke/finetuned-ckpt \
  --holdout-rows 3 \
  --device-map auto \
  --output data/processed/chronos2_smoke_holdout_forecast.csv
```

The holdout output adds `actual`, `error`, and `abs_error` columns.

### Base vs Fine-Tuned Validation

To compare the pretrained Chronos-2 model against a fine-tuned version on the same validation split, first run a dry check:

```bash
python scripts/compare_chronos2_validation.py --smoke-test --prepare-only
```

Then run the smoke comparison:

```bash
python scripts/compare_chronos2_validation.py --smoke-test --device-map auto
```

This workflow:

- reserves the final validation rows chronologically,
- evaluates `amazon/chronos-2` as-is on rolling validation windows,
- fine-tunes Chronos-2 on the train split only,
- evaluates the fine-tuned model on the exact same validation windows,
- writes `models/chronos2-validation-comparison/validation_predictions.csv`,
- writes `models/chronos2-validation-comparison/validation_metrics.csv`.

The main comparison metrics are `mae_log_return`, `rmse_log_return`, `bias_log_return`, `directional_accuracy`, and `q10_q90_coverage`. Lower MAE/RMSE is better; higher directional accuracy and interval coverage are better.

Summarize the comparison into a compact report:

```bash
python scripts/summarize_chronos2_comparison.py
```

This reads the comparison outputs and writes:

- `models/chronos2-validation-comparison/summary_report.md`
- `models/chronos2-validation-comparison/summary_deltas.csv`

The summary report includes an overall verdict, per-horizon metric deltas, and the validation dates where fine-tuning helped or hurt the most.

## Output Data Structure

The processed output files are saved in `data/processed/` and share a common `date` column (in `YYYY-MM-DD` format) so they can easily be joined together. Sections A-F are the original sources; G-L were added in the data-v2 pass.

### A. QQQ Prices (`data/processed/qqq_ohlcv_daily.csv`) 🟢 Success
This file contains the core market data for the Nasdaq-100 ETF, aligned to US trading days.
- **Base columns**: `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`
- **Features**:
  - `return_1d`: The 1-day percentage change (using adjusted close).
  - `return_5d`: The 5-day percentage change.
  - `log_return_1d`: The natural logarithm of the 1-day return.
  - `volatility_20d`: A rolling 20-day standard deviation of the log returns.
  - `volume_change_1d`: The percentage change in trading volume from the previous day.

### B. FRED Macro Indicators (`data/processed/fred_macro_daily.csv`) 🟢 Success
Calendar-day file of FRED series stored on their reference dates; the feature table applies the publication lags (H.15 rates post the next day, breakevens the same afternoon). The ICE BofA high-yield spread is not included because FRED now serves only its last three years; the HYG return in `cross_asset_daily.csv` is the credit signal.
- **Base columns**: `date`
- **Features**:
  - `dff`, `dgs10`, `dgs2`, `t10y2y`: fed funds, 10y, 2y, and the 10y-2y spread (H.15).
  - `dfii10`, `t5yie`: 10-year real yield and 5-year breakeven inflation (Treasury-sourced, same day).
  - `cpi`, `cpi_yoy`, `cpi_ref_month`: the CPI level and year-over-year change **as published on or before each day** (ALFRED vintages), not forward-filled by reference month.
  - `cpi_release_day`: 1 on BLS CPI publication days.
  - `vix_fred`: kept for compatibility; the builder uses the CBOE close.

### C. CBOE VIX (`data/processed/vix_daily.csv`) 🟢 Success
This file contains the official CBOE Volatility Index, which tracks market expectations for volatility over the next 30 days.
- **Base columns**: `date`
- **Features**: 
  - `vix_open`, `vix_high`, `vix_low`, `vix_close`

### D. GDELT News Features (`data/processed/gdelt_topic_daily.csv`) 🟢 Success (2017 onward)
Calendar-day counts and average tone from the GDELT DOC 2.0 timeline API for six topics: `ai`, `semiconductor`, `fed`, `inflation`, `big_tech_earnings`, `recession`.
- **Base columns**: `date`
- **Features** per topic: `<topic>_news_count`, `<topic>_avg_tone` (GDELT's -100..100 tone, typically -7..+3), and `<topic>_news_share` (count divided by all articles GDELT monitored that day, which removes coverage growth from the raw counts).
- Days GDELT did not cover (a 17-day outage in June 2025, the partial final day) are NaN, not zero. The API only reaches back to 2017, so the feature table has NaN before then.

### E. SEC Filings (`data/processed/sec_filings_daily.csv`) 🟢 Success
This file tracks major corporate filing events (10-K, 10-Q, 8-K) for the top Nasdaq companies.
- **Base columns**: `date`
- **Features**:
  - `sec_10k_count`, `sec_10q_count`, `sec_8k_count`, plus `sec_20f_count` and `sec_6k_count` for foreign issuers (TSMC).
  - `sec_total_filings`: Sum of the above.
  - `{ticker}_filing_event`: 1 if that company filed one of the tracked forms that day.
  - `ndx_earnings_count`, `ndx_earnings_premarket`, `ndx_earnings_postmarket`: 8-K filings with Item 2.02 (earnings release), split by EDGAR acceptance time; `{ticker}_earnings_event` per company. Post-market releases are attached to the next trading day by the builder.
  - History is complete from 2006 (the paginated older submission files are fetched, not only the most recent thousand filings). Alphabet's filings before its 2015 holding-company reorganization come from the Google Inc registrant, and Broadcom's before 2018 from Broadcom Ltd / Avago.
  - TSMC reports earnings via 6-K rather than 8-K Item 2.02, so it has no `tsm_earnings_event` column.

### F. FOMC Statements (`data/processed/fomc_events_daily.csv`) 🟢 Success
This file parses Federal Reserve monetary policy statements.
- **Base columns**: `date`
- **Features**:
  - `is_fomc_day`: 1 on each of the 168 policy-statement days since 2006 (eight scheduled meetings a year plus the 2008 and 2020 emergency meetings; non-policy releases such as the Statement on Longer-Run Goals are excluded).
  - `days_since_fomc`: Calendar days since the last statement (NaN before the first one).
  - `fomc_statement_length`, `fomc_hawkish_keyword_count`, `fomc_dovish_keyword_count`: legacy keyword measures, carried forward between meetings. The classifier-based tone in `fomc_tone_daily.csv` supersedes them.

### G. CBOE volatility indices (`cboe_vol_daily.csv`)
One row per CBOE trading day from 2006. VIX OHLC plus `vxn_close` (Nasdaq-100 implied vol; 2006-2009 backfilled from FRED `VXNCLS`), `vix3m_close` (from Dec 2007), `vix9d_close` (from 2011), `vvix`, and the derived `vxn_minus_vix`, `vix_term_ratio` (VIX / VIX3M, above 1 = backwardation) and `vix9d_ratio`. Same-evening availability.

### H. Cross-asset closes (`cross_asset_daily.csv`)
Daily closes and log returns for TLT, HYG (from Apr 2007), SPY, IWM, SMH, GLD, WTI crude (`oil_`) and the dollar index (`dxy_`). The feature table uses TLT and HYG directly and turns SPY, IWM and SMH into relative returns against QQQ.

### I. Economic Policy Uncertainty (`epu_daily.csv`)
Calendar-day `epu_daily` and `emu_daily` (Equity Market Uncertainty) from policyuncertainty.com, 2005 onward (source runs from 1985). Published the next morning; only the trailing month is revised. In the feature table `epu_log` / `emu_log` are the log of the trailing 7-day mean (single weekend days are very noisy) and `epu_log_1d` / `emu_log_1d` the raw prior day.

### J. FOMC tone (`fomc_tone_daily.csv`)
Sentence-level hawkish / dovish classification of every policy statement (168 statements, 2006-2026). Statement days carry `fomc_hawkish_share`, `fomc_dovish_share`, `fomc_net_hawkish`, `fomc_tone_change`; `fomc_net_hawkish_ewma` and `fomc_net_hawkish_last` are carried forward between meetings. Raw sentence labels are in `data/raw/fomc/fomc_sentence_labels.csv`.

### K. Event calendar (`event_calendar_daily.csv`)
Known-in-advance flags per NYSE trading day, extended 120 days past today: `is_fomc_day`, `is_fomc_day_before`, `days_to_fomc`, `fomc_cycle_week`, `is_cpi_day`, `is_nfp_day` (need `FRED_API_KEY`), `days_to_month_end` (-3..+3 window), `is_month_end`, `is_quarter_end`, `is_opex_day`, `is_opex_week`, `is_quad_witching`, `day_of_week`, `is_pre_holiday`.

### L. Feature table groups (`daily_feature_table_groups.json`)
Column to `{group, role}` map written by the builder. Roles: `target`, `past`, `known_future`, `raw`.
