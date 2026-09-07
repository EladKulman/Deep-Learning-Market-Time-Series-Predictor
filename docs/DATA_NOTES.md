# Data notes

Working notes for the data layer, written 2026-09-07 alongside the covariate research brief.
Read this before adding or changing a processed CSV.

## Rules every processed file follows

1. **Never zero-fill a missing observation.** Chronos-2 builds an observation mask from NaN
   and standardizes each series over observed points only. A zero written into a day the
   source did not cover is a fake observation. Counts are zero only when the source was
   live and nothing matched (e.g. no filings that day).
2. **Attach by release time, not reference date.** The value attached to trading day T must
   have been knowable at the 16:00 ET close on T. `build_daily_feature_table.py` applies the
   lags listed below; the source files themselves store values on their natural dates.
3. **Levels stay in the source files; transforms happen in the builder.** Rates and spreads
   are first-differenced, prices and volatility indices log-differenced, bounded ratios and
   trailing z-scores kept as levels.
4. **Known-future covariates are complete through the forecast horizon.** The calendar file
   runs 120 days past today so there are no NaNs in the future block.

## Availability lags applied in the builder

| Source | Day-T value known | Builder action |
|---|---|---|
| QQQ, cross-asset ETFs, CBOE indices | at the close | none |
| FRED H.15 rates (DFF, DGS2, DGS10, T10Y2Y) | next day 16:15 ET | shift +1 day |
| FRED breakevens / real yields (T5YIE, DFII10) | same afternoon | none |
| CPI (ALFRED vintage) | release day 08:30 ET | already vintage-aligned |
| EPU / EMU (calendar days) | next morning | use value dated T-1; weekend values roll into Monday |
| GDELT (calendar days) | after 00:00 UTC | use value dated T-1; weekends roll into Monday |
| FOMC statement / tone | 14:00 ET same day | none for a close-to-close target |
| SEC 8-K earnings, post-market | after the close | shift +1 day (pre-market stays on T) |

## Options researched but not built (for Tom)

These were judged "worth trying" in the brief but need a key, a GPU, or a cloud project.
Each is a self-contained ablation arm; none is required for the core experiment.

- **FinBERT daily headline sentiment from the NYT Archive API.** Free API key at
  developer.nytimes.com; about 250 monthly calls cover 2006-2026. Filter to the Business
  and Technology sections, score headline + abstract with `yiyanghkust/finbert-tone`
  (Apache-2.0), aggregate per trading day with a prior-close cutoff. Terms forbid caching
  raw article text, so publish only the derived index. Expect a few CPU hours or under an
  hour on a T4. The FNSPID dataset on Hugging Face (`Zihan1004/FNSPID`, CC BY-NC) adds a
  ticker-tagged layer for QQQ constituents but stops in 2023.
- **GDELT through BigQuery instead of the DOC API.** `gdelt-bq.gdeltv2.gkg_partitioned`
  gives 2015+ with no rate limit under the 1 TB/month free tier. Select only `DATE`,
  `V2Tone`, `V2Themes`, one year per query, and dry-run first because `V2Themes` is wide.
  `gdelt-bq.full.events` is about ten times cheaper and reaches 2013.
- **SF Fed Daily News Sentiment Index.** One xlsx download
  (frbsf.org/wp-content/uploads/news_sentiment_data.xlsx). Include only as a low-priority
  arm: the whole history is revised between vintages and one published test finds no
  return-forecasting power. Lag at least seven days if used.
- **The reference FOMC tone model is gated.** `gtfintechlab/FOMC-RoBERTa` (Shah, Paturi &
  Chava, ACL 2023, CC BY-NC 4.0) needs you to accept its terms on Hugging Face and run
  `huggingface-cli login`; then rerun `score_fomc_tone.py --model-id gtfintechlab/FOMC-RoBERTa`.
  The pipeline currently uses `LorenzoAleCon29/roberta-large-fomc-hawkish-dovish`, an
  ungated RoBERTa-large trained on the same sentence dataset; the script checks three
  probe sentences at start-up so a wrong label map fails loudly.
- **Jarocinski-Karadi monetary policy shocks** (github.com/marekjarocinski/jkshocks_update_fed,
  CC BY 4.0, 1988-2024) as a per-meeting surprise series alongside the RoBERTa tone.
- **OFR Financial Stress Index** (financialresearch.gov, daily from 2000, two-day lag).

## Sources deliberately skipped

CBOE put/call ratios (free history ends 2019-10), SKEW, foreign indices, NFCI and other
weekly stress composites (revised, duplicate VIX plus spreads), Michigan and Conference
Board levels, monthly EMV, Twitter uncertainty (discontinued 2023), Google Trends (no
stable API, non-reproducible samples), margin debt, short interest, ETF flows.
