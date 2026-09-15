# Data notes

Updated 2026-09-08. Read before changing a source or selecting model covariates.

## Modeling contract

The sample begins January 3, 2006 and currently ends September 4, 2026: 5,201 NYSE
sessions and 77 columns. QQQ source history starts in 2005 so rolling price features are
already initialized at the modeling start. The target is the adjusted-close one-day log
return. An origin at the close of T predicts subsequent session returns, not T itself.

1. Missing past values remain NaN and reach Chronos-2's observation mask. Never drop a
   trading session because a covariate is missing or replace unavailable news with zero.
2. Attach observations by availability. Zero event counts mean a covered period with no
   matching event, not a failed source request.
3. The roles file controls inputs: 55 past covariates, 14 known-future calendar values,
   six excluded raw columns, and one target. Core selects 21 past plus six known future.
4. Future input arrays contain only the selected known-future columns. Historical future
   returns and news are withheld. Incomplete future schedules stop prediction.
5. Keep raw observations and provenance. Rebuilding from a cache does not make its raw
   snapshot newer. Freshness gates and publication lags do not prove a vintage backtest.

## Availability rules

| Source | Alignment used |
|---|---|
| QQQ and US cross-asset ETFs | Same session's close; 2005 QQQ history supplies rolling warm-up. |
| CBOE VIX/VXN/term ratios/VVIX | Prior calendar day's observation, then as-of join. Daily values can finalize after the 16:00 ET cutoff. This also covers next-day FRED backfills. |
| CBOE same-session variants (`market_t0` group: `vxn_log_chg_t0`, `vix_term_ratio_t0`) | Day-T CBOE close, no lag. The 16:00 ET index value is observable even though CBOE finalizes the close at 16:15, so the one-session lag above is a conservative choice, not a fact; these two columns exist so the lag can be ablated. Never combine both versions in one run. |
| Yahoo oil and dollar index | One-day conservative lag because daily-bar cutoff is not guaranteed before the NYSE close. |
| FRED H.15: DFF, DGS2, DGS10, DFII10 | Publication is after the next federal business day’s close. Use the first NYSE session after that release, accounting for federal holidays and early closes. |
| FRED spreads: T10Y2Y, T5YIE | Before June 21, 2019 use the H.15 rule. Thereafter direct Treasury sourcing supports same-evening publication, available at the next NYSE close. The legacy `t10y2y_lag1` name remains. |
| CPI | ALFRED release vintages. Both monthly prints and pure historical revisions update the published level/yoy. Only a new reference-month print sets the monthly-release flag. |
| EPU/EMU | Trailing seven-calendar-day average through T−1; prior-day raw log also retained. |
| GDELT | Completed prior UTC dates, lagged one day; weekend values aggregate into Monday. Outages stay missing. |
| FOMC text/tone | Actual policy-statement dates, available in past context. Scheduled meetings and emergency statements are separate inventories. |
| SEC, all forms and earnings | Parse acceptance timestamp as UTC; assign to the first NYSE close strictly after acceptance. Exact-close, post-close, weekends and early closes move correctly. Missing timestamp uses the next session after filing date. No second shift in the builder. |
| Future calendar | Scheduled Fed meetings, BLS release dates, and deterministic NYSE/month/expiration structure. Earnings filing counts are past-only. |

SEC session counts preserve each filing once. The legacy `premarket` column now means accepted before that session’s close (including
intraday filings); `postmarket` counts filings moved to a subsequent session.
The fetcher writes `sec_filings_daily.metadata.json`; the builder rejects legacy alignment.

## Repairs and remaining coverage

| Area | Status and limit |
| FOMC 2012 heading (2026-09-15) | The historical page's cross-month heading "July 31-August 1 Meeting" was parsed as 2012-07-31; the statement was 2012-08-01. `parse_meeting_heading` now uses the explicitly named end month; unit-tested for both heading forms. |
|---|---|
| Price rolling warm-up | Filled with 2005 observations before trimming the modeling sample. |
| Fed statements | 170 statements. Added January 22 and December 16, 2008 (`b` URLs); corrected June 2007's URL typo to the printed June 28 date. Cached non-policy exclusions make offline replay complete. |
| Fed tone | All 170 statements scored with the pinned ungated RoBERTa classifier. Metadata records input/output hashes, model revision and label probes. |
| Scheduled FOMC flags | Exclude emergency meetings and the August 2025 notation vote. Preserve the originally planned March 18, 2020 date. Historical meeting pages reconstruct schedules; they are not an archive of every advance announcement or cancellation. |
| CPI/NFP future flags | Official BLS schedules merged with historical releases. Coverage currently runs through December 2026; later flags are NaN. NFP history was migrated from the previous committed FRED-derived NYSE calendar and is not a complete calendar-day release archive. |
| GDELT recent tail | The AI and semiconductor topics now reach September 7; fed, inflation, big-tech earnings, and recession still end June 30. Individual successful requests are saved so the refresh can resume from the remaining topics. Repeated paced retries from both the local and TAU networks received HTTP 429 on September 8 and again from TAU on September 9. Consult `data_readiness.json` for the saved snapshot status. |
| GDELT historical holes | Saved DOC history begins in 2017 and contains a June 2025 outage (13 affected modeling sessions). Retry the outage explicitly; unavailable data cannot be reconstructed by zero-fill. Pre-2017 coverage requires another source. |
| Market source inception | HYG starts April 2007, VIX3M late 2007, VIX9D 2011, VVIX March 2006. Preserve missing history rather than invent values before inception. |
| Fed initial state | Tone starts with the first 2006 statement; tone-change requires a second statement. Earlier missing values remain masked. |

The complete feature table is retained for all ablations. A reduced target-only experiment
can use the full return history without requiring news. The Slurm infrastructure smoke
uses four past and two calendar features, excludes GDELT, and explicitly records its stale
news exception. Full validation requires the strict source-readiness gate.

`check_data_readiness.py` compares source tails with the latest completed NYSE session
and checks the exact forecast dates. Use `--as-of 2026-09-08T12:00:00Z` to reproduce a
historical readiness decision; do not confuse a frozen snapshot with fresh market data.
The validator currently reports 48 passes, one correlated-volatility warning, and no failures.

## Reproducibility and research limits

The September 8 check of official H.15/FRED pages showed September 3 nominal/real yields
published September 4 at 16:15–16:16 ET, after the equity close. T10Y2Y/T5YIE instead
showed September 4 values published that evening; their notes date the source change to
June 21, 2019. The builder reconstructs these schedules conservatively. Exceptional
historical delays and revisions still require a vintage archive. When reference dates
share an availability session, the latest reference observation wins deterministically.


The official BLS endpoint returned HTTP 403 to the fetcher. The saved CPI and employment
CSVs were transcribed from the official tables accessed on September 8, 2026; each stores
its source URL, retrieval date, and published coverage horizon. Refresh falls back to these
caches and does not claim an unpublished horizon is known.

- BLS CPI schedule: https://www.bls.gov/schedule/news_release/cpi.htm
- BLS employment schedule: https://www.bls.gov/schedule/news_release/empsit.htm
- Fed calendar and linked historical pages: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- H.15 publication: https://www.federalreserve.gov/releases/h15/
- Real yields: https://fred.stlouisfed.org/series/DFII10
- Spread timing/source change: https://fred.stlouisfed.org/series/T10Y2Y and https://fred.stlouisfed.org/series/T5YIE
- CBOE timing: https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Cboe_Volatility_Index.pdf

EPU, Yahoo adjustments, non-CPI FRED history, reconstructed release calendars, the modern
SEC issuer basket and the tone classifier all carry revision, hindsight or model-training
limitations. A one-day lag cannot eliminate them. The classifier is a substitute for the
gated reference model and its historical outputs are a retrospective feature experiment.
These limits belong in the report; passing ingestion checks is not proof of a fully
point-in-time dataset. Source alternatives and optional future experiments remain in
`docs/REPORT_DECISIONS.md`.
