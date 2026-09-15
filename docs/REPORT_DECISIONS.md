# Decisions and ideas for the final report

Running log of the design decisions, evidence, and open ideas that the final report should
state. Each entry says what was decided, why, and what to write. Update this file whenever a
decision is made or reversed; the changelog at the end records when.

Group 13: Tom Weitman, Elad Kulman. Workshop on Deep Learning.
Companion documents: `docs/DATA_NOTES.md` (data conventions) and the covariate research
brief (published 2026-09-07; source notes in the session that produced it).

The literature summaries below are inherited from the research brief; the September 8 engineering audit did not independently verify every citation or empirical claim.

---

## 1. Framing

### 1.1 Research question
**Decision.** The project asks: *can text-derived uncertainty and Fed tone improve a
foundation model's calibrated forecasts of Nasdaq-100 returns, and which narratives matter?*
It is framed around calibration and covariate ranking, not point-return accuracy.

**Why.** Every published test of time-series foundation models on daily equity returns,
including Chronos-2, finds point forecasts collapse toward zero, directional accuracy near
51%, and skill over a zero forecast of order 10^-3 (Rahimikia et al. 2025, arXiv 2511.18578;
Noguer i Alonso & Franklin 2026, arXiv 2606.27100). A project that promises to predict
direction would fail on its own terms. A project that asks whether news tells the model
*how much to worry* is answerable with the data and maps onto the proposal's promise to rank
which narratives move the market.

**Write.** State the negative prior explicitly in the introduction and position the
contribution as (a) a publication-aware feature table with explicit vintage limitations, (b) a covariate ablation
ladder that ranks feature groups, and (c) event-conditioned calibration analysis.

### 1.2 Departures from the proposal
| Proposal | Final | Reason |
|---|---|---|
| Bitcoin | QQQ (Nasdaq-100 ETF) | Richer free covariates (macro, Fed, SEC filings); the news-and-narrative question is sharper for tech equities |
| Darts + Temporal Fusion Transformer | Amazon Chronos-2 (foundation model, LoRA fine-tuning) | Zero-shot baseline for free; native past-only and known-future covariates; group attention over covariates |
| Attention-weight interpretability | Covariate-group ablation ladder | Chronos-2 does not expose per-covariate attention the way TFT does; ablation is model-agnostic and directly answers "which narratives matter" |
| Kaggle pre-labelled sentiment dataset | GDELT counts/tone, daily EPU, classifier-scored FOMC statements | Long daily history from 2006 (2017 for GDELT); pre-labelled corpora are short or stock-level |

**Write.** A short "changes from the proposal" paragraph in the introduction; examiners
expect it and it shows the reasoning matured.

---

## 2. Target and horizon
**Decision.** Target is the one-day log return of QQQ, forecast 1 to 5 trading days ahead,
on NYSE trading days from 2006-01-03 to the present (5,201 rows as of 2026-09-04; 2005 price history supplies warm-up).

**Why.** Log returns are the standard stationary target; a 1 to 5 day horizon matches the
proposal's "next week" and Chronos-2's direct multi-step output. QQQ's own history starts in
1999 but several covariates (CBOE VVIX, HYG) begin 2006 to 2007, so 2006 is the practical
start.

---

## 3. How Chronos-2 shaped the data design
These facts from the Chronos-2 paper and source code (arXiv 2510.15821;
github.com/amazon-science/chronos-forecasting) drove concrete data decisions. **Write** them
in the methodology section; they justify choices an examiner will otherwise question.

| Model fact | Data decision |
|---|---|
| Missing values are handled by an observation mask; the model standardizes over observed points and zero-fills *in standardized space* | **Never zero-fill.** Empty means unknown. Pre-2017 GDELT and outage days are NaN, not 0. (The original pipeline zero-filled; this was reversed.) |
| Each covariate becomes an extra series in the target's attention group; pretraining groups were small | **Target 10 to 20 past-only covariates per run and ablate.** The full table has 55 past features so groups can be swapped in and out; the `core` profile has 21. |
| Every series is standardized over its own context window and passed through asinh | **Difference drifting levels** (rates, spreads, volatility indices, prices); keep bounded ratios, trailing z-scores and regime levels. |
| Known-future covariates are supported natively and are the strongest documented channel | **Build a calendar of scheduled events** (scheduled FOMC, CPI, NFP, month-end, opex) and pass it through `known_covariates_names`. |
| `predict_df` requires regularly spaced timestamps | Use array-based prediction inputs and generate forecast dates from the NYSE calendar; do not invent holiday rows. |
| Fine-tuning on a single short series may not beat zero-shot (official guidance) | Zero-shot is rung 0 of the ladder and every fine-tuned run must beat it. Use tested `chronos-forecasting==2.3.1` and record exact runtime versions (v2.1.0 fixed a past-covariate masking bug). |

---

## 4. Data sources: chosen, and why

### 4.1 Kept from the original pipeline
- **QQQ OHLCV** (yfinance) and derived features. Raw price levels are no longer model inputs; instead: overnight gap, Parkinson range volatility (1-day and 22-day), 20-day volume z-score, 21/63-day momentum, distance to 50/200-day averages. `return_1d` and `return_5d` were dropped as linear duplicates of the target.
- **FRED rates**: fed funds, 2y, 10y, 10y-2y. Entered as one-day basis-point changes plus the lagged slope level. 10y level dropped (slope = 10y - 2y exactly).
- **CPI**: rebuilt from ALFRED release vintages (see section 5).
- **FOMC statements**: scraper rewritten (94 to 168, then 170 after the September 8 suffix/date audit); keyword counts superseded by a classifier (section 6).
- **SEC filings**: rebuilt with full history from 2006, foreign-issuer forms for TSMC, and past earnings-event counts from 8-K Item 2.02.
- **GDELT**: kept from 2017 (API limit) with share-of-coverage normalization; outages left empty.

### 4.2 Added
| Source | Why it earned a place | Availability |
|---|---|---|
| **Daily Economic Policy Uncertainty and Equity Market Uncertainty** (Baker-Bloom-Davis, policyuncertainty.com, CC BY 4.0) | Longest free daily uncertainty series (1985); improves 1-day-ahead S&P realized-volatility forecasts (Liu & Zhang 2015); only trailing 30 days revised (verified by diffing vintages) | T+1 morning |
| **VXN** (Nasdaq-100 implied vol) and **VIX/VIX3M** term-structure ratio (CBOE) | VXN is the index-specific fear gauge; the term-structure slope is a documented short-horizon signal (Johnson 2017 JFQA) | one-day conservative lag at the 16:00 ET cutoff |
| **TLT and HYG returns** | Stock-bond correlation regime and same-day credit; FRED's spread series post a day later | close |
| **Relative returns** QQQ-SPY, IWM-SPY, SMH-QQQ | Raw SPY is a near-duplicate of QQQ (corr ~0.95); the spreads carry growth, size and semiconductor tilts | close |
| **Real yield (DFII10) and 5y breakeven (T5YIE)** | Real-yield shocks are the cleanest macro driver of long-duration growth stocks post-2020; source timestamps must precede the forecast origin | H.15 publication rule for DFII10; post-close Treasury-sourced spread rule for T5YIE |
| **Scheduled-event calendar** | Known-future channel; 11.4 bp average return on CPI/NFP/FOMC days vs 1.1 bp otherwise (Savor & Wilson 2013); pre-FOMC drift (Lucca & Moench 2015); turn-of-month (McConnell & Xu 2008); opex weeks (Stivers & Sun 2013) | known in advance |
| **Fed tone classifier** | Purpose-trained hawkish/dovish sentence model replaces a six-word list | 14:00 ET same day |

### 4.3 Considered and rejected (state these; they show judgement)
| Source | Reason rejected |
|---|---|
| **SF Fed Daily News Sentiment Index** | Entire history is revised between vintages (verified: year 2000 correlates 0.94 between the 2024 and 2026 files), no vintage archive, 13.5-day half-life smoothing, and the one direct test finds no return-forecasting power (Glasserman, Mamaysky & Qin 2023). Kept as an optional low-priority arm only. |
| **ICE BofA high-yield OAS** (FRED BAMLH0A0HYM2) | Since April 2026 FRED serves only three years of history. HYG return substitutes. |
| CBOE put/call ratios | Free history ends October 2019. |
| SKEW, foreign indices, NQ futures, DIA, XLK, EEM | Redundant with QQQ or with series already included; foreign closes precede the US close so add nothing for a close-to-close target. |
| Chicago Fed NFCI, St. Louis Fed FSI, ADS | Weekly composites of VIX and spreads already included; fully re-estimated each week (needs ALFRED to un-leak). |
| Michigan / Conference Board sentiment levels | Monthly; FRED posts Michigan a month late so vintages are wrong. Release dates kept as flags only. |
| Monthly EMV, Twitter uncertainty | Monthly, or discontinued in 2023. |
| Google Trends | No stable API (pytrends dead 2025), non-reproducible samples, 269-day daily windows need stitching. |
| Margin debt, short interest, ETF flows | Published weeks after the fact. |

### 4.4 Sources deferred (future work, see section 11)
FinBERT daily sentiment from NYT Archive API and FNSPID headlines; GDELT via BigQuery
(2015+, no rate limit); Jarocinski-Karadi monetary-policy shocks; OFR Financial Stress Index;
AAII survey; CFTC positioning.

---

## 5. Leakage discipline
**Decision.** Every covariate is attached to trading day T by the time it became knowable at
T's 16:00 ET close, not by its reference date. Concretely:

| Series | Rule applied |
|---|---|
| CPI | ALFRED vintages: the value on day T is the one published on or before T; year-over-year computed within that vintage. The original pipeline forward-filled by reference month, which exposed each CPI print about six weeks early (July CPI is dated 07-01 but published ~08-12). |
| FRED H.15 (DFF, DGS2, DGS10, DFII10) | Next federal business day at 16:15 ET: use the first NYSE close after publication, not merely T+1 calendar day. |
| T10Y2Y/T5YIE spreads | H.15 timing before June 21, 2019; same-evening Treasury sourcing thereafter, available next NYSE session. |
| Calendar-day news counts (EPU, GDELT) | Complete only after midnight: value dated T-1; weekend days roll into Monday as a mean. |
| Releases at or after 14:00 ET (FOMC statement) | Before the close, so day-T flag is legitimate for a close-to-close target. |
| All SEC filings | Parse UTC acceptance times and assign to the first NYSE close strictly after acceptance, including early closes. Earnings counts are past-only. |
| CBOE indices, oil, dollar bars | One-day conservative lag; CBOE daily finalization may extend beyond the origin cutoff. |
| Known-future flags | Only events on a published schedule before T (scheduled FOMC, CPI, NFP, calendar structure); historical schedules are reconstructed, not a full vintage archive. |

**Evidence to cite.** `scripts/audit_processed_data.py --leakage` regresses each covariate on
the same-day return: no next-day-only column explains more than 2% of today's return, while
close-measured ETF and QQQ features show same-day correlation. The regression is a diagnostic, not proof against leakage.
`scripts/validate_feature_table.py` now runs 49 checks: 48 pass, one correlated-volatility warning, no failures.

**Write.** A dedicated "point-in-time alignment" subsection with this table. It is the part
of the pipeline most likely to be questioned and the easiest to get wrong.

---

## 6. Fed tone: classifier over keywords
**Decision.** Each of the 170 statements is split into sentences and classified
hawkish / dovish / neutral by a RoBERTa-large fine-tuned on the FOMC hawkish-dovish sentence
dataset (Shah, Paturi & Chava, ACL 2023). Per statement: hawkish share, dovish share, net
hawkishness, change versus the previous statement, and an exponentially weighted level
carried forward between meetings.

**Why.** The original six-word keyword list counted "inflation" as hawkish regardless of
context. The classifier reproduces the known cycle: net hawkish in 2006 to 2007, dovish
through 2009 to 2021 with a trough in 2020, hawkish in the 2022 to 2023 hiking cycle.

**Caveat to state.** The reference model (`gtfintechlab/FOMC-RoBERTa`) is gated on Hugging
Face; the run used an ungated model trained on the same dataset
(`LorenzoAleCon29/roberta-large-fomc-hawkish-dovish`), verified with probe sentences. The scorer filters voting/media boilerplate, so dissent sentences matching those filters are excluded. The pinned revision is `f4759d4ad3f1182f81d87e47ba603261740d36cf`; model/input/output hashes are saved. This retrospective classifier is not a historical model-vintage archive.

---

## 7. Feature transforms and the two profiles
**Decision.** Rates and spreads enter as basis-point changes; prices, VXN, VVIX, TLT, HYG as
log changes; VIX/VIX3M, VIX9D/VIX, distance-to-average and volume z-score as bounded levels;
CPI yoy, slope, VXN level, VXN-VIX as regime levels; EPU/EMU as log of trailing 7-day mean
(single weekend days are too noisy, see changelog). The builder writes a `groups.json` giving
each column a group (`qqq`, `market`, `rates`, `uncertainty`, `news`, `fed`, `sec`,
`calendar`) and a role (`target`, `past`, `known_future`, `raw`).

Two profiles: `full` (55 past + 14 known-future) for ablation; `core` (21 past + six
known-future) for the initial comparison. Target-only and named group filters are implemented.

**Write.** A feature table in the appendix generated from `groups.json`, plus the transform
rules.

---

## 8. Experiment design: the ablation ladder
**Decision.** Same validation windows at every rung; each rung is one row of the results table.

| Rung | Features | Question answered |
|---|---|---|
| 0 | Zero-shot Chronos-2, target only | The bar everything must beat |
| 1 | + QQQ-derived transforms | Does the model need help seeing its own volatility structure? |
| 2 | + known-future calendar flags | The channel Chronos-2 is documented to exploit |
| 3 | + market covariates (VXN, term ratio, TLT, HYG, rate changes) | Market-based control arm |
| 4 | + uncertainty and text (EPU, EMU, Fed tone, GDELT shares) | "Does news move the needle?" |
| 5 | Rungs 3 and 4 repeated with LoRA fine-tuning; optionally SPY and XLK as co-targets | Fine-tuning and same-domain grouping (the one multivariate setting shown to help, Das et al. 2026) |
| baseline | LightGBM quantile regressor on the same table | Honest non-foundation comparison (beats zero-shot TSFMs on daily returns in Rahimikia et al.) |

**Also.** Per-group leave-one-out at the best rung gives the feature ranking the proposal
promised; regime cuts (2008, 2020, 2022, 2025) show where news helps or hurts.

---

## 9. Evaluation
**Decision.** Primary: weighted quantile loss over Chronos-2's 21 quantiles against a
zero-mean, rolling-volatility Gaussian baseline; coverage of the 10/90 and 1/99 bands.
Secondary: directional accuracy with a binomial test; Diebold-Mariano test of MAE against
the zero forecast; event-conditioned interval width around FOMC and CPI days.

**Why.** MAE alone rewards predicting zero. Quantile loss and coverage measure what a
return forecaster can actually deliver. The comparison now implements the Gaussian baseline, weighted quantile loss, mean pinball loss, MAE, RMSE, bias, directional accuracy and both coverage bands. Statistical significance tests and event-conditioned analysis remain planned.

**Charts to plan.** (1) Ladder table with deltas per rung. (2) Predicted 10 to 90 band
around FOMC/CPI days with and without known-future flags, overlaid on realized absolute
returns. (3) Narrative overlay: which GDELT topic shares were elevated on the days where the
model's uncertainty should have been higher.

---

## 10. Limitations to state honestly
- **GDELT covers 2017 onward only**; news columns are empty for 55% of the sample. Tone is a generic lexicon score with known noise (~55% key-field accuracy in audits).
- **Ticker survivorship**: the ten SEC companies are today's Nasdaq leaders, chosen with hindsight. Their filing and earnings flags describe the current index, not the 2006 index.
- **EPU** revises its trailing 30 days; training values for the last month of any vintage are not exactly point-in-time. Single-day values are noisy, hence the 7-day mean.
- **Daily volatility timing**: corrected by lagging all CBOE inputs, including historical FRED backfills, one day. Source inception gaps remain masked.
- **Fed tone model** is a substitute for the gated reference model; its training data may overlap historical statements used as covariates.
- **Calendar vintages**: historical CPI/NFP and scheduled Fed flags are reconstructed from release history/current archives, not a complete log of what was announced at every origin.
- **Other revisions**: Yahoo adjusted prices and non-CPI macro history are current snapshots. These limitations prevent an unqualified claim of a fully point-in-time dataset.
- **High-yield spread** dropped because FRED truncated its history; HYG (from April 2007) stands in.
- **Pretraining overlap**: the pretrained model’s exposure to historical market series is not ruled out by our chronological LoRA split. Use a separate test interval after model release for stronger claims.
- **Single-series fine-tuning** is below the regime the Chronos-2 authors recommend; results may not beat zero-shot, and that is a legitimate finding.
- **No transaction costs or tradability claims** are made.

---

## 11. Open ideas (not built; candidates for "future work" or a stretch goal)
- FinBERT daily headline sentiment from the NYT Archive API (2006 onward, free key) and FNSPID for ticker-level layer to 2023.
- GDELT through BigQuery to extend news to 2015 and remove the rate limit.
- Jarocinski-Karadi monetary-policy shocks (1988 to 2024) as a per-meeting surprise series alongside tone.
- SF Fed sentiment as a low-priority arm with the revision caveat.
- Co-targets (SPY, XLK) in the same Chronos-2 group; Takens-style argument in the paper suggests small gains.
- Predicting the open gap instead of close-to-close would let foreign indices contribute.

---

## 12. Reproducibility notes for the report
- Pipeline: `README.md` lists the eleven commands in order; `docs/DATA_NOTES.md` lists conventions.
- Keys: FRED API key (free) and an SEC contact email in `.env` (git-ignored).
- Environment: tested locally on Python 3.12.10. TAU Slurm smoke job 869927 used Python 3.12.3, PyTorch 2.11.0+cu128, CUDA 12.8, Chronos Forecasting 2.3.2, Transformers 5.16.1, PEFT 0.20.0 and Accelerate 1.14.0 on an RTX 2080 Ti. Exact cluster setup is recorded in `slurm/README.md`; each model run records its package versions.
- Commits: data layer v2 `d19acf9` (2026-09-07), FRED/SEC completion `7db1129`, validation fixes `182cfa6`.
- Validation: the corrected working table has 5,201 rows x 77 columns. The current historical checks have no failures; the separate freshness gate still tracks the GDELT refresh.

---

## 13. Review of the first modeling round (2026-09-15)

Independent code review of commits e86e689 and 25d865f (data and modeling sides), with the
key claims re-verified by hand. **Write** the confirmed parts in the methods section and the
caveats in the limitations section.

**Confirmed sound.** Feature selection by profile/group/explicit list with role validation;
known-future columns go through `known_covariates_names` and only into the future block;
chronological split with LoRA trained strictly on rows before the validation period and no
checkpoint selection on validation; rolling 50 non-overlapping five-session windows with a
strictly trailing 60-session Gaussian baseline; WQL implemented as the standard
2 * sum(pinball) / (Q * sum|y|); paired-window bootstrap; SEC acceptance times now mapped to
the first NYSE close after acceptance (the old UTC-hour comparison was a real bug); CPI
revisions now enter on their publication date; two missed 'b'-suffix FOMC statements
(2008-01-22, 2008-12-16) recovered; emergency meetings excluded from the known-future flag.

**Must fix before further runs.**
1. *Effective batch size was one window.* Chronos-2 counts target plus covariate rows toward
   `batch_size`; with 22 to 43 rows per window and `train_batch_size=8` every optimizer step
   used exactly one randomly sampled window (verified in chronos 2.3.2 `dataset.py`). All 18
   fits were 200 single-window updates at lr 1e-5, which explains why LoRA barely moved WQL
   (0.6907 to 0.6894) and why seed spread (std 0.0036 to 0.0052) exceeds most treatment deltas.
   Set `train_batch_size` to at least four times the number of variates (e.g. 128 for 23
   variates) and use 500 or more steps.
2. *FOMC date parsing bug:* the 2012 heading "July 31-August 1" is parsed as 2012-07-31; the
   statement was 2012-08-01. Verified in the table.
3. *Rate lags re-checked and confirmed correct (2026-09-15):* the H.15 series (DFF, DGS2,
   DGS10, DFII10) are attached two sessions late because the Fed posts day T's values at
   16:15 ET on T+1, after that session's close; the Treasury-sourced T10Y2Y and T5YIE post the
   same evening and are one session late. The apparent inconsistency is the sources' own
   publication schedules. A same-day 2y/10y from treasury.gov would be a data improvement.
4. *VXN one-session lag is a judgment call, not a fact.* CBOE settles index closes at 16:15 ET,
   but the 16:00 value is observable and nearly identical; the lag discards the strongest
   same-day volatility signal (the 2020-03-16 row carries 51.8 instead of 80.1). Make it an
   ablation arm, not a silent default.

**Results status.** The screen results are not yet reportable as findings beyond "no reliable
effect":
- The GDELT topic ranking is one seed compared against the seed-42 control, which was the
  worst of the three control seeds; against the control mean the six topic gains compress to
  0.7 to 1.0% and become indistinguishable from an unlucky control draw.
- Eleven treatments were compared to one control with no multiplicity correction; the same
  per-run interval that now calls all six topics significant called EPU significant at seed 42
  before it failed at seeds 43 and 44.
- Directional accuracy of 56 to 58% equals the up-day base rate of the validation year (the
  zero-forecast baseline scores 57.2%); it must be reported against that base rate.
- The aggregate all_external-versus-control confidence interval is quoted in the changelog
  but not printed in the results document.
- Run artifacts live in the ignored `models/` directory; nothing in the docs can be regenerated
  without the cluster.

**What the round does establish.** Most of the covariate gain appears in the *pretrained*
model given the covariates in context, before any fine-tuning; twelve GDELT columns at once
do worse than any single topic; the news effect is small relative to training noise. All three
are findings about model dynamics and belong in the report.

**Refocus (agreed 2026-09-15).** The research question is which inputs change the model's
behaviour and when, not whether the forecast beats the market. Planned tools: leave-one-out
from the all-external model; permutation importance at inference (no retraining; per-window
scores give importance over time); walk-forward folds over 2022 to 2025 for regime diversity;
per-horizon effects from the saved predictions; seeds 42 to 47 for the finalists.

---

## Changelog
- **2026-09-07** Research brief completed; Chronos-2 facts, source verdicts, leakage rules recorded (sections 1, 3, 4, 5).
- **2026-09-07** Data layer rebuilt: new fetchers, FOMC scraper fixed (94 to 168), tone classifier, calendar, publication-aware builder, no zero-fill policy (sections 4, 6, 7).
- **2026-09-07** FRED rebuilt with ALFRED-vintaged CPI; HY OAS dropped (FRED serves three years only). SEC rebuilt with full history, predecessor registrants for Alphabet and Broadcom, earnings 8-K items.
- **2026-09-07** Validation pass found and fixed: Monday-NaN bug in the as-of join (FRED weekend rows), EPU weekend-day outliers (switched to 7-day mean), undefined FOMC cycle before the first 2006 meeting (seeded with December 2005). 47 history checks pass.
- **2026-09-08** Verified H.15 and spread publication timestamps against official pages. Replaced the insufficient calendar-day rate lag and same-day real-yield assumption with release-session mapping. Stable as-of ordering keeps the latest reference observation when several releases become available together.
- **2026-09-08** Repaired UTC/early-close SEC alignment and removed earnings from known-future inputs; included pure CPI revisions; corrected Fed inventory to 170 and rebuilt scheduled FOMC/BLS flags. Lagged CBOE and uncertain daily commodity/FX bars. Added QQQ warm-up history.
- **2026-09-08** Wired past masks and known-future inputs through training, rolling evaluation and prediction. Added target-only and group selection, a Gaussian baseline, quantile loss, provenance, source-readiness checks and Slurm batch files.
- **2026-09-08** Local infrastructure runs completed: two-step LoRA on two three-session validation windows, checkpoint reload, and a one-window target-only pretrained run. The tiny comparison had WQL 0.7098 pretrained, 0.7143 LoRA, 0.6806 Gaussian; this is execution evidence only, not model selection. Artifacts are under ignored `models/data-v3-smoke/` and `models/data-v3-target-smoke/`; their metadata identifies the exact earlier table snapshot. A further five-step/five-window smoke test, full 69-covariate masked-input test, and offline pinned-adapter reload also passed (`models/data-v3-smoke-pinned/`, `models/data-v3-full-contract-smoke/`). These runs precede the final macro availability correction; their hashes preserve the actual snapshots used.
- **2026-09-08** TAU Slurm GPU smoke job `869927` completed on `s-005` in 4:00 with exit code 0. It used one RTX 2080 Ti, loaded the pinned Chronos-2 revision, ran five LoRA steps, evaluated 15 forecast points, saved and reloaded the adapter offline, and generated the holdout forecast and summary. WQL was 0.734899 pretrained, 0.710572 LoRA and 0.667505 Gaussian; LoRA improved WQL 3.31%, MAE 1.84% and RMSE 2.08% over pretrained, but did not beat the Gaussian baseline. The sample is an infrastructure test, not evidence of forecasting skill. Job `869909` exposed the incompatible default PyTorch 2.14/CUDA 13 wheel; job `869924` was cancelled after a SIGBUS during overlay validation. The successful runtime is pinned in `slurm/README.md`, and artifacts are under ignored `models/slurm-smoke-869927/`.
- **2026-09-08** Prepared the controlled news-source screen in `configs/news_ablation.json`: a common 21-feature market/calendar control versus EPU/EMU, FOMC tone, SEC disclosures, all GDELT topics, and all external sources. All setups use the same 2017-01-03 to 2026-06-30 sample, 252-session development window, 50 five-session origins, 200 LoRA steps, and seed 42. The top setups will be repeated with two more seeds before topic-level GDELT ablations. See `docs/NEWS_ABLATION_PLAN.md`.
- **2026-09-08** The semiconductor GDELT tail was completed through September 7 and the feature snapshot rebuilt. Four topic families remain at June 30; this does not affect the news-source screen because its common sample is frozen at that date.
- **2026-09-09** TAU Slurm array `869989` completed the six-setup seed-42 news screen; array `871482` completed control, EPU/EMU, and all-external repeats at seeds 43 and 44. All-external led across seeds with mean WQL 0.689439 versus 0.693582 for control (0.60% lower), mean MAE 0.008420 versus 0.008467, and 57.47% versus 56.67% direction. It won two of three seeds, while the aggregate paired-window 95% interval (-0.010021, 0.001424) still crossed zero. EPU/EMU's strong seed-42 result did not repeat. See `docs/NEWS_ABLATION_RESULTS.md`.
- **2026-09-09** Historical experiment jobs now freeze the readiness timestamp to the September 4 feature snapshot. This avoids wall-clock failures after a new trading session while preserving the strict current-date gate for live forecasts. The successful robustness job requested 6 GB host RAM after a 24 GB request was blocked by other allocations.
- **2026-09-10** TAU Slurm array `874192` completed six single-topic GDELT fits from the pinned Chronos-2 base, each using the common 21 controls plus one topic's share and tone. All six beat the seed-42 fine-tuned control on WQL. Fed ranked first at 0.686526 (1.28% below control), narrowly ahead of recession at 0.686796; their direct paired interval crossed zero. Repeat both with seeds 43 and 44 before selecting a topic. See `docs/GDELT_TOPIC_RESULTS.md`.
- **2026-09-15** Independent review of the first modeling round (section 13): pipeline sound; effective LoRA batch was one window (batch_size counts covariate rows); 2012-08-01 FOMC mis-dated (fixed same day); rate lags confirmed correct on re-check; VXN lag made an ablation arm (market_t0 group); screen results not yet reportable beyond "no reliable effect". Research focus restated as feature dynamics: leave-one-out, permutation importance over time, walk-forward folds.
- **2026-09-15** Zero-shot covariate ladder (`docs/PRETRAINED_LADDER_RESULTS.md`, 9 rungs x 5 folds x 50 windows, inference only): all covariate effects within ±0.7% of the 21-feature control; 42 features significantly worse pooled (+0.0045 WQL); calendar flags as past-only beat them as known-future; Fed tone the only news input with a consistent small gain; Gaussian baseline wins the 2022 bear year; direction equals the up-day base rate everywhere. Conclusion: in-context use of covariates is weak, so any news effect must come from fine-tuning. Also found that the per-window normalization in `summarize_news_ablation.py` inflates quiet weeks; fold-level normalization used instead.
- *(next: fix batch size and the two data issues, then rerun control and finalists with seeds 42-47; permutation importance on the all-external model; walk-forward folds)*
