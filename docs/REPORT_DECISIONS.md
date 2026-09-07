# Decisions and ideas for the final report

Running log of the design decisions, evidence, and open ideas that the final report should
state. Each entry says what was decided, why, and what to write. Update this file whenever a
decision is made or reversed; the changelog at the end records when.

Group 13: Tom Weitman, Elad Kulman. Workshop on Deep Learning.
Companion documents: `docs/DATA_NOTES.md` (data conventions) and the covariate research
brief (published 2026-09-07; source notes in the session that produced it).

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
contribution as (a) a leak-free, publication-aware feature table, (b) a covariate ablation
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
on NYSE trading days from 2006-01-04 to the present (5,200 rows as of 2026-09-04).

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
| Each covariate becomes an extra series in the target's attention group; pretraining groups were small | **Target 10 to 20 past-only covariates per run and ablate.** The full table has 54 so groups can be swapped in and out; the `core` profile has about 20. |
| Every series is standardized over its own context window and passed through asinh | **Difference drifting levels** (rates, spreads, volatility indices, prices); keep bounded ratios, trailing z-scores and regime levels. |
| Known-future covariates are supported natively and are the strongest documented channel | **Build a calendar of scheduled events** (FOMC, CPI, NFP, earnings, month-end, opex) and pass it through `known_covariates_names`. |
| `predict_df` requires regularly spaced timestamps | Pass `freq="B"` or reindex; trading-day gaps otherwise raise. |
| Fine-tuning on a single short series may not beat zero-shot (official guidance) | Zero-shot is rung 0 of the ladder and every fine-tuned run must beat it. Pin `chronos-forecasting>=2.3.1` (v2.1.0 fixed a past-covariate masking bug). |

---

## 4. Data sources: chosen, and why

### 4.1 Kept from the original pipeline
- **QQQ OHLCV** (yfinance) and derived features. Raw price levels are no longer model inputs; instead: overnight gap, Parkinson range volatility (1-day and 22-day), 20-day volume z-score, 21/63-day momentum, distance to 50/200-day averages. `return_1d` and `return_5d` were dropped as linear duplicates of the target.
- **FRED rates**: fed funds, 2y, 10y, 10y-2y. Entered as one-day basis-point changes plus the lagged slope level. 10y level dropped (slope = 10y - 2y exactly).
- **CPI**: rebuilt from ALFRED release vintages (see section 5).
- **FOMC statements**: scraper rewritten (94 to 168 statements); keyword counts superseded by a classifier (section 6).
- **SEC filings**: rebuilt with full history from 2006, foreign-issuer forms for TSMC, and an earnings calendar from 8-K Item 2.02.
- **GDELT**: kept from 2017 (API limit) with share-of-coverage normalization; outages left empty.

### 4.2 Added
| Source | Why it earned a place | Availability |
|---|---|---|
| **Daily Economic Policy Uncertainty and Equity Market Uncertainty** (Baker-Bloom-Davis, policyuncertainty.com, CC BY 4.0) | Longest free daily uncertainty series (1985); improves 1-day-ahead S&P realized-volatility forecasts (Liu & Zhang 2015); only trailing 30 days revised (verified by diffing vintages) | T+1 morning |
| **VXN** (Nasdaq-100 implied vol) and **VIX/VIX3M** term-structure ratio (CBOE) | VXN is the index-specific fear gauge; the term-structure slope is a documented short-horizon signal (Johnson 2017 JFQA) | same evening |
| **TLT and HYG returns** | Stock-bond correlation regime and same-day credit; FRED's spread series post a day later | close |
| **Relative returns** QQQ-SPY, IWM-SPY, SMH-QQQ | Raw SPY is a near-duplicate of QQQ (corr ~0.95); the spreads carry growth, size and semiconductor tilts | close |
| **Real yield (DFII10) and 5y breakeven (T5YIE)** | Real-yield shocks are the cleanest macro driver of long-duration growth stocks post-2020; Treasury-sourced so same-day on FRED | same afternoon |
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
| FRED H.15 rates (DFF, DGS2, DGS10, T10Y2Y) | Posted at 16:15 ET the *next* day: lagged one day. |
| Calendar-day news counts (EPU, GDELT) | Complete only after midnight: value dated T-1; weekend days roll into Monday as a mean. |
| Releases at or after 14:00 ET (FOMC statement) | Before the close, so day-T flag is legitimate for a close-to-close target. |
| Post-market earnings 8-Ks | Attached to T+1; pre-market to T (EDGAR acceptance timestamp). |
| Known-future flags | Only events on a published schedule before T (FOMC, CPI, NFP, earnings dates, calendar structure). |

**Evidence to cite.** `scripts/audit_processed_data.py --leakage` regresses each covariate on
the same-day return: no next-day-only column explains more than 2% of today's return, while
close-measured series (VXN change 54%, HYG 36%, overnight gap 35%) correlate as they should.
`scripts/validate_feature_table.py` runs 47 checks against known history (all pass).

**Write.** A dedicated "point-in-time alignment" subsection with this table. It is the part
of the pipeline most likely to be questioned and the easiest to get wrong.

---

## 6. Fed tone: classifier over keywords
**Decision.** Each of the 168 statements is split into sentences and classified
hawkish / dovish / neutral by a RoBERTa-large fine-tuned on the FOMC hawkish-dovish sentence
dataset (Shah, Paturi & Chava, ACL 2023). Per statement: hawkish share, dovish share, net
hawkishness, change versus the previous statement, and an exponentially weighted level
carried forward between meetings.

**Why.** The original six-word keyword list counted "inflation" as hawkish regardless of
context. The classifier reproduces the known cycle: net hawkish in 2006 to 2007, dovish
through 2009 to 2021 with a trough in 2020, hawkish in the 2022 to 2023 hiking cycle.

**Caveat to state.** The reference model (`gtfintechlab/FOMC-RoBERTa`) is gated on Hugging
Face; the run used an ungated model trained on the same dataset
(`LorenzoAleCon29/roberta-large-fomc-hawkish-dovish`), verified with probe sentences. Dissent
sentences at the end of statements are scored along with the policy text.

---

## 7. Feature transforms and the two profiles
**Decision.** Rates and spreads enter as basis-point changes; prices, VXN, VVIX, TLT, HYG as
log changes; VIX/VIX3M, VIX9D/VIX, distance-to-average and volume z-score as bounded levels;
CPI yoy, slope, VXN level, VXN-VIX as regime levels; EPU/EMU as log of trailing 7-day mean
(single weekend days are too noisy, see changelog). The builder writes a `groups.json` giving
each column a group (`qqq`, `market`, `rates`, `uncertainty`, `news`, `fed`, `sec`,
`calendar`) and a role (`target`, `past`, `known_future`, `raw`).

Two profiles: `full` (54 past + 15 known-future) for ablation; `core` (about 20 past + 7
known-future) as the recommended single run.

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
return forecaster can actually deliver. The existing comparison script already reports MAE,
bias, directional accuracy and 10/90 coverage, so this extends it.

**Charts to plan.** (1) Ladder table with deltas per rung. (2) Predicted 10 to 90 band
around FOMC/CPI days with and without known-future flags, overlaid on realized absolute
returns. (3) Narrative overlay: which GDELT topic shares were elevated on the days where the
model's uncertainty should have been higher.

---

## 10. Limitations to state honestly
- **GDELT covers 2017 onward only**; news columns are empty for 55% of the sample. Tone is a generic lexicon score with known noise (~55% key-field accuracy in audits).
- **Ticker survivorship**: the ten SEC companies are today's Nasdaq leaders, chosen with hindsight. Their filing and earnings flags describe the current index, not the 2006 index.
- **EPU** revises its trailing 30 days; training values for the last month of any vintage are not exactly point-in-time. Single-day values are noisy, hence the 7-day mean.
- **VXN and VIX3M before their CBOE files start** (2009) are backfilled from FRED, which posts next-day; for 2006 to 2009 those values were strictly available one day later than assumed.
- **Fed tone model** is a substitute for the gated reference model.
- **High-yield spread** dropped because FRED truncated its history; HYG (from April 2007) stands in.
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
- Environment: Python 3.11, `requirements.txt`; the tone model downloads about 1.4 GB once.
- Commits: data layer v2 `d19acf9` (2026-09-07), FRED/SEC completion `7db1129`, validation fixes `182cfa6`.
- Validation: `audit_processed_data.py --leakage` and `validate_feature_table.py` both pass on the committed table (5,200 rows x 77 columns).

---

## Changelog
- **2026-09-07** Research brief completed; Chronos-2 facts, source verdicts, leakage rules recorded (sections 1, 3, 4, 5).
- **2026-09-07** Data layer rebuilt: new fetchers, FOMC scraper fixed (94 to 168), tone classifier, calendar, publication-aware builder, no zero-fill policy (sections 4, 6, 7).
- **2026-09-07** FRED rebuilt with ALFRED-vintaged CPI; HY OAS dropped (FRED serves three years only). SEC rebuilt with full history, predecessor registrants for Alphabet and Broadcom, earnings 8-K items.
- **2026-09-07** Validation pass found and fixed: Monday-NaN bug in the as-of join (FRED weekend rows), EPU weekend-day outliers (switched to 7-day mean), undefined FOMC cycle before the first 2006 meeting (seeded with December 2005). 47 history checks pass.
- *(next: modeling runs; record rung results and any reversal of the decisions above here)*
