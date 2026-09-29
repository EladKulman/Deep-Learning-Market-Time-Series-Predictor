# 3  Data and information availability

The archived feature table contains 5,201 exchange sessions from January 2006 through September 4, 2026, with 2005 prices supplying indicator warm-up. Its 77 columns include the date and target, 55 past-covariate fields, 14 eligible future-calendar fields, and six raw fields. This is the available feature inventory, not the number of inputs in every experiment. GDELT availability limits all news-training comparisons to January 2017 onward. The selected control has 21 covariates and the all-external setup has 42, both excluding the target.

::: table ../tables/01_sources.csv widths=38,100,28 font=8.8
Source families in the executed study. Appendix B gives the exact selected fields and topic definitions.
:::

## 3.1  Publication time determines the usable date

Every feature is assigned to the first permitted exchange close after its information becomes available. A July CPI observation is not known on July 1: its release arrives later. The pipeline uses ALFRED release vintages and computes CPI growth within the available vintage. This prevents the reference month from exposing an unreleased value to a historical forecast.

H.15 series use the next publication-business-day 16:15 release and then the next eligible exchange close. SEC filings use UTC acceptance timestamps, including early closes. Complete daily news aggregates are shifted by a calendar day, then rolled to exchange sessions; weekend observations are averaged. CBOE-derived inputs receive a conservative one-session lag. Only scheduled calendar information enters the future-covariate channel. Future news, prices and realized returns are masked.

## 3.2  News construction and missing data

GDELT queries cover AI, semiconductors, the Federal Reserve, inflation, big-tech earnings and recession. Each topic contributes news share and average tone [[6]](#ref6). These are numeric summaries, not raw text embeddings or headlines. Queries overlap, including Nvidia in both AI and semiconductor searches. The original common sample has 13 missing sessions in each selected GDELT column (0.55%). Missing values remain masked rather than becoming zero news.

The statement scorer uses a pinned public RoBERTa FOMC classifier related to the task introduced by Shah, Paturi and Chava [[4]](#ref4). It summarizes 170 statements. Publication alignment does not solve every vintage problem: revised prices, some macro series, reconstructed schedules and retrospectively selected SEC firms remain limitations. The two experiment rounds record different feature-table hashes.
