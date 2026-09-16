# Ten-minute presentation

Slides 1-12 total 10:00. Slides 13-16 are optional backups. Times are rehearsal targets.

## 1. Can news improve QQQ forecasts?

Elad - 45 seconds

Our project asks whether broad news narratives add information about QQQ beyond market prices and scheduled events. QQQ tracks the Nasdaq-100, so our unit of prediction is an index-level daily return, rather than the reaction of a single company to one headline. We chose a pretrained time-series model, Chronos-2, and compared the same model with different inputs, before and after adaptation. The first experiments suggested small gains. The larger study asks whether those gains persist across years and what the model changes during training. Our conclusion is deliberately scoped to the features and training budget that we tested.

Sources: https://arxiv.org/abs/2510.15821; https://arxiv.org/abs/2106.09685; docs/REPORT_DECISIONS.md

## 2. Daily returns and forecast uncertainty

Elad - 75 seconds

A time-series forecaster uses an ordered history to predict future values. We supply up to 512 trading sessions and ask for five separate daily log returns at once. This is not one cumulative weekly return. After five sessions, we insert all five actual observations and issue the next forecast. The scored blocks do not overlap. Each day has 21 predicted quantiles, including a median and the 10th and 90th percentiles. The central band should contain about 80 percent of outcomes over repeated forecasts. This real example was issued after June 27, 2025. All five forecasts share that cutoff. We evaluate the quantiles with weighted quantile loss, or WQL. Lower is better. WQL measures a forecast distribution, so a value of 0.69 is not 69 percent accuracy. We separately report direction and coverage. Prior financial evaluations motivate this distinction between general forecasting capability and reliable equity-return prediction.

Sources: models/news-ablation-869989/gdelt_fed-seed42/validation_predictions.csv; scripts/compare_chronos2_validation.py; https://arxiv.org/abs/2511.18578

## 3. Chronos-2

Tom - 60 seconds

Chronos-2 is a pretrained time-series foundation model. The checkpoint has about 119.5 million base parameters, conventionally rounded to 120 million. We did not pretrain that model ourselves. It takes numeric time series, not the raw headlines. Each covariate becomes another series in a group with the QQQ target. Time attention processes history and group attention exchanges information across those series. The model scales observed context, forms patches of 16 observations, and produces quantile forecasts. Our context is capped at 512 sessions. Known future calendar values can be supplied, while future news and prices remain masked. The published model learns covariate relationships in pretraining. Whether those relationships transfer to financial news is our empirical question, not an assumption that we can settle from the architecture.

Sources: https://arxiv.org/abs/2510.15821; scripts/build_project_report.py; configs/walk_forward.json

## 4. LoRA adaptation and training windows

Tom - 60 seconds

LoRA learns small low-rank weight updates while freezing the base checkpoint. Here that means roughly 1.2 million trainable adapter parameters, rank eight, and a learning rate of ten to the minus five. Every experiment starts from the same original weights. During training, the model sees sampled historical windows and their future return targets. Test observations never enter the fitting call. There is an important batching detail: Chronos counts target and covariate series, not independent forecast windows. Our first batch argument of eight yielded effectively one complete window per update. The later runner explicitly requests eight windows, which is 176 series rows for the 21-feature control. The larger study uses 4,000 sampled windows per fit. We freeze the adapter during evaluation. Because other settings and periods also changed, we cannot attribute every difference between rounds to batching alone.

Sources: https://arxiv.org/abs/2106.09685; scripts/compare_chronos2_validation.py; configs/walk_forward.json

## 5. Public data and the availability rule

Elad - 60 seconds

The archived data inventory spans about twenty years and contains 77 columns, including the date, target and auxiliary raw fields. We select 21 control covariates or 42 in the full external setup. The sources include QQQ and related markets, rates, release calendars, uncertainty indices, FOMC statements, SEC activity, and six GDELT topics. GDELT provides news share and average tone. It limits the comparable news-training history to 2017 onward. The central data rule is availability, not the period a number describes. A July CPI observation cannot be used on July first if it is published in August. We align CPI vintages to release dates and apply source-specific delays to other inputs. Missing news stays masked. These rules reduce look-ahead, although revised series and incomplete historical vintages remain limitations.

Sources: docs/DATA_NOTES.md; configs/news_ablation.json; https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

## 6. Two completed experiment rounds

Elad - 60 seconds

The first round adds one source or one topic to a fixed control and evaluates 250 days from mid-2025 to mid-2026. It contains eighteen fits, with seed repetitions only for selected configurations. The later round expands to four calendar years, 2022 through 2025. Each year trains on earlier data beginning in 2017, then scores fifty five-day windows. The ladder starts with the target and progressively adds QQQ transforms, calendars, markets, uncertainty, statement tone and news. Three seeds cover every trained configuration. There are 108 fits including the matched four-window controls. These are 1,000 distinct test days, not three times as many observations because there are three seeds. We compare original and adapted weights on identical dates. A separate swap test exchanges one feature history between evaluation windows and measures the resulting change in loss.

Sources: configs/news_ablation.json; configs/walk_forward.json; scripts/permutation_importance.py

## 7. The initial source gains were small

Elad - 30 seconds

The first screen identified candidates rather than a reliable winner. Fed news improved WQL by 1.28 percent against the fine-tuned control in seed 42. Across three seeds, all-external improved the mean by 0.60 percent, but its uncertainty interval included no gain. EPU and EMU lost their first-seed advantage on repetition. The larger study was designed to test whether those small effects carried across years with a better specified training budget.

Sources: docs/GDELT_TOPIC_RESULTS.md; docs/NEWS_ABLATION_RESULTS.md

## 8. Overall adaptation gains remain unresolved

Tom - 45 seconds

This chart uses the cumulative ladder from the four-year study. Each point averages the four calendar-year WQL scores, and the trained points also average three seeds. Across the seven standard rungs, the gain from LoRA is small. Every pooled confidence interval for adapted minus pretrained loss includes zero. That does not prove the effects are exactly zero. It means this experiment does not establish a reliable aggregate advantage. We keep the 42-feature model separate on the next slide because it requires a different batch recipe and an identically trained control.

Sources: docs/results/walk_forward/wql_pretrained.csv; docs/results/walk_forward/wql_fine_tuned.csv; docs/results/walk_forward/finetune_gain.csv

## 9. All external does not outperform matched control

Tom - 35 seconds

The full 42-feature model uses four windows per step and a thousand steps. We trained a control with exactly that recipe. All-external improves from about 0.704 to 0.699 with adaptation, but the matched control is about 0.698. The difference is positive 0.00126, and the longer-block confidence interval spans zero. The result is no demonstrated extra value from the external features. It is not evidence that training has no effect at all: training improves the full model relative to its own starting point.

Sources: docs/results/walk_forward/wql_pretrained.csv; docs/results/walk_forward/wql_fine_tuned.csv; final report/evidence.json

## 10. Input-importance patterns change relatively little

Tom - 35 seconds

These are paired swap effects for the all-external model in calendar 2025. A negative bar means that exchanging the feature history improved the score. The major patterns persist after adaptation. Across all six checkpoint comparisons, the importance-profile correlation lies between 0.91 and 0.98. Most news effects are small. We should not call the profiles identical, or conclude that the network literally reads an input backwards. This is a prediction perturbation test, and correlated features and unusual donor histories limit its interpretation.

Sources: docs/results/permutation_finetuned/runs/; scripts/permutation_importance.py; final report/evidence.json

## 11. Fine-tuning widens the forecast intervals

Elad - 35 seconds

The clearest change is forecast dispersion. Across all 108 trained cells, intervals widen by about 7.8 percent and aggregate 80 percent coverage rises from 77.7 to 80.1 percent. The chart shows the seven standard rungs by year. Coverage increases in every year, although increasing coverage does not always improve calibration: 2025 was already near the target. Post-hoc analysis finds that adaptation helps in volatile weeks and hurts in quiet weeks. That supports a widening explanation, but it does not prove that widening is the only thing learned.

Sources: docs/results/walk_forward/coverage_direction.csv; final report/evidence.json; docs/FINETUNED_LADDER_RESULTS.md

## 12. Conclusions and the next experiments

Elad - 60 seconds

Across four years, the tested news groups have not established reliable incremental value beyond the market control. At this LoRA budget, forecast intervals widen while the main input-importance patterns remain broadly similar. The lesson is to measure training windows explicitly and to evaluate small source gains across periods, seeds, and uncertainty analyses. This is a bounded result about one ETF, numeric news summaries, and this recipe. It is not proof that news contains no information or that all foundation models fail. The next experiments are a controlled rerun of individual topics, daily one-step forecasts, and a calibrated-scale or supervised quantile baseline. We should select settings on development data and then freeze the model for an untouched prospective test. The model also beats our simple Gaussian baseline on pooled loss, so lack of a demonstrated news benefit should not be confused with lack of all predictive skill.

Sources: final report/evidence.json; docs/results/independent_review_2026-09-15.md

## 13. Backup: the original topic comparison

Backup

All values are from the original five-day, seed-42 screen. The all-external model includes sources beyond GDELT. The later cumulative ladder does not replace a matched rerun of each individual topic. Direction must be compared with the 143/250 always-up rate.

Sources: docs/GDELT_TOPIC_RESULTS.md

## 14. Backup: batch correction and comparison limits

Backup

For the 21-feature control, the newer effective batch has 8 times 22 equals 176 series rows. For 42 external features, the four-window recipe has 4 times 43 equals 172 rows. The old argument of eight resulted in effectively one full group per update. The later runs differ in more than batch size: training steps, test periods, feature snapshot, and Transformers/PEFT versions also change. Therefore the statement that the batch bug caused all original improvements is not isolated by these experiments.

Sources: scripts/compare_chronos2_validation.py; configs/walk_forward.json; docs/results/walk_forward/cells/

## 15. Backup: gains depend on realized volatility

Backup

Each observed week first averages seeds and rungs 3 through 6. We then divide the 200 distinct weeks into terciles using realized mean absolute daily returns. The quiet group has 67 weeks, the middle group 66, and the volatile group 67. Average WQL change is positive 0.0090, negative 0.0021, and negative 0.0103. The analysis is post-hoc and uses future realized movement, so it is not a forecast-time decision rule. It is consistent with wider intervals helping during volatile outcomes and hurting during quiet outcomes.

Sources: docs/results/walk_forward/gain_by_window_volatility.csv; final report/evidence.json

## 16. Backup: evidence limits and Gaussian baseline

Backup

The final paper recomputes three contrasts with five-window circular blocks drawn within each year and 5,000 bootstrap replicates. The fine-tuned market control beats the Gaussian pooled by 0.01088 WQL, with interval negative 0.02269 to negative 0.00106. Its 2022 point estimate is worse. The all-external versus matched control interval includes zero. All these intervals are exploratory and not multiplicity-adjusted. We have one ETF and no untouched prospective test. The base checkpoint publication follows most historical folds, so the adaptation split does not establish absence of pretraining overlap. The original VXN scramble effect is concentrated in five windows and cannot identify the model mechanism.

Sources: final report/evidence.json; docs/results/independent_review_2026-09-15.md; https://arxiv.org/abs/2510.15821; https://arxiv.org/abs/2106.09685; https://arxiv.org/abs/2511.18578