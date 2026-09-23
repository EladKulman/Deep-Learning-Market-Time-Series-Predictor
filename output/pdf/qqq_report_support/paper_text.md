Can News Improve QQQ Forecasts?

Covariate Ablations, LoRA Adaptation andFour-Year Evaluation with Chronos-2

Elad Kulman and Tom Weitman

Group 13 - Workshop on Deep Learning - Tel Aviv UniversitySeptember 2026

Abstract. We study whether public news-derived time series improve probabilistic forecasts of QQQ daily log returns. A publication-aware data pipeline combines market, macroeconomic, calendar, uncertainty, disclosure and news-topic variables. An initial 18-fit source screen yields small candidate improvements over a market control, including a 1.28% weighted quantile loss reduction for Federal Reserve news in one seed. A subsequent study evaluates 108 LoRA fits across four calendar years with larger, explicitly measured training batches. In that study, the all-external model improves from 0.703724 pretrained WQL to 0.698957 after adaptation, but does not outperform its identically trained control (0.697701). Its paired difference is +0.001257 with a confidence interval spanning zero. Across the trained configurations, intervals widen by about 7.8%, while permutation-importance profiles remain broadly similar before and after training. These results do not establish reliable incremental news value under the tested recipe. They also do not establish that news is intrinsically uninformative or that Chronos-2 cannot use it. We provide reproducible comparisons, investigate the limits of input perturbation, and identify the need for an untouched prospective evaluation.

1  Introduction

Index-level returns aggregate responses to many firms, policy announcements and changing market conditions. The project asks whether broad news narratives supply information that is not already present in market prices and scheduled events. We use QQQ, an exchange-traded fund tracking the Nasdaq-100, as a single, consistently observed target. The objective is to evaluate daily return distributions and source contributions, rather than to attribute an index move to one headline.

The research questions are: RQ1, do news-derived features add value beyond a fixed market and calendar control? RQ2, does LoRA adaptation improve the same model supplied with the same features? RQ3, do input perturbations reveal a change in predictive reliance after adaptation? An accurate return distribution can represent uncertainty even when its median offers little directional discrimination.

Our contribution is an empirical study with explicit information-availability rules, a first-round source screen, a larger chronological evaluation, and diagnostic analysis of the saved forecasts. We use an existing foundation model and adaptation method. The report develops the experiments in our project presentation, with the distinction between promising initial rankings and evidence that survives broader evaluation made explicit.


---


2  Background and related work

2.1  Why a time-series model for an index?

A time series is a sequence of observations ordered in time. A forecasting model learns relationships between past observations and future outcomes while respecting that order. For QQQ, inputs can include return history, recent volatility and information known at the forecast origin. Randomly shuffling daily rows would destroy this timing and could allow future information into training. We therefore split chronologically and reveal new observations only after their dates have passed.

An index reflects many companies and common macroeconomic exposures. Topic-level news summaries offer a tractable hypothesis: market-wide narratives could affect the distribution of the next return even when they do not determine its sign. The hypothesis is incremental. News must improve forecasts beyond a control that already observes market conditions, rates and scheduled announcements. A time-series model is an appropriate way to test that hypothesis, but temporal structure alone does not make returns predictable.

2.2  From specialized models to foundation forecasting

Temporal Fusion Transformers combine temporal processing with variable selection and quantile prediction for multi-horizon forecasting [5]. PatchTST represents a time series as patches and uses Transformer processing, illustrating how architecture can exploit local temporal structure [9]. These are relevant methodological predecessors, not executed baselines in our project.

Chronos-2 provides a pretrained model with support for multiple related series and covariates through group attention [1]. This is useful experimentally: we can keep one base checkpoint fixed while varying which inputs it receives, then compare inference before and after adaptation. The model supplies a distribution through quantiles, so evaluation need not reduce to whether tomorrow is up or down.

Rahimikia, Ni and Wang report weaknesses of off-the-shelf time-series foundation models on financial excess returns and motivate domain-specific evaluation [3]. Their benchmark does not prove failure on every financial task. Our daily QQQ log-return study differs in target, inputs and scoring; it tests its own baselines and does not infer performance from general forecasting benchmarks.

2.3  News representation and parameter-efficient adaptation

GDELT provides query-dependent measures of news volume and tone [6]. Economic Policy Uncertainty measures reporting about uncertainty [7], whereas FOMC-language classification captures hawkish or dovish policy communication [4]. These measurements represent distinct hypotheses. An average tone score is neither the sentiment of investors toward QQQ nor a direct estimate of its expected return.

LoRA freezes the original weights and learns low-rank changes to selected matrices [2]. Its modest trainable parameter count makes repeated source comparisons practical on our available GPU. We use it to ask whether QQQ-specific training changes forecast quality or measured input reliance. The experiment does not assume that low-rank adaptation is sufficient to recover every useful relation in the news.

The resulting design separates three questions: whether adding inputs helps before training, whether adaptation helps with a fixed input set, and whether news still helps when both news and control models receive the same adaptation recipe. Answering only the second question would not establish the incremental value of news.


---


3  Data and information availability

The archived feature table contains 5,201 exchange sessions from January 2006 through September 4, 2026, with 2005 prices supplying indicator warm-up. Its 77 columns include the date and target, 55 past-covariate fields, 14 eligible future-calendar fields, and six raw fields. This is the available feature inventory, not the number of inputs in every experiment. GDELT availability limits all news-training comparisons to January 2017 onward. The selected control has 21 covariates and the all-external setup has 42, both excluding the target.


Source family | Selected representation | Role
QQQ and related markets | Returns, gaps, range volatility, volume, momentum, VXN, TLT, HYG | Past only
Rates and macroeconomics | Rate changes, curve slope, real yields, CPI growth | Past only
Event calendar | FOMC, CPI, payroll dates, month-end and option expiry | Known future
EPU and EMU | Log trailing seven-day uncertainty averages | Past only
FOMC statements | Classifier tone, change, smoothed state, elapsed time | Past only
SEC disclosures | Accepted filings and past earnings-event counts | Past only
GDELT topics | Normalized news share and average tone per topic | Past only
Source families in the executed study. Appendix B gives the exact selected fields and topic definitions.

3.1  Publication time determines the usable date

Every feature is assigned to the first permitted exchange close after its information becomes available. A July CPI observation is not known on July 1: its release arrives later. The pipeline uses ALFRED release vintages and computes CPI growth within the available vintage. This prevents the reference month from exposing an unreleased value to a historical forecast.

H.15 series use the next publication-business-day 16:15 release and then the next eligible exchange close. SEC filings use UTC acceptance timestamps, including early closes. Complete daily news aggregates are shifted by a calendar day, then rolled to exchange sessions; weekend observations are averaged. CBOE-derived inputs receive a conservative one-session lag. Only scheduled calendar information enters the future-covariate channel. Future news, prices and realized returns are masked.

3.2  News construction and missing data

GDELT queries cover AI, semiconductors, the Federal Reserve, inflation, big-tech earnings and recession. Each topic contributes news share and average tone [6]. These are numeric summaries, not raw text embeddings or headlines. Queries overlap, including Nvidia in both AI and semiconductor searches. The original common sample has 13 missing sessions in each selected GDELT column (0.55%). Missing values remain masked rather than becoming zero news.

The statement scorer uses a pinned public RoBERTa FOMC classifier related to the task introduced by Shah, Paturi and Chava [4]. It summarizes 170 statements. Publication alignment does not solve every vintage problem: revised prices, some macro series, reconstructed schedules and retrospectively selected SEC firms remain limitations. The two experiment rounds record different feature-table hashes.


---


4  Model, adaptation and forecast protocol

4.1  Foundation forecasting and related work

Chronos-2 supports forecasting with related series and covariates through group attention [1]. Its pretrained capability provides an inference-only reference before any QQQ adaptation. LoRA freezes base weights and trains low-rank updates [2]. Financial evaluations of time-series foundation models motivate testing domain-specific behavior directly rather than assuming that general forecasting performance transfers to daily equity returns [3]. Our single-ETF experiment is narrower than those benchmarks.

The pinned checkpoint contains 119,477,664 base parameters. Its configuration has 12 layers, width 768, 12 attention heads and 16-step patches. Each supplied numeric series is scaled over observed context and transformed with asinh. The model returns 21 quantiles at levels 0.01, 0.05, 0.10 through 0.95 in 0.05 increments, and 0.99. We cap historical context at 512 exchange sessions.

4.2  LoRA and the training batch

The adapters target attention query, key, value and output projections and the output patch projection. They contain 1,206,912 trainable parameters, approximately 1% of the combined model, with adapter dropout zero. Query/key/value/output projections occur in both attention sublayers of each of the 12 blocks; counting only one would understate the adapter size. A uses the default random Kaiming-uniform initialization and B starts at zero, so the initial update is zero [10]. Each run starts from the same original checkpoint, samples historical training slices, and optimizes quantile loss. We do not continue training a previous control adapter. No validation examples are supplied to fit(), and the final scheduled checkpoint is used without validation-based early stopping.

The Chronos training batch counts target and covariate series. In the first screen, a batch argument of eight admitted effectively one complete QQQ window with its covariates. The later runner requests a number of windows explicitly: eight windows with 21 covariates requires 176 series rows. This corrects an under-specified training budget. It does not by itself prove that the earlier performance differences were caused by batch size.

4.3  What a five-day forecast means

At an origin t, the model receives observed history through t and forecasts five separate daily log returns, not one cumulative weekly return. Every lead uses the same information cutoff. After five exchange sessions, we refresh history with all five actual observations and issue the next forecast. Historical inputs overlap substantially; the five-day blocks of scored dates do not overlap. No predicted return replaces a realized observation. We freeze adapter weights throughout each evaluation year.

One first-round Fed-topic forecast. The July 4 holiday shifts the fifth scored session to July 7. Later leads do not observe intervening news or returns. Inclusion inside a broad interval and a correct directional prediction are different events.


---


5  Experimental design and scoring


Property | Round A: source screen | Round B: chronological ladder
Training starts | January 3, 2017 | January 3, 2017
Training ends | June 27, 2025 | Last session before each test year
Training sessions | 2,133 | 1,259 to 2,012, expanding by year
Scored interval | June 30, 2025-June 26, 2026 | Calendar years 2022, 2023, 2024, 2025
Scored days | 250 in 50 windows | 250 per year, 1,000 distinct days
Completed LoRA fits | 18 | 108 including matched controls
Updates and windows | 200 updates, about 1 window each | 500 x 8; all-external and matched control 1,000 x 4
Seeds | 42; 43/44 for three source setups | 42, 43, 44 for every trained setup
The rounds answer related questions but do not isolate one training change. Both use context 512, five-step horizon, stride five and learning rate 1e-5.

Round A adds one source family or one GDELT topic to the same 21-feature control. Round B follows a cumulative ladder: target only; six QQQ transforms; six calendar flags; nine market/rate fields; EPU/EMU; FOMC tone; GDELT Fed and recession; then all remaining external fields. The resulting counts are 0, 6, 12, 21, 23, 27, 31 and 42 covariates. Seven rungs use eight windows for 500 steps (84 fits). The 42-feature rung and a matched control use four windows for 1,000 steps (24 fits), because eight all-external windows exceed the GPU memory budget.

A separate pretrained-only exploration includes four calendar years plus the original mid-2025 to mid-2026 interval. That fifth interval overlaps calendar 2025. Its intermediate source arms are not all cumulative. We use the four-year walk-forward arm for the main paired pretrained/LoRA comparison, avoiding both mixed feature definitions and treating overlapping intervals as independent years.

5.1  Weighted quantile loss

WQL averages the asymmetric quantile loss across the 21 levels and normalizes by realized absolute returns. Lower is better; 0.70 does not mean 70% accuracy. The implementation scores each requested quantile [8]. We also report median MAE, direction, and the coverage and width of the nominal 80% and 98% intervals. A zero-mean Gaussian with trailing 60-session sample standard deviation supplies quantiles without news or training. It forecasts daily, not cumulative, uncertainty.

Reading a reduction. Relative WQL reduction is 100 x (control WQL - news WQL) / control WQL. A decrease from 0.695413 to 0.686526 is an absolute loss reduction of 0.008887, or 1.28% relative. A 1% reduction from 0.70 would give 0.693. It describes a better quantile score on the same outcomes, not a one-percentage-point increase in direction accuracy or a financial return.

5.2  Pairing and uncertainty

We pair forecasts on the same realized dates and average seeds before comparing windows. Four-year estimates average fold-normalized losses, giving each calendar year equal weight. Published ladder intervals use 5,000 window-bootstrap draws. For three headline contrasts we additionally resample circular blocks of five consecutive forecast windows separately within each year (5,000 draws, seed 20260916). These intervals remain exploratory and unadjusted for multiple comparisons; they condition on observed years and seeds.


---


6  Initial news-source screen

The first round established end-to-end GPU execution and generated candidate source rankings. A separate five-update smoke test produced only 15 outcomes; it verified training and checkpoint reload, not forecasting skill. Table 3 summarizes the substantive single-topic comparison. All configurations retain the common market/calendar control.


Added news | Fine-tuned WQL | Reduction vs control | Correct direction
None | 0.695413 | -- | 140/250
Federal Reserve | 0.686526 | 1.28% | 144/250
Recession | 0.686796 | 1.24% | 144/250
Semiconductors | 0.687499 | 1.14% | 144/250
AI | 0.687580 | 1.13% | 143/250
Inflation | 0.688131 | 1.05% | 142/250
All external sources | 0.688303 | 1.02% | 145/250
Big-tech earnings | 0.688750 | 0.96% | 142/250
Round A, seed 42. The all-external arm includes non-GDELT sources. All-GDELT alone is a different 33-feature arm (WQL 0.692348).

Fed has the lowest point estimate, 1.28% below the fine-tuned control; recession differs by only 0.000269 WQL. A paired interval for their difference includes zero. The six topics each have one seed, so their apparent ranking does not demonstrate a uniquely informative narrative. Longer blocks in the first-round audit weaken several topic-versus-control comparisons as well.

6.1  Replication changes the interpretation


Source setup | Mean LoRA WQL | Seed SD | Seeds beating control
Control | 0.693582 | 0.001656 | --
EPU / EMU | 0.693473 | 0.005212 | 1 / 3
All external | 0.689439 | 0.003583 | 2 / 3
Round A seed repetitions. Each seed evaluates the same 250 market outcomes; these are not 750 independent days.

All-external improves mean loss by 0.60% relative to control, but its paired interval includes no gain. Its own pretrained WQL is 0.690730, so LoRA adds only a 0.19% relative improvement. EPU/EMU looks strong in the first seed and loses that advantage in repetitions. These observations motivated a larger training budget and evaluation across different years.

Direction is particularly weak evidence of skill in this interval: 143 of 250 actual returns are non-negative. Always predicting up scores 57.2%, versus 57.6% for Fed and 58.0% for all-external at seed 42. The Gaussian median is zero, which the scoring code counts as up. The result therefore supports candidate probabilistic comparisons, not a dependable trading rule.

6.2  Why the later study is not a batch-only experiment

The original API batch argument gave much less training than intended. Round B increases both windows per update and update count, evaluates different periods, changes the feature snapshot and uses updated Transformers/PEFT versions. Its evidence can overturn confidence in the original ranking, but cannot identify a batch-size defect as the sole causal explanation. A controlled rerun on the original snapshot and dates would be required for that claim.


---


7  Four-year results

Round B cumulative ladder. Dotted curves use the original checkpoint; solid curves average three LoRA seeds. Each year scores the same 250 dates across rungs. The all-external arm is shown separately in Table 5 because its training recipe requires a matched control.

Across the seven standard rungs, pooled fine-tuned-minus-pretrained WQL ranges from -0.001813 to +0.000371. Every reported pooled interval includes zero. This is insufficient evidence of a reliable adaptation benefit at these settings, rather than an equivalence test proving an exactly zero effect. Small changes in individual years are exploratory among many examined contrasts.


Model and recipe | 2022 | 2023 | 2024 | 2025 | Mean
Pretrained, all external | 0.6977 | 0.6947 | 0.7026 | 0.7199 | 0.7037
LoRA, all external, 4 x 1,000 | 0.6904 | 0.6881 | 0.7026 | 0.7147 | 0.6990
LoRA, control, 4 x 1,000 | 0.6885 | 0.6855 | 0.7034 | 0.7135 | 0.6977
LoRA, control, 8 x 500 | 0.6865 | 0.6843 | 0.7021 | 0.7139 | 0.6967
Matched all-external comparison. The four-window and eight-window recipes each sample 4,000 training windows, but differ in optimizer-update granularity.

All-external reduces WQL by 0.004766 relative to its own pretrained arm. Yet it remains +0.001257 above the identically trained control, with a five-window-block 95% interval [-0.00095, +0.00369]. Its advantage from adaptation also remains unresolved under that block analysis: [-0.01013, +0.00032]. Thus the larger input set has not established incremental predictive value over the market control.

The first-round mid-2025 to mid-2026 interval favored the pretrained all-external setup. Most additional calendar-year comparisons show the opposite direction. Period dependence and a small initial training budget make the original improvement a development observation, not a general result. The later cumulative GDELT arm combines Fed and recession with other sources; it does not independently replicate all six single-topic fits.


---


8  Input perturbation and predictive reliance

The swap test replaces one feature's context with the same feature from another evaluation window and re-forecasts without retraining. For known-future calendar inputs it also replaces the future calendar block. Three donor assignments are averaged. The loss difference is scrambled minus original: a positive value indicates that the original alignment was helpful under this intervention; a negative value indicates that the swap improved the score.

This is an offline diagnostic. Donor windows can be later than the recipient origin, so the perturbed series is not a feasible historical trading input. Its history retains temporal structure but loses its original relationship to the target and other covariates. It is not literal feature removal, a causal intervention on the market, or a direct measure of internal semantic understanding.

Selected features of the 42-covariate pretrained model in the original evaluation interval. Means and exploratory 95% window-bootstrap intervals use locally normalized window WQL, as in the saved swap experiment. These values are not directly interchangeable with the fold-normalized ladder deltas.

Swapping VXN change lowers mean locally normalized WQL by 0.0134 in the control. Its median effect is much smaller (-0.00234), and five windows account for approximately 94% of the signed total. In the 42-feature setup, the mean effect is -0.0138. A same-session VXN variant has a similar effect, which weakens the hypothesis that a one-session lag alone explains the result. The perturbation may alter forecast dispersion ahead of shocks; the evidence does not establish that the model reads volatility backwards.

8.1  News effects and adaptation

The 42-feature pretrained test has one small positive news result: GDELT Fed average tone, +0.00175 WQL, with an unadjusted interval excluding zero. Forty-two feature tests create a multiple-comparison problem. This is also a different variable from the FOMC-statement classifier used in another ladder arm, so those two findings do not independently confirm the same signal. Correlated features can share information and make individual perturbations appear weak.


---


8.2  Does adaptation change measured input reliance?

Paired perturbations, all-external model, calendar 2025, seed 42. Ten selected inputs illustrate the dominant market effects and representative news families. The correlation of 0.908 is computed over all 42 features. Negative values mean the scrambled forecast had lower loss; the plot does not show feature deletion or retraining.

Six fine-tuned checkpoints cover the control, FOMC rung and all-external rung in 2022 and 2025, using seed 42. Their matched pretrained/LoRA importance-vector correlations range from 0.908 to 0.978. Dominant effects persist, but profiles are not identical: the two-year yield change moves toward zero in both tested 2022 control/FOMC cells. The supported conclusion is limited change in the measured importance pattern at this budget, rather than proof that training cannot alter covariate use.


Input setup | 2022 correlation | 2025 correlation
Market/calendar control | 0.948 | 0.970
Control + uncertainty + FOMC | 0.963 | 0.978
All external | 0.938 | 0.908
Pearson correlation between pretrained and adapted mean importance vectors. Each cell uses seed 42. High correlation describes the profile; it does not establish equality of individual feature effects.

A near-zero swap effect can arise because the model makes little use of the feature, because another correlated feature carries similar information, or because helpful and harmful effects cancel across origins. Whole-context swaps may also create unlikely combinations of market and news histories. The result therefore supports limited measured change under this diagnostic, not the categorical claim that the model cannot understand news.

A stronger follow-up would combine group-level swaps, retrained feature ablations and conditional donor matching. Agreement across those tests would make an input-reliance conclusion more convincing. All donors and normalization choices should be held fixed between the pretrained and adapted arms.


---


9  Calibration, regimes and baseline performance

Left: average nominal 80% coverage for the seven standard rungs, with equal rung weight. Right: post-hoc mean LoRA-minus-pretrained WQL across rungs 3-6 after averaging their repeated measurements per observed week. The terciles contain 67, 66 and 67 distinct weeks, not 800 independent market observations.

Across all 108 trained cells, including the matched-recipe reruns, mean 80% coverage increases from 77.69% to 80.09%, and mean interval width increases by 7.80%. Those cells repeatedly evaluate the same market dates, so this aggregate describes model behavior rather than providing an independent-observation count. The standard-rung plot shows a rise in every year, but higher coverage is not always closer to 80%: in 2025 it moves from approximately 80.3% to 81.8%. Tail coverage also remains imperfect.

Widening is consistent with the post-hoc loss pattern. After averaging seeds and rungs 3-6 within each observed week, fine-tuning raises WQL by 0.0090 in the quiet third and lowers it by 0.0103 in the volatile third. The group assignment uses realized future absolute returns, so this analysis explains realized performance; it cannot select a trading action in advance. We do not attach independence-based intervals to repeated rung observations.

These findings support a dispersion-change hypothesis, not a complete identification of an unconditional scale mechanism. A forecast-time regime model or a simple calibrated widening baseline, fitted only on training data, would be needed to test whether LoRA adds more than interval rescaling. In particular, improvement in one realized-volatility group and deterioration in another can occur without revealing what information the network used.

9.1  Comparison with a simple forecasting solution


Contrast | Mean WQL difference | Five-window-block 95% interval
All-external LoRA minus matched control | +0.001257 | [-0.00095, +0.00369]
All-external LoRA minus own pretrained | -0.004766 | [-0.01013, +0.00032]
Control LoRA minus Gaussian | -0.010878 | [-0.02269, -0.00106]
New report sensitivity analysis: seeds averaged, equal year weights, circular blocks within years, 5,000 draws. Negative differences favor the first model. Intervals are exploratory and not multiplicity-adjusted.

The fine-tuned market control beats the Gaussian on the pooled four-year score, while its 2022 point estimate is worse. Therefore, absence of demonstrated incremental news benefit does not mean that the complete model has no forecasting skill. This baseline is deliberately simple; no supervised quantile regression, tree model, or learned volatility baseline was executed in this study. Comparisons with those solutions remain future work.


---


10  Discussion, limitations and future work

10.1  What the experiments answer

The initial screen suggested small source-specific improvements. The broader study does not establish an incremental benefit from the tested news groups across four calendar years under a common recipe. All-external adaptation reduces its own pretrained loss but does not surpass a matched market control. Permutation tests show broadly persistent input-importance patterns, and interval widening is a clearer measured effect than an improvement in overall WQL. These findings answer RQ1 and RQ2 with limited positive evidence and RQ3 with a qualified observation of similarity.

The relevant claim is about this representation, target, evaluation protocol and training budget. Numeric topic averages may discard event timing, novelty or firm-specific relevance that richer text representations retain. Shared topics may be redundant with market features. The model may require a different adaptation procedure. None of those explanations has been isolated, and a nonsignificant contrast is not proof of no possible news information.

10.2  Limits on generalization and causal interpretation

The study uses one ETF, a five-day direct forecast, limited training seeds and a fixed learning rate/rank. Calendar-year folds provide distinct evaluation outcomes, but still form one market history. The first-round evaluation overlaps part of the later development history. All examined periods informed model choices, so none is an untouched final test. The October 2025 publication date of Chronos-2 follows most historical fold dates; the LoRA split does not rule out overlap with foundation-model pretraining or establish a historically deployable backtest.

Point-in-time feature alignment limits look-ahead in the data pipeline but cannot recover unavailable historical vintages. The FOMC classifier is retrospective, SEC company selection reflects current leaders, and some series may contain revisions. Missing-news intervals remain masked. The second round also changes the feature snapshot and some library versions, which prevents attributing the change from the first round to the batch correction alone.

Uncertainty intervals depend on normalization, pairing and resampling. We distinguish fold-normalized ladder deltas from locally normalized permutation deltas, average seed replicas before resampling, and show a longer-block sensitivity check for headline comparisons. Unadjusted exploration over many rungs, years, features and conditions can produce chance discoveries. The positive Fed-tone permutation result and single-year adaptation gains therefore require replication.

10.3  Next experiments

First, repeat all six individual topics with explicit window batches on the same snapshot and evaluation dates to separate recipe changes from period changes. Second, align training and evaluation to a one-day horizon with daily origins if the operational goal is tomorrow's return. Compare share-only, tone-only and a small topic combination against matched controls. Third, compare LoRA with supervised quantile and calibrated-scale baselines, using an inner chronological validation split to select a limited budget of training settings. Use a controlled budget comparison of longer LoRA training and full fine-tuning; add event novelty and firm-relevant text representations before expanding the input count. Finally, freeze the selected specification and record forecasts before outcomes arrive in a new prospective block. Event-conditioned calibration should use conditions available at the forecast origin.

10.4  Conclusion

Our completed experiments do not demonstrate reliable incremental news value for QQQ under the tested Chronos-2 LoRA recipe. They do demonstrate why an apparent source ranking needs replication, why training budgets must count actual forecast windows, and why aggregate loss should be read alongside calibration and perturbation diagnostics. These findings support a bounded empirical conclusion; they do not establish that foundation models cannot use financial news.


---


References

[1] Ansari, A. F., et al. (2025). Chronos-2: From Univariate to Universal Forecasting. arXiv:2510.15821. Source.

[2] Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. Source.

[3] Rahimikia, E., Ni, H., and Wang, W. (2025). Re(Visiting) Time Series Foundation Models in Finance. arXiv:2511.18578. Source.

[4] Shah, A., Paturi, S., and Chava, S. (2023). Trillion Dollar Words: A New Financial Dataset, Task &amp; Market Analysis. ACL, 6664-6679. Source.

[5] Lim, B., Arik, S. O., Loeff, N., and Pfister, T. (2019). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. arXiv:1912.09363. Source.

[6] GDELT Project (2017). GDELT DOC 2.0 API Debuts. API documentation. Source.

[7] Baker, S. R., Bloom, N., and Davis, S. J. (2016). Measuring Economic Policy Uncertainty. QJE, 131(4), 1593-1636. Source.

[8] Amazon Science. Chronos Forecasting, version 2.3.2, source implementation. Source.

[9] Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2022). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. arXiv:2211.14730. Source.

[10] Hugging Face. PEFT LoRA configuration and initialization documentation, version 0.21.0. Source.

Appendix A  Execution record and reproducibility


Round | Slurm jobs | Completed work
Infrastructure | 869927 | Five LoRA updates, 15 outcomes, adapter reload
A: source screen | 869989; 871482; 874192 | 18 fits, 200 updates per fit
B: ladder | 896510; 896540; 896541 | 84 fits, seven rungs x four years x three seeds
B: matched recipe | 896751; 896752 | 24 fits, all-external and control, 4 x 1,000
B: diagnostics | 896501; 896762 | 32 pretrained cells; six paired checkpoint perturbation tests
The 126 substantive LoRA fits comprise 18 first-round fits plus 108 later fits. Additional pretrained exploration and perturbation inference ran on MPS; these are not additional training seeds.

Both rounds pin model revision 29ec3766d36d6f73f0696f85560a422f50e8498c. Round A data SHA-256 is e0c8d48b38059eb70d9f91931bff68678dbcc46aeadab1502ded1b9ee11df037; Round B is b3beb046933b15f1f75c098bfa6df30af4106395302ea4d8b68711322a262e34. Training used an NVIDIA RTX 2080 Ti (11 GB) through Slurm. Both GPU runtimes record Chronos 2.3.2 and PyTorch 2.11.0+cu128. Transformers/PEFT change from 5.16.1/0.20.0 to 5.17.0/0.21.0. Later metadata records dirty working trees, making source hashes material to reproduction.

Round A predictions and adapters reside in models/news-ablation-869989/. Round B local exports in docs/results/walk_forward/ include per-window loss summaries and 140 cell metadata/metric pairs (108 trained, 32 pretrained-only). Full later adapters and per-day trained predictions are not in that local export. The final report analyzer checks cell metrics against window means to 3.4e-16 and records source hashes in output/pdf/qqq_report_support/evidence.json. Historical recipes come from saved run metadata; the current configuration files have since changed. No new model was trained to prepare this report.


---


Appendix B  Exact inputs and source definitions


Control family | Exact fields
QQQ (6) | overnight_gap, parkinson_vol_1d, parkinson_vol_22d, volume_z_20d, mom_21d, dist_200dma
Market (5) | vxn_log_chg, vxn_minus_vix, vix_term_ratio, tlt_log_ret, hyg_log_ret
Rates / macro (4) | d_dgs2_bp, t10y2y_lag1, d_dfii10_bp, cpi_yoy
Calendar (6) | is_fomc_day, is_cpi_day, is_nfp_day, days_to_fomc, days_to_month_end, is_opex_week
The control contains 15 past-only and six known-future fields. Every experiment also includes the target history.


GDELT topic | Query before the English-language filter
ai | ("artificial intelligence" OR "generative AI" OR "machine learning" OR OpenAI OR Nvidia)
semiconductor | (semiconductor OR chipmaker OR Nvidia OR AMD OR TSMC OR ASML OR Intel)
fed | ("Federal Reserve" OR FOMC OR "interest rates" OR "rate hike" OR "rate cut")
inflation | (inflation OR CPI OR "consumer prices" OR "core inflation")
big tech earnings | (Apple OR Microsoft OR Amazon OR Alphabet OR Meta OR Nvidia) (earnings OR revenue OR guidance)
recession | (recession OR "hard landing" OR "economic contraction")
Each topic adds gdelt_TOPIC_news_share and gdelt_TOPIC_avg_tone. Queries use sourcelang:eng and no timeline smoothing. Article counts contribute to share construction but are not selected model inputs.


Other external family | Exact fields
EPU / EMU (2) | epu_log, emu_log
FOMC (4) | fomc_net_hawkish_last, fomc_net_hawkish_ewma, fomc_tone_change_last, days_since_fomc
SEC (3) | ndx_earnings_count, sec_8k_count, sec_total_filings
All external combines nine non-GDELT fields and 12 GDELT fields with the 21 controls.

EPU and EMU derive from the policy uncertainty data family [7]. The FOMC model used in the pipeline is LorenzoAleCon29/roberta-large-fomc-hawkish-dovish, revision f4759d4ad3f1182f81d87e47ba603261740d36cf. Missing observations use the model's observation mask. The array-based Chronos interface preserves exchange-session order without fabricating holiday rows [8].