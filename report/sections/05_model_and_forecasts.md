# 4  Model, adaptation and forecast protocol

## 4.1  Foundation forecasting and related work

Chronos-2 supports forecasting with related series and covariates through group attention [[1]](#ref1). Its pretrained capability provides an inference-only reference before any QQQ adaptation. LoRA freezes base weights and trains low-rank updates [[2]](#ref2). Financial evaluations of time-series foundation models motivate testing domain-specific behavior directly rather than assuming that general forecasting performance transfers to daily equity returns [[3]](#ref3). Our single-ETF experiment is narrower than those benchmarks.

The pinned checkpoint contains 119,477,664 base parameters. Its configuration has 12 layers, width 768, 12 attention heads and 16-step patches. Each supplied numeric series is scaled over observed context and transformed with asinh. The model returns 21 quantiles at levels 0.01, 0.05, 0.10 through 0.95 in 0.05 increments, and 0.99. We cap historical context at 512 exchange sessions.

## 4.2  LoRA and the training batch

::: equation 1
W=W_0+(\alpha/r)BA,\quad r=8,\quad\alpha=16
:::

The adapters target attention query, key, value and output projections and the output patch projection. They contain 1,206,912 trainable parameters, approximately 1% of the combined model, with adapter dropout zero. Query/key/value/output projections occur in both attention sublayers of each of the 12 blocks; counting only one would understate the adapter size. A uses the default random Kaiming-uniform initialization and B starts at zero, so the initial update is zero [10]. Each run starts from the same original checkpoint, samples historical training slices, and optimizes quantile loss. We do not continue training a previous control adapter. No validation examples are supplied to fit(), and the final scheduled checkpoint is used without validation-based early stopping.

The Chronos training batch counts target and covariate series. In the first screen, a batch argument of eight admitted effectively one complete QQQ window with its covariates. The later runner requests a number of windows explicitly: eight windows with 21 covariates requires 176 series rows. This corrects an under-specified training budget. It does not by itself prove that the earlier performance differences were caused by batch size.

## 4.3  What a five-day forecast means

::: equation 2
r_t=\log(P_t^{adj}/P_{t-1}^{adj}),\qquad \widehat q_{t+h,\tau},\ h=1,\ldots,5
:::

At an origin t, the model receives observed history through t and forecasts five separate daily log returns, not one cumulative weekly return. Every lead uses the same information cutoff. After five exchange sessions, we refresh history with all five actual observations and issue the next forecast. Historical inputs overlap substantially; the five-day blocks of scored dates do not overlap. No predicted return replaces a realized observation. We freeze adapter weights throughout each evaluation year.

::: figure ../figures/forecast_example.png
One first-round Fed-topic forecast. The July 4 holiday shifts the fifth scored session to July 7. Later leads do not observe intervening news or returns. Inclusion inside a broad interval and a correct directional prediction are different events.
:::
