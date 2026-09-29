# 2  Background and related work

## 2.1  Why a time-series model for an index?

A time series is a sequence of observations ordered in time. A forecasting model learns relationships between past observations and future outcomes while respecting that order. For QQQ, inputs can include return history, recent volatility and information known at the forecast origin. Randomly shuffling daily rows would destroy this timing and could allow future information into training. We therefore split chronologically and reveal new observations only after their dates have passed.

An index reflects many companies and common macroeconomic exposures. Topic-level news summaries offer a tractable hypothesis: market-wide narratives could affect the distribution of the next return even when they do not determine its sign. The hypothesis is incremental. News must improve forecasts beyond a control that already observes market conditions, rates and scheduled announcements. A time-series model is an appropriate way to test that hypothesis, but temporal structure alone does not make returns predictable.

## 2.2  From specialized models to foundation forecasting

Temporal Fusion Transformers combine temporal processing with variable selection and quantile prediction for multi-horizon forecasting [[5]](#ref5). PatchTST represents a time series as patches and uses Transformer processing, illustrating how architecture can exploit local temporal structure [[9]](#ref9). These are relevant methodological predecessors, not executed baselines in our project.

Chronos-2 provides a pretrained model with support for multiple related series and covariates through group attention [[1]](#ref1). This is useful experimentally: we can keep one base checkpoint fixed while varying which inputs it receives, then compare inference before and after adaptation. The model supplies a distribution through quantiles, so evaluation need not reduce to whether tomorrow is up or down.

Rahimikia, Ni and Wang report weaknesses of off-the-shelf time-series foundation models on financial excess returns and motivate domain-specific evaluation [[3]](#ref3). Their benchmark does not prove failure on every financial task. Our daily QQQ log-return study differs in target, inputs and scoring; it tests its own baselines and does not infer performance from general forecasting benchmarks.

## 2.3  News representation and parameter-efficient adaptation

GDELT provides query-dependent measures of news volume and tone [[6]](#ref6). Economic Policy Uncertainty measures reporting about uncertainty [[7]](#ref7), whereas FOMC-language classification captures hawkish or dovish policy communication [[4]](#ref4). These measurements represent distinct hypotheses. An average tone score is neither the sentiment of investors toward QQQ nor a direct estimate of its expected return.

LoRA freezes the original weights and learns low-rank changes to selected matrices [[2]](#ref2). Its modest trainable parameter count makes repeated source comparisons practical on our available GPU. We use it to ask whether QQQ-specific training changes forecast quality or measured input reliance. The experiment does not assume that low-rank adaptation is sufficient to recover every useful relation in the news.

The resulting design separates three questions: whether adding inputs helps before training, whether adaptation helps with a fixed input set, and whether news still helps when both news and control models receive the same adaptation recipe. Answering only the second question would not establish the incremental value of news.
