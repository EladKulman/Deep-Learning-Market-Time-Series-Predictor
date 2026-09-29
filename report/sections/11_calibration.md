# 9  Calibration, regimes and baseline performance

::: figure ../figures/calibration.png
Left: average nominal 80% coverage for the seven standard rungs, with equal rung weight. Right: post-hoc mean LoRA-minus-pretrained WQL across rungs 3-6 after averaging their repeated measurements per observed week. The terciles contain 67, 66 and 67 distinct weeks, not 800 independent market observations.
:::

Across all 108 trained cells, including the matched-recipe reruns, mean 80% coverage increases from 77.69% to 80.09%, and mean interval width increases by 7.80%. Those cells repeatedly evaluate the same market dates, so this aggregate describes model behavior rather than providing an independent-observation count. The standard-rung plot shows a rise in every year, but higher coverage is not always closer to 80%: in 2025 it moves from approximately 80.3% to 81.8%. Tail coverage also remains imperfect.

Widening is consistent with the post-hoc loss pattern. After averaging seeds and rungs 3-6 within each observed week, fine-tuning raises WQL by 0.0090 in the quiet third and lowers it by 0.0103 in the volatile third. The group assignment uses realized future absolute returns, so this analysis explains realized performance; it cannot select a trading action in advance. We do not attach independence-based intervals to repeated rung observations.

These findings support a dispersion-change hypothesis, not a complete identification of an unconditional scale mechanism. A forecast-time regime model or a simple calibrated widening baseline, fitted only on training data, would be needed to test whether LoRA adds more than interval rescaling. In particular, improvement in one realized-volatility group and deterioration in another can occur without revealing what information the network used.

## 9.1  Comparison with a simple forecasting solution

::: table ../tables/07_paired_contrasts.csv widths=68,39,59 font=8.6
New report sensitivity analysis: seeds averaged, equal year weights, circular blocks within years, 5,000 draws. Negative differences favor the first model. Intervals are exploratory and not multiplicity-adjusted.
:::

The fine-tuned market control beats the Gaussian on the pooled four-year score, while its 2022 point estimate is worse. Therefore, absence of demonstrated incremental news benefit does not mean that the complete model has no forecasting skill. This baseline is deliberately simple; no supervised quantile regression, tree model, or learned volatility baseline was executed in this study. Comparisons with those solutions remain future work.
