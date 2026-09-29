# 5  Experimental design and scoring

::: table ../tables/02_training_recipes.csv widths=34,58,74 font=8.7
The rounds answer related questions but do not isolate one training change. Both use context 512, five-step horizon, stride five and learning rate 1e-5.
:::

Round A adds one source family or one GDELT topic to the same 21-feature control. Round B follows a cumulative ladder: target only; six QQQ transforms; six calendar flags; nine market/rate fields; EPU/EMU; FOMC tone; GDELT Fed and recession; then all remaining external fields. The resulting counts are 0, 6, 12, 21, 23, 27, 31 and 42 covariates. Seven rungs use eight windows for 500 steps (84 fits). The 42-feature rung and a matched control use four windows for 1,000 steps (24 fits), because eight all-external windows exceed the GPU memory budget.

A separate pretrained-only exploration includes four calendar years plus the original mid-2025 to mid-2026 interval. That fifth interval overlaps calendar 2025. Its intermediate source arms are not all cumulative. We use the four-year walk-forward arm for the main paired pretrained/LoRA comparison, avoiding both mixed feature definitions and treating overlapping intervals as independent years.

## 5.1  Weighted quantile loss

::: equation 3
\rho_\tau(u)=\max(\tau u,(\tau-1)u)
:::

::: equation 4
\mathrm{WQL}=\frac{2}{|Q|\sum_{t=1}^{N}|r_t|}\sum_{\tau\in Q}\sum_{t=1}^{N}\rho_\tau(r_t-\widehat q_{t,\tau})
:::

WQL averages the asymmetric quantile loss across the 21 levels and normalizes by realized absolute returns. Lower is better; 0.70 does not mean 70% accuracy. The implementation scores each requested quantile [[8]](#ref8). We also report median MAE, direction, and the coverage and width of the nominal 80% and 98% intervals. A zero-mean Gaussian with trailing 60-session sample standard deviation supplies quantiles without news or training. It forecasts daily, not cumulative, uncertainty.

**Reading a reduction.** Relative WQL reduction is 100 x (control WQL - news WQL) / control WQL. A decrease from 0.695413 to 0.686526 is an absolute loss reduction of 0.008887, or 1.28% relative. A 1% reduction from 0.70 would give 0.693. It describes a better quantile score on the same outcomes, not a one-percentage-point increase in direction accuracy or a financial return.

## 5.2  Pairing and uncertainty

We pair forecasts on the same realized dates and average seeds before comparing windows. Four-year estimates average fold-normalized losses, giving each calendar year equal weight. Published ladder intervals use 5,000 window-bootstrap draws. For three headline contrasts we additionally resample circular blocks of five consecutive forecast windows separately within each year (5,000 draws, seed 20260916). These intervals remain exploratory and unadjusted for multiple comparisons; they condition on observed years and seeds.
