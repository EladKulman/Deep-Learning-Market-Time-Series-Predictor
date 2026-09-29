# 7  Four-year results

::: figure ../figures/ladder.png
Round B cumulative ladder. Dotted curves use the original checkpoint; solid curves average three LoRA seeds. Each year scores the same 250 dates across rungs. The all-external arm is shown separately in Table 5 because its training recipe requires a matched control.
:::

Across the seven standard rungs, pooled fine-tuned-minus-pretrained WQL ranges from -0.001813 to +0.000371. Every reported pooled interval includes zero. This is insufficient evidence of a reliable adaptation benefit at these settings, rather than an equivalence test proving an exactly zero effect. Small changes in individual years are exploratory among many examined contrasts.

::: table ../tables/05_four_year_results.csv widths=66,20,20,20,20,20 font=8.6
Matched all-external comparison. The four-window and eight-window recipes each sample 4,000 training windows, but differ in optimizer-update granularity.
:::

All-external reduces WQL by 0.004766 relative to its own pretrained arm. Yet it remains +0.001257 above the identically trained control, with a five-window-block 95% interval [-0.00095, +0.00369]. Its advantage from adaptation also remains unresolved under that block analysis: [-0.01013, +0.00032]. Thus the larger input set has not established incremental predictive value over the market control.

The first-round mid-2025 to mid-2026 interval favored the pretrained all-external setup. Most additional calendar-year comparisons show the opposite direction. Period dependence and a small initial training budget make the original improvement a development observation, not a general result. The later cumulative GDELT arm combines Fed and recession with other sources; it does not independently replicate all six single-topic fits.
