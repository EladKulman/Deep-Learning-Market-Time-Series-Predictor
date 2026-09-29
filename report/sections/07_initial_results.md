# 6  Initial news-source screen

The first round established end-to-end GPU execution and generated candidate source rankings. A separate five-update smoke test produced only 15 outcomes; it verified training and checkpoint reload, not forecasting skill. Table 3 summarizes the substantive single-topic comparison. All configurations retain the common market/calendar control.

::: table ../tables/03_topic_comparison.csv widths=51,35,43,37 font=9
Round A, seed 42. The all-external arm includes non-GDELT sources. All-GDELT alone is a different 33-feature arm (WQL 0.692348).
:::

Fed has the lowest point estimate, 1.28% below the fine-tuned control; recession differs by only 0.000269 WQL. A paired interval for their difference includes zero. The six topics each have one seed, so their apparent ranking does not demonstrate a uniquely informative narrative. Longer blocks in the first-round audit weaken several topic-versus-control comparisons as well.

## 6.1  Replication changes the interpretation

::: table ../tables/04_seed_comparison.csv widths=51,40,35,40 font=9
Round A seed repetitions. Each seed evaluates the same 250 market outcomes; these are not 750 independent days.
:::

All-external improves mean loss by 0.60% relative to control, but its paired interval includes no gain. Its own pretrained WQL is 0.690730, so LoRA adds only a 0.19% relative improvement. EPU/EMU looks strong in the first seed and loses that advantage in repetitions. These observations motivated a larger training budget and evaluation across different years.

Direction is particularly weak evidence of skill in this interval: 143 of 250 actual returns are non-negative. Always predicting up scores 57.2%, versus 57.6% for Fed and 58.0% for all-external at seed 42. The Gaussian median is zero, which the scoring code counts as up. The result therefore supports candidate probabilistic comparisons, not a dependable trading rule.

## 6.2  Why the later study is not a batch-only experiment

The original API batch argument gave much less training than intended. Round B increases both windows per update and update count, evaluates different periods, changes the feature snapshot and uses updated Transformers/PEFT versions. Its evidence can overturn confidence in the original ranking, but cannot identify a batch-size defect as the sole causal explanation. A controlled rerun on the original snapshot and dates would be required for that claim.
