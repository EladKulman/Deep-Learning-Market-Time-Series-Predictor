# 8.2  Does adaptation change measured input reliance?

::: figure ../figures/paired_permutation_2025.png
Paired perturbations, all-external model, calendar 2025, seed 42. Ten selected inputs illustrate the dominant market effects and representative news families. The correlation of 0.908 is computed over all 42 features. Negative values mean the scrambled forecast had lower loss; the plot does not show feature deletion or retraining.
:::

Six fine-tuned checkpoints cover the control, FOMC rung and all-external rung in 2022 and 2025, using seed 42. Their matched pretrained/LoRA importance-vector correlations range from 0.908 to 0.978. Dominant effects persist, but profiles are not identical: the two-year yield change moves toward zero in both tested 2022 control/FOMC cells. The supported conclusion is limited change in the measured importance pattern at this budget, rather than proof that training cannot alter covariate use.

::: table ../tables/06_importance_correlations.csv widths=86,40,40 font=9
Pearson correlation between pretrained and adapted mean importance vectors. Each cell uses seed 42. High correlation describes the profile; it does not establish equality of individual feature effects.
:::

A near-zero swap effect can arise because the model makes little use of the feature, because another correlated feature carries similar information, or because helpful and harmful effects cancel across origins. Whole-context swaps may also create unlikely combinations of market and news histories. The result therefore supports limited measured change under this diagnostic, not the categorical claim that the model cannot understand news.

A stronger follow-up would combine group-level swaps, retrained feature ablations and conditional donor matching. Agreement across those tests would make an input-reliance conclusion more convincing. All donors and normalization choices should be held fixed between the pretrained and adapted arms.
