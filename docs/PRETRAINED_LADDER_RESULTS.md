# Zero-shot covariate ladder: what the pretrained model does with each input

Run 2026-09-15 on a Mac (MPS), inference only, no fine-tuning. Artifacts:
`docs/results/pretrained_ladder/` (per-fold predictions, `ladder_metrics.csv`,
`paired_deltas_vs_control.csv`, `paired_deltas_pooled.csv`, `ladder_summary.md`).
Regenerate with `python scripts/run_pretrained_ladder.py` then `python scripts/summarize_ladder.py`.

## Question

Before any training noise enters, which covariate groups change the pretrained Chronos-2's
quantile forecasts of QQQ daily log returns, and in which direction? This isolates the
in-context channel from the fine-tuning channel.

## Setup

- Model: `amazon/chronos-2`, revision `29ec3766`, context 512, horizon 5, the 21 quantiles
  used everywhere else in the project.
- Folds: validation years 2022, 2023, 2024, 2025, plus Elad's 2025-07 to 2026-06 window. Each
  fold uses all rows from 2017-01-03 up to its start as context. Stride 5, so 50 windows and
  250 daily outcomes per fold, 1,250 outcomes in total.
- Rungs are cumulative and built from the same 21-feature control as the fine-tuning screen:
  `r0_target` (no covariates), `r1_qqq` (6 QQQ transforms), `r2_qqq_calendar` (+6 known-future
  calendar flags), `r3_control` (+9 market and rate signals, the full control),
  `r4_plus_uncertainty` (+EPU, EMU), `r5_plus_fed` (+4 FOMC tone features),
  `r6_plus_gdelt_fed_recession` (+4 GDELT columns), `r7_all_external` (control + every
  external source, 42 features). `r3_control_nofuture` passes the same 21 features with the
  calendar flags as past-only. `gaussian` is the zero-mean, trailing-60-session baseline.
- Paired statistic: per-window pinball loss difference versus `r3_control`, normalized by the
  fold's mean absolute return so window means equal the aggregate WQL delta; 5,000-draw
  bootstrap over windows.

## Result in one table

Mean WQL delta versus the control rung, pooled over the five folds (negative is better).

| Rung | Delta WQL | 95% interval | Windows better | Read |
|---|---:|---|---:|---|
| r2_qqq_calendar | -0.0020 | [-0.0045, +0.0004] | 46% | best single configuration, not significant |
| r3_control_nofuture | -0.0015 | [-0.0030, -0.0000] | 50% | calendar flags as past-only beat them as known-future |
| r1_qqq | -0.0012 | [-0.0041, +0.0015] | 50% | |
| r5_plus_fed | -0.0009 | [-0.0020, +0.0001] | 54% | the only news source with a consistent (small) gain |
| r4_plus_uncertainty | +0.0006 | [-0.0003, +0.0016] | 48% | EPU and EMU do nothing zero-shot |
| r6_plus_gdelt_fed_recession | +0.0008 | [-0.0012, +0.0029] | 54% | GDELT does nothing zero-shot |
| r7_all_external | +0.0045 | [+0.0004, +0.0088] | 52% | **42 covariates hurt** |
| r0_target | +0.0047 | [-0.0002, +0.0098] | 44% | the control helps, mostly via 2025 |

Control WQL by fold: 0.689 (2022), 0.683 (2023), 0.704 (2024), 0.713 (2025), 0.692 (2025H2).
All covariate effects are inside ±0.7% of the control loss.

## Findings

1. **Zero-shot, covariates barely move the forecast.** Every rung is within 0.7% of the
   control. The pretrained model's in-context use of these inputs is weak. Any news effect the
   project reports must therefore come from fine-tuning, which makes the corrected-batch
   walk-forward ladder the decisive experiment.
2. **More covariates is worse.** The 42-feature configuration is significantly worse than
   the 21-feature control pooled (+0.0045, interval excludes zero) and significantly worse in
   2023 (+0.012). This is the "large attention group" concern from the Chronos-2 review,
   observed directly: the pretrained model degrades as unrelated series are added.
3. **The known-future channel is not exploited for binary calendar flags.** Passing the
   same calendar columns as past-only is slightly *better* than passing them as known-future
   (pooled -0.0015, interval just excludes zero; significant in the 2025H2 fold). Zero-shot,
   the model does not use "FOMC tomorrow" the way the documentation suggests it could. Whether
   fine-tuning teaches it to is an open question for the walk-forward run.
4. **Fed tone is the only news input with a consistent zero-shot benefit**, small (-0.0009
   pooled) and significant only in the 2025H2 fold (-0.0031, 66% of windows). EPU/EMU and
   GDELT fed/recession are indistinguishable from zero.
5. **Regime matters more than any covariate.** The Gaussian baseline beats every model
   configuration in 2022 (bear market, high volatility) by about 1.3% and loses by 5.4% in
   2025. The pretrained model's 10-90 band covers 76-78% of outcomes against an 80% target,
   so it runs slightly narrow; the Gaussian's trailing volatility adapts faster in a
   volatility regime shift.
6. **Direction is the up-day base rate in every fold** (44% in 2022, 57-59% otherwise) for
   every configuration including the zero forecast. Directional accuracy carries no
   information here and should be reported only against that base rate.

## Consistency with the fine-tuning screen

In Elad's 2025H2 window the zero-shot all-external configuration is 0.4% better than the
zero-shot control (-0.0027), matching his observation that most of the all-external gain
appeared in the pretrained arm. The other four folds show the opposite sign, so that window
was the favourable one for the news inputs.

## Caveats

- 50 windows per fold gives paired intervals about ±0.003 wide; effects smaller than that are
  undetectable at this sample size.
- The per-window paired statistic in `scripts/summarize_news_ablation.py` normalizes each
  window by its own mean absolute return, which inflates quiet weeks and makes the mean of
  window deltas disagree with the aggregate WQL; the intervals here use fold-level
  normalization. Elad's reported intervals should be recomputed the same way.
- Rungs are cumulative, so each row is the marginal effect of one group on top of the
  previous ones, not the group in isolation.
