# Walk-forward fine-tuning ladder: does fine-tuning teach Chronos-2 what the inputs mean?

Runs 2026-09-15 on the TAU Slurm cluster (RTX 2080 Ti). Slurm jobs 896501 (pretrained arm),
896510 / 896540 / 896541 (fine-tuned ladder, seeds 42 / 43 / 44), 896751 / 896752
(all-external rung and matched control with the 4-window recipe), 896762 (permutation
importance on fine-tuned checkpoints). Artifacts: `docs/results/walk_forward/` (summary
tables, per-window losses, per-cell metrics and metadata) and
`docs/results/permutation_finetuned/`. Regenerate with `scripts/summarize_walk_forward.py`
and `scripts/summarize_permutation_importance.py`.

## Question

The zero-shot ladder and permutation runs (`PRETRAINED_LADDER_RESULTS.md`,
`PERMUTATION_PRETRAINED_RESULTS.md`) showed that the pretrained model is misled by fast
market covariates and uses almost no news. Does LoRA fine-tuning on the QQQ history repair
that prior, and if so which inputs does the fine-tuned model come to rely on?

## Setup

- Folds: validation calendar years 2022, 2023, 2024, 2025; each fold trains on every session
  from 2017-01-03 up to the day before the validation year (1,259 to 2,012 sessions) and
  evaluates 50 non-overlapping five-session windows in the year (250 outcomes per fold).
- Rungs are cumulative, built from the 21-feature control of the news screen: `r0_target`,
  `r1_qqq` (+6 QQQ transforms), `r2_qqq_calendar` (+6 known-future flags),
  `r3_qqq_calendar_market` (+9 market and rate signals; the control), `r4_plus_uncertainty`
  (+EPU, EMU), `r5_plus_fed` (+4 FOMC tone), `r6_plus_gdelt_fed_recession` (+4 GDELT),
  `r7_all_external` (42 features).
- Recipe: LoRA (rank 8), learning rate 1e-5, context 512, **8 windows per optimizer step**
  (effective Chronos-2 batch = 8 × (1 + covariates) rows, i.e. 176 rows for the control) for
  500 steps, three seeds. The all-external rung does not fit in 11 GB at 8 windows and was
  run at 4 windows for 1,000 steps (same number of windows seen); the control was rerun with
  that recipe as a matched comparison.
- Statistics: per-window pinball loss normalized by the fold's mean absolute return, seeds
  averaged per window, 5,000-draw paired bootstrap over windows. Pretrained and fine-tuned
  arms are scored on identical windows.

## Result 1: fine-tuning changes the loss very little

Pooled over four years and three seeds, fine-tuned minus pretrained WQL:

| Rung | Gain | 95% interval | Windows better |
|---|---:|---|---:|
| r0_target | -0.0001 | [-0.0019, +0.0017] | 47% |
| r1_qqq | -0.0002 | [-0.0018, +0.0013] | 51% |
| r2_qqq_calendar | +0.0004 | [-0.0019, +0.0026] | 49% |
| r3_control | -0.0008 | [-0.0039, +0.0023] | 46% |
| r4_plus_uncertainty | -0.0009 | [-0.0041, +0.0023] | 48% |
| r5_plus_fed | -0.0010 | [-0.0039, +0.0018] | 53% |
| r6_plus_gdelt_fed_recession | -0.0018 | [-0.0057, +0.0017] | 44% |

No rung's pooled gain excludes zero. The gain grows slightly with the number of covariates
and is concentrated in 2022: with uncertainty added, fine-tuning gains -0.0063 in the bear
year (interval [-0.0126, -0.0005]), with Fed and GDELT close behind (-0.0053, -0.0056, both
just touching zero). In 2023 to 2025 every gain is within ±0.002.

The marginal effect of each added group on the fine-tuned model is null in every year
(all pooled intervals include zero; the largest pooled point estimate is +0.0007). No rung
differs from the control after fine-tuning.

Seed spread is now small: the standard deviation of aggregate WQL across the three seeds is
0.0013 to 0.0024 per rung, below the treatment effects being tested. This is the corrected
batch size at work; the earlier screen, where every step saw one window, had seed spread of
0.004 to 0.005.

## Result 2: fine-tuning improves calibration, not covariate use

Coverage of the 10-90 band, averaged over rungs:

| Fold | Pretrained | Fine-tuned |
|---|---:|---:|
| 2022 | 0.760 | 0.787 |
| 2023 | 0.790 | 0.812 |
| 2024 | 0.755 | 0.781 |
| 2025 | 0.800 | 0.818 |

The pretrained model runs narrow; fine-tuning widens its intervals toward the 80% target in
every year and on every rung, including the target-only rung with no covariates at all. The
1-99 band moves the same way. This is the mechanism behind the small loss gains: fine-tuning
learns the scale of QQQ returns better, not the meaning of the inputs.

## Result 3: the fine-tuned model uses its inputs exactly like the pretrained one

Permutation importance (same swap method as the pretrained runs, 3 repeats, 50 windows) on
six fine-tuned checkpoints, each paired with the pretrained model on identical features and
windows. Loss change when the feature is removed; negative means the feature was misleading.

| Feature | Control 2022 pre → ft | Control 2025 pre → ft | Fed rung 2025 pre → ft | All-external 2025 pre → ft |
|---|---:|---:|---:|---:|
| vxn_log_chg | -0.0040 → -0.0033 | -0.0071 → -0.0091 | -0.0083 → -0.0098 | -0.0084 → -0.0087 |
| overnight_gap | -0.0030 → -0.0030 | -0.0046 → -0.0060 | -0.0048 → -0.0058 | -0.0023 → -0.0023 |
| tlt_log_ret | -0.0001 → +0.0002 | -0.0007 → -0.0020 | -0.0016 → -0.0027 | -0.0015 → -0.0011 |
| days_to_fomc | -0.0002 → -0.0001 | +0.0003 → -0.0004 | +0.0019 → +0.0007 | +0.0013 → -0.0001 |
| epu_log | | | -0.0004 → -0.0003 | -0.0005 → -0.0002 |
| fomc_net_hawkish_ewma | | | -0.0001 → -0.0001 | -0.0001 → -0.0001 |
| gdelt_fed_avg_tone | | | | +0.0002 → +0.0002 |
| gdelt_fed_news_share | | | | +0.0008 → -0.0002 |

The pattern is the same in all six cells: the features that misled the pretrained model
(VXN change, overnight gap, real-yield change, TLT, volume) still mislead the fine-tuned one,
by the same or slightly larger amounts, and every uncertainty, Fed-tone and GDELT column
remains within ±0.001 of zero. In 2025 the fine-tuned control is significantly hurt by the
overnight gap, TLT and volume z-score (intervals exclude zero), which the pretrained model
was not. The only feature whose removal significantly hurts a fine-tuned checkpoint is the
distance to the 200-day average (+0.0025 in 2025). Nothing flips sign.

## Result 4: the all-external rung, matched recipe

Run at 4 windows × 1,000 steps (same windows seen) with a control trained identically.

| | 2022 | 2023 | 2024 | 2025 | mean |
|---|---:|---:|---:|---:|---:|
| Pretrained, 42 features | 0.6977 | 0.6947 | 0.7026 | 0.7199 | 0.7037 |
| Fine-tuned, 42 features | 0.6904 | 0.6881 | 0.7026 | 0.7147 | 0.6990 |
| Fine-tuned control, matched recipe | 0.6885 | 0.6855 | 0.7034 | 0.7135 | 0.6977 |
| Fine-tuned control, 8-window recipe | 0.6865 | 0.6843 | 0.7021 | 0.7139 | 0.6967 |

- **Largest fine-tuning gain of any rung:** -0.0048 pooled, interval [-0.0106, +0.0008];
  -0.0073 in 2022 and -0.0066 in 2023. This is the rung on which the pretrained model was hurt
  most by its inputs, and fine-tuning recovers most of that damage.
- **No gain over the matched control:** fine-tuned all-external minus fine-tuned matched
  control is +0.0013 pooled, interval [-0.0012, +0.0039], worse in three of four years.
  Fine-tuning brings the 42-feature model back to where the 21-feature market-only model
  already was; the 21 extra news, uncertainty, tone and filing columns add nothing.
- **Recipe effect is negligible:** the 4-window control differs from the 8-window control by
  +0.0010 on average, inside its interval, so the comparison is like for like.
- Coverage of the fine-tuned all-external model is 0.78 to 0.82 across years, in line with
  the other fine-tuned rungs.

## Result 5: where fine-tuning pays (added after the independent review)

Per-window fine-tuning gain (fine-tuned minus pretrained, seeds averaged) split by the
realised volatility of the forecast week, rungs r3 to r6 pooled, 800 window-rungs
(`docs/results/walk_forward/gain_by_window_volatility.csv`, `scripts/analyze_gain_conditions.py`):

| Weeks by realised volatility | Gain | 95% interval |
|---|---:|---|
| quiet third | **+0.0090** | [+0.0072, +0.0108] |
| middle third | -0.0021 | [-0.0045, +0.0002] |
| volatile third | **-0.0104** | [-0.0140, -0.0069] |

Rank correlation of gain with realised volatility is -0.38 to -0.46 on every rung. By horizon,
the pooled gain is -0.0049 [-0.0068, -0.0030] at one day ahead and within ±0.0015 at days two
to five. The fine-tuned control beats the zero-mean Gaussian baseline pooled over the four
years by -0.0109 [-0.0184, -0.0037], losing only in 2022.

**Reading.** LoRA learns one thing: make the bands about 8% wider. That is unconditional, so
it helps exactly when a week turns out volatile and hurts when it turns out quiet, and the two
cancel to the near-zero headline gain. It is also why the 2022 bear year shows the largest gain
and why the earlier "concentrated in 2022" observation was really "concentrated in volatile
weeks". A fine-tuning that learned *when* to widen would show a gain in both terciles; this
one does not. The small pretrained-to-fine-tuned change in the 2y-yield input in both 2022
cells (from -0.0020 to -0.0003, the hiking year) is the only trace of covariate re-weighting.

## What this means

1. **LoRA fine-tuning on a single index history does not teach Chronos-2 what its covariates
   mean.** At this budget (500 to 1,000 steps, 1,300 to 2,000 sessions) the covariate usage of
   the fine-tuned model is indistinguishable from the pretrained one. The "unlearn the wrong
   prior" hypothesis is rejected.
2. **What fine-tuning does learn is the return distribution's scale, unconditionally.**
   Coverage improves on every rung including target-only; the widening helps in volatile weeks
   (-0.010) and hurts in quiet ones (+0.009), which is why the headline gain is near zero.
3. **News sources are inert both before and after fine-tuning**, with the single exception of
   Fed-topic tone, which the pretrained model uses slightly and the fine-tuned model no more.
   Their marginal effects on the loss are null in every year, and the full external set
   fine-tuned with a matched recipe is no better than the market-only control.
4. **The regime matters more than any input or any training.** The 2022 bear year is where
   fine-tuning with covariates gains the most and where the naive Gaussian beats every model.
5. The earlier screen's "all external beats control by 0.6%" was training noise: with the
   batch corrected and three seeds averaged per window, no covariate group changes the loss.

## Caveats

- Four folds of 50 windows each; effects smaller than about ±0.002 are below the resolution
  of this design even pooled.
- One learning rate and one LoRA rank. A larger budget (full fine-tuning, higher rate, more
  steps) might move covariate usage; the corrected-batch recipe here follows the official
  guidance and already shows tiny seed spread, so more steps alone are unlikely to change the
  conclusion.
- Permutation importance was run on seed-42 checkpoints only, for three rungs and two years.
- The one significant per-fold gain (uncertainty rung, 2022, -0.0063) is one of 28 cells and a
  single window (origin 2022-03-08) carries a third of it; it is a volatility effect, not a
  covariate effect (Result 5).
- An independent re-derivation of every number in this document from the raw CSVs is in
  `docs/results/independent_review_2026-09-15.md`.
- The all-external rung uses a different step granularity (4 windows × 1,000 steps); compare it
  with the matched 4-window control, not the 8-window one.
