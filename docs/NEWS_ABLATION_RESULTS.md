# News-source fine-tuning results

## What exactly did we predict?

At each test date, the model was given everything available through the previous QQQ close.
It then predicted **five separate daily QQQ returns**: the return on the next trading day,
the day after that, and so on through day five.

It did **not** predict the QQQ price and did **not** predict one cumulative five-day return.
For example, a prediction of `+0.0020` means approximately **+0.20% for that individual
trading day**.

We made one five-day forecast, moved forward five trading sessions, and made the next one.
This produced 50 non-overlapping forecast blocks and 250 tested daily returns:

```text
data through June 27, 2025
            │
            └── predict June 30, July 1, July 2, July 3 and July 7

then include those five realized days and make the next five-day forecast
            │
            └── repeat until the final June 22–26, 2026 block
```

Here is the first block from the final fine-tuned run. “Predicted” is the median forecast.

| Trading day | Actual QQQ return | Predicted return | Direction correct? |
|---|---:|---:|---:|
| 2025-06-30 | +0.65% | +0.21% | Yes |
| 2025-07-01 | -0.84% | +0.08% | No |
| 2025-07-02 | +0.70% | +0.12% | Yes |
| 2025-07-03 | +0.98% | +0.17% | Yes |
| 2025-07-07 | -0.75% | +0.15% | No |

## What did we compare?

For every data setup, we performed the same sequence:

1. Give the original pretrained Chronos-2 model that setup's data and predict the same
   250 returns.
2. Start again from the original weights, train LoRA weights using only data ending
   June 27, 2025, and predict those same 250 returns again.
3. Compare both models with a simple forecast centred at zero.

The data setups were control data alone, control plus EPU/EMU, control plus FOMC tone,
control plus SEC activity, control plus GDELT, and control plus all external sources.

## What was the understandable result?

The best average setup was **all external sources together**. That means market/control
features plus EPU/EMU, FOMC tone, SEC activity and GDELT.

| Question | Result |
|---|---|
| Did all external data beat the control setup? | Yes on average: probabilistic loss was 0.60% lower. It won 2 of 3 training seeds. |
| How often was the up/down direction right? | 57.47% with all external data, versus 56.67% for control. Across three runs, this was 431 versus 425 correct directions out of 750. |
| How large was the typical point-prediction error? | About 0.842 percentage points per day with all external data, versus 0.847 for control. |
| Did LoRA itself help the all-external model? | Only slightly on probabilistic loss. Mean WQL changed from 0.690730 pretrained to 0.689439 fine-tuned. Point error and direction did not improve. |
| Which single news source won? | None reliably. EPU/EMU looked best in seed 42 but failed to repeat in seeds 43 and 44. |
| Is the improvement proven? | No. The uncertainty interval still includes no improvement. We need an untouched future test period. |

The main score, weighted quantile loss (WQL), evaluates the complete predicted probability
distribution. Lower is better. It rewards a model that puts sensible probabilities around
both ordinary and extreme returns; it is different from simply counting correct directions.

This distinction matters: most of the measured gain came from **giving the pretrained model
the external covariates**. LoRA then made a much smaller additional improvement to WQL.

The honest conclusion is: external information may help Chronos predict the distribution of
future QQQ returns a little, but these runs do not yet show a dependable trading signal or a
dependable best individual news source.

## Technical record

TAU Slurm array `869989` ran the six seed-42 source configurations. Array `871482` repeated
control, EPU/EMU, and `all_external` with seeds 43 and 44. All 12 model fits completed on an
NVIDIA GeForce RTX 2080 Ti using the pinned Chronos-2 revision. Each fit used:

- 2,133 training sessions from 2017-01-03 through 2025-06-27;
- 252 validation sessions from 2025-06-30 through 2026-06-30;
- 50 non-overlapping five-session origins, giving 250 forecast outcomes;
- context length 512, 200 LoRA steps, learning rate 1e-5, and batch size 8;
- the same 21 quantiles and the same feature-table hash
  `e0c8d48b38059eb70d9f91931bff68678dbcc46aeadab1502ded1b9ee11df037`.

The September 4 feature snapshot is checked with a frozen `--as-of` timestamp for experiment
replay. Live forecast jobs still use the current-date readiness gate.

## Phase 1 source screen

These are seed-42 fine-tuned results. Lower WQL, MAE, and RMSE are better; 10%-90% coverage
should be close to 0.80.

| Setup | WQL | Delta vs control | MAE | Direction | 10%-90% coverage |
|---|---:|---:|---:|---:|---:|
| EPU/EMU uncertainty | 0.688112 | -0.007301 | 0.008436 | 57.2% | 77.6% |
| All external sources | 0.688303 | -0.007109 | 0.008420 | 58.0% | 79.6% |
| All GDELT topics | 0.692348 | -0.003065 | 0.008445 | 55.6% | 77.6% |
| SEC disclosures | 0.692848 | -0.002565 | 0.008458 | 58.0% | 79.2% |
| FOMC tone | 0.694609 | -0.000804 | 0.008483 | 57.6% | 77.6% |
| Control | 0.695413 | 0.000000 | 0.008484 | 56.0% | 78.0% |

At seed 42, EPU/EMU beat control in 64% of validation windows and had a paired bootstrap
95% interval entirely below zero (-0.013826 to -0.001294). The result did not survive the
seed repetitions: seed-43 WQL was 0.698521 and seed-44 WQL was 0.693786.

## Robustness across seeds

| Setup | Seeds | Mean WQL | WQL std | Mean delta vs control | Seeds beating control | Mean MAE | Mean direction | Mean 10%-90% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All external sources | 3 | 0.689439 | 0.003583 | -0.004144 | 2/3 | 0.008420 | 57.47% | 79.07% |
| EPU/EMU uncertainty | 3 | 0.693473 | 0.005212 | -0.000109 | 1/3 | 0.008467 | 56.13% | 77.60% |
| Control | 3 | 0.693582 | 0.001656 | 0.000000 | 0/3 | 0.008467 | 56.67% | 77.87% |

`all_external` also beat the zero-mean Gaussian baseline on mean WQL (0.689439 versus
0.703413) and MAE (0.008420 versus 0.008545). Its mean RMSE was slightly worse
(0.011433 versus 0.011391). Mean directional accuracy was 57.47%, only 0.27 percentage
points above the Gaussian baseline.

The direct three-seed `all_external` minus EPU/EMU paired-window difference was -0.004034,
with a 95% bootstrap interval of -0.010470 to 0.002141. This supports `all_external` as the
next candidate but does not prove it is better than EPU/EMU.

## Single-topic GDELT screen

Slurm array `874192` added six seed-42 fits, each using the same 21 controls plus only one
GDELT topic's news share and average tone. All six beat the fine-tuned seed-42 control on
WQL. Fed ranked first at 0.686526, followed closely by recession at 0.686796,
semiconductors at 0.687499, AI at 0.687580, inflation at 0.688131, and big-tech earnings at
0.688750. Control was 0.695413.

For the requested eight-case view, the earlier seed-42 `all_external` model is also included
in the focused report. Its WQL was 0.688303, placing it between inflation and big-tech
earnings; its 58.0% directional accuracy was the highest of the eight cases.

The paired interval for every topic versus control was below zero, but direct topic-to-topic
intervals included zero. The focused explanation, readable table, and exact artifact paths
are in `docs/GDELT_TOPIC_RESULTS.md`.

## Interpretation and next test

The infrastructure smoke test and the full source screen answer different questions. Smoke
job `869927` established that model loading, CUDA, LoRA training, checkpoint save/reload,
rolling evaluation, and holdout forecasting all work. Its 15 outcomes and five optimizer
steps could not measure forecasting skill. The full arrays increase this to 250 outcomes and
200 optimizer steps per fit, use a fixed common control, and repeat the leaders across seeds.

The next confirmation experiment should repeat the Fed and recession topic setups with
seeds 43 and 44. Fed has the best point estimate, but its small lead over recession is not
resolved by this single-seed screen. The selected configuration must then be evaluated once
on a future block after 2026-09-04 that was not used for these choices.

Raw checkpoints, predictions, metrics, environments, and readiness reports are stored under
the ignored directory `models/news-ablation-869989/`. Regenerate the aggregate tables with:

```bash
python scripts/summarize_news_ablation.py --root models/news-ablation-869989
```
