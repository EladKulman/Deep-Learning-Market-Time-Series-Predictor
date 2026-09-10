# News-source fine-tuning plan

The Phase 1 screen, three-seed leader check, and six single-topic GDELT fits are complete.
See `docs/NEWS_ABLATION_RESULTS.md` and `docs/GDELT_TOPIC_RESULTS.md` for results and the
current decision.

## What the GPU smoke test established

TAU Slurm job `869927` completed the complete execution path on one RTX 2080 Ti: load the
pinned pretrained Chronos-2 revision, forecast with it, train a LoRA adapter, evaluate the
adapter on the same rolling windows, save and reload the adapter offline, and generate a
holdout forecast. The tested runtime and raw artifacts are documented in `slurm/README.md`
and `models/slurm-smoke-869927/`.

The smoke sample was deliberately small: 452 training sessions, 60 validation sessions,
five non-overlapping three-session windows, five optimizer steps, and six covariates. It
produced only 15 outcomes. LoRA lowered weighted quantile loss from 0.734899 to 0.710572
(3.31%), MAE from 0.019400 to 0.019042 (1.84%), and RMSE from 0.021941 to 0.021483 (2.08%).
It had lower pinball loss on 10 of 15 outcomes and lower absolute error on 9 of 15. The
Gaussian baseline still won overall, and directional accuracy was unchanged at 26.7%.

These results prove that the pipeline runs. They do not identify a useful news source: the
six-feature setup mixed EPU and FOMC tone, excluded GDELT and SEC activity, and was too small
for statistical conclusions.

## Phase 1: common-control source screen

Every setup starts with the same 21-feature control: six QQQ transforms, five market
signals, four rate/macro signals, and six known-future calendar signals. Each news/event
source is added to that fixed control, and Chronos-2 is fine-tuned again from the same pinned
base revision. This measures whether the source adds value beyond information already in
market prices and the calendar.

| Setup | Added source | Total features |
|---|---|---:|
| `control` | none | 21 |
| `uncertainty` | EPU and EMU | 23 |
| `fomc_tone` | four classifier-derived Fed tone/state features | 25 |
| `sec_disclosures` | earnings, 8-K, and total filing activity | 24 |
| `gdelt_all` | share and tone for six GDELT topics | 33 |
| `gdelt_ai` | AI share and tone only | 23 |
| `gdelt_semiconductor` | semiconductor share and tone only | 23 |
| `gdelt_fed` | Federal Reserve share and tone only | 23 |
| `gdelt_inflation` | inflation share and tone only | 23 |
| `gdelt_big_tech_earnings` | big-tech earnings share and tone only | 23 |
| `gdelt_recession` | recession share and tone only | 23 |
| `all_external` | every source above | 42 |

The common sample is frozen to 2017-01-03 through 2026-06-30, when every GDELT topic cache
has coverage. Training uses 2,133 sessions through 2025-06-27. The final 252 sessions,
2025-06-30 through 2026-06-30, form 50 non-overlapping five-session validation windows and
250 forecast outcomes. Each run uses context length 512, 200 LoRA steps, batch size 8,
learning rate 1e-5, seed 42, and the same 21 quantiles.

The job evaluates source/build integrity using the feature snapshot's September 4, 2026
cutoff. This keeps historical experiment replay independent of the wall-clock date; live
forecast jobs continue to use the current-date freshness gate.

The primary comparison is fine-tuned weighted quantile loss versus `control`, paired on the
same forecast windows. MAE, RMSE, directional accuracy, 10%-90% coverage, 1%-99% coverage,
and interval width are secondary. A source advances only if it improves the primary metric
without materially degrading calibration and the gain appears across windows rather than
coming from one market shock.

## Phase 2: robustness and GDELT topics

The best two source setups and the control were repeated with seeds 43 and 44. Array
`874192` then fine-tuned six additional seed-42 models that add one GDELT topic pair at a
time (share plus tone) to the same 21 controls. Every run started from the original pinned
pretrained Chronos-2 weights and used the same training split, validation windows, and
optimization settings as the control. Fed ranked first, narrowly ahead of recession; both
should now be repeated with seeds 43 and 44.

The 2025-2026 window is development validation. The smoke test already exposed a few dates
inside it, so it is not an untouched final test. Freeze a later, newly arriving block after
2026-09-04 and evaluate the selected configuration once before making a final performance
claim.

## Running the screen

The source matrix is `configs/news_ablation.json`. All source-family and single-topic dry-run
plans pass locally. After staging the current tree in shared Slurm storage, submit:

```bash
sbatch slurm/chronos_news_ablation.sbatch
```

The array runs one setup at a time (`0-5%1`) to avoid competing for student GPUs. Each task
saves its environment, readiness report, checkpoint, predictions, metrics, and provenance
under `models/news-ablation-ARRAY_JOB_ID/`.

After ranking Phase 1, repeat control and the two leading setups with seeds 43 and 44:

```bash
NEWS_ABLATION_OUTPUT_ROOT=models/news-ablation-PHASE1_JOB_ID \
  sbatch slurm/chronos_news_robustness.sbatch
```

Run the six single-topic GDELT fits beside the existing seed-42 control artifacts:

```bash
GDELT_TOPIC_OUTPUT_ROOT=models/news-ablation-PHASE1_JOB_ID \
  sbatch slurm/chronos_gdelt_topic_ablation.sbatch
```

The topic array is serial (`0-5%1`). It adds six new model directories and leaves the
existing control and source-screen artifacts unchanged.
