# Academic report review and refinement

Reviewed 2026-09-10. Final artifact: `output/pdf/qqq_chronos2_project_report.pdf`.
The previous PDF is preserved under `output/pdf/archive/`.

## Comparison with the example

The example is an eight-page workshop paper with an abstract, introduction,
related work, method and loss equations, illustrated experiments, limitations,
references, and an appendix. Its useful pattern is explaining each experiment's
method beside its evidence. It is a format reference, not a grading rubric.

The initial QQQ draft followed the section structure but omitted several details
needed to evaluate the research: complete pretrained-versus-adapted results,
formal target and metric definitions, LoRA configuration, an interpretable
forecast example, and diagnostics beyond aggregate ranking. Its source bar chart
also used a truncated baseline, exaggerating visually small differences.

The refined paper has eight pages of main text and two pages of references and
reproducibility material. It retains the academic style while adding:

- Explicit research questions and a contribution tied to the original proposal.
- Correct 15-past + 6-calendar control counts, separate from the default core profile.
- Exact GDELT queries, all selected covariates, observed missingness, and timing rules.
- Target, LoRA, pinball-loss, and WQL equations; an architecture diagram.
- A numerical five-day forecast example with origin, daily returns, and intervals.
- Every source's pretrained and adapted WQL, three-seed replication, and all eight
  requested topic-comparison configurations.
- Paired uncertainty plots with explicit block-resampling assumptions, replacing
  a visually amplified ranking chart.
- Directional class imbalance, lead-time sensitivity, calibration drift, event-day
  behavior, and quantile-order checks.
- Verified primary references and a run ledger with reproducible analysis scripts.

## New analysis of already-executed forecasts

`scripts/analyze_report_results.py` reads existing artifacts only. It asserts the
common dates, realized outcomes, 50-by-5 structure, selected features, model/data
hashes, and fitting settings for 18 runs. Recomputed WQL, MAE, and direction counts
match the saved metrics. The 54 forecast sets are 18 runs times pretrained,
fine-tuned, and Gaussian outputs; they are not 54 independent market experiments.

Findings added to the paper:

- 143/250 outcomes are non-negative. Always-up direction accuracy is 57.2%; the
  Fed model has only one extra correct day, and all external has two at seed 42.
  The scorer treats a zero forecast as up, explaining the Gaussian direction score.
- A 20,000-draw paired circular-block analysis (seed 20260910) checks block lengths
  1, 2, 5, and 10 forecast windows. At length 5, the AI, inflation, and big-tech
  topic intervals include zero. Fed and recession remain indistinguishable.
- All-external mean WQL improvement over control is 0.60%; its interval includes
  zero. Its additional improvement over its own pretrained arm is 0.19%, also
  unresolved. Seed-aggregate intervals condition on the three observed seeds.
- Fed 80% interval coverage changes from 109/125 (87.2%) in the first chronological
  half to 86/125 (68.8%) in the second. The event-day slice is descriptive only.
- All 12 GDELT fields have 13 missing sessions in the common sample; the outage
  overlaps the first forecast origin. Those values remain masked.
- Training-only Fed/inflation news-share correlation is 0.711, and AI/semiconductor
  is 0.585. These support redundancy as a hypothesis, not a demonstrated cause.
- Three fine-tuned forecast sets contain one quantile crossing each. Metrics and
  raw quantiles are preserved; no postprocessing was silently applied.

The analysis settings, complete derived statistics, and input hashes are saved in
`output/pdf/report_analysis.json`. The paper builder is
`scripts/build_project_report.py`. Rebuilding requires matplotlib, numpy, pandas,
ReportLab, and the bundled Liberation Serif and Mono fonts available in this workspace.

## Final quality checks

All ten final pages were rendered with Poppler and visually checked, including a
second check after embedding the appendix's monospace font. The PDF contains five
figures, twelve numbered tables, four equations, nine external reference links,
and working internal citations. Text extraction and margin checks pass. The
analysis reproduces the saved numerical metrics; no raw training or prediction
artifacts were changed.

## Remaining research limitations

A strong presentation cannot replace missing empirical evidence. No untouched
prospective test, multiple-seed topic confirmation, stronger supervised quantile
baseline, shuffled-news control, or share-versus-tone isolation is claimed. The
paper describes these as future work. It also acknowledges selection/multiplicity,
base-pretraining overlap, retrospective data vintages, and SEC survivorship.
No numerical course grade can be certified without an assessment rubric.
