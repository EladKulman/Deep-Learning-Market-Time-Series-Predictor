# Final report package

Implemented `report_plan.md` using both completed training rounds and the example paper's academic structure.

## Deliverables

- [Final paper PDF](../output/pdf/qqq_chronos2_final_report.pdf): 11 pages, four figures, ten tables, methods and WQL equations, results, limitations, references, reproducibility details, and exact selected feature/topic inventory.
- [Presentation](qqq_chronos2_presentation_v2.pptx): editable PowerPoint using the selected Simple Light Mode template. Slides 1–12 form the ten-minute talk; slides 13–16 are backups. Six charts include editable data workbooks and five tables are native PowerPoint tables.
- [Presenter notes](presenter_notes.md): spoken script, Elad/Tom assignments, timing targets, and slide-level sources. These notes are also embedded in the PPTX.
- [Timing schedule](presentation_timing.json): 600 seconds for the main talk. The suggested outline in the plan totaled eleven minutes, so the introduction was shortened to meet the ten-minute requirement.
- [Paper text](paper_text.md): generated text/table extract for searching. The PDF and its builder are authoritative for equations, formatting, and figures.

## Evidence and interpretation

Round A is the original 18-fit source/topic screen, archived under `models/news-ablation-869989/`. Round B is the later four-year study with 108 LoRA fits, archived under `docs/results/walk_forward/`. Paired input diagnostics are under `docs/results/permutation_finetuned/runs/`.

The final report distinguishes these rounds instead of treating the later study as a controlled batch-only correction. Dates, data snapshots, training budgets, and some library versions also changed. The original per-topic ranking is presented as exploratory; the later cumulative ladder does not replace a matched rerun of all six topics.

The supported conclusion is **no reliable incremental news benefit under the tested recipe**, with wider forecast intervals and broadly similar input-importance patterns after adaptation. The report does not claim that news has no information, that input effects are exactly unchanged, or that a batch-size issue alone caused the initial results. It also reports the comparison with the simple Gaussian baseline.

[evidence.json](evidence.json) contains source-derived metrics and SHA-256 hashes of the analyzed files. The analysis reconciles window contributions with saved cell WQL to floating-point precision. Headline uncertainty intervals use 5,000 within-year circular-block bootstrap draws, average seeds before resampling, and remain exploratory. No new model training was performed for these deliverables.

## Rebuild

Run from the repository root. `analyze_evidence.py` requires NumPy and pandas; `build_paper.py` also requires Matplotlib and ReportLab. The paper builder reuses rendering helpers from `scripts/build_project_report.py` and its existing analysis artifact. The presentation builder requires the configured Codex Artifact Tool runtime and the installed Simple Light Mode reference deck; its local paths are declared at the top of the script.

```sh
.venv/bin/python 'final report/analyze_evidence.py'
```

For the PDF in this workspace:

```sh
env FONTCONFIG_FILE="$PWD/tmp/pdfs/final_report/fontconfig.xml" \
  XDG_CACHE_HOME=/tmp/qqq-report-xdg-cache \
  MPLCONFIGDIR=/tmp/qqq-report-mpl-cache \
  PYTHONPATH='/Users/eladkulman/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages' \
  .venv/bin/python 'final report/build_paper.py'
```

For a new presentation revision, use a new output filename so the validated deliverable is preserved:

```sh
env DECK_FINAL_NAME=qqq_chronos2_presentation_v3.pptx \
  '/Users/eladkulman/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node' \
  'final report/build_presentation.mjs'
```

Build previews and validation receipts live in `tmp/pdfs/final_report/` and `tmp/final_report_deck/`. All paper pages and all slides were visually reviewed. The final PPTX passed package, layout, font-policy, embedded chart-data, and Artifact Tool import checks. These checks do not claim a native Microsoft PowerPoint execution test.

The earlier `output/pdf/qqq_chronos2_project_report.pdf` remains available as the historical report. Use `qqq_chronos2_final_report.pdf` for the combined final paper.
