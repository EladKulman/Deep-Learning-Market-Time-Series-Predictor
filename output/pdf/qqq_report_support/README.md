# QQQ / Chronos-2 report, 23 September 2026

Final PDF: `../qqq_chronos2_project_report_20260923.pdf`.

The report expands `refrences/project presentation.pptx` into an academic paper, following the single-column structure of `refrences/Example Project.pdf`. It covers both executed experiment rounds; no training was performed for this report.

## Evidence

- Round A: `models/news-ablation-869989/`, 18 LoRA fits and saved daily quantiles.
- Round B: `docs/results/walk_forward/`, 108 LoRA fits plus 32 pretrained cells. The local export includes window scores and metadata, not full later adapters or daily trained quantiles.
- Perturbations: `docs/results/permutation_pretrained/`, `permutation_control21/`, and `permutation_finetuned/`.
- `evidence.json`: computed aggregates, source SHA-256 hashes, paired bootstrap sensitivity checks, and an audit of metrics against window contributions.
- `source_screen_audit.json`: independent recomputation of first-round forecasts, parameter counts, and seed/topic comparisons.
- `figures/`: publication-resolution plots and equation images used by the report.
- `paper_text.md`: searchable manuscript text (equations and numeric plot values are in the PDF/figures and evidence).

The main four-year confidence intervals average seeds before circular block resampling, use five consecutive forecast windows per block within each year, equal year weights, 5,000 draws and seed 20260916. They are exploratory and not adjusted for multiple comparisons. The archived first-round audit uses its separately documented resampling settings.

## Clarifications relative to the slides

- The prediction target is adjusted-close daily log return, not index price level.
- Five daily returns are predicted together; target blocks do not overlap, but their historical contexts do.
- A relative WQL reduction measures quantile loss, not direction accuracy or profit.
- LoRA has 1,206,912 trainable parameters (about 1%), not approximately 0.6 million.
- The 57.2% always-up result belongs to June 2025-June 2026, not calendar 2025.
- Swapping a feature history is an offline perturbation, not removing it or proving how the model understands it.
- A small positive GDELT Fed-tone permutation effect exists, with an unadjusted interval; 'all news has zero effect' is too strong.
- Larger batches were not the only change between rounds, so the earlier rankings cannot be causally attributed to the batch issue alone.
- Wider intervals and similar importance profiles support a limited interpretation, not proof that training only learns unconditional scale.
- All examined periods are development evidence; an untouched prospective block remains future work.

## Editing and rebuilding

This folder preserves evidence and the original September 23 manuscript. The live editable
source is now `report/sections/`, with CSV tables and figures under `report/`.

```sh
./scripts/build_report.sh
```

This writes `output/pdf/qqq_chronos2_project_report.pdf`. See `report/README.md` for instructions.
The normal build does not run the evidence analyzer, modify source files, refresh plots,
or start training. The dated September 23 PDF remains an archived copy.

To intentionally recompute the evidence aggregates (separate from editing), run
`scripts/analyze_presentation_report.py`. This does not synchronize manually edited prose,
tables or figures. Historical hyperparameters come from saved run metadata, not current
configuration defaults.
