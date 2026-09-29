# Appendix B  Exact inputs and source definitions

::: table ../tables/09_control_features.csv widths=32,134 font=8.5
The control contains 15 past-only and six known-future fields. Every experiment also includes the target history.
:::

::: table ../tables/10_gdelt_queries.csv widths=34,132 font=8.5
Each topic adds gdelt_TOPIC_news_share and gdelt_TOPIC_avg_tone. Queries use sourcelang:eng and no timeline smoothing. Article counts contribute to share construction but are not selected model inputs.
:::

::: table ../tables/11_external_features.csv widths=34,132 font=8.5
All external combines nine non-GDELT fields and 12 GDELT fields with the 21 controls.
:::

::: note
EPU and EMU derive from the policy uncertainty data family [[7]](#ref7). The FOMC model used in the pipeline is [LorenzoAleCon29/roberta-large-fomc-hawkish-dovish](https://huggingface.co/LorenzoAleCon29/roberta-large-fomc-hawkish-dovish), revision f4759d4ad3f1182f81d87e47ba603261740d36cf. Missing observations use the model's observation mask. The array-based Chronos interface preserves exchange-session order without fabricating holiday rows [[8]](#ref8).
:::
