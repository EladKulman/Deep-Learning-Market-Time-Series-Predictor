# Appendix A  Execution record and reproducibility

::: table ../tables/08_execution_record.csv widths=34,49,83 font=8.3
The 126 substantive LoRA fits comprise 18 first-round fits plus 108 later fits. Additional pretrained exploration and perturbation inference ran on MPS; these are not additional training seeds.
:::

::: note
Both rounds pin model revision <font name="PaperMono" size="7.7">29ec3766d36d6f73f0696f85560a422f50e8498c</font>. Round A data SHA-256 is <font name="PaperMono" size="7.2">e0c8d48b38059eb70d9f91931bff68678dbcc46aeadab1502ded1b9ee11df037</font>; Round B is <font name="PaperMono" size="7.2">b3beb046933b15f1f75c098bfa6df30af4106395302ea4d8b68711322a262e34</font>. Training used an NVIDIA RTX 2080 Ti (11 GB) through Slurm. Both GPU runtimes record Chronos 2.3.2 and PyTorch 2.11.0+cu128. Transformers/PEFT change from 5.16.1/0.20.0 to 5.17.0/0.21.0. Later metadata records dirty working trees, making source hashes material to reproduction.
:::

::: note
Round A predictions and adapters reside in <font name="PaperMono" size="8">models/news-ablation-869989/</font>. Round B local exports in <font name="PaperMono" size="8">docs/results/walk_forward/</font> include per-window loss summaries and 140 cell metadata/metric pairs (108 trained, 32 pretrained-only). Full later adapters and per-day trained predictions are not in that local export. The final report analyzer checks cell metrics against window means to 3.4e-16 and records source hashes in <font name="PaperMono" size="8">output/pdf/qqq_report_support/evidence.json</font>. Historical recipes come from saved run metadata; the current configuration files have since changed. No new model was trained to prepare this report.
:::
