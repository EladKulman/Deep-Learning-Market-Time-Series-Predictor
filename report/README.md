# Edit and rebuild the report

**Edit the Markdown files in `sections/`, then run this command from the repository root:**

```bash
./scripts/build_report.sh
```

The updated PDF is saved to **`output/pdf/qqq_chronos2_project_report.pdf`**. Each build replaces that file. The dated September 23 PDF remains an archived copy.

You do not need to edit Python, run training, download data, or recalculate the experiments. The compiler reads your files directly and never rewrites them. If compilation fails, it prints an error and preserves the last successfully built PDF. Source parsing errors include the file involved.

## Where to edit

| Content | File |
|---|---|
| Title, authors, date | [sections/00_title.md](sections/00_title.md) |
| Abstract | [sections/01_abstract.md](sections/01_abstract.md) |
| Introduction and research questions | [sections/02_introduction.md](sections/02_introduction.md) |
| Background and related work | [sections/03_related_work.md](sections/03_related_work.md) |
| Data sources and availability | [sections/04_data.md](sections/04_data.md) |
| Chronos-2, LoRA, five-day forecasts | [sections/05_model_and_forecasts.md](sections/05_model_and_forecasts.md) |
| Experimental design and WQL | [sections/06_experimental_design.md](sections/06_experimental_design.md) |
| Initial training and topic results | [sections/07_initial_results.md](sections/07_initial_results.md) |
| Four-year training results | [sections/08_four_year_results.md](sections/08_four_year_results.md) |
| Pretrained permutation results | [sections/09_permutation_results.md](sections/09_permutation_results.md) |
| Before/after adaptation | [sections/10_adaptation_results.md](sections/10_adaptation_results.md) |
| Calibration and baseline comparison | [sections/11_calibration.md](sections/11_calibration.md) |
| Discussion | [sections/12_discussion.md](sections/12_discussion.md) |
| Limitations | [sections/13_limitations.md](sections/13_limitations.md) |
| Future work | [sections/14_future_work.md](sections/14_future_work.md) |
| Conclusion | [sections/15_conclusion.md](sections/15_conclusion.md) |
| Bibliography | [sections/16_references.md](sections/16_references.md) |
| Execution/reproducibility appendix | [sections/17_execution_record.md](sections/17_execution_record.md) |
| Exact feature and query appendix | [sections/18_feature_inventory.md](sections/18_feature_inventory.md) |

## Normal text

Use a blank line between paragraphs. Common Markdown formatting works:

```markdown
# Main section heading

## Subsection heading

A paragraph with **bold**, *italic*, `code`, and [a link](https://example.com).

- First point
- Second point

1. First step
2. Second step
```

`<!-- comments like this -->` on their own lines are hidden in the PDF. Existing `<br/>`, `<font ...>` and reference anchors are formatting helpers; keep them when editing the text around them.

Some blocks select a specific style. Edit the text between the markers and keep both markers:

```markdown
::: abstract
**Abstract.** Your revised abstract here.
:::
```

Other styles include `title`, `subtitle`, `author`, `meta`, `note`, and `reference`.

## Tables

Edit the corresponding CSV file in `tables/`. Each file's first row is its header. A CSV editor or spreadsheet app can edit these; when using a text editor, keep cells containing commas inside double quotes.

The section file contains the caption and layout:

```markdown
::: table ../tables/03_topic_comparison.csv widths=51,35,43,37 font=9
Your table caption here. Do not type a table number.
:::
```

Widths are in millimeters, one per column, with a maximum total of 166. Omit `widths` for equal columns. The compiler numbers tables in their appearance order. Tables currently stay together with their captions; split a table into two if it grows beyond one page.

Values are editable snapshots of the audited results. Rebuilding does **not** recalculate them, and it does not update numbers mentioned in prose or plotted in figures. If you revise a result, update all relevant places deliberately.

## Figures and equations

Figures are in `figures/`. Replace an image, or change its path and caption in the section:

```markdown
::: figure ../figures/ladder.png width=166
Your figure caption here. Do not type a figure number.
:::
```

Paths are relative to the section file. Figure numbers are automatic. PNG and JPEG files work; keep images short enough to fit on a page with their caption.

Equations use Matplotlib's LaTeX-style math syntax:

```markdown
::: equation 1
W = W_0 + (\alpha/r)BA
:::
```

The number after `equation` is the printed equation number. Full LaTeX packages are not required or supported.

The optional `scripts/refresh_report_figures.py` recreates the five original charts from archived evidence and predictions. It **overwrites those five images** and is not part of normal compilation. Run it only if you intentionally want those data-generated figures back; it uses the project's NumPy/Pandas dependencies.

## Section order, page breaks, references

[report.json](report.json) controls section order, whether a section starts on a new page, PDF metadata, running headers and the output path. To add a section, create its Markdown file and add it to the `sections` list. Text automatically flows onto another page when necessary.

A `::: pagebreak` block ending with `:::` adds a manual page break inside a file. A `::: spacer 9` block adds 9 mm of vertical space.

Section numbers, equation numbers, bibliography numbers, and prose mentions such as "Table 5" are manually maintained. The compiler automatically numbers only the figures and tables. A citation such as `[[1]](#ref1)` links to the `<a name="ref1"/>` anchor in the references file. Preserve unique matching anchors when editing references.

If you change the title or authors, edit both `00_title.md` (visible title page) and `report.json` (PDF metadata).

## Environment and alternate commands

On the current computer, the shell command uses `.venv` and the installed Codex ReportLab bundle automatically. It can also be invoked using its absolute path from another directory.

On another computer, Python 3.11+ with these small report-only dependencies is sufficient:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r report/requirements.txt
./scripts/build_report.sh
```

Liberation Serif is used when available; otherwise the compiler uses the DejaVu fonts included with Matplotlib. `REPORT_FONT_DIR` can point to a directory containing the Liberation Serif and Mono TTF files. Font changes may change pagination.

```bash
# Save a separate draft:
./scripts/build_report.sh --output output/pdf/my_draft.pdf

# Select another Python environment:
REPORT_PYTHON=/path/to/python ./scripts/build_report.sh

# Direct Python entry point:
.venv/bin/python scripts/build_report.py
```

The old `scripts/build_presentation_report.py` entry point now delegates to this compiler. The archived evidence and manuscript in `output/pdf/qqq_report_support/` are provenance for the original report, not the current editable source.
