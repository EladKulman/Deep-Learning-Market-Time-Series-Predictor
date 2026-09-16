#!/usr/bin/env python3
"""Create the final workshop-style PDF report for the QQQ Chronos-2 project."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "models/news-ablation-869989"
TMP = ROOT / "tmp/pdfs/qqq_report_assets"
OUTPUT = ROOT / "output/pdf/qqq_chronos2_project_report.pdf"

NAVY = "#17365D"
BLUE = "#4F81BD"
LIGHT_BLUE = "#DCE6F1"
ORANGE = "#F4B183"
GREY = "#777777"
LIGHT_GREY = "#E7E6E6"
RED = "#B6403A"


def setup_dirs() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def overall(path: Path, model: str = "fine_tuned") -> pd.Series:
    frame = pd.read_csv(path / "validation_metrics.csv", dtype={"horizon": str})
    row = frame[(frame["model"] == model) & (frame["horizon"] == "overall")]
    if len(row) != 1:
        raise ValueError(f"Missing overall {model} metrics in {path}")
    return row.iloc[0]


def save_source_chart() -> Path:
    rows = []
    labels = {
        "control": "Control",
        "fomc_tone": "FOMC tone",
        "sec_disclosures": "SEC disclosures",
        "gdelt_all": "All GDELT topics",
        "all_external": "All external sources",
        "uncertainty": "EPU / EMU",
    }
    for setup, label in labels.items():
        metric = overall(ARTIFACT_ROOT / f"{setup}-seed42")
        rows.append((label, float(metric["weighted_quantile_loss"]), setup))
    frame = pd.DataFrame(rows, columns=["label", "wql", "setup"]).sort_values("wql", ascending=False)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    palette = [NAVY if s == "all_external" else GREY if s == "control" else BLUE for s in frame.setup]
    bars = ax.barh(frame.label, frame.wql, color=palette)
    ax.set_xlim(0.684, 0.699)
    ax.set_xlabel("Weighted quantile loss (lower is better)")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, frame.wql):
        ax.text(value + 0.0002, bar.get_y() + bar.get_height() / 2, f"{value:.6f}", va="center", fontsize=8)
    fig.tight_layout()
    path = TMP / "source_screen.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_robustness_chart() -> Path:
    ranking = pd.read_csv(ARTIFACT_ROOT / "news_ablation_ranking.csv")
    wanted = {
        "control": ("Control", GREY),
        "uncertainty": ("EPU / EMU", ORANGE),
        "all_external": ("All external", NAVY),
    }
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    for setup, (label, color) in wanted.items():
        rows = ranking[ranking.setup == setup].sort_values("seed")
        ax.plot(rows.seed, rows.weighted_quantile_loss, marker="o", linewidth=1.8,
                markersize=5, label=label, color=color)
    ax.set_xticks([42, 43, 44])
    ax.set_xlabel("Fine-tuning seed")
    ax.set_ylabel("WQL")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    fig.tight_layout()
    path = TMP / "seed_robustness.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_topic_chart() -> Path:
    frame = pd.read_csv(ARTIFACT_ROOT / "gdelt_topic_plus_all_news_comparison.csv")
    frame = frame[frame.setup != "control"].sort_values("improvement_vs_control_percent")
    labels = frame.topic.replace({"All external/news sources combined": "All external sources"})
    palette = []
    for setup in frame.setup:
        if setup == "gdelt_fed":
            palette.append(NAVY)
        elif setup == "all_external":
            palette.append(GREY)
        else:
            palette.append(BLUE)
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    bars = ax.barh(labels, frame.improvement_vs_control_percent, color=palette)
    ax.set_xlim(0, 1.42)
    ax.set_xlabel("WQL reduction versus seed-42 control (%)")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, frame.improvement_vs_control_percent):
        ax.text(value + 0.025, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%", va="center", fontsize=8)
    fig.tight_layout()
    path = TMP / "topic_screen.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_forecast_chart() -> Path:
    frame = pd.read_csv(ARTIFACT_ROOT / "gdelt_fed-seed42/validation_predictions.csv")
    frame = frame[frame.model == "fine_tuned"].copy()
    date_col = "timestamp" if "timestamp" in frame else "date"
    frame[date_col] = pd.to_datetime(frame[date_col])
    qcols = [c for c in frame if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()]
    levels = {float(c[1:]): c for c in qcols}
    q10 = levels[min(levels, key=lambda x: abs(x - 0.10))]
    q50 = levels[min(levels, key=lambda x: abs(x - 0.50))]
    q90 = levels[min(levels, key=lambda x: abs(x - 0.90))]
    x = frame[date_col]
    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    ax.fill_between(x, frame[q10] * 100, frame[q90] * 100, color=LIGHT_BLUE,
                    alpha=0.80, label="10%-90% forecast interval")
    ax.plot(x, frame.actual * 100, color=GREY, linewidth=0.8, alpha=0.85,
            label="Actual QQQ return")
    ax.plot(x, frame[q50] * 100, color=NAVY, linewidth=1.0, label="Median forecast")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Daily log return (%)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.19))
    fig.tight_layout()
    path = TMP / "fed_forecasts.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_timeline_chart() -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 1.45))
    train_start = pd.Timestamp("2017-01-03")
    train_end = pd.Timestamp("2025-06-27")
    val_start = pd.Timestamp("2025-06-30")
    val_end = pd.Timestamp("2026-06-30")
    ax.barh([0], [(train_end - train_start).days], left=mdates.date2num(train_start),
            height=0.45, color=BLUE, label="Training: 2,133 sessions")
    ax.barh([0], [(val_end - val_start).days], left=mdates.date2num(val_start),
            height=0.45, color=ORANGE, label="Validation allocation: 252 sessions")
    ax.set_xlim(train_start, val_end)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.tight_layout()
    path = TMP / "experiment_timeline.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def report_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Times-Roman", fontSize=18,
            leading=22, alignment=TA_CENTER, textColor=colors.black, spaceAfter=12,
        ),
        "author": ParagraphStyle(
            "Author", parent=base["Normal"], fontName="Times-Roman", fontSize=11.5,
            leading=14, alignment=TA_CENTER, spaceAfter=5,
        ),
        "meta": ParagraphStyle(
            "Meta", parent=base["Normal"], fontName="Times-Roman", fontSize=9.2,
            leading=11, alignment=TA_CENTER, textColor=colors.HexColor(GREY), spaceAfter=16,
        ),
        "abstract_head": ParagraphStyle(
            "AbstractHead", parent=base["Heading2"], fontName="Times-Bold", fontSize=10,
            leading=12, alignment=TA_CENTER, spaceBefore=2, spaceAfter=4,
        ),
        "abstract": ParagraphStyle(
            "Abstract", parent=base["Normal"], fontName="Times-Roman", fontSize=8.7,
            leading=10.5, alignment=TA_JUSTIFY, leftIndent=28, rightIndent=28, spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Times-Bold", fontSize=14,
            leading=17, spaceBefore=8, spaceAfter=6, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Times-Bold", fontSize=11.2,
            leading=13.5, spaceBefore=6, spaceAfter=4, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Times-Roman", fontSize=9.15,
            leading=11.2, alignment=TA_JUSTIFY, firstLineIndent=13, spaceAfter=5.2,
            allowWidows=0, allowOrphans=0,
        ),
        "body_noindent": ParagraphStyle(
            "BodyNoIndent", parent=base["BodyText"], fontName="Times-Roman", fontSize=9.15,
            leading=11.2, alignment=TA_JUSTIFY, firstLineIndent=0, spaceAfter=5.2,
        ),
        "appendix": ParagraphStyle(
            "Appendix", parent=base["BodyText"], fontName="Times-Roman", fontSize=9.15,
            leading=11.2, alignment=TA_LEFT, firstLineIndent=0, spaceAfter=5.2,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName="Times-Roman", fontSize=8.8,
            leading=10.7, leftIndent=13, firstLineIndent=-7, bulletIndent=4, spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["Normal"], fontName="Times-Roman", fontSize=8.2,
            leading=9.8, alignment=TA_CENTER, spaceBefore=2, spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["Normal"], fontName="Times-Roman", fontSize=7.7,
            leading=9.2, alignment=TA_LEFT, spaceAfter=3,
        ),
        "reference": ParagraphStyle(
            "Reference", parent=base["Normal"], fontName="Times-Roman", fontSize=8.1,
            leading=9.8, leftIndent=18, firstLineIndent=-18, spaceAfter=5,
        ),
    }


def P(text: str, style, **kwargs) -> Paragraph:
    return Paragraph(text, style, **kwargs)


def make_table(data, widths, styles, font_size=7.2, header=True, aligns=None):
    rows = []
    for r, row in enumerate(data):
        converted = []
        for cell in row:
            cell_style = ParagraphStyle(
                f"Cell-{r}-{len(converted)}", parent=styles["small"], fontSize=font_size,
                leading=font_size + 1.6, alignment=TA_LEFT, textColor=colors.white if r == 0 and header else colors.black,
            )
            converted.append(P(str(cell), cell_style))
        rows.append(converted)
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7B7B7")),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ])
        if len(rows) > 1:
            commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]))
    if aligns:
        for col, alignment in aligns.items():
            commands.append(("ALIGN", (col, 1 if header else 0), (col, -1), alignment))
    table.setStyle(TableStyle(commands))
    return table


def pipeline_drawing() -> Drawing:
    d = Drawing(470, 90)
    boxes = [
        (4, "Market and QQQ"),
        (101, "Rates and calendar"),
        (198, "News, Fed and SEC"),
        (295, "Availability alignment"),
        (392, "Chronos-2 + LoRA"),
    ]
    for x, label in boxes:
        fill = colors.HexColor(LIGHT_BLUE if x < 295 else "#E2F0D9" if x == 295 else "#FFF2CC")
        d.add(Rect(x, 34, 78, 34, rx=4, ry=4, fillColor=fill, strokeColor=colors.HexColor(NAVY), strokeWidth=0.8))
        words = label.split()
        if len(words) > 2:
            split = len(words) // 2
            lines = [" ".join(words[:split]), " ".join(words[split:])]
        else:
            lines = [label]
        for i, line in enumerate(lines):
            d.add(String(x + 39, 53 - i * 10 + (5 if len(lines) == 1 else 0), line,
                         fontName="Times-Roman", fontSize=7.8, textAnchor="middle"))
    for x in [82, 179, 276, 373]:
        d.add(Line(x + 3, 51, x + 17, 51, strokeColor=colors.HexColor(NAVY), strokeWidth=1.0))
        d.add(Polygon([x + 17, 51, x + 12, 54, x + 12, 48], fillColor=colors.HexColor(NAVY), strokeColor=None))
    d.add(String(235, 12, "Five separate daily QQQ return distributions at each forecast origin",
                 fontName="Times-Italic", fontSize=8.3, textAnchor="middle"))
    return d


def image_with_caption(path: Path, width: float, height: float, caption: str, number: int, styles):
    return KeepTogether([
        Image(str(path), width=width, height=height),
        P(f"<b>Fig. {number}:</b> {caption}", styles["caption"]),
    ])


def add_bullet(story, text, styles):
    story.append(P(f"- {text}", styles["bullet"]))


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(A4[0] / 2, 14 * mm, str(doc.page))
    canvas.restoreState()


def build_report() -> None:
    setup_dirs()
    source_chart = save_source_chart()
    robustness_chart = save_robustness_chart()
    topic_chart = save_topic_chart()
    forecast_chart = save_forecast_chart()
    timeline_chart = save_timeline_chart()
    styles = report_styles()

    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=29 * mm, rightMargin=29 * mm,
        topMargin=23 * mm, bottomMargin=23 * mm,
        title="Workshop On Deep Learning - QQQ Forecasting with Chronos-2",
        author="Elad Kulman and Tom Weitman",
        subject="Chronos-2 pretrained inference and LoRA fine-tuning for QQQ return forecasting",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=page_number)])
    story = []

    # Page 1
    story.extend([
        Spacer(1, 12 * mm),
        P("Workshop On Deep Learning - QQQ Forecasting with Chronos-2", styles["title"]),
        P("Elad Kulman and Tom Weitman", styles["author"]),
        P("Group 13 - Tel Aviv University", styles["meta"]),
        P("Abstract", styles["abstract_head"]),
        P(
            "We study whether market, macroeconomic, calendar, disclosure, and news signals improve "
            "probabilistic forecasts of QQQ daily log returns. The forecasting model is Amazon "
            "Chronos-2, evaluated both with its original pretrained weights and after parameter-efficient "
            "LoRA fine-tuning. All full experiments use 2,133 training sessions and predict five separate "
            "daily returns at each of 50 non-overlapping validation origins, giving 250 outcomes. Across "
            "three seeds, the model using all external sources achieved mean weighted quantile loss 0.689439 "
            "versus 0.693582 for the control. In the single-topic GDELT screen, Federal Reserve news ranked "
            "first at 0.686526, narrowly ahead of recession news at 0.686796. The differences between the "
            "leading topics are not yet statistically resolved, so the results support further testing "
            "rather than a trading claim.",
            styles["abstract"],
        ),
        P("1&nbsp;&nbsp;Introduction", styles["h1"]),
        P(
            "Daily equity returns are noisy, weakly predictable, and dominated by changing volatility "
            "regimes. These properties create a difficult test for time-series foundation models. A useful "
            "forecast must describe uncertainty as well as a central estimate; a model that predicts a "
            "small return near zero can look competitive under point-error metrics while contributing little "
            "information about risk.", styles["body"],
        ),
        P(
            "Our project asks whether external information tells a pretrained forecasting model how the "
            "distribution of future Nasdaq-100 returns should change. The target is the one-day adjusted-close "
            "log return of QQQ. At every origin, the model directly predicts distributions for trading days "
            "one through five. This is a five-output forecast, not a cumulative weekly-return prediction.",
            styles["body"],
        ),
        P(
            "The original proposal considered Bitcoin and a Temporal Fusion Transformer. We changed the "
            "asset to QQQ because public macro, Federal Reserve, SEC, volatility, and technology-news data "
            "provide a stronger setting for controlled covariate experiments. Chronos-2 replaced the custom "
            "architecture because it supplies a pretrained baseline, supports past and known-future "
            "covariates, and can be adapted with a small LoRA checkpoint [1, 2].",
            styles["body"],
        ),
        P(
            "The main contribution is an executed ablation study. Every news setup starts from the same "
            "pinned Chronos-2 revision, keeps a common 21-feature market and calendar control, trains only on "
            "data before the validation interval, and is scored on the same 250 realized returns. This design "
            "lets us ask which source or narrative adds information beyond observable market conditions.",
            styles["body"],
        ),
    ])

    # Page 2
    story.extend([
        PageBreak(),
        P("2&nbsp;&nbsp;Related work", styles["h1"]),
        P("2.1&nbsp;&nbsp;Time-series foundation models", styles["h2"]),
        P(
            "Chronos-2 is a pretrained universal forecasting model that accepts univariate series, "
            "multivariate groups, and covariates. Its group-attention mechanism shares information among "
            "the target and related series, enabling zero-shot covariate-informed forecasting [1]. We use "
            "the original pretrained model as a baseline and keep its revision fixed across every experiment.",
            styles["body"],
        ),
        P(
            "Finance is a particularly strict application. Rahimikia, Ni, and Wang report that off-the-shelf "
            "time-series foundation models struggle on daily excess returns and that domain adaptation, "
            "dataset scale, and tuning materially affect performance [3]. This motivates our emphasis on "
            "calibration, matched controls, and repeated seeds instead of interpreting small point-estimate "
            "differences as dependable return predictability.", styles["body"],
        ),
        P("2.2&nbsp;&nbsp;Low-rank adaptation", styles["h2"]),
        P(
            "LoRA freezes the pretrained parameters and learns low-rank updates inside selected transformer "
            "layers [2]. It reduces the number of trainable parameters and allows one compact checkpoint per "
            "feature configuration. In this project, every setup receives a fresh LoRA adapter trained from "
            "the original base weights. Topic models never inherit the previously fine-tuned control adapter.",
            styles["body"],
        ),
        P("2.3&nbsp;&nbsp;News and policy signals", styles["h2"]),
        P(
            "News volume and tone can represent attention and sentiment, but they also contain overlapping "
            "and noisy signals. GDELT's DOC API reports topic coverage as a share of monitored global news and "
            "provides a tone timeline [5]. We retain both daily share and average tone for six topics: artificial "
            "intelligence, semiconductors, the Federal Reserve, inflation, big-tech earnings, and recession.",
            styles["body"],
        ),
        P(
            "Economic Policy Uncertainty and Equity Market Uncertainty provide longer-running daily measures "
            "of news-based uncertainty [6]. Federal Reserve communication is represented separately by a "
            "sentence classifier that scores 170 policy statements as hawkish, dovish, or neutral [7]. SEC "
            "filing and earnings-event activity supplies a disclosure channel that differs from media coverage.",
            styles["body"],
        ),
        P("2.4&nbsp;&nbsp;Evaluation principle", styles["h2"]),
        P(
            "The primary score is weighted quantile loss (WQL), computed over 21 forecast quantiles and "
            "normalized by the mean absolute realized return. Lower WQL rewards a distribution that is both "
            "sharp and calibrated. We report mean absolute error (MAE), root mean squared error, directional "
            "accuracy, 10%-90% and 1%-99% interval coverage, and paired bootstrap intervals as supporting "
            "diagnostics.", styles["body"],
        ),
        P(
            "WQL = 2 x mean pinball loss / mean absolute realized return. Directional accuracy uses the median "
            "forecast only; it therefore answers a different question from WQL.", styles["body_noindent"],
        ),
    ])

    # Page 3
    story.extend([
        PageBreak(),
        P("3&nbsp;&nbsp;Dataset and data collection", styles["h1"]),
        P(
            "The processed table contains 5,201 NYSE sessions from January 2006 through September 4, 2026, "
            "with 2005 QQQ history used to initialize rolling features. The controlled news experiments use "
            "the common interval January 3, 2017 through June 30, 2026 because all six GDELT topics are "
            "available through that date. The target is QQQ adjusted-close one-day log return.", styles["body"],
        ),
        P("3.1&nbsp;&nbsp;Sources and roles", styles["h2"]),
    ])
    data_table = [
        ["Source", "Signals used", "Role at forecast time"],
        ["QQQ / Yahoo", "Gap, range volatility, momentum, volume z-score, distance to 200-day mean", "Past only"],
        ["CBOE / ETFs", "VXN change, VXN-VIX spread, VIX term ratio, TLT and HYG returns", "Past only"],
        ["FRED / ALFRED", "2-year and real-yield changes, 10y-2y slope, vintage CPI year-over-year", "Past only"],
        ["Calendar", "FOMC, CPI, NFP, month-end, options-expiration timing", "Known future"],
        ["GDELT", "Daily news share and average tone for six technology and macro topics", "Past only, T-1"],
        ["Policy / filings", "EPU, EMU, FOMC classifier state, earnings and SEC filing counts", "Past only"],
    ]
    story.extend([
        make_table(data_table, [27 * mm, 91 * mm, 30 * mm], styles, font_size=7.0),
        P("<b>Table 1:</b> Main data sources and their role in the forecasting contract.", styles["caption"]),
        P("3.2&nbsp;&nbsp;Point-in-time alignment", styles["h2"]),
        P(
            "Each observation is assigned by the first QQQ close at which it could be known. Rates follow "
            "their publication schedule, CPI uses ALFRED release vintages, SEC forms use UTC acceptance "
            "timestamps, and completed daily news is lagged to the next market session. CBOE indices and "
            "daily commodity or currency bars receive a conservative one-day lag. Scheduled calendar fields "
            "are the only values allowed in the future input horizon.", styles["body"],
        ),
        P(
            "Missing past data remains missing and reaches Chronos-2 through its observation mask. A failed "
            "news request is never converted into a zero count. This matters because a zero means that the "
            "source was observed and contained no matching news, whereas a missing value means the source was "
            "unavailable.", styles["body"],
        ),
        pipeline_drawing(),
        P("<b>Fig. 1:</b> Data flow from source-specific observations to the five-day probabilistic forecast.", styles["caption"]),
        P("3.3&nbsp;&nbsp;Feature profiles", styles["h2"]),
        P(
            "The complete table contains 55 past covariates and 14 known-future calendar values. The common "
            "control uses 21 selected signals: six QQQ transforms, five market variables, four rate and macro "
            "variables, and six calendar variables. A single-topic GDELT run adds exactly two columns - that "
            "topic's news share and tone - for 23 total features. The all-external setup contains 42 features.",
            styles["body"],
        ),
    ])

    # Page 4
    story.extend([
        PageBreak(),
        P("4&nbsp;&nbsp;Methodology", styles["h1"]),
        P("4.1&nbsp;&nbsp;Forecasting protocol", styles["h2"]),
        P(
            "Every full run uses the pinned <i>amazon/chronos-2</i> revision "
            "29ec3766d36d6f73f0696f85560a422f50e8498c. At a validation origin, the model receives "
            "observations through the previous QQQ close and produces 21 quantiles for each of the next five "
            "trading-day returns. After those five realized sessions, the origin advances by five sessions. "
            "Fifty non-overlapping origins therefore score 250 daily outcomes; the final two sessions of the "
            "252-session allocation cannot form a complete block.", styles["body"],
        ),
        image_with_caption(timeline_chart, 150 * mm, 31 * mm,
                           "Chronological train and development-validation split used by every full run.", 2, styles),
        P("4.2&nbsp;&nbsp;Pretrained and fine-tuned arms", styles["h2"]),
        P(
            "For each feature configuration we first evaluate the original pretrained weights on all 250 "
            "outcomes. We then restart from those same weights, train a fresh LoRA adapter on the 2,133 "
            "pre-validation sessions, and repeat the identical evaluation. Training uses context length 512, "
            "200 optimizer steps, learning rate 1e-5, batch size 8, and seed 42 unless a robustness run "
            "explicitly selects seed 43 or 44. No validation outcome is used for checkpoint selection.",
            styles["body"],
        ),
        P("4.3&nbsp;&nbsp;Experiment ladder", styles["h2"]),
    ])
    experiment_table = [
        ["Stage", "Executed configurations", "Purpose"],
        ["GPU smoke", "Pretrained, five-step LoRA, Gaussian; 15 outcomes", "Verify CUDA, training, save/reload and prediction"],
        ["Source screen", "Control; EPU/EMU; FOMC tone; SEC; all GDELT; all external", "Measure source-family value at seed 42"],
        ["Robustness", "Control, EPU/EMU and all external at seeds 43 and 44", "Check whether source ranking repeats"],
        ["Topic screen", "Six GDELT topics, one pair of share/tone columns per run", "Identify the most informative narrative"],
    ]
    story.extend([
        make_table(experiment_table, [25 * mm, 76 * mm, 47 * mm], styles, font_size=7.0),
        P("<b>Table 2:</b> Executed experiment stages. The full stages comprise 18 LoRA fits.", styles["caption"]),
        P("4.4&nbsp;&nbsp;Baselines and uncertainty", styles["h2"]),
        P(
            "A zero-mean Gaussian baseline estimates volatility from the last 60 target returns at each "
            "origin. Paired comparisons preserve the 50 common forecast blocks and bootstrap their WQL "
            "differences. A 95% interval entirely below zero favors the candidate over control. Reusing the "
            "same development period for several comparisons still creates selection risk, so a later "
            "untouched block is required for a final claim.", styles["body"],
        ),
    ])

    # Page 5
    story.extend([
        PageBreak(),
        P("5&nbsp;&nbsp;Executed pretrained and fine-tuning results", styles["h1"]),
        P("5.1&nbsp;&nbsp;Infrastructure smoke test", styles["h2"]),
        P(
            "TAU Slurm job 869927 completed on an NVIDIA GeForce RTX 2080 Ti. On its deliberately small "
            "15-outcome sample, WQL decreased from 0.734899 with pretrained weights to 0.710572 after five "
            "LoRA steps, a 3.31% improvement. MAE improved 1.84% and RMSE improved 2.08%. The Gaussian "
            "baseline remained best at WQL 0.667505. This run establishes execution correctness, not "
            "forecasting skill.", styles["body"],
        ),
        P("5.2&nbsp;&nbsp;Seed-42 source-family screen", styles["h2"]),
        image_with_caption(source_chart, 154 * mm, 66 * mm,
                           "Fine-tuned seed-42 WQL for the common control and five source configurations.", 3, styles),
    ])
    source_results = [
        ["Setup", "Features", "WQL", "Delta vs control", "Direction", "80% coverage"],
        ["EPU / EMU", "23", "0.688112", "-0.007301", "57.2%", "77.6%"],
        ["All external", "42", "0.688303", "-0.007109", "58.0%", "79.6%"],
        ["All GDELT", "33", "0.692348", "-0.003065", "55.6%", "77.6%"],
        ["SEC disclosures", "24", "0.692848", "-0.002565", "58.0%", "79.2%"],
        ["FOMC tone", "25", "0.694609", "-0.000804", "57.6%", "77.6%"],
        ["Control", "21", "0.695413", "0.000000", "56.0%", "78.0%"],
    ]
    story.extend([
        make_table(source_results, [35 * mm, 17 * mm, 23 * mm, 30 * mm, 22 * mm, 25 * mm], styles, font_size=6.7),
        P("<b>Table 3:</b> Seed-42 source-family results. Lower WQL is better.", styles["caption"]),
        P(
            "EPU/EMU ranked first at seed 42 and beat control in 64% of forecast blocks. Its paired interval "
            "was below zero. This initial result did not repeat at seeds 43 and 44, motivating the explicit "
            "robustness stage.", styles["body"],
        ),
    ])

    # Page 6
    story.extend([
        PageBreak(),
        P("6&nbsp;&nbsp;Robustness and GDELT topic attribution", styles["h1"]),
        P("6.1&nbsp;&nbsp;Three-seed robustness", styles["h2"]),
        image_with_caption(robustness_chart, 154 * mm, 56 * mm,
                           "WQL across fine-tuning seeds 42, 43, and 44 for the repeated configurations.", 4, styles),
        P(
            "Across three seeds, all external sources produced mean WQL 0.689439 versus 0.693582 for control, "
            "a 0.60% improvement, and won two of three seeds. Its mean directional accuracy was 57.47% versus "
            "56.67% for control. The aggregate paired interval (-0.010021, 0.001424) crossed zero. EPU/EMU's "
            "strong seed-42 advantage disappeared in the repetitions, with mean WQL 0.693473.", styles["body"],
        ),
        P("6.2&nbsp;&nbsp;Single-topic GDELT screen", styles["h2"]),
        image_with_caption(topic_chart, 154 * mm, 78 * mm,
                           "WQL reduction versus the matched seed-42 control. The grey bar is the previous all-external model.", 5, styles),
    ])
    topic_results = [
        ["Rank", "News data", "WQL", "Reduction", "Correct direction"],
        ["1", "Federal Reserve", "0.686526", "1.28%", "144 / 250"],
        ["2", "Recession", "0.686796", "1.24%", "144 / 250"],
        ["3", "Semiconductors", "0.687499", "1.14%", "144 / 250"],
        ["4", "AI", "0.687580", "1.13%", "143 / 250"],
        ["5", "Inflation", "0.688131", "1.05%", "142 / 250"],
        ["6", "All external sources", "0.688303", "1.02%", "145 / 250"],
        ["7", "Big-tech earnings", "0.688750", "0.96%", "142 / 250"],
        ["-", "Control: no news", "0.695413", "-", "140 / 250"],
    ]
    story.extend([
        make_table(topic_results, [15 * mm, 57 * mm, 25 * mm, 25 * mm, 30 * mm], styles, font_size=6.7),
        P("<b>Table 4:</b> Eight-case topic comparison on the same 250 daily outcomes.", styles["caption"]),
    ])

    # Page 7
    story.extend([
        PageBreak(),
        P("7&nbsp;&nbsp;Interpretation", styles["h1"]),
        P(
            "Federal Reserve news has the best seed-42 WQL point estimate, with recession news nearly tied. "
            "All six isolated topics beat the fine-tuned control and their paired intervals against control "
            "were below zero. Direct comparisons among the topics remained unresolved: the Fed-minus-recession "
            "difference was -0.000269 with a 95% interval from -0.002163 to 0.001596.", styles["body"],
        ),
        image_with_caption(forecast_chart, 154 * mm, 61 * mm,
                           "Fed-topic fine-tuned forecasts across the 250 development outcomes.", 6, styles),
        P(
            "The all-external model was not the worst case. It ranked sixth by WQL, ahead of big-tech "
            "earnings and control, and achieved the highest direction count at 145 of 250. Its distribution "
            "was wider than the Fed model and its improvement over its own pretrained arm was smaller. This "
            "pattern is consistent with redundant or noisy covariates diluting a cleaner signal, while still "
            "helping the sign of the median forecast.", styles["body"],
        ),
        P("7.1&nbsp;&nbsp;Limitations", styles["h2"]),
    ])
    limitations = [
        "The 2025-2026 interval is development validation and was reused for source selection.",
        "The topic screen has one fine-tuning seed; Fed and recession require repetitions.",
        "GDELT begins in 2017 and contains a June 2025 outage; unavailable observations remain masked.",
        "GDELT share and generic tone measure media coverage, not verified market-relevant sentiment.",
        "Economic, price, and filing histories can be revised; the pipeline applies conservative availability rules but is not a complete vintage archive.",
        "The SEC issuer basket uses current technology leaders and therefore contains survivorship hindsight.",
        "Chronos-2 pretraining may include historical market patterns; overlap cannot be ruled out.",
        "No transaction costs, portfolio construction, or tradability claim is evaluated.",
    ]
    for item in limitations:
        add_bullet(story, item, styles)
    story.extend([
        P("7.2&nbsp;&nbsp;Next experiments", styles["h2"]),
        P(
            "The immediate next run should repeat Fed and recession at seeds 43 and 44, followed by a small "
            "Fed-plus-recession combination. The selected configuration should then be frozen and evaluated "
            "once on newly arriving sessions after September 4, 2026. Later work can test SPY and XLK as "
            "related targets, compare a quantile tree baseline, and obtain a more reliable point-in-time news "
            "archive.", styles["body"],
        ),
        P("8&nbsp;&nbsp;Conclusion", styles["h1"]),
        P(
            "The project establishes a reproducible Chronos-2 forecasting pipeline and executes 18 full LoRA "
            "fits plus a GPU smoke test. External information produced small improvements in probabilistic "
            "QQQ forecasts, and the topic screen points to Federal Reserve and recession coverage as the most "
            "promising narratives. The evidence is exploratory: the leading topics remain statistically tied "
            "and require repeated seeds and an untouched future evaluation before a final forecasting claim.",
            styles["body"],
        ),
    ])

    # Page 8
    story.extend([
        PageBreak(),
        P("References", styles["h1"]),
        P("[1]&nbsp;&nbsp;Ansari, A. F., et al.: <i>Chronos-2: From Univariate to Universal Forecasting</i>. arXiv:2510.15821 (2025). <link href='https://arxiv.org/abs/2510.15821' color='blue'>arxiv.org/abs/2510.15821</link>", styles["reference"]),
        P("[2]&nbsp;&nbsp;Hu, E. J., et al.: <i>LoRA: Low-Rank Adaptation of Large Language Models</i>. arXiv:2106.09685 (2021). <link href='https://arxiv.org/abs/2106.09685' color='blue'>arxiv.org/abs/2106.09685</link>", styles["reference"]),
        P("[3]&nbsp;&nbsp;Rahimikia, E., Ni, H., Wang, W.: <i>Re(Visiting) Time Series Foundation Models in Finance</i>. arXiv:2511.18578 (2025). <link href='https://arxiv.org/abs/2511.18578' color='blue'>arxiv.org/abs/2511.18578</link>", styles["reference"]),
        P("[4]&nbsp;&nbsp;Amazon Science: <i>Chronos Forecasting</i>, source code and model documentation. <link href='https://github.com/amazon-science/chronos-forecasting' color='blue'>github.com/amazon-science/chronos-forecasting</link>", styles["reference"]),
        P("[5]&nbsp;&nbsp;The GDELT Project: <i>GDELT DOC 2.0 API</i>. <link href='https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/' color='blue'>blog.gdeltproject.org/gdelt-doc-2-0-api-debuts</link>", styles["reference"]),
        P("[6]&nbsp;&nbsp;Baker, S. R., Bloom, N., Davis, S. J.: Measuring Economic Policy Uncertainty. <i>Quarterly Journal of Economics</i> 131(4), 1593-1636 (2016).", styles["reference"]),
        P("[7]&nbsp;&nbsp;Shah, A., Paturi, A., Chava, S.: Trillion Dollar Words: A New Financial Dataset, Task and Market Analysis for Central Bank Communications. <i>ACL</i> (2023).", styles["reference"]),
        P("Appendix A&nbsp;&nbsp;Executed run ledger", styles["h1"]),
    ])
    ledger = [
        ["Slurm job", "Runs", "Configuration", "Outcome"],
        ["869927", "1 smoke", "5 LoRA steps; 15 outcomes", "Completed; adapter reload and holdout prediction passed"],
        ["869989", "6", "Source screen, seed 42", "Completed; 250 outcomes per run"],
        ["871482", "6", "Control, EPU/EMU, all external; seeds 43-44", "Completed; three-seed robustness assembled"],
        ["874192", "6", "One GDELT topic per run, seed 42", "Completed; Fed ranked first"],
    ]
    story.extend([
        make_table(ledger, [24 * mm, 20 * mm, 60 * mm, 48 * mm], styles, font_size=6.8),
        P("<b>Table 5:</b> Successful TAU Slurm executions included in this report.", styles["caption"]),
        P("Appendix B&nbsp;&nbsp;Reproducibility record", styles["h1"]),
        P(
            "The full experiments use feature-table SHA-256 "
            "e0c8d48b38059eb70d9f91931bff68678dbcc46aeadab1502ded1b9ee11df037. "
            "The verified cluster runtime was Python 3.12.3, PyTorch 2.11.0+cu128, CUDA 12.8, Chronos "
            "Forecasting 2.3.2, Transformers 5.16.1, PEFT 0.20.0, and Accelerate 1.14.0 on an RTX 2080 Ti. "
            "Each run directory stores its environment, data-readiness record, metadata, 250 predictions, "
            "per-horizon metrics, and LoRA adapter.", styles["appendix"],
        ),
        P(
            "Primary local artifacts: <font name='Courier' size='7'>models/news-ablation-869989/</font>. "
            "Experiment definitions: <font name='Courier' size='7'>configs/news_ablation.json</font>. "
            "Readable analysis: <font name='Courier' size='7'>docs/NEWS_ABLATION_RESULTS.md</font> and "
            "<font name='Courier' size='7'>docs/GDELT_TOPIC_RESULTS.md</font>.", styles["appendix"],
        ),
    ])

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
