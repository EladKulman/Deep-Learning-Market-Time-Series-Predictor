#!/usr/bin/env python3
"""Build the workshop paper from audited, already-executed Chronos-2 runs.

Run analyze_report_results.py first. Requires reportlab, matplotlib, numpy,
and pandas; rendering the resulting PDF is a separate visual QA step.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qqq-report-mpl-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "models/news-ablation-869989"
OUT = ROOT / "output/pdf/qqq_chronos2_project_report.pdf"
TMP = ROOT / "tmp/pdfs/refined_report"
A = json.loads((OUT.parent / "report_analysis.json").read_text())
CFG = json.loads((ROOT / "configs/news_ablation.json").read_text())
NAVY, BLUE, TEAL, GOLD, GRAY = "#193650", "#3579AA", "#217D77", "#B9792B", "#757B82"
PALE = "#EEF3F6"
WIDTH = 166 * mm
STORY = []
FIGURE = TABLE = 0


def fonts():
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype"
    for label, suffix in [("Regular", "Regular"), ("Bold", "Bold"), ("Italic", "Italic"), ("BoldItalic", "BoldItalic")]:
        pdfmetrics.registerFont(TTFont("Paper" + label, str(bundled / f"LiberationSerif-{suffix}.ttf")))
    pdfmetrics.registerFontFamily("PaperRegular", normal="PaperRegular", bold="PaperBold", italic="PaperItalic", boldItalic="PaperBoldItalic")
    pdfmetrics.registerFont(TTFont("PaperMono", str(bundled / "LiberationMono-Regular.ttf")))


def styles():
    base = dict(fontName="PaperRegular", fontSize=10.4, leading=13.0,
                textColor=colors.HexColor("#182028"))
    return {
        "body": ParagraphStyle("Body", **base, alignment=TA_JUSTIFY, spaceAfter=6),
        "left": ParagraphStyle("Left", **base, alignment=TA_LEFT, spaceAfter=6),
        "h1": ParagraphStyle("H1", fontName="PaperBold", fontSize=14, leading=17, spaceBefore=9, spaceAfter=7, keepWithNext=True),
        "h2": ParagraphStyle("H2", fontName="PaperBold", fontSize=11.3, leading=14, spaceBefore=7, spaceAfter=5, keepWithNext=True),
        "title": ParagraphStyle("Title", fontName="PaperBold", fontSize=22, leading=26, alignment=TA_CENTER, spaceAfter=13, textColor=colors.HexColor(NAVY)),
        "subtitle": ParagraphStyle("Subtitle", fontName="PaperRegular", fontSize=12, leading=15, alignment=TA_CENTER, spaceAfter=13),
        "author": ParagraphStyle("Author", fontName="PaperRegular", fontSize=11.5, leading=15, alignment=TA_CENTER, spaceAfter=4),
        "meta": ParagraphStyle("Meta", fontName="PaperRegular", fontSize=9.4, leading=12, alignment=TA_CENTER, spaceAfter=17, textColor=colors.HexColor(GRAY)),
        "abstract": ParagraphStyle("Abstract", fontName="PaperRegular", fontSize=9.6, leading=12.0, alignment=TA_JUSTIFY, leftIndent=14, rightIndent=14, spaceAfter=12),
        "caption": ParagraphStyle("Caption", fontName="PaperRegular", fontSize=8.8, leading=10.6, alignment=TA_LEFT, spaceBefore=4, spaceAfter=7),
        "note": ParagraphStyle("Note", fontName="PaperRegular", fontSize=8.9, leading=11.0, alignment=TA_LEFT, spaceAfter=6),
        "reference": ParagraphStyle("Reference", fontName="PaperRegular", fontSize=9.2, leading=11.4, leftIndent=17, firstLineIndent=-17, spaceAfter=6),
    }


def p(text, style="body"):
    return Paragraph(text.replace("name='Courier'", "name='PaperMono'"), S[style])


def add(text, style="body"):
    STORY.append(p(text, style))


def h(text):
    add(text, "h1")


def sub(text):
    add(text, "h2")


def page():
    STORY.append(PageBreak())


def ref(n):
    return f'<link href="#ref{n}" color="{BLUE}">[{n}]</link>'


def metric(setup, model="fine_tuned", seed=42):
    return A["runs"][f"{setup}-seed{seed}"][model]


def ci(v, digits=5):
    return f"[{v[0]:.{digits}f}, {v[1]:.{digits}f}]"


def table(data, widths, caption, font=9.0, numeric=()):
    global TABLE
    TABLE += 1
    cells = []
    for i, row in enumerate(data):
        cells.append([Paragraph(str(v).replace("name='Courier'", "name='PaperMono'"), ParagraphStyle(f"Cell{TABLE}-{i}-{j}",
            fontName="PaperBold" if i == 0 else "PaperRegular", fontSize=font,
            leading=font+2.0, alignment=TA_RIGHT if j in numeric else TA_LEFT,
            textColor=colors.HexColor(NAVY) if i == 0 else colors.HexColor("#182028")))
            for j,v in enumerate(row)])
    t = Table(cells, colWidths=[w*mm for w in widths], hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LINEABOVE", (0,0), (-1,0), .8, colors.HexColor(NAVY)),
        ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor(NAVY)),
        ("LINEBELOW", (0,-1), (-1,-1), .65, colors.HexColor(NAVY)),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor(PALE)]),
    ]))
    STORY.append(KeepTogether([t, p(f"<b>Table {TABLE}.</b> {caption}", "caption")]))


def savefig(fig, name):
    path = TMP / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.055, facecolor="white")
    plt.close(fig)
    return path


def figure(path, caption, width=166):
    global FIGURE
    FIGURE += 1
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width*mm, width*mm*ratio
    STORY.append(KeepTogether([img, p(f"<b>Figure {FIGURE}.</b> {caption}", "caption")]))


def equation(text, number):
    height = .72 if number==4 else .39
    fig = plt.figure(figsize=(6.5,height))
    fig.text(.46,.50,text,fontsize=14,ha="center",va="center")
    fig.text(.99,.50,f"({number})",fontsize=10,ha="right",va="center")
    path=TMP/f"equation{number}.png"
    fig.savefig(path,dpi=400,facecolor="white")
    plt.close(fig)
    img=Image(str(path), width=165.1*mm, height=height*25.4*mm)
    STORY.extend([img,Spacer(1,3)])


def plot_settings():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9,
        "axes.labelsize":9, "axes.titlesize":10,"legend.fontsize":8,
        "axes.spines.top":False, "axes.spines.right":False,
        "axes.edgecolor":"#8B9298", "grid.color":"#CBD3D9", "grid.alpha":.5,
        "axes.titleweight":"bold", "xtick.labelsize":8, "ytick.labelsize":8,
        "mathtext.fontset":"stix"})


def architecture():
    d=Drawing(WIDTH,111)
    ys=49
    boxes=[(0,84,"Observed series","QQQ + controls + news"),
           (99,80,"Scale and patch","Masks + time index"),
           (194,91,"Chronos-2 blocks","Time + group attention"),
           (300,77,"Quantile head","21 levels x 5 days")]
    for i,(x,w,a,b) in enumerate(boxes):
        d.add(Rect(x,ys,w,43,rx=3,ry=3,fillColor=colors.HexColor(PALE),strokeColor=colors.HexColor(NAVY),strokeWidth=.6))
        d.add(String(x+w/2,ys+27,a,fontName="PaperBold",fontSize=8.7,textAnchor="middle"))
        d.add(String(x+w/2,ys+13,b,fontName="PaperRegular",fontSize=7.1,textAnchor="middle"))
        if i < len(boxes)-1:
            end=boxes[i+1][0]-3
            d.add(Line(x+w+3,ys+21,end,ys+21,strokeColor=colors.HexColor(NAVY)))
            d.add(Polygon([end,ys+21,end-4,ys+24,end-4,ys+18],fillColor=colors.HexColor(NAVY),strokeColor=None))
    d.add(Rect(392,ys,78,43,rx=3,ry=3,fillColor=colors.HexColor("#EDF4EF"),strokeColor=colors.HexColor(TEAL),strokeWidth=.6))
    d.add(String(431,ys+27,"QQQ forecasts",fontName="PaperBold",fontSize=8.5,textAnchor="middle"))
    d.add(String(431,ys+13,"Daily return quantiles",fontName="PaperRegular",fontSize=7.1,textAnchor="middle"))
    d.add(Line(380,70,389,70,strokeColor=colors.HexColor(NAVY)))
    d.add(Polygon([389,70,385,73,385,67],fillColor=colors.HexColor(NAVY),strokeColor=None))
    d.add(String(235,19,"Train: update LoRA matrices only     |     Evaluate: freeze weights; refresh observed context",fontName="PaperRegular",fontSize=8.3,textAnchor="middle"))
    d.add(String(235,103,"Future inputs: scheduled calendar only; future returns and news remain masked",fontName="PaperItalic",fontSize=8.5,textAnchor="middle"))
    global FIGURE
    FIGURE+=1
    STORY.append(KeepTogether([d,p(f"<b>Figure {FIGURE}.</b> Implemented forecasting system. The adapter changes selected projections inside the pretrained model. The six known calendar signals also enter the masked forecast horizon.","caption")]))


def seed_chart():
    fig,ax=plt.subplots(figsize=(6.5,1.75))
    for setup,label,c in [("control","Control",GRAY),("uncertainty","EPU / EMU",GOLD),("all_external","All external",NAVY)]:
        ys=[metric(setup,seed=s)["wql"] for s in [42,43,44]]
        ax.plot([42,43,44],ys,"o-",label=label,color=c,lw=1.5,ms=4)
    ax.set_xticks([42,43,44]);ax.set_ylabel("WQL");ax.set_xlabel("Fine-tuning seed")
    ax.legend(frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(.5,1.0))
    ax.grid(axis="y");fig.tight_layout(pad=.3)
    return savefig(fig,"seed_robustness")


TOPICS=[("gdelt_fed","Federal Reserve"),("gdelt_recession","Recession"),
        ("gdelt_semiconductor","Semiconductors"),("gdelt_ai","AI"),
        ("gdelt_inflation","Inflation"),("all_external","All external"),
        ("gdelt_big_tech_earnings","Big-tech earnings"),("control","Control")]


def forest():
    fig,ax=plt.subplots(figsize=(6.5,2.3))
    for y,(setup,label) in enumerate(TOPICS[:-1]):
        v=A["paired"][setup+"-seed42"]
        for block,shift,c,marker in [(1,-.12,BLUE,"o"),(5,.12,GOLD,"s")]:
            lo,hi=v["ci95"][str(block)]; m=v["mean_delta"]
            ax.errorbar(m*1000,y+shift,xerr=np.array([[m-lo],[hi-m]])*1000,fmt=marker,
                color=c,markersize=3.7,lw=1.1,capsize=2,label=f"{block} window" if y==0 and block==1 else "5 consecutive windows" if y==0 else None)
    ax.axvline(0,color=GRAY,lw=.8,ls="--");ax.set_yticks(range(7),[x[1] for x in TOPICS[:-1]])
    ax.invert_yaxis();ax.set_xlabel("WQL difference from control x 1,000 (left favors added news)")
    ax.legend(frameon=False,ncol=2,loc="lower center",bbox_to_anchor=(.5,1.0),fontsize=8)
    ax.grid(axis="x");fig.tight_layout(pad=.25)
    return savefig(fig,"topic_intervals")


def diagnostic_chart():
    fig,(ax,bx)=plt.subplots(1,2,figsize=(6.5,2.1),gridspec_kw={"width_ratios":[1,1]})
    for setup,label,c in [("gdelt_fed","Fed news",TEAL),("all_external","All external",NAVY)]:
        values=[]
        for h in range(1,6):
            own=A["runs"][setup+"-seed42"]["by_horizon"][str(h)]["wql"]
            con=A["runs"]["control-seed42"]["by_horizon"][str(h)]["wql"]
            values.append(100*(1-own/con))
        ax.plot(range(1,6),values,"o-",color=c,lw=1.3,ms=4,label=label)
    ax.axhline(0,color=GRAY,lw=.7);ax.set_xticks(range(1,6));ax.set_xlabel("Lead time (trading days)")
    ax.set_ylabel("WQL reduction vs control (%)");ax.set_title("(a) Lead-time sensitivity",loc="left");ax.grid(axis="y")
    ax.legend(frameon=False,fontsize=7,loc="upper right")
    xx=np.array([0,1]); w=.22
    for j,(setup,label,c) in enumerate([("control","Control",GRAY),("all_external","All external",NAVY),("gdelt_fed","Fed news",TEAL)]):
        vals=[100*A["halves"][setup][s]["coverage80"] for s in ["first_25_windows","last_25_windows"]]
        bx.bar(xx+(j-1)*w,vals,width=w,color=c,label=label)
    bx.axhline(80,ls="--",color=GOLD,lw=1);bx.set_ylim(0,100)
    bx.set_xticks(xx,["First 125 days","Last 125 days"]);bx.set_ylabel("80% interval coverage (%)")
    bx.set_title("(b) Calibration changes over time",loc="left");bx.legend(frameon=False,fontsize=6.7,ncol=3,loc="lower center",bbox_to_anchor=(.5,-.39))
    fig.tight_layout(pad=.45,w_pad=1.7)
    return savefig(fig,"diagnostics")


def forecast_chart():
    f=pd.read_csv(RUNS/"gdelt_fed-seed42/validation_predictions.csv")
    f=f[f.model=="fine_tuned"].sort_values(["window","horizon"])
    xx=np.arange(250)
    fig,ax=plt.subplots(figsize=(6.5,1.9))
    ax.fill_between(xx,100*f["q0.1"],100*f["q0.9"],color="#DCEBE9",label="10%-90% interval")
    ax.plot(xx,100*f.actual,color=GRAY,lw=.65,label="Observed return")
    ax.plot(xx,100*f.prediction,color=TEAL,lw=.9,label="Median forecast")
    ax.axhline(0,lw=.4,color=GRAY);ax.set_ylabel("Daily log return (%)")
    ax.set_xticks([0,64,125,188,249],["Jun 30\n2025","Sep 30","Dec 26","Mar 30\n2026","Jun 26"])
    # Read the true dates for the selected session indices, never invent calendar labels.
    ticks=[0,64,125,188,249]
    ax.set_xticklabels([pd.Timestamp(f.iloc[i].date).strftime("%b %d\n%Y") for i in ticks])
    ax.set_xlim(0,249);ax.grid(axis="y");ax.legend(frameon=False,ncol=3,fontsize=7.5,loc="lower center",bbox_to_anchor=(.5,1.0))
    fig.tight_layout(pad=.3)
    return savefig(fig,"fed_forecasts")


def footer(canvas,doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("PaperRegular",8)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(22*mm,281*mm,"News Covariates for QQQ Return Forecasting")
        canvas.drawRightString(188*mm,281*mm,"Workshop on Deep Learning | Group 13")
        canvas.setStrokeColor(colors.HexColor("#D7DEE3"));canvas.setLineWidth(.4)
        canvas.line(22*mm,278.5*mm,188*mm,278.5*mm)
    canvas.setFont("PaperRegular",9)
    canvas.setFillColor(colors.HexColor(GRAY))
    canvas.drawCentredString(A4[0]/2,15*mm,str(doc.page))
    canvas.restoreState()


def manuscript():
    # Page 1: research question, contribution, and a self-contained abstract.
    STORY.append(Spacer(1,14*mm))
    add("Which News Helps Forecast QQQ?", "title")
    add("Controlled Covariate Experiments with Chronos-2 and LoRA", "subtitle")
    add("Elad Kulman and Tom Weitman", "author")
    add("Group 13 | Workshop on Deep Learning | Tel Aviv University<br/>September 2026", "meta")
    add("<b>Abstract.</b> We investigate whether news-derived inputs improve probabilistic forecasts of QQQ daily returns beyond market, macroeconomic, and calendar controls. Starting from a fixed pretrained Chronos-2 checkpoint, we execute 18 LoRA fits: six source configurations, six seed repetitions, and six isolated GDELT topics. Each fit uses 2,133 training sessions and forecasts five separate daily returns at 50 non-overlapping origins, yielding 250 common evaluation outcomes. Across three seeds, all external sources lower mean weighted quantile loss (WQL) from 0.693582 to 0.689439, a 0.60% reduction whose paired interval includes zero. Federal Reserve news has the best single-seed WQL, 0.686526, but is statistically indistinguishable from recession news. Longer-block resampling weakens several topic comparisons. An always-up rule scores 57.2% direction accuracy, and interval coverage deteriorates in the second half of evaluation. The evidence supports small, conditional distributional gains and identifies candidates for confirmation; it does not establish reliable direction prediction or a uniquely informative news topic.","abstract")
    h("1  Introduction")
    add("Daily equity forecasting is difficult because a small conditional return signal must be recovered from much larger day-to-day variation. A prediction close to zero can achieve modest point error while failing to anticipate large moves. This motivates a distributional question: can information about news coverage and uncertainty help a pretrained model assign more useful probabilities to future QQQ returns?")
    add("QQQ, an exchange-traded fund tracking the Nasdaq-100, provides a focused setting for this question. Its technology exposure motivates news about AI, semiconductors, and large-company earnings, while interest rates, inflation, and recession concerns provide macroeconomic narratives. We compare these inputs against the same market and calendar information, so an added source must contribute beyond signals already observable in prices.")
    add("<b>Research questions.</b> (RQ1) Do external covariates improve forecasts beyond the common control? (RQ2) Does LoRA adaptation add value beyond supplying those covariates to the original pretrained model? (RQ3) Which individual GDELT topic has the strongest incremental evidence, and how stable is its ranking?")
    add("<b>Contributions.</b> We implement an availability-aware feature pipeline, execute a matched source and topic ablation study on Slurm, and audit the resulting forecasts for seed sensitivity, serial dependence, directional class imbalance, and calibration drift. Every result is traceable to saved predictions and a frozen feature snapshot.")
    add("The original proposal used Bitcoin, a Temporal Fusion Transformer, and attention-based feature interpretation. We retained the narrative-ranking objective but moved to QQQ and Chronos-2 to use richer public covariates and a common pretrained baseline. Add-one-source experiments measure incremental predictive usefulness under this pipeline; they do not identify a headline's causal effect on the market.")

    # Page 2: literature is connected directly to design, followed by data scope.
    page();h("2  Related work and data")
    sub("2.1  Why a foundation model, and why probabilistic evaluation?")
    add(f"Chronos-2 supports multivariate and covariate-informed forecasting through attention across related series {ref(1)}. This makes it possible to change the supplied feature group while preserving a common pretrained checkpoint. LoRA learns low-rank updates to selected weight matrices {ref(2)}, allowing separate configurations to be adapted with modest GPU memory. We use these existing methods as experimental tools; the contribution is their controlled evaluation on a finance-specific data contract.")
    add(f"Strong general forecasting results do not guarantee useful equity-return forecasts. Rahimikia, Ni, and Wang find weaknesses in off-the-shelf time-series foundation models on financial excess returns and show that domain, scale, and training choices matter {ref(3)}. Our single-ETF, log-return task is narrower and not directly comparable with their benchmark. We therefore compare both pretrained and adapted arms and avoid treating fine-tuning as an automatic improvement.")
    add(f"News coverage offers distinct measurements. Economic Policy Uncertainty (EPU) and Equity Market Uncertainty (EMU) summarize uncertainty-related reporting {ref(4)}; Federal Open Market Committee (FOMC) text classification measures policy language {ref(5)}; GDELT supplies query-specific volume and tone {ref(6)}. None directly measures return predictability. Following probabilistic scoring principles {ref(7)}, we evaluate quantile accuracy together with empirical interval coverage and report direction as a secondary diagnostic.")
    sub("2.2  Data scope and common sample")
    add("The processed snapshot contains 5,201 NYSE sessions from January 2006 through September 4, 2026; 2005 price history initializes rolling features. The experiments use the common GDELT interval, January 3, 2017 through June 30, 2026: 2,385 sessions in total. This date restriction avoids comparing models on different market periods. The full schema has one target, 55 past covariates, 14 known-future calendar fields, and six raw price/volume fields; experiments select only the columns defined in Appendix B.")
    table([
        ["Source", "Model information", "Availability treatment"],
        ["QQQ / Yahoo; other ETFs", "QQQ transforms; TLT and HYG returns", "Observed through the origin close"],
        ["CBOE volatility indices", "VXN change and VXN-VIX; VIX/VIX3M", "Conservative one-session lag"],
        ["FRED / ALFRED", "Yield changes, slope, vintage CPI growth", "Release-based alignment; CPI vintages"],
        ["GDELT", "News share and tone for six queries", "Next-session alignment of daily news"],
        ["EPU/EMU; FOMC; SEC", "Uncertainty, policy tone, filing activity", "Source-specific publication rules"],
        ["Scheduled calendar", "FOMC, CPI, NFP, month-end, options expiry", "Only declared known-future fields"],
    ],[36,65,65],"Sources used in the controlled experiments. Raw OHLCV columns are retained for audit and are not direct model covariates.",font=9)
    sub("2.3  Missing observations and what coverage means")
    add("Each of the 12 GDELT input columns is missing on 13 of the 2,385 common-sample sessions (0.55%): October 21, 2020; March 24, 2023; and 11 sessions between June 17 and July 2, 2025. Missing news remains masked, never replaced by a zero article count. The outage spans the first evaluation origin, so a common date range does not imply uninterrupted news availability. Four topic caches ended in June 2026 when this snapshot was frozen; their later refresh is unnecessary for replaying these historical results.")

    # Page 3: implementation specifics and exactly what is adapted.
    page();h("3  Feature contract and model")
    sub("3.1  Inputs available at a forecast origin")
    add("The origin is the last observed NYSE close. H.15 rates are attached to the first exchange close after their reconstructed publication time, including federal-holiday differences. CPI uses ALFRED release vintages, with year-over-year change computed within each vintage. SEC forms use UTC acceptance timestamps and the next eligible exchange close, including early closes. Completed calendar-day news is lagged by one calendar day and assigned to the next exchange session; observations rolling into the same session are averaged.")
    add("The common control contains <b>15 past covariates and six known-future calendar signals</b> (21 in total), in addition to the QQQ target history. One GDELT topic adds two past columns for 23 total covariates. All six topics give 33; all external sources give 42. Counts exclude the target. Future returns, realized news, and realized filing activity are masked; only the six calendar columns may populate the five-step forecast horizon.")
    sub("3.2  From numeric news features to return quantiles")
    add(f"Chronos-2 receives numeric time series, not raw headlines. Each series is scaled within the available context, transformed with asinh, and divided into 16-step patches. Time attention processes a series' history; group attention exchanges information among the target and its covariates {ref(1)}. The saved checkpoint specifies 12 layers, model width 768, and 12 attention heads. We cap context at 512 sessions and retain the 21 native quantiles for each of the next five daily returns.")
    architecture()
    sub("3.3  Low-rank fine-tuning")
    equation(r"$W_{\mathrm{adapted}}=W_0+\frac{\alpha}{r}BA,\qquad r=8,\quad\alpha=16$",1)
    add("The original weights W<sub>0</sub> remain frozen; A and B are learned low-rank matrices. Adapters are applied to the query, key, value, and output attention projections and the output patch projection. The checkpoint contains 119,477,664 base parameters and 1,206,912 adapter parameters, about 1.00% of the combined model. LoRA dropout is zero; this does not remove the base architecture's dropout setting.")
    table([
        ["Training item", "Executed setting"],
        ["Initialization", "Fresh adapter from the same original pretrained revision for every fit"],
        ["Data and objective", "One QQQ series; random training slices; masked quantile-loss training"],
        ["Optimization", "AdamW; linear LR decay from 1e-5; 200 steps; batch_size argument 8"],
        ["Context / forecast horizon", "At most 512 past sessions / five future daily returns"],
        ["Validation during fitting", "None; final-step adapter used; no early stopping or checkpoint selection"],
        ["Compute", "One NVIDIA GeForce RTX 2080 Ti per Slurm task"],
    ],[42,124],"Shared fitting protocol. Library defaults and exact package versions are recorded in Appendix A.",font=9)
    add(f"The 200 steps are optimizer updates, not 200 epochs over the history. In the Chronos API, batch size counts target and covariate series rather than eight independent QQQ windows {ref(8)}. Each feature setup is supplied as one QQQ task. The QQQ training data were used for adaptation only; this project did not pretrain a foundation model from scratch.","note")

    # Page 4: explicit dates, a numerical example, metrics, and uncertainty estimand.
    page();h("4  Forecasting and evaluation protocol")
    sub("4.1  Target and chronological split")
    equation(r"$r_t=\log(P^{\mathrm{adj}}_t/P^{\mathrm{adj}}_{t-1}),\qquad \widehat q_{t,h}(\tau),\ h\in\{1,\ldots,5\}$",2)
    add("The target is the <b>one-day adjusted-close log return</b>. A five-step forecast contains five distinct daily returns, not a price level or one cumulative weekly return. For example, 0.002 is a log return of 0.20%, approximately a 0.20% simple return. All percentage-return displays below multiply log returns by 100.")
    table([
        ["Partition", "Dates", "Sessions / use"],
        ["LoRA training", "2017-01-03 to 2025-06-27", "2,133; adaptation only"],
        ["Development allocation", "2025-06-30 to 2026-06-30", "252; source and model comparison"],
        ["Scored outcomes", "2025-06-30 to 2026-06-26", "250; 50 blocks x five horizons"],
    ],[40,68,58],"All full runs share this split. June 29 and June 30, 2026 are not scored because they cannot form a complete five-session block.",font=9.1)
    add("At the June 27, 2025 close, each model forecasts June 30, July 1, July 2, July 3, and July 7 (July 4 is a market holiday). After July 7 closes, those realized observations enter the next context and the model issues another five-step forecast. <b>Weights stay fixed throughout evaluation.</b> Within each block, later horizons receive no intervening realized news or returns. Every date is scored once, at one lead time; the study is not 250 one-step-ahead forecasts.")
    f=pd.read_csv(RUNS/"gdelt_fed-seed42/validation_predictions.csv")
    f=f[(f.model=="fine_tuned") & (f.window==0)].sort_values("horizon")
    rows=[["Date", "Horizon", "Actual (%)", "Median (%)", "80% interval (%)"]]
    for _,r in f.iterrows():
        rows.append([r.date,str(int(r.horizon)),f"{100*r.actual:+.3f}",f"{100*r.prediction:+.3f}",f"[{100*r['q0.1']:+.3f}, {100*r['q0.9']:+.3f}]"])
    table(rows,[33,20,29,30,54],"Worked example: the first Fed-topic LoRA forecast, issued once on June 27. All five returns fall inside the interval, but only three median signs are correct.",font=8.8,numeric=(1,2,3,4))
    sub("4.2  Scores and paired uncertainty")
    equation(r"$\rho_\tau(u)=\max(\tau u,(\tau-1)u),\quad u=r_{t+h}-\widehat q_{t,h}(\tau)$",3)
    equation(r"$\mathrm{WQL}=\frac{2\sum_{i=1}^{N}\sum_{\tau\in\mathcal{Q}}\rho_\tau(r_i-\widehat q_i(\tau))}{|\mathcal{Q}|\sum_{i=1}^{N}|r_i|},\quad N=250,\ |\mathcal{Q}|=21$",4)
    add("We average equally over quantile levels 0.01, 0.05, 0.10, ..., 0.95, 0.99. WQL is dimensionless; lower is better. MAE and RMSE use the median forecast. The Gaussian reference has zero mean and a sample standard deviation estimated from the last 60 observed returns at each origin, with no cumulative-return horizon scaling. Coverage is the fraction of realized returns inside the nominal 80% or 98% interval.")
    add("Paired WQL differences use a fixed full-sample denominator and preserve each five-day forecast block. We use 20,000 percentile-bootstrap draws (seed 20260910), resampling one window or circular blocks of 2, 5, and 10 consecutive windows. For repeated fits, differences are averaged across seeds within each window before resampling. These exploratory intervals condition on the observed seeds and are unadjusted for multiple comparisons.","note")

    # Page 5: the missing distinction between data and weight adaptation.
    page();h("5  Executed source experiments")
    sub("5.1  Infrastructure smoke test and source screen")
    add("Slurm job 869927 first verified pretrained inference, five LoRA updates, adapter save/reload, and holdout prediction on an RTX 2080 Ti. Its 15 scored outcomes and three-day horizon were deliberately small: WQL changed from 0.734899 to 0.710572, while the Gaussian reference scored 0.667505. The smoke result validates execution, not forecasting skill. All results below use the full five-day protocol.")
    data=[["Source added to control", "Covariates", "Pretrained WQL", "LoRA WQL", "LoRA - pretrained"]]
    for setup,label in [("control","None (control)"),("uncertainty","EPU / EMU"),("fomc_tone","FOMC tone"),("sec_disclosures","SEC disclosures"),("gdelt_all","All six GDELT topics"),("all_external","All external sources")]:
        base,fine=metric(setup,"base_pretrained"),metric(setup)
        data.append([label,str(A["runs"][setup+"-seed42"]["features"]),f"{base['wql']:.6f}",f"{fine['wql']:.6f}",f"{fine['wql']-base['wql']:+.6f}"])
    table(data,[48,20,33,30,35],"Source screen at seed 42. Every pretrained arm receives that row's covariates. A negative last-column value indicates improvement from adapting the weights.",font=9,numeric=(1,2,3,4))
    add("EPU/EMU is best at seed 42, followed by all external sources. However, the effects of inputs and fine-tuning differ. Adding all external inputs to the pretrained model lowers WQL from 0.694294 to 0.690730 (0.51%). LoRA then lowers its seed-42 WQL by another 0.002427 (0.35%). Control and FOMC tone become worse after adaptation. Thus neither more covariates nor fine-tuning guarantees an improvement.")
    sub("5.2  Repeating the control and two leading sources")
    data=[["Configuration", "Seed 42", "Seed 43", "Seed 44", "Mean ± SD", "Wins"]]
    for setup,label in [("control","Control"),("uncertainty","EPU / EMU"),("all_external","All external")]:
        g=A["aggregate"][setup]
        data.append([label,*[f"{metric(setup,seed=s)['wql']:.6f}" for s in [42,43,44]],f"{g['mean_wql']:.6f} ± {g['sd_wql']:.6f}","-" if setup=="control" else f"{g['seeds_beating_control']}/3"])
    table(data,[30,25,25,25,47,14],"Full LoRA WQL over three seeds. SD is sample standard deviation across fits; wins count seeds with lower WQL than the matched control.",font=8.6,numeric=(1,2,3,4,5))
    figure(seed_chart(),"Seed changes reverse the initial EPU/EMU result. All external sources beat control in two seeds and are slightly worse in seed 44.",width=154)
    g=A["aggregate"]["all_external"]
    add(f"All external sources achieve mean WQL {g['mean_wql']:.6f}, 0.60% below control. The paired mean difference is {g['delta']:.6f}, with a 95% one-window interval {ci(g['ci95']['1'])}; five consecutive windows give {ci(g['ci95']['5'])}. Both include zero. Against its own pretrained arm, mean WQL changes by -0.001292 (a 0.19% reduction), with 95% WQL-difference interval {ci(g['adaptation_vs_own_pretrained']['ci95']['1'])}. Pretrained predictions are identical across seeds. These fits reuse 250 outcomes, not 750 independent observations.")

    # Page 6: requested eight-arm table with properly qualified inference.
    page();h("6  Which GDELT topic contributes most?")
    add("Six additional seed-42 fits each start from the original pretrained weights, retain all 21 control covariates, and add exactly one topic's news share and average tone. They never inherit the previously fine-tuned control adapter. Table 7 compares these topics with the control and the earlier all-external model, which includes EPU/EMU, FOMC tone, and SEC activity as well as all six GDELT topics.")
    data=[["News setup", "Pretrained WQL", "LoRA WQL", "WQL gain vs control", "Direction correct", "Blocks won"]]
    control=metric("control")["wql"]
    for setup,label in TOPICS:
        b,f=metric(setup,"base_pretrained"),metric(setup)
        win=A["paired"][setup+"-seed42"]["winning_windows"]
        data.append([label,f"{b['wql']:.6f}",f"{f['wql']:.6f}","-" if setup=="control" else f"{100*(1-f['wql']/control):.2f}%",f"{f['direction_correct']}/250","-" if setup=="control" else f"{win}/50"])
    table(data,[34,28,28,29,25,22],"Eight configurations on identical dates. Positive WQL gain means improvement over the seed-42 fine-tuned control; block wins use paired quantile loss.",font=8.8,numeric=(1,2,3,4,5))
    add("Fed news ranks first by WQL (0.686526), followed closely by recession (0.686796). All six isolated topics outperform both the fine-tuned control and their own pretrained arm. Fed's advantage over the pretrained control is 1.12%; its improvement over its own pretrained Fed arm is 1.06%. These are different comparisons from the 1.28% gain against the fine-tuned control.")
    figure(forest(),"Exploratory 95% intervals for each setup minus the fine-tuned control (seed 42). Points are identical; intervals change with the resampling block. Intervals crossing zero do not resolve the sign of the difference.")
    add("All six topic intervals exclude zero when individual forecast windows are resampled. With blocks of five consecutive windows (25 trading sessions), Fed, recession, and semiconductors still exclude zero; AI, inflation, and big-tech earnings do not. Semiconductors is close to the boundary. This sensitivity shows why non-overlapping forecast outcomes should not be treated as independent market observations.")
    d=A["direct_fed_minus"]["gdelt_recession"]
    add(f"The direct Fed-minus-recession difference is {d['mean_delta']:.6f}, with a one-window interval {ci(d['ci95']['1'])}. The Fed-minus-all-external interval also includes zero. Therefore, the observed ordering is a candidate ranking rather than evidence that Fed is uniquely informative. The six topics have only one training seed, and their comparisons were selected within the same development period.")

    # Page 7: diagnostics beyond a headline score.
    page();h("7  Forecast behavior and failure analysis")
    sub("7.1  A lower quantile loss is not a directional trading signal")
    data=[["Forecast", "WQL", "MAE (pp)", "RMSE (pp)", "80% cov.", "98% cov.", "Width (pp)"]]
    for setup,label,model in [("control","Zero Gaussian","zero_gaussian"),("control","Control LoRA","fine_tuned"),("all_external","All external LoRA","fine_tuned"),("gdelt_fed","Fed LoRA","fine_tuned")]:
        m=metric(setup,model)
        data.append([label,f"{m['wql']:.6f}",f"{m['mae_pp']:.3f}",f"{m['rmse_pp']:.3f}",f"{100*m['coverage80']:.1f}%",f"{100*m['coverage98']:.1f}%",f"{m['width80_pp']:.3f}"])
    table(data,[38,26,21,22,19,19,21],"Seed-42 point error and calibration. pp denotes percentage points of daily log return; width is the mean 10%-90% interval width. Coverage targets are 80% and 98%.",font=8.8,numeric=(1,2,3,4,5,6))
    add("There are 143 non-negative and 107 negative evaluation returns. Always predicting up therefore scores <b>57.2%</b>. Fed gets 144 directions correct (57.6%); all external gets 145 (58.0%). These are only one and two additional correct days. Fed predicts a negative median on just seven days, while 107 actual returns are negative. The existing scorer counts a zero prediction as up, which explains the Gaussian's 57.2% direction score despite its zero median. The model's main measured benefit is in quantile loss, not reliable anticipation of market direction.")
    figure(diagnostic_chart(),"Post-hoc diagnostics from saved predictions. (a) WQL improvement varies by forecast lead; each lead contains 50 different realized days. (b) Full-period coverage hides a substantial decline across the two chronological halves.")
    sub("7.2  Calibration drift and scheduled-event behavior")
    add("Fed's 80% interval covers 109/125 outcomes (87.2%) through December 24, 2025 but only 86/125 (68.8%) from December 26 onward. Its average interval width rises from 2.479 to 2.687 percentage points while MAE rises from 0.684 to 0.999. The control and all-external model also lose coverage. Thus the aggregate 78.0% Fed coverage conceals periods of both overcoverage and undercoverage; it cannot establish conditional calibration.")
    add("On the 29 evaluation days flagged for a scheduled FOMC, CPI, or NFP event, Fed interval coverage is 22/29 (75.9%), versus 173/221 (78.3%) on other days. Mean width is 2.555 versus 2.587 percentage points, despite larger event-day MAE (0.962 versus 0.826). This small, post-hoc slice suggests a useful follow-up calibration test; without a calendar ablation it cannot determine whether the known-future calendar helped.")
    add("A quantile-order audit finds one crossed forecast in each of three fine-tuned runs: all external at seed 43, all GDELT at seed 42, and EPU/EMU at seed 44. The largest adjacent reversal is 0.0401 percentage points. No quantiles were reordered for this report; published metrics remain those of the executed models. Monotone quantile postprocessing is a candidate for a separately evaluated revision.","note")

    # Page 8: mechanisms, threats, next experiments, and a bounded conclusion.
    page();h("8  Discussion, limitations, and conclusion")
    sub("8.1  Why did combining news fail to dominate?")
    add("The all-external model is not the worst configuration: it ranks sixth of the eight cases by WQL and has the highest direction count. The GDELT-only combination is a separate 33-covariate model, with WQL 0.692348; every isolated topic has a lower point estimate. Neither pattern establishes that additional topics are useless.")
    add("Redundancy is one plausible explanation with observable support. In the training period, Fed and inflation news shares have Pearson correlation 0.711, while AI and semiconductor shares correlate 0.585 (2,123 jointly observed sessions). The exact queries overlap, including Nvidia in both AI and semiconductors. More columns therefore need not supply proportionally more independent information. Under a fixed 200-step budget, noisy and correlated inputs may be harder to use. These are hypotheses: we did not run shuffled-news controls, matched-size random features, or a training-budget sweep to isolate the mechanism.")
    figure(forecast_chart(),"All 250 Fed-topic forecasts in chronological order. The median is usually near zero and positive; large realized moves are much less predictable. Each group of five forecasts was issued at a single origin.",width=158)
    sub("8.2  What limits the strength of the evidence?")
    add("<b>Selection and replication.</b> Source choices and topic rankings reuse one development interval, which the smoke run also partially exposed. Seeds do not provide new market histories. Bootstrap intervals are conditional and unadjusted for the many examined configurations; they do not certify a winner. A single ETF and 50 forecast windows give limited evidence about other assets, market regimes, or longer horizons.")
    add(f"<b>Historical availability.</b> Conservative timestamp rules reduce obvious look-ahead, but Yahoo adjustments, non-CPI macro snapshots, EPU revisions, and reconstructed event schedules are not a complete vintage archive. The FOMC classifier is a retrospective substitute model {ref(9)}, and the SEC basket uses present-day technology leaders, creating survivorship hindsight. Chronos-2's paper appeared in October 2025 {ref(1)}, after our evaluation starts; the split controls LoRA training leakage but does not rule out base-model pretraining overlap or establish a live historical backtest.")
    add("<b>Attribution and baselines.</b> Each topic combines share and tone, so their individual roles are unidentified. Add-one-source experiments do not measure marginal value conditional on every other source or causal market influence. A quantile tree model, target-only full-period baseline, costs, and trading rules were not included in this 18-fit study. WQL gains therefore establish neither superiority over strong supervised finance baselines nor economic profitability.")
    sub("8.3  Next experiments and conclusion")
    add("First repeat Fed and recession with additional seeds, then compare their combination, share-only and tone-only variants, shuffled news, and a quantile tree baseline under the same split. Select and freeze one specification on development data. Evaluate it once on a newly arriving block after this selection and the September 4, 2026 snapshot, with preregistered metrics and a missing-data rule; assess event-conditioned calibration and direction against always-up.")
    add("The study establishes a reproducible forecasting pipeline and evidence of small, conditional improvements in QQQ return distributions. Fed and recession merit confirmation; neither is a proven winner. Stronger claims require fresh market outcomes, competitive baselines, and stable calibration across time.")

    # Page 9: primary references and concise, exact execution ledger.
    page();h("References")
    references=[
        ("Ansari, A. F., et al. (2025). <i>Chronos-2: From Univariate to Universal Forecasting.</i> arXiv:2510.15821.","https://arxiv.org/abs/2510.15821","arxiv.org/abs/2510.15821"),
        ("Hu, E. J., et al. (2021). <i>LoRA: Low-Rank Adaptation of Large Language Models.</i> arXiv:2106.09685.","https://arxiv.org/abs/2106.09685","arxiv.org/abs/2106.09685"),
        ("Rahimikia, E., Ni, H., and Wang, W. (2025). <i>Re(Visiting) Time Series Foundation Models in Finance.</i> arXiv:2511.18578.","https://arxiv.org/abs/2511.18578","arxiv.org/abs/2511.18578"),
        ("Baker, S. R., Bloom, N., and Davis, S. J. (2016). Measuring Economic Policy Uncertainty. <i>Quarterly Journal of Economics</i>, 131(4), 1593-1636. EPU/EMU data and documentation.","https://www.policyuncertainty.com/research.html","policyuncertainty.com/research.html"),
        ("Shah, A., Paturi, S., and Chava, S. (2023). Trillion Dollar Words: A New Financial Dataset, Task &amp; Market Analysis. <i>ACL</i>, 6664-6679.","https://aclanthology.org/2023.acl-long.368/","aclanthology.org/2023.acl-long.368"),
        ("The GDELT Project (2017). <i>GDELT DOC 2.0 API Debuts!</i> Query, volume-normalization, and tone documentation.","https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/","blog.gdeltproject.org/gdelt-doc-2-0-api-debuts"),
        ("Gneiting, T., and Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. <i>Journal of the American Statistical Association</i>, 102(477), 359-378.","https://doi.org/10.1198/016214506000001437","doi.org/10.1198/016214506000001437"),
        ("Amazon Science. <i>Chronos Forecasting</i>, version 2.3.2. Source code and fitting API.","https://github.com/amazon-science/chronos-forecasting/tree/v2.3.2","github.com/amazon-science/chronos-forecasting/tree/v2.3.2"),
        ("LorenzoAleCon29. <i>roberta-large-fomc-hawkish-dovish.</i> Classifier used by this pipeline; pinned revision f4759d4ad3f1182f81d87e47ba603261740d36cf.","https://huggingface.co/LorenzoAleCon29/roberta-large-fomc-hawkish-dovish","Hugging Face model card"),
    ]
    for n,(entry,url,label) in enumerate(references,1):
        add(f'<a name="ref{n}"/>[{n}]  {entry} <link href="{url}" color="{BLUE}">{label}</link>',"reference")
    h("Appendix A  Executions and reproducibility")
    table([
        ["Slurm job", "Completed fits", "Configuration"],
        ["869927", "1 smoke", "Five LoRA steps; 15 scored outcomes; reload checked"],
        ["869989", "6 full fits", "Source screen, seed 42"],
        ["871482", "6 full fits", "Control, EPU/EMU, all external; seeds 43 and 44"],
        ["874192", "6 full fits", "One GDELT topic per fit, seed 42"],
    ],[29,34,103],"Successful GPU executions. The full study comprises 18 fits of 200 optimizer steps each, plus the separate smoke run.",font=8.9)
    add("<b>Model:</b> amazon/chronos-2, revision 29ec3766d36d6f73f0696f85560a422f50e8498c.<br/><b>Feature-table SHA-256:</b><br/><font size='8.1'>e0c8d48b38059eb70d9f91931bff68678dbcc46aeadab1502ded1b9ee11df037</font>","note")
    add("The recorded cluster runtime used Python 3.12.3, PyTorch 2.11.0+cu128, CUDA 12.8, Chronos Forecasting 2.3.2, Transformers 5.16.1, PEFT 0.20.0, and Accelerate 1.14.0. Per-run metadata stores feature roles, model and script hashes, package versions, and the split. Its git-commit field is empty, so the source hashes, rather than a recorded cluster commit, identify the executed scripts.","note")
    add("Artifacts are under <font name='Courier' size='8'>models/news-ablation-869989/</font>, with predictions, metrics, metadata, and adapters in each setup/seed directory. Run <font name='Courier' size='8'>scripts/analyze_report_results.py</font> to audit all 54 forecast sets and regenerate the report's 20,000-draw sensitivity analysis; then run <font name='Courier' size='8'>scripts/build_project_report.py</font>. The companion <font name='Courier' size='8'>output/pdf/report_analysis.json</font> records the analysis settings and prediction-file hashes. These scripts analyze existing runs; they do not retrain models.","note")

    # Page 10: exact features and the actual news search definitions.
    page();h("Appendix B  Exact covariates and news queries")
    add("Feature names below match the executed configuration. Every count excludes the target <font name='Courier' size='8.5'>log_return_1d</font>. The fixed control is 15 past fields plus six known-future fields; it is a selected experiment configuration, distinct from the repository's broader default core profile.")
    controls=CFG["control_features"]
    data=[["Control group", "Exact column names"]]
    for label,cols in [("QQQ (6 past)",controls[:6]),("Market (5 past)",controls[6:11]),("Rates / macro (4 past)",controls[11:15]),("Calendar (6 known future)",controls[15:])]:
        data.append([label,", ".join(f"<font name='Courier' size='8'>{c}</font>" for c in cols)])
    table(data,[40,126],"Common 21-covariate control. No raw price levels or other default-profile news columns are selected implicitly.",font=9)
    add("GDELT queries use <font name='Courier' size='8.5'>sourcelang:eng</font> and no timeline smoothing. News share is matching articles divided by all monitored articles in the response; tone is the API's average tone. The article count is used to construct share but is not itself an input. News share and tone are separate channels, and overlapping query memberships are permitted.")
    tree=ast.parse((ROOT/"scripts/fetch_gdelt_news.py").read_text())
    topics=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="TOPICS" for t in n.targets))
    query_rows=[["Topic", "Exact query before the English-language filter"]]
    for key,label in [("ai","AI"),("semiconductor","Semiconductors"),("fed","Federal Reserve"),("inflation","Inflation"),("big_tech_earnings","Big-tech earnings"),("recession","Recession")]:
        query_rows.append([label,escape(topics[key])])
    table(query_rows,[35,131],"The six executed news definitions. Each run adds gdelt_TOPIC_news_share and gdelt_TOPIC_avg_tone. Topic keys in row order: ai, semiconductor, fed, inflation, big_tech_earnings, recession.",font=9)
    sub("B.1  Other external-source columns and transforms")
    data=[["Added source", "Columns added to the same control"]]
    for key,label in [("uncertainty","EPU / EMU (+2)"),("fomc_tone","FOMC tone (+4)"),("sec_disclosures","SEC disclosures (+3)")]:
        data.append([label,", ".join(f"<font name='Courier' size='8'>{x}</font>" for x in CFG['setups'][key]['add_features'])])
    table(data,[40,126],"The all-external setup combines these nine fields with 12 GDELT fields: 21 controls + 21 external covariates = 42.",font=9)
    add("EPU/EMU inputs are the logarithm of trailing seven-calendar-day means, subsequently aligned to the exchange session. FOMC tone uses sentence-level hawkish/dovish/neutral classification across 170 statements; the four selected fields summarize the most recent state, its exponentially weighted level, its change, and elapsed time. SEC activity counts accepted filings and past earnings announcements; it is not a future earnings calendar.","note")
    add("QQQ momentum and distance-to-average are log ratios; Parkinson range volatility uses high/low ranges; volume is standardized over 20 sessions. Rate changes are in basis points; slope and CPI growth retain regime information. Calendar flags include Consumer Price Index (CPI) and nonfarm payrolls (NFP) releases. These transforms precede Chronos-2's context scaling.","note")


def main():
    TMP.mkdir(parents=True,exist_ok=True)
    fonts();plot_settings()
    global S
    S=styles()
    manuscript()
    doc=BaseDocTemplate(str(OUT),pagesize=A4,leftMargin=22*mm,rightMargin=22*mm,
        topMargin=23*mm,bottomMargin=23*mm,
        title="Which News Helps Forecast QQQ? Controlled Covariate Experiments with Chronos-2 and LoRA",
        author="Elad Kulman and Tom Weitman",
        subject="Workshop on Deep Learning: executed QQQ forecasting experiments and sensitivity analysis")
    frame=Frame(22*mm,23*mm,WIDTH,251*mm,id="paper",leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="Paper",frames=[frame],onPage=footer))
    doc.build(STORY)
    print(f"Wrote {OUT} ({FIGURE} figures, {TABLE} tables)")


if __name__=="__main__":
    main()
