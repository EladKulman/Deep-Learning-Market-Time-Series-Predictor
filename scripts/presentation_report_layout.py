#!/usr/bin/env python3
"""Typography, charts, and layout helpers for the presentation-based report."""
from __future__ import annotations

import os
from pathlib import Path
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qqq-report-mpl-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
OUT = ROOT / "output/pdf/qqq_chronos2_project_report.pdf"
TMP = ROOT / "tmp/pdfs/refined_report"
NAVY, BLUE, TEAL, GOLD, GRAY = "#193650", "#3579AA", "#217D77", "#B9792B", "#757B82"
PALE = "#EEF3F6"
HEADER_LEFT = "News Covariates for QQQ Return Forecasting"
HEADER_RIGHT = "Workshop on Deep Learning | Group 13"
WIDTH = 166 * mm
STORY = []
FIGURE = TABLE = 0


def fonts():
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype"
    candidates = ([Path(os.environ['REPORT_FONT_DIR'])] if os.environ.get('REPORT_FONT_DIR') else []) + [
        bundled, Path('/usr/share/fonts/truetype/liberation2'), Path('/usr/share/fonts/truetype/liberation')]
    directory = next((p for p in candidates if (p/'LiberationSerif-Regular.ttf').exists()), None)
    if directory:
        for suffix in ['Regular', 'Bold', 'Italic', 'BoldItalic']:
            pdfmetrics.registerFont(TTFont('Paper'+suffix, str(directory/f'LiberationSerif-{suffix}.ttf')))
        mono = directory/'LiberationMono-Regular.ttf'
    else:
        directory = Path(matplotlib.get_data_path())/'fonts/ttf'
        for label, suffix in [('Regular',''), ('Bold','-Bold'), ('Italic','-Italic'), ('BoldItalic','-BoldItalic')]:
            pdfmetrics.registerFont(TTFont('Paper'+label, str(directory/f'DejaVuSerif{suffix}.ttf')))
        mono = directory/'DejaVuSansMono.ttf'
    pdfmetrics.registerFontFamily('PaperRegular', normal='PaperRegular', bold='PaperBold', italic='PaperItalic', boldItalic='PaperBoldItalic')
    pdfmetrics.registerFont(TTFont('PaperMono', str(mono)))


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

def footer(canvas,doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("PaperRegular",8)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(22*mm,281*mm,HEADER_LEFT)
        canvas.drawRightString(188*mm,281*mm,HEADER_RIGHT)
        canvas.setStrokeColor(colors.HexColor("#D7DEE3"));canvas.setLineWidth(.4)
        canvas.line(22*mm,278.5*mm,188*mm,278.5*mm)
    canvas.setFont("PaperRegular",9)
    canvas.setFillColor(colors.HexColor(GRAY))
    canvas.drawCentredString(A4[0]/2,15*mm,str(doc.page))
    canvas.restoreState()

