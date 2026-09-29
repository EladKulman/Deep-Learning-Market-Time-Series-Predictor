#!/usr/bin/env python3
"""Compile editable report/ Markdown, CSV tables and figures into an atomic PDF output."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
from xml.sax.saxutils import escape, quoteattr

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'tmp/pdfs/editable_report'
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(CACHE / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(CACHE / 'cache'))
# The local project environment lacks ReportLab; use the installed desktop bundle
# if needed, without requiring a network download or changing that environment.
try:
    import reportlab
except ImportError:
    bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib'
    for site in sorted(bundled.glob('python*/site-packages')):
        sys.path.append(str(site))
try:
    from markdown_it import MarkdownIt
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Spacer, ListFlowable, ListItem
    from reportlab.lib.units import mm
    from reportlab.lib.pagesizes import A4
    import presentation_report_layout as layout
except ImportError as exc:
    raise SystemExit(f'Missing report dependency: {exc.name}. Install with:\n'
                     f'  {sys.executable} -m pip install -r {ROOT / "report/requirements.txt"}') from exc

MD = MarkdownIt('commonmark', {'html': True})


def inline(text: str) -> str:
    """Translate CommonMark inline tokens to ReportLab paragraph markup."""
    result = []
    for token in MD.parseInline(text)[0].children or []:
        kind = token.type
        if kind == 'text':
            result.append(escape(token.content))
        elif kind in ('softbreak', 'hardbreak'):
            result.append(' ' if kind == 'softbreak' else '<br/>')
        elif kind == 'code_inline':
            result.append(f'<font name="PaperMono" size="8.5">{escape(token.content)}</font>')
        elif kind in ('strong_open', 'strong_close', 'em_open', 'em_close'):
            result.append({'strong_open':'<b>', 'strong_close':'</b>', 'em_open':'<i>', 'em_close':'</i>'}[kind])
        elif kind == 'link_open':
            result.append(f'<link href={quoteattr(token.attrGet("href"))} color="{layout.BLUE}">')
        elif kind == 'link_close':
            result.append('</link>')
        elif kind == 'html_inline':
            # Existing typography (fonts, anchors, line breaks) remains editable.
            if not re.match(r'^</?(?:a|font|br|b|i|super|sub)\b', token.content):
                raise ValueError(f'Unsupported inline HTML: {token.content}')
            result.append(token.content)
        else:
            raise ValueError(f'Unsupported inline Markdown: {kind}; use a figure directive for images')
    return ''.join(result)


def markdown(text: str) -> None:
    tokens = MD.parse(text)
    i = 0

    def blocks(end=None):
        nonlocal i
        output = []
        while i < len(tokens):
            t = tokens[i]
            if t.type == end:
                i += 1
                return output
            if t.type in ('paragraph_open', 'heading_open'):
                style = 'h1' if t.tag == 'h1' else 'h2' if t.type == 'heading_open' else 'body'
                output.append(layout.p(inline(tokens[i+1].content), style))
                i += 3
            elif t.type in ('bullet_list_open', 'ordered_list_open'):
                closing = t.type.replace('_open', '_close')
                ordered = t.type == 'ordered_list_open'
                start = t.attrGet('start') or 1
                i += 1
                items = []
                while i < len(tokens) and tokens[i].type != closing:
                    if tokens[i].type != 'list_item_open':
                        raise ValueError('Malformed list')
                    i += 1
                    items.append(ListItem(blocks('list_item_close')))
                if i == len(tokens):
                    raise ValueError('Unclosed list')
                i += 1
                output.append(ListFlowable(items, bulletType='1' if ordered else 'bullet',
                                           start=start if ordered else None, leftIndent=14,
                                           bulletFontName='PaperRegular', bulletFontSize=9))
            elif t.type == 'html_block' and t.content.strip().startswith('<!--'):
                i += 1  # Author notes do not appear in the paper.
            else:
                raise ValueError(f'Unsupported Markdown block: {t.type}')
        if end:
            raise ValueError(f'Unclosed Markdown block: {end}')
        return output

    layout.STORY.extend(blocks())


def directive(header: str, body: str, source: Path) -> None:
    words = shlex.split(header)
    name, args = words[0], words[1:]
    if name in layout.S:
        if args:
            raise ValueError(f'{name} takes no arguments')
        layout.add(inline(body), name)
        return
    if name == 'spacer':
        layout.STORY.append(Spacer(1, float(args[0]) * mm))
    elif name == 'pagebreak':
        layout.page()
    elif name == 'equation':
        layout.equation('$' + body.strip().strip('$') + '$', int(args[0]))
    elif name in ('table', 'figure'):
        path = (source.parent / args[0]).resolve()
        if not path.is_file():
            raise ValueError(f'Missing {name} file: {path}')
        opts = dict(arg.split('=', 1) for arg in args[1:])
        if name == 'figure':
            if set(opts) - {'width'}:
                raise ValueError(f'Unknown figure options: {opts}')
            layout.figure(path, inline(body), width=float(opts.get('width', 166)))
        else:
            if set(opts) - {'widths', 'font'}:
                raise ValueError(f'Unknown table options: {opts}')
            with path.open(newline='', encoding='utf-8-sig') as f:
                rows = list(csv.reader(f))
            if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
                raise ValueError(f'Table must have a header and equally sized rows: {path}')
            widths = [float(x) for x in opts['widths'].split(',')] if 'widths' in opts else [166/len(rows[0])]*len(rows[0])
            if len(widths) != len(rows[0]) or any(w <= 0 for w in widths) or sum(widths) > 166.01:
                raise ValueError(f'Table widths must match {len(rows[0])} columns and total at most 166 mm')
            layout.table([[inline(c) for c in row] for row in rows], widths, inline(body),
                         font=float(opts.get('font', 9)))
    else:
        raise ValueError(f'Unknown directive: {name}')


def section(path: Path) -> None:
    lines = path.read_text(encoding='utf-8').splitlines()
    pending = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(':::'):
            if pending:
                markdown('\n'.join(pending)); pending = []
            line_no = i+1
            header = lines[i][3:].strip()
            if not header:
                raise ValueError(f'{path}:{line_no}: unexpected directive terminator')
            body = []
            i += 1
            while i < len(lines) and lines[i].strip() != ':::':
                body.append(lines[i]); i += 1
            if i == len(lines):
                raise ValueError(f'{path}:{line_no}: unclosed {header!r} directive')
            try:
                directive(header, '\n'.join(body), path)
            except (ValueError, IndexError, KeyError) as exc:
                raise ValueError(f'{path}:{line_no}: {exc}') from exc
        else:
            pending.append(lines[i])
        i += 1
    if pending:
        markdown('\n'.join(pending))


class ReportDocument(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, 'style') and flowable.style.name == 'H1':
            self.section_index = getattr(self, 'section_index', 0) + 1
            key = f'section-{self.section_index}'
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0, closed=False)


def build(config_path: Path, output: Path | None = None) -> Path:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    output = (output or config_path.parent / config['output']).resolve()
    layout.STORY = []; layout.FIGURE = 0; layout.TABLE = 0
    layout.TMP = CACHE / 'equations'; layout.TMP.mkdir(parents=True, exist_ok=True)
    layout.fonts(); layout.S = layout.styles(); layout.plot_settings()
    layout.HEADER_LEFT = config.get('header_left', '')
    layout.HEADER_RIGHT = config.get('header_right', '')
    if not config.get('sections'):
        raise ValueError('report.json must list at least one section')
    for n, entry in enumerate(config['sections']):
        source = config_path.parent / entry['file']
        if n and entry.get('page_break_before', False):
            layout.page()
        try:
            section(source)
        except Exception as exc:
            raise ValueError(f'Cannot compile {source}: {exc}') from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the last good PDF if malformed content or an oversized image fails.
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.pdf', delete=False) as f:
        temporary = Path(f.name)
    try:
        doc = ReportDocument(str(temporary), pagesize=A4, leftMargin=22*mm, rightMargin=22*mm,
                             topMargin=23*mm, bottomMargin=23*mm,
                             title=config.get('title', ''), author=config.get('author', ''))
        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal',
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        doc.addPageTemplates(PageTemplate(id='paper', frames=frame, onPage=layout.footer))
        doc.build(layout.STORY)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Built {output}\n{len(config["sections"])} section files, {layout.FIGURE} figures, {layout.TABLE} tables.')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'report/report.json')
    parser.add_argument('--output', type=Path, help='Override the configured PDF output path')
    args = parser.parse_args()
    try:
        build(args.config, args.output)
    except Exception as exc:
        parser.exit(1, f'Report build failed: {exc}\n')


if __name__ == '__main__':
    main()
