# -*- coding: utf-8 -*-
"""
FB2 and DOCX inputs.

- FB2 / .fb2.zip (FictionBook 2): the usual format of Russian e-books. Title, authors, language and cover from the
  description; section titles become chapters; notes bodies and footnote links are not read; epigraphs, poems
  (one stanza = one paragraph) and "* * *" subtitles (scene breaks) are kept.
- DOCX (Word, Google Docs export): the Title style is the title, Heading 1-6 are chapters.

Both return the same structure as the other inputs (see core.load_source).
"""
import base64
import html
import io
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

from audiblez.text import blocks_to_text, detect_language, html_to_blocks
from audiblez.textinput import TextDocument

FB2_SUFFIXES = ('.fb2', '.fb2.zip')


def is_fb2(path):
    return Path(path).name.lower().endswith(FB2_SUFFIXES)


def is_docx(path):
    return Path(path).name.lower().endswith('.docx')



def _local(tag):
    return tag.rsplit('}', 1)[-1] if isinstance(tag, str) else ''


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _first(el, *path):
    for name in path:
        if el is None:
            return None
        el = next(iter(_children(el, name)), None)
    return el


def _text_of(el):
    return re.sub(r'\s+', ' ', ''.join(el.itertext())).strip() if el is not None else ''


def _href(el):
    for k, v in el.attrib.items():
        if _local(k) == 'href':
            return v
    return ''


def _fb2_inline(el):
    """Inline content of an FB2 paragraph as HTML (footnote links are kept as <a> so html_to_blocks drops them)."""
    out = [html.escape(el.text or '')]
    for c in el:
        name = _local(c.tag)
        if name == 'a' and (c.get('type') == 'note' or _href(c).startswith('#')):
            out.append(f'<a href="{html.escape(_href(c))}" epub:type="noteref">{html.escape(_text_of(c))}</a>')
        elif name == 'image':
            pass
        else:
            out.append(_fb2_inline(c))
        out.append(html.escape(c.tail or ''))
    return ''.join(out)


def _fb2_section_html(el, depth, out):
    for c in el:
        name = _local(c.tag)
        if name == 'title':
            text = '. '.join(t for t in (_text_of(p) for p in _children(c, 'p')) if t) or _text_of(c)
            if text:
                level = min(depth, 6)
                out.append(f'<h{level}>{html.escape(text)}</h{level}>')
        elif name == 'section':
            _fb2_section_html(c, depth + 1, out)
        elif name == 'p':
            out.append(f'<p>{_fb2_inline(c)}</p>')
        elif name == 'subtitle':
            text = _text_of(c)
            out.append('<hr/>' if text and not re.search(r'\w', text) else f'<h6>{_fb2_inline(c)}</h6>')
        elif name in ('epigraph', 'cite', 'annotation'):
            _fb2_section_html(c, depth, out)
        elif name == 'text-author':
            out.append(f'<p>{_fb2_inline(c)}</p>')
        elif name == 'poem':
            for part in c:
                pname = _local(part.tag)
                if pname == 'title':
                    out.append(f'<p>{html.escape(_text_of(part))}</p>')
                elif pname == 'stanza':
                    verses = [_fb2_inline(v) for v in part if _local(v.tag) == 'v']
                    if verses:
                        out.append('<p>' + '<br/>'.join(verses) + '</p>')
                elif pname == 'text-author':
                    out.append(f'<p>{_fb2_inline(part)}</p>')
        elif name == 'table':
            for row in c.iter():
                if _local(row.tag) in ('td', 'th'):
                    out.append(f'<p>{_fb2_inline(row)}</p>')
        # empty-line, image: nothing to read


def _document(name, blocks, index=0):
    doc = TextDocument(name=name, blocks=blocks)
    doc.chapter_index = index
    return doc


def read_fb2(path):
    from lxml import etree
    path = Path(path)
    data = path.read_bytes()
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = [n for n in z.namelist() if n.lower().endswith('.fb2')] or z.namelist()
            data = z.read(names[0])
    root = etree.fromstring(data, parser=etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False))
    info = _first(root, 'description', 'title-info')
    title = _text_of(_first(info, 'book-title'))
    authors = []
    for a in _children(info, 'author') if info is not None else []:
        name = ' '.join(t for t in (_text_of(_first(a, n)) for n in ('first-name', 'middle-name', 'last-name')) if t)
        authors.append(name or _text_of(_first(a, 'nickname')))
    language = (_text_of(_first(info, 'lang')) or '').lower()[:2] or None

    binaries = {b.get('id'): b for b in root if _local(b.tag) == 'binary'}
    cover = b''
    cover_image = _first(info, 'coverpage', 'image')
    if cover_image is not None:
        binary = binaries.get(_href(cover_image).lstrip('#'))
        if binary is not None and binary.text:
            try:
                cover = base64.b64decode(re.sub(r'\s+', '', binary.text))
            except Exception:
                cover = b''

    documents = []
    bodies = [b for b in root if _local(b.tag) == 'body' and (b.get('name') or '').lower() not in ('notes', 'comments',
                                                                                                    'footnotes')]
    for body in bodies:
        # One document per body: the chapters are found from the section titles (h1/h2), like in single-file EPUBs,
        # and a body-level epigraph is merged with the first chapter instead of becoming a chapter of its own.
        html_parts = []
        for c in body:
            name = _local(c.tag)
            if name == 'section':
                _fb2_section_html(c, 1, html_parts)
            elif name == 'epigraph':
                _fb2_section_html(c, 1, html_parts)
            # the body <title> is the title page (title and author are announced from the metadata)
        blocks = html_to_blocks('<html><body>' + ''.join(html_parts) + '</body></html>') if html_parts else []
        if blocks:
            documents.append(_document(f'{path.name}#{len(documents) + 1}', blocks, len(documents)))
    if not language:
        language = detect_language(' '.join(d.extracted_text[:5000] for d in documents[:5]))
    return SimpleNamespace(kind='fb2', title=title, creator=', '.join(a for a in authors if a), language=language,
                           cover=cover, documents=documents)


# ---------------------------------------------------------------------------------------------------------------------
# DOCX (Word, Google Docs export)
# ---------------------------------------------------------------------------------------------------------------------

_W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def _docx_heading_levels(z) -> dict:
    """styleId -> heading level (0 = title) from word/styles.xml (style names are English even in localized Word)."""
    from lxml import etree
    levels = {}
    try:
        root = etree.fromstring(z.read('word/styles.xml'))
    except KeyError:
        return levels
    for st in root.iter(_W + 'style'):
        sid = st.get(_W + 'styleId')
        name_el = st.find(_W + 'name')
        name = (name_el.get(_W + 'val') if name_el is not None else '') or ''
        m = re.fullmatch(r'(?i)heading\s*(\d)', name.strip())
        if m:
            levels[sid] = int(m.group(1))
        elif name.strip().lower() == 'title':
            levels[sid] = 0
    return levels


def read_docx(path):
    from lxml import etree
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        styles = _docx_heading_levels(z)
        root = etree.fromstring(z.read('word/document.xml'))
        title = ''
        try:
            core = etree.fromstring(z.read('docProps/core.xml'))
            title = next((_text_of(e) for e in core.iter() if _local(e.tag) == 'title'), '')
        except KeyError:
            pass
    body = root.find(_W + 'body')
    out, doc_title = [], ''
    for p in body.iter(_W + 'p') if body is not None else []:
        style = p.find(f'{_W}pPr/{_W}pStyle')
        level = styles.get(style.get(_W + 'val')) if style is not None else None
        outline = p.find(f'{_W}pPr/{_W}outlineLvl')
        if level is None and outline is not None:
            try:
                level = int(outline.get(_W + 'val')) + 1
            except (TypeError, ValueError):
                level = None
        parts = []
        for el in p.iter(_W + 't', _W + 'tab', _W + 'br', _W + 'cr'):
            if next(el.iterancestors(_W + 'p'), None) is not p:
                continue  # text of a nested paragraph (text box): read on its own
            tag = el.tag
            if tag == _W + 't':
                parts.append(html.escape(el.text or ''))
            elif tag == _W + 'tab':
                parts.append(' ')
            elif tag in (_W + 'br', _W + 'cr'):
                parts.append('<br/>')
        content = ''.join(parts)
        if not content.strip():
            continue
        if level == 0:
            doc_title = doc_title or re.sub(r'<br/>', ' ', html.unescape(content)).strip()
            continue
        if level is not None and 1 <= level <= 6:
            out.append(f'<h{level}>{content}</h{level}>')
        else:
            out.append(f'<p>{content}</p>')
    blocks = html_to_blocks('<html><body>' + ''.join(out) + '</body></html>')
    title = doc_title or title or path.stem
    return SimpleNamespace(kind='text', title=title, creator='', language=detect_language(blocks_to_text(blocks)),
                           cover=b'', documents=[_document(path.name, blocks)] if blocks else [])
