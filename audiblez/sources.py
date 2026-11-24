# -*- coding: utf-8 -*-
"""
Inputs other than EPUB: plain-text / Markdown story scripts, FB2 e-books and Word documents.

Story channels publish original stories, written as plain text (or Markdown, or a Google Docs / Word export), and
most Russian e-books are distributed as FB2. Every reader returns a `Source` whose documents expose the small part
of the ebooklib item interface used by the pipeline (get_name, get_type, blocks, extracted_text...), so the rest
of audiblez (chapter splitting, normalization, synthesis, export) is identical for every format.
"""
import base64
import html
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from audiblez.text import (BREAK, HEADING, PARA, Block, blocks_to_text, clean_inline_text, html_to_blocks,
                           is_chapter_heading_text, postprocess_blocks, _SCENE_BREAK_RE)

TEXT_SUFFIXES = ('.txt', '.text')
MARKDOWN_SUFFIXES = ('.md', '.markdown')
SUPPORTED_SUFFIXES = TEXT_SUFFIXES + MARKDOWN_SUFFIXES + ('.fb2', '.fb2.zip', '.docx')


class SourceDocument:
    """One document of a non-EPUB source, compatible with what the pipeline reads from ebooklib items."""

    def __init__(self, name, blocks: List[Block], toc_title=None, index=0):
        import ebooklib
        self._name = name
        self._type = ebooklib.ITEM_DOCUMENT
        self.blocks = blocks
        self.extracted_text = blocks_to_text(blocks)
        self._text_signature = self.extracted_text
        self.toc_title = toc_title
        self.chapter_index = index

    def get_name(self):
        return self._name

    def get_type(self):
        return self._type

    def get_content(self):
        return self.extracted_text.encode('utf-8')


@dataclass
class Source:
    title: str
    creator: str
    language: Optional[str]
    documents: List[SourceDocument] = field(default_factory=list)
    cover: bytes = b''


def is_supported(path) -> bool:
    name = Path(path).name.lower()
    return any(name.endswith(s) for s in SUPPORTED_SUFFIXES)


def read_source(path) -> Source:
    path = Path(path)
    name = path.name.lower()
    if name.endswith('.fb2.zip') or name.endswith('.fb2'):
        return read_fb2(path)
    if name.endswith('.docx'):
        return read_docx(path)
    if name.endswith(MARKDOWN_SUFFIXES):
        return read_text(path, markdown=True)
    if name.endswith(TEXT_SUFFIXES):
        return read_text(path, markdown=False)
    raise ValueError(f'Unsupported input format: {path.name} (supported: .epub, {", ".join(SUPPORTED_SUFFIXES)})')


# ---------------------------------------------------------------------------------------------------------------------
# Language detection (used when the file has no language metadata)
# ---------------------------------------------------------------------------------------------------------------------

def detect_language(text) -> Optional[str]:
    counts = dict(cyr=0, uk=0, lat=0, kana=0, hangul=0, han=0)
    for ch in text[:200000]:
        o = ord(ch)
        if 0x0400 <= o <= 0x04FF:
            counts['cyr'] += 1
            if ch in 'іїєґІЇЄҐ':
                counts['uk'] += 1
        elif ch.isascii() and ch.isalpha():
            counts['lat'] += 1
        elif 0x3040 <= o <= 0x30FF:
            counts['kana'] += 1
        elif 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
            counts['hangul'] += 1
        elif 0x4E00 <= o <= 0x9FFF:
            counts['han'] += 1
    if not any(counts.values()):
        return None
    if counts['kana'] > 0.05 * (counts['han'] + counts['kana']) and counts['kana'] + counts['han'] > counts['lat']:
        return 'ja'
    script = max(('cyr', 'lat', 'hangul', 'han'), key=lambda k: counts[k])
    if script == 'cyr':
        return 'uk' if counts['uk'] > 0.005 * counts['cyr'] else 'ru'
    return {'lat': 'en', 'hangul': 'ko', 'han': 'zh'}[script]


# ---------------------------------------------------------------------------------------------------------------------
# Plain text and Markdown
# ---------------------------------------------------------------------------------------------------------------------

def decode_text(data: bytes) -> str:
    """UTF-8 (with or without BOM), UTF-16 with BOM, otherwise Windows-1251 (most legacy Russian text files)."""
    if data[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return data.decode('utf-16')
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError:
        pass
    text = data.decode('utf-8-sig', errors='replace')
    if text.count('\ufffd') <= max(2, len(text) // 2000):  # UTF-8 with a few broken bytes
        return text
    try:
        return data.decode('cp1251')
    except UnicodeDecodeError:
        return data.decode('latin-1')


_DIALOGUE_START_RE = re.compile(r'^\s*(?:[—–]|-{1,2})\s')
_TERMINAL_RE = re.compile(r'[.!?…:;»"”’)\]]\s*$')
_MD_HEADING_RE = re.compile(r'^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$')
_MD_RULE_RE = re.compile(r'^\s{0,3}(?:[-*_]\s*){3,}$')


def _is_wrapped(groups) -> bool:
    """Hard-wrapped text (a paragraph spread over several lines, e.g. copied from a PDF or a Gutenberg file)?"""
    total = continued = 0
    for g in groups:
        for a, b in zip(g, g[1:]):
            total += 1
            if not _TERMINAL_RE.search(a) and re.match(r'\s*[a-zа-яё,(]', b):
                continued += 1  # the line ends mid-sentence and the next one goes on in lower case
    return total >= 3 and continued / total > 0.3


def _unwrap(group, width):
    """Join the lines of a hard-wrapped block into paragraphs."""
    paragraphs, cur = [], ''
    for i, line in enumerate(group):
        indented = bool(re.match(r'^(?:\t| {2,})\S', line))
        prev = group[i - 1] if i else ''
        new_para = (not cur or _DIALOGUE_START_RE.match(line)
                    or (indented and not re.match(r'^(?:\t| {2,})\S', prev))
                    or (_TERMINAL_RE.search(prev) and len(prev.strip()) < 0.6 * width))
        if new_para:
            if cur:
                paragraphs.append(cur)
            cur = line.strip()
        elif cur.endswith('-') and not cur.endswith(' -') and re.match(r'\s*[a-zа-яё]', line):
            cur = cur[:-1] + line.strip()  # word hyphenated at the end of the line
        else:
            cur += ' ' + line.strip()
    if cur:
        paragraphs.append(cur)
    return paragraphs


def _clean_markdown_inline(line):
    line = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', line)  # images
    line = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', line)  # links -> their text
    line = re.sub(r'<[^>]+>', '', line)  # inline HTML
    line = re.sub(r'(\*\*|__|~~|`)', '', line)
    line = re.sub(r'(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])', r'\1', line)  # *emphasis*
    line = re.sub(r'(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])', r'\1', line)
    line = re.sub(r'^\s{0,3}>\s?', '', line)  # blockquote
    return line


def _dialogue_dash(text):
    """'- Привет, - сказал он.' (hyphens typed instead of dashes in story scripts) -> '— Привет, — сказал он.'"""
    text = re.sub(r'^\s*(?:-{1,2}|–)\s+', '— ', text)
    return re.sub(r'(?<=\s)(?:-{1,2}|–)(?=\s)', '—', text)


def text_to_source_blocks(text, markdown=False) -> List[Block]:
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\f', '\n')
    groups, cur = [], []
    for line in text.split('\n'):
        if line.strip():
            cur.append(line.rstrip())
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    wrapped = _is_wrapped(groups)
    widths = sorted(len(line) for g in groups for line in g)
    width = widths[int(len(widths) * 0.9)] if widths else 80

    blocks: List[Block] = []
    for g in groups:
        lines = []
        for line in g:
            if markdown:
                m = _MD_HEADING_RE.match(line)
                if m:
                    if lines:
                        blocks.extend(_lines_to_blocks(_unwrap(lines, width) if wrapped else lines))
                        lines = []
                    title = clean_inline_text(_clean_markdown_inline(m.group(2)))
                    if title:
                        blocks.append(Block(HEADING, title, len(m.group(1))))
                    continue
                if _MD_RULE_RE.match(line):
                    if lines:
                        blocks.extend(_lines_to_blocks(_unwrap(lines, width) if wrapped else lines))
                        lines = []
                    blocks.append(Block(BREAK))
                    continue
                line = _clean_markdown_inline(line)
            lines.append(line)
        if lines:
            blocks.extend(_lines_to_blocks(_unwrap(lines, width) if wrapped else lines))
    return postprocess_blocks(blocks)


def _lines_to_blocks(lines) -> List[Block]:
    out = []
    for line in lines:
        line = clean_inline_text(line)
        if not line:
            continue
        if not re.search(r'\w', line):
            if _SCENE_BREAK_RE.match(line) and re.search(r'[*•·⁂~#◆◇■□●○✱✲✳❖※]|(?:[—–\-]\s*){3}', line):
                out.append(Block(BREAK))
            continue
        line = _dialogue_dash(line)
        if is_chapter_heading_text(line):
            out.append(Block(HEADING, line, 2))
        else:
            out.append(Block(PARA, line))
    return out


def _pop_title(blocks: List[Block]):
    """A first line that looks like a title (short, no sentence punctuation) becomes the title of the story."""
    if not blocks:
        return '', blocks
    first = blocks[0]
    if first.kind == HEADING and first.level == 1 and not any(b.kind == HEADING and b.level == 1 for b in blocks[1:]):
        return first.text, blocks[1:]
    if (first.kind == PARA and len(first.text) <= 90 and not re.search(r'[.!?…:;,]$', first.text)
            and not first.text.startswith(('—', '-', '«', '"', '„', '“')) and len(blocks) > 1
            and not is_chapter_heading_text(first.text)):
        return first.text, blocks[1:]
    return '', blocks


def read_text(path, markdown=False) -> Source:
    path = Path(path)
    text = decode_text(path.read_bytes())
    blocks = text_to_source_blocks(text, markdown=markdown)
    title, blocks = _pop_title(blocks)
    return Source(title=title, creator='', language=detect_language(text),
                  documents=[SourceDocument(path.name, blocks)] if blocks else [])


# ---------------------------------------------------------------------------------------------------------------------
# FB2 (FictionBook 2), also zipped (.fb2.zip)
# ---------------------------------------------------------------------------------------------------------------------

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


def read_fb2(path) -> Source:
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
            documents.append(SourceDocument(f'{path.name}#{len(documents) + 1}', blocks, index=len(documents)))
    if not language:
        language = detect_language(' '.join(d.extracted_text[:5000] for d in documents[:5]))
    return Source(title=title, creator=', '.join(a for a in authors if a), language=language, documents=documents,
                  cover=cover)


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


def read_docx(path) -> Source:
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
    blocks = [Block(b.kind, _dialogue_dash(b.text), b.level) if b.kind == PARA else b for b in blocks]
    title = doc_title or title
    if not title:
        title, blocks = _pop_title(blocks)
    text = blocks_to_text(blocks)
    return Source(title=title, creator='', language=detect_language(text),
                  documents=[SourceDocument(path.name, blocks)] if blocks else [])
