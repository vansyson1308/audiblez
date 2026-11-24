# -*- coding: utf-8 -*-
"""
Text extraction and segmentation for audiobook synthesis.

- EPUB (X)HTML -> ordered list of blocks (headings, paragraphs, scene breaks), without
  footnote markers, footnote bodies, scripts, ruby annotations, soft hyphens, etc.
- Splitting of single documents that contain several chapters (by headings).
- Language-aware sentence splitting and packing of sentences into TTS-sized chunks.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

import warnings

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import Comment, Doctype, ProcessingInstruction, Declaration, CData

try:
    from bs4 import XMLParsedAsHTMLWarning
except ImportError:  # pragma: no cover
    XMLParsedAsHTMLWarning = UserWarning

HEADING = 'heading'
PARA = 'para'
BREAK = 'break'


@dataclass
class Block:
    kind: str  # HEADING | PARA | BREAK
    text: str = ''
    level: int = 0  # heading level (1..6), 0 for non headings


@dataclass
class Section:
    """A chapter of the audiobook: a title and its blocks."""
    title: Optional[str]
    blocks: List[Block] = field(default_factory=list)

    @property
    def text_length(self):
        return sum(len(b.text) for b in self.blocks)


# ---------------------------------------------------------------------------------------------------------------------
# HTML -> blocks
# ---------------------------------------------------------------------------------------------------------------------

BLOCK_TAGS = {
    'address', 'article', 'aside', 'blockquote', 'body', 'caption', 'center', 'dd', 'details', 'dialog', 'div', 'dl',
    'dt', 'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header',
    'hgroup', 'li', 'main', 'ol', 'p', 'pre', 'section', 'summary', 'table', 'tbody', 'td', 'tfoot', 'th', 'thead',
    'title', 'tr', 'ul', 'html',
}
SKIP_TAGS = {
    'script', 'style', 'head', 'rt', 'rp', 'svg', 'math', 'img', 'audio', 'video', 'object', 'iframe', 'template',
    'noscript', 'nav', 'button', 'input', 'select', 'textarea', 'map', 'area', 'canvas', 'embed', 'source', 'track',
}
HEADING_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

_NOTE_TYPES = ('footnote', 'endnote', 'rearnote', 'footnotes', 'endnotes', 'rearnotes', 'note')
_NOTE_ROLES = ('doc-footnote', 'doc-endnote', 'doc-endnotes', 'doc-footnotes', 'doc-backlink')
_NOTE_CLASSES = ('mw-references', 'references', 'mw-cite-backlink', 'footnotes', 'endnotes', 'reflist',
                 'mw-references-wrap', 'references-small')
# A footnote marker: "1", "[12]", "*", "**", "†", "(3)", "i", "iv"...
_MARKER_RE = re.compile(r'^\s*[\[(]?\s*(?:\d{1,4}|[*†‡§¶]{1,4}|[ivxlc]{1,6})\s*[\])]?\s*$', re.I)

LINE_BREAK = ' '  # internal marker for <br>
_INVISIBLE_RE = re.compile('[­​‌‍‎‏⁠﻿᠎]')
_SPACES_RE = re.compile(r'[ \t\r\n\f\v   -   　]+')
_SCENE_BREAK_RE = re.compile(r'^[\s*•·⁂~_=#◆◇■□●○✱✲✳❖※—–\-.]+$')

_ROMAN_RE = r'(?=[MDCLXVI])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})'
_ORDINAL_WORDS = (
    r'(?:(?:перв|втор|трет|четв[её]рт|пят|шест|седьм|восьм|девят|десят|одиннадцат|двенадцат|тринадцат|'
    r'четырнадцат|пятнадцат|шестнадцат|семнадцат|восемнадцат|девятнадцат|двадцат|тридцат|сороков)'
    r'(?:ая|ый|ое|ой|ий|ья|ье|ей)|последн(?:яя|ий|ее)|'
    r'first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|last|'
    r'one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
    r'premier|première|deuxième|primero|primera|segundo|segunda|primo|prima|secondo|seconda|erste[rs]?|zweite[rs]?)'
)
_NUMBERED_WORDS = (r'(?:глава|часть|книга|том|раздел|chapter|part|book|volume|capítulo|capitulo|parte|libro|'
                   r'chapitre|partie|livre|kapitel|teil|buch|capitolo)')
_STANDALONE_WORDS = (r'(?:пролог|эпилог|предисловие|послесловие|интерлюдия|вступление|заключение|prologue|epilogue|'
                     r'preface|interlude|introduction|conclusion|prólogo|epílogo|prologo|epilogo|épilogue|prolog|'
                     r'epilog)')
_HEADING_NUM = rf'(?:{_ROMAN_RE}|\d{{1,4}}|{_ORDINAL_WORDS})'
_CHAPTER_HEADING_RE = re.compile(
    rf'^\s*(?:{_NUMBERED_WORDS}\.?\s+{_HEADING_NUM}|{_HEADING_NUM}|{_STANDALONE_WORDS})'
    r'\s*\.?(?:\s*[.:—–\-]\s*\S.{0,79})?\s*$',
    re.IGNORECASE)
# Cyrillic letters often typed instead of Latin ones in Roman numerals (e.g. "ХII" with a Cyrillic "Х")
_ROMAN_LOOKALIKES = str.maketrans({'Х': 'X', 'х': 'x', 'І': 'I', 'і': 'i', 'С': 'C', 'с': 'c', 'М': 'M', 'м': 'm',
                                   'Ⅰ': 'I', 'Ⅴ': 'V', 'Ⅹ': 'X'})
_NOTES_HEADINGS = {
    'примечания', 'примечание', 'сноски', 'комментарии', 'примечания автора', 'примечания переводчика',
    'примечания редактора', 'notes', 'footnotes', 'endnotes', 'note', 'references', 'notes and references',
    'anmerkungen', 'fußnoten', 'notas', 'note', 'remarques', 'notes de bas de page',
}


def _attr_str(tag: Tag, name: str) -> str:
    v = tag.attrs.get(name)
    if v is None:
        return ''
    if isinstance(v, (list, tuple)):
        return ' '.join(v).lower()
    return str(v).lower()


def _is_note_container(tag: Tag) -> bool:
    etype = _attr_str(tag, 'epub:type') or _attr_str(tag, 'type')
    if any(t in etype.split() for t in _NOTE_TYPES):
        return True
    role = _attr_str(tag, 'role')
    if any(r in role.split() for r in _NOTE_ROLES):
        return True
    classes = _attr_str(tag, 'class').split()
    return any(c in classes for c in _NOTE_CLASSES)


def _is_note_marker(tag: Tag) -> bool:
    """Inline footnote references like <sup><a href="#n1">1</a></sup> or <a epub:type="noteref">[1]</a>."""
    text = tag.get_text()
    if not _MARKER_RE.match(text or ''):
        return False
    if tag.name == 'sup':
        return True
    etype = _attr_str(tag, 'epub:type')
    role = _attr_str(tag, 'role')
    classes = _attr_str(tag, 'class')
    href = _attr_str(tag, 'href')
    return ('noteref' in etype or 'doc-noteref' in role or 'footnote' in classes or 'noteref' in classes
            or 'reference' in classes or (tag.name == 'a' and href.startswith('#') and text.strip() != ''))


def clean_inline_text(text: str) -> str:
    """Normalize unicode and whitespace inside a block; handle <br> line breaks."""
    text = unicodedata.normalize('NFC', text)
    text = _INVISIBLE_RE.sub('', text)
    lines = [_SPACES_RE.sub(' ', line).strip() for line in text.split(LINE_BREAK)]
    lines = [line for line in lines if line]
    out = ''
    for line in lines:
        if out:
            # A line break (poetry, addresses...) is read as a short pause: add a comma if the line has no punctuation.
            if re.search(r'[\w»”"’)\]]$', out):
                out += ','
            out += ' '
        out += line
    return out


def _split_double_breaks(text: str) -> List[str]:
    """Two or more <br> in a row separate paragraphs."""
    parts = re.split(rf'(?:{LINE_BREAK}\s*){{2,}}', text)
    return [p for p in parts if p.strip()]


def html_to_blocks(html) -> List[Block]:
    """Convert an (X)HTML document to an ordered list of Blocks."""
    if isinstance(html, bytes):
        html = html.decode('utf-8', errors='replace')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, features='lxml')
    root = soup.body or soup
    blocks: List[Block] = []
    buf: List[str] = []

    def flush(kind, level):
        raw = ''.join(buf)
        buf.clear()
        for part in _split_double_breaks(raw):
            text = clean_inline_text(part)
            if not text or not re.search(r'\w', text):
                if text and _SCENE_BREAK_RE.match(text) and re.search(r'[*•·⁂~#◆◇■□●○✱✲✳❖※]|[—–\-]{3}', text):
                    _add_break(blocks)
                continue
            if kind == HEADING:
                blocks.append(Block(HEADING, text, level))
            else:
                blocks.append(Block(PARA, text, 0))

    def walk(el, kind, level):
        for child in el.children:
            if isinstance(child, (Comment, Doctype, ProcessingInstruction, Declaration, CData)):
                continue
            if isinstance(child, NavigableString):
                buf.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = (child.name or '').lower().split(':')[-1]
            if name in SKIP_TAGS:
                continue
            if name == 'br':
                buf.append(LINE_BREAK)
                continue
            if name == 'hr':
                flush(kind, level)
                _add_break(blocks)
                continue
            if _is_note_container(child) or _is_note_marker(child):
                continue
            if name in BLOCK_TAGS:
                flush(kind, level)
                if name in HEADING_TAGS:
                    walk(child, HEADING, int(name[1]))
                    flush(HEADING, int(name[1]))
                else:
                    walk(child, kind if kind == HEADING else PARA, level)
                    flush(kind if kind == HEADING else PARA, level)
            else:
                walk(child, kind, level)

    walk(root, PARA, 0)
    flush(PARA, 0)
    return postprocess_blocks(blocks)


def _add_break(blocks: List[Block]):
    if blocks and blocks[-1].kind != BREAK:
        blocks.append(Block(BREAK))


def latinize_roman(text: str) -> str:
    """'ХII' (Cyrillic Х) -> 'XII' when the token only contains roman-numeral-like characters."""
    t = text.strip().rstrip('.')
    cand = t.translate(_ROMAN_LOOKALIKES)
    if t and re.fullmatch(_ROMAN_RE, cand):
        return cand
    return text


def is_chapter_heading_text(text: str) -> bool:
    t = text.strip()
    if not 0 < len(t) <= 100:
        return False
    if _CHAPTER_HEADING_RE.match(t):
        return True
    words = t.split()
    if words and len(words) <= 2:  # "Глава ХII" with Cyrillic lookalikes
        fixed = ' '.join(words[:-1] + [latinize_roman(words[-1])])
        return fixed != t and bool(_CHAPTER_HEADING_RE.match(fixed))
    return False


def is_notes_heading(text: str) -> bool:
    t = re.sub(r'[^\w\s]', '', text).strip().lower()
    return t in _NOTES_HEADINGS


def postprocess_blocks(blocks: List[Block]) -> List[Block]:
    out: List[Block] = []
    skip_level = None  # when inside a "Notes" section, the heading level that started it
    for b in blocks:
        if b.kind == HEADING or (b.kind == PARA and is_chapter_heading_text(b.text)):
            level = b.level if b.kind == HEADING else 2
            if skip_level is not None and level > skip_level:
                continue
            skip_level = None
            if is_notes_heading(b.text):
                skip_level = level
                continue
            out.append(Block(HEADING, b.text, level))
            continue
        if skip_level is not None:
            continue
        if b.kind == PARA:
            # Drop footnote back-links leftovers ("↑ text")
            if b.text.startswith(('↑', '^')):
                continue
            if _SCENE_BREAK_RE.match(b.text) and not re.search(r'\w', b.text):
                _add_break(out)
                continue
            out.append(b)
        elif b.kind == BREAK:
            _add_break(out)
    # Remove leading/trailing breaks
    while out and out[0].kind == BREAK:
        out.pop(0)
    while out and out[-1].kind == BREAK:
        out.pop()
    return out


def blocks_to_text(blocks: List[Block]) -> str:
    """Plain text representation (one block per line), used for display/editing and chapter heuristics."""
    lines = []
    for b in blocks:
        if b.kind == BREAK:
            lines.append('* * *')
        else:
            lines.append(b.text)
    return '\n'.join(lines) + ('\n' if lines else '')


def text_to_blocks(text: str) -> List[Block]:
    """Parse plain text (e.g. edited in the GUI): one paragraph per line, chapter-like lines become headings."""
    blocks = []
    for line in text.splitlines():
        line = clean_inline_text(line)
        if not line:
            continue
        if not re.search(r'\w', line):
            if _SCENE_BREAK_RE.match(line):
                _add_break(blocks)
            continue
        if is_chapter_heading_text(line):
            blocks.append(Block(HEADING, line, 2))
        else:
            blocks.append(Block(PARA, line))
    return postprocess_blocks(blocks)


def split_sections(blocks: List[Block], default_title: Optional[str] = None, min_chars: int = 300) -> List[Section]:
    """Split the blocks of one document into chapters at chapter-level headings (h1/h2 or 'Chapter N'-like)."""
    sections: List[Section] = []
    current = Section(title=None)
    for b in blocks:
        is_split = b.kind == HEADING and (b.level <= 2 or is_chapter_heading_text(b.text))
        if is_split and any(x.kind != HEADING for x in current.blocks):
            sections.append(current)
            current = Section(title=None)
        if b.kind == HEADING and is_split:
            if current.title is None:
                current.title = b.text
            elif all(x.kind == HEADING for x in current.blocks) and current.title.count('. ') < 1:
                current.title = f'{current.title}. {b.text}'  # "Часть первая" + "Глава 1"
        current.blocks.append(b)
    if current.blocks:
        sections.append(current)

    # Merge sections that are too small to be chapters (e.g. "Part One" heading followed by "Chapter 1").
    merged: List[Section] = []
    carry: Optional[Section] = None
    for s in sections:
        if carry is not None:
            titles = [t for t in (carry.title, s.title) if t]
            s = Section(title='. '.join(titles) if titles else None, blocks=carry.blocks + s.blocks)
            carry = None
        body = sum(len(b.text) for b in s.blocks if b.kind != HEADING)
        if body < min_chars:
            carry = s
        else:
            merged.append(s)
    if carry is not None:
        if merged:
            merged[-1].blocks.extend(carry.blocks)
        else:
            merged.append(carry)
    if merged and merged[0].title is None:
        merged[0].title = default_title
    return merged


# ---------------------------------------------------------------------------------------------------------------------
# Sentence splitting and chunking
# ---------------------------------------------------------------------------------------------------------------------

_ABBREVIATIONS = {
    'ru': {'т', 'е', 'д', 'п', 'г', 'гг', 'им', 'ул', 'пр', 'др', 'см', 'стр', 'напр', 'проф', 'акад', 'тов', 'гр',
           'кв', 'обл', 'руб', 'коп', 'млн', 'млрд', 'тыс', 'ок', 'св', 'ст', 'ср', 'мин', 'сек', 'вв', 'н', 'э',
           'изд', 'рис', 'табл', 'гл', 'ч', 'с', 'в', 'к', 'л', 'м', 'р', 'о', 'пер', 'наб', 'пл', 'просп', 'б',
           'доц', 'канд', 'инж', 'ген', 'полк', 'кап', 'лейт', 'кн', 'гос', 'тт', 'вл'},
    'en': {'mr', 'mrs', 'ms', 'dr', 'prof', 'st', 'jr', 'sr', 'vs', 'etc', 'e.g', 'i.e', 'mt', 'fig', 'vol',
           'ch', 'pp', 'capt', 'col', 'gen', 'gov', 'lt', 'sgt', 'rev', 'hon', 'ave', 'inc', 'ltd', 'co', 'cf',
           'approx', 'dept', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov',
           'dec', 'u.s', 'u.k'},
    'fr': {'m', 'mm', 'mme', 'mlle', 'dr', 'st', 'ste', 'etc', 'cf', 'p', 'pp', 'vol', 'chap'},
    'es': {'sr', 'sra', 'srta', 'dr', 'dra', 'etc', 'ud', 'uds', 'p', 'pp', 'vol', 'cap'},
    'it': {'sig', 'sigg', 'dott', 'prof', 'ecc', 'pag', 'p', 'pp', 'vol', 'cap', 'sig.ra'},
    'pt': {'sr', 'sra', 'dr', 'dra', 'etc', 'p', 'pp', 'vol', 'cap'},
    'de': {'hr', 'fr', 'dr', 'prof', 'bzw', 'ca', 'usw', 'z.b', 'd.h', 'u.a', 'vgl', 'nr', 's'},
}
_SINGLE_LETTER_WORDS = {'ru': set('яаиоу'), 'en': {'a', 'i'}}
_CJK_LANGS = ('zh', 'ja', 'ko')

_BOUNDARY_RE = re.compile(
    r'([.!?…]+|\.{3})'                    # terminal punctuation
    r'([»"”’)\]]*)'                        # closing quotes/brackets
    r'(\s+)'                               # whitespace
    r'(?=[«"“„(\[—–\-]?\s*[^\W\d_]|[«"“„(\[]?\d)',  # next sentence starts (optionally with quote or dialogue dash)
)
_CJK_BOUNDARY_RE = re.compile(r'([。！？!?…]+[」』”"’）)]*)')


def _word_before(text: str, end: int) -> str:
    m = re.search(r'([^\W\d_][\w.\-]*)$', text[:end])
    return m.group(1) if m else ''


def split_sentences(text: str, lang: str = 'en') -> List[str]:
    text = text.strip()
    if not text:
        return []
    lang = (lang or 'en')[:2].lower()
    if lang in _CJK_LANGS:
        parts = _CJK_BOUNDARY_RE.split(text)
        sentences, cur = [], ''
        for i, p in enumerate(parts):
            cur += p
            if i % 2 == 1:
                sentences.append(cur.strip())
                cur = ''
        if cur.strip():
            sentences.append(cur.strip())
        return [s for s in sentences if s]

    abbrevs = _ABBREVIATIONS.get(lang, set())
    single_words = _SINGLE_LETTER_WORDS.get(lang, set())
    sentences = []
    start = 0
    for m in _BOUNDARY_RE.finditer(text):
        punct = m.group(1)
        if punct == '.':
            word = _word_before(text, m.start(1))
            low = word.lower().rstrip('.')
            # Initials ("А. С. Пушкин") and abbreviations ("т. е.", "Mr.") do not end sentences
            if len(word) == 1 and word.isupper():
                continue
            if len(low) == 1 and low not in single_words:
                continue
            if low in abbrevs or (('.' in low) and len(low) <= 6):
                continue
        # The next sentence must start with an uppercase letter or a digit ("Сурин? — спросил хозяин" goes on)
        nxt = re.match(r'[\s«"“„(\[—–\-]*(.)', text[m.end(3):])
        if nxt and nxt.group(1).isalpha() and not nxt.group(1).isupper():
            continue
        end = m.end(2)
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = m.end(3)
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


_SPLIT_CLASSES = [
    re.compile(r'[;:]\s+'),
    re.compile(r'\s+[—–]\s+'),
    re.compile(r',\s+'),
    re.compile(r'\s+'),
]


def split_long_text(text: str, max_chars: int) -> List[str]:
    """Split a sentence longer than max_chars at the most natural point (; : — , then whitespace)."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    n = len(text)
    lo, hi = int(n * 0.2), int(n * 0.8)
    for pattern in _SPLIT_CLASSES:
        candidates = [m.end() for m in pattern.finditer(text) if lo <= m.end() <= hi]
        if candidates:
            pos = min(candidates, key=lambda p: abs(p - n / 2))
            left, right = text[:pos].strip(), text[pos:].strip()
            if left and right:
                return split_long_text(left, max_chars) + split_long_text(right, max_chars)
    # No whitespace at all in the central part (very unusual): hard split
    return [text[:max_chars]] + split_long_text(text[max_chars:], max_chars)


@dataclass
class Chunk:
    text: str
    sentence_end: bool  # True when the chunk ends at a sentence boundary (False: split inside a long sentence)


def chunk_paragraph(text: str, lang: str, max_chars: int) -> List[Chunk]:
    """Pack the sentences of a paragraph into chunks of at most max_chars characters (never crossing paragraphs)."""
    pieces: List[Chunk] = []
    for sentence in split_sentences(text, lang):
        parts = split_long_text(sentence, max_chars)
        for i, p in enumerate(parts):
            pieces.append(Chunk(p, sentence_end=(i == len(parts) - 1)))
    chunks: List[Chunk] = []
    for piece in pieces:
        if chunks and chunks[-1].sentence_end and len(chunks[-1].text) + 1 + len(piece.text) <= max_chars:
            chunks[-1] = Chunk(chunks[-1].text + ' ' + piece.text, piece.sentence_end)
        else:
            chunks.append(piece)
    return chunks


def count_letters(text: str) -> int:
    return sum(1 for c in text if c.isalnum())
