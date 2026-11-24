# -*- coding: utf-8 -*-
"""
Plain text and Markdown input, for original stories written by the channel owner.

Accepted inputs:
- a .txt file: one paragraph per line (web-novel style) or paragraphs separated by blank lines (hard-wrapped text),
- a .md file: Markdown headings (#, ##...) are chapters, blank lines separate paragraphs, `---` is a scene break,
- a folder: every .txt/.md file inside, in natural order (chapter 2 before chapter 10), one file per chapter.

Optional front matter at the top of a file (or in `book.md`/`book.txt` of a folder):
    ---
    title: 星を継ぐ者
    author: 山田太郎
    language: ja
    cover: cover.jpg
    ---

Japanese furigana written the way web-novel platforms and Aozora Bunko do it is read with its reading:
｜漢字《かんじ》, 漢字《かんじ》, 漢字（かんじ）; emphasis 《《言葉》》 and annotations ［＃…］ are cleaned.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import ebooklib

from audiblez.text import (BREAK, HEADING, PARA, Block, _SCENE_BREAK_RE, blocks_to_text, clean_inline_text,
                           detect_language, is_chapter_heading_text, postprocess_blocks)

TEXT_EXTENSIONS = ('.txt', '.text', '.md', '.markdown')
_LEGACY_ENCODINGS = ('cp932', 'euc-jp', 'cp949', 'gb18030', 'big5', 'cp1251', 'koi8-r')


@dataclass
class TextDocument:
    """A chapter source with the small interface the pipeline uses on EPUB documents."""
    name: str
    blocks: List[Block] = field(default_factory=list)
    toc_title: Optional[str] = None

    def __post_init__(self):
        self.extracted_text = blocks_to_text(self.blocks)
        self._text_signature = self.extracted_text
        self.chapter_index = 0

    def get_name(self):
        return self.name

    def get_type(self):
        return ebooklib.ITEM_DOCUMENT


def _plausibility(text):
    """
    How much a decoded text looks like real prose. Byte-level validity is not enough: Shift_JIS bytes also decode
    as (meaningless) GB18030 Chinese, so real Japanese is recognized by its hiragana, Korean by its Hangul, Chinese
    by its most common function characters, Russian by its Cyrillic.
    """
    if not text:
        return 0.0
    n = len(text)
    good = len(re.findall(r'[\sA-Za-z0-9.,!?;:\'"()\-\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af\u0400-\u04ff'
                          r'\u3000-\u303f\uff01-\uff5e«»—…]', text)) / n
    odd = len(re.findall(r'[\ue000-\uf8ff\uff61-\uff9f\u0080-\u00bf]', text)) / n
    hiragana = len(re.findall(r'[\u3041-\u309f]', text)) / n
    hangul = len(re.findall(r'[\uac00-\ud7af]', text)) / n
    cyrillic = len(re.findall(r'[\u0430-\u044f]', text)) / n  # lowercase Cyrillic: real Russian prose
    zh_common = len(re.findall(r'[的了是不我他在有这人们一个说你她到和就也]', text)) / n
    return good - 5 * odd + 1.5 * max(hiragana, hangul, cyrillic, 4 * zh_common)


def read_text_file(path) -> str:
    """Read a text file in UTF-8/UTF-16 or a legacy encoding (Shift_JIS, EUC-KR, GB18030, cp1251...)."""
    raw = Path(path).read_bytes()
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        text = raw.decode('utf-16')
    else:
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            candidates = []
            for enc in _LEGACY_ENCODINGS:
                try:
                    decoded = raw.decode(enc)
                except UnicodeDecodeError:
                    continue
                candidates.append((_plausibility(decoded[:20000]), decoded))
            text = max(candidates)[1] if candidates else raw.decode('latin-1')
    return text.replace('\r\n', '\n').replace('\r', '\n')


def split_front_matter(text):
    """'---\\nkey: value\\n---' at the top -> (dict, rest)."""
    m = re.match(r'^﻿?---[ \t]*\n(.*?)\n---[ \t]*\n', text, flags=re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        kv = re.match(r'^\s*([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$', line)
        if kv:
            meta[kv.group(1).lower()] = kv.group(2).strip().strip('"\'')
    return meta, text[m.end():]


# ---------------------------------------------------------------------------------------------------------------------
# Japanese ruby (furigana) notation
# ---------------------------------------------------------------------------------------------------------------------

_KANJI = r'[一-鿿㐀-䶿豈-﫿々〆ヶヵ〇]'
_KANA = r'[ぁ-ゟ゠-ヿー]'


def convert_ruby(text, mode='reading'):
    """Narou / Kakuyomu / Aozora furigana -> the reading (mode='reading') or the base text (mode='base')."""
    def pick(base, reading):
        if mode != 'reading' or re.fullmatch(r'[・﹅﹆•●◦。、]+', reading):  # ｜傍《・》: emphasis dots, not a reading
            return base
        return reading
    text = re.sub(r'［＃[^］\n]*］', '', text)  # Aozora annotations
    text = re.sub(r'《《([^《》\n]+)》》', r'\1', text)  # emphasis dots (傍点)
    text = re.sub(r'[｜|]([^｜|《》\n]{1,40})《([^《》\n]{1,40})》', lambda m: pick(m.group(1), m.group(2)), text)
    text = re.sub(rf'({_KANJI}+)《([^《》\n]{{1,40}})》', lambda m: pick(m.group(1), m.group(2)), text)
    text = re.sub(rf'({_KANJI}+)[（(]({_KANA}{{1,20}})[）)]', lambda m: pick(m.group(1), m.group(2)), text)
    return text.replace('｜', '')


# ---------------------------------------------------------------------------------------------------------------------
# Markdown and plain text -> blocks
# ---------------------------------------------------------------------------------------------------------------------

# Scripts written without spaces between words (Japanese, Chinese). Korean is written with spaces.
_CJK_CHAR = r'[\u3000-\u303f\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]'


def _join_lines(lines):
    out = ''
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if out and not (re.search(_CJK_CHAR + '$', out) and re.match(_CJK_CHAR, line)):
            out += ' '
        out += line
    return out


def _strip_markdown_inline(t):
    t = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', t)  # images
    t = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', t)  # links
    t = re.sub(r'<!--.*?-->', '', t)
    t = re.sub(r'<[^>]+>', '', t)
    t = re.sub(r'(\*\*|__)(.+?)\1', r'\2', t)
    t = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'\1', t)
    t = re.sub(r'(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])', r'\1', t)
    t = re.sub(r'`([^`]+)`', r'\1', t)
    return t


def _is_break_line(line):
    t = line.strip()
    return bool(t) and bool(_SCENE_BREAK_RE.match(t)) and not re.search(r'\w', t) and len(t) >= 3


def markdown_to_blocks(text, ruby_mode='reading'):
    blocks: List[Block] = []
    para: List[str] = []
    in_code = False

    def flush():
        if para:
            t = clean_inline_text(convert_ruby(_join_lines([_strip_markdown_inline(x) for x in para]), ruby_mode))
            if t and re.search(r'\w', t):
                blocks.append(Block(HEADING, t, 2) if is_chapter_heading_text(t) else Block(PARA, t))
            para.clear()

    for line in text.splitlines():
        if line.strip().startswith('```'):
            flush()
            in_code = not in_code
            continue
        if in_code:
            continue
        h = re.match(r'^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$', line)
        if h:
            flush()
            t = clean_inline_text(convert_ruby(_strip_markdown_inline(h.group(2)), ruby_mode))
            if t:
                blocks.append(Block(HEADING, t, len(h.group(1))))
            continue
        if re.match(r'^\s{0,3}([-*_])(\s*\1){2,}\s*$', line) or _is_break_line(line):
            flush()
            blocks.append(Block(BREAK))
            continue
        if not line.strip():
            flush()
            continue
        line = re.sub(r'^\s{0,3}>\s?', '', line)  # blockquote
        # "- Привет, - сказала она." is a line of dialogue typed with a hyphen, not a list item: keep it as a dash
        line = re.sub(r'^\s*[-–]\s+(?=[«"„“]?[А-ЯЁ])', '— ', line)
        line = re.sub(r'^\s*(?:[-*+]|\d+[.)])\s+', '', line)  # list items
        para.append(line)
    flush()
    return postprocess_blocks(blocks)


def _looks_hard_wrapped(lines):
    """Gutenberg-style text: paragraphs of several lines wrapped at ~70 characters, separated by blank lines."""
    groups, cur = [], []
    for line in lines:
        if line.strip():
            cur.append(line)
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    multi = [g for g in groups if len(g) >= 2]
    if not multi or len(multi) < 0.3 * len(groups):
        return False
    inner = [line for g in multi for line in g[:-1]]
    unterminated = sum(1 for line in inner if not re.search(r'[.!?…:;»"”。！？」』]\s*$', line))
    return unterminated >= 0.6 * len(inner)


def plain_text_to_blocks(text, ruby_mode='reading'):
    lines = text.splitlines()
    blocks: List[Block] = []
    if _looks_hard_wrapped(lines):
        groups, cur = [], []
        for line in lines + ['']:
            if line.strip():
                cur.append(line)
            elif cur:
                groups.append(_join_lines(cur))
                cur = []
        candidates = groups
    else:
        candidates = lines
    for raw in candidates:
        if _is_break_line(raw):
            blocks.append(Block(BREAK))
            continue
        t = clean_inline_text(convert_ruby(raw, ruby_mode))
        if not t or not re.search(r'\w', t):
            continue
        blocks.append(Block(HEADING, t, 2) if is_chapter_heading_text(t) else Block(PARA, t))
    return postprocess_blocks(blocks)


def _strip_aozora_wrapper(text):
    """Remove the Aozora Bunko symbol legend and colophon (底本：...) if present."""
    text = re.sub(r'\n-{20,}\n.*?\n-{20,}\n', '\n', text, count=1, flags=re.S)
    m = re.search(r'\n\s*底本[：:]', text)
    return text[:m.start()] if m else text


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path.name)]


def _first_heading(blocks):
    return next((b.text for b in blocks if b.kind == HEADING), None)


def read_text_book(path):
    """
    Read a .txt/.md file or a folder of them.
    Returns (metadata dict with title/author/language/cover, list of TextDocument).
    """
    path = Path(path)
    if path.is_dir():
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in TEXT_EXTENSIONS and p.is_file()],
                       key=_natural_key)
        meta_file = next((p for p in files if p.stem.lower() in ('book', 'meta', 'index', '_book')), None)
        if not files:
            raise SystemExit(f'No .txt or .md files in {path}')
        base_dir = path
    else:
        files, meta_file, base_dir = [path], path, path.parent
    texts = {f: read_text_file(f) for f in files}
    meta = {}
    if meta_file is not None:
        meta, texts[meta_file] = split_front_matter(texts[meta_file])
    for f in files:
        if f is not meta_file:
            extra, texts[f] = split_front_matter(texts[f])
            for k, v in extra.items():
                meta.setdefault(k, v)
    language = (meta.get('language') or meta.get('lang') or detect_language(' '.join(texts.values()))).lower()[:2]
    ruby_mode = 'reading' if language == 'ja' else 'base'
    docs = []
    for f in files:
        text = _strip_aozora_wrapper(texts[f])
        if f.suffix.lower() in ('.md', '.markdown'):
            blocks = markdown_to_blocks(text, ruby_mode)
        else:
            blocks = plain_text_to_blocks(text, ruby_mode)
        if not blocks:
            continue
        docs.append(TextDocument(name=f.name, blocks=blocks, toc_title=_first_heading(blocks) or f.stem))
    for i, d in enumerate(docs):
        d.chapter_index = i
    title = meta.get('title')
    if not title and len(files) == 1:
        h1 = [b for b in docs[0].blocks if b.kind == HEADING and b.level == 1] if docs else []
        if len(h1) == 1 and not is_chapter_heading_text(h1[0].text):
            title = h1[0].text
            docs[0].blocks.remove(h1[0])  # the title is announced separately
            docs[0].__post_init__()
    title = title or (path.name if path.is_dir() else path.stem)
    cover = meta.get('cover')
    cover_path = None
    if cover and (base_dir / cover).is_file():
        cover_path = base_dir / cover
    else:
        for name in ('cover.jpg', 'cover.jpeg', 'cover.png', 'cover.webp'):
            if (base_dir / name).is_file():
                cover_path = base_dir / name
                break
    return dict(title=title, author=meta.get('author', ''), language=language, cover=cover_path), docs
