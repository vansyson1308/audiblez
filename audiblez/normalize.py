# -*- coding: utf-8 -*-
"""Language-dispatching text normalization and user pronunciation lexicons."""
import re
import unicodedata
from pathlib import Path

NORMALIZATION_VERSION = '1'  # bump when normalization output changes (invalidates the synthesis cache)


class Lexicon:
    """
    User pronunciation dictionary, applied before any other normalization.

    File format (UTF-8), one rule per line:
        # comment
        Германн = Герман                  whole word / phrase, case-insensitive, е and ё are interchangeable
        ИИ = искусственный интеллект
        re:\\bг\\.\\s?Москва = город Москва   regular expression (Python syntax)
    The separator can be "=", "=>", "→" or a tab.
    """

    def __init__(self, rules=None):
        self.rules = rules or []

    @classmethod
    def load(cls, path):
        rules = []
        for lineno, raw in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            m = re.match(r'^(.*?)\s*(?:=>|→|=|\t)\s*(.*)$', line)
            if not m or not m.group(1):
                raise ValueError(f'{path}:{lineno}: cannot parse lexicon rule "{raw}" (expected "word = replacement")')
            src, dst = m.group(1).strip(), m.group(2).strip()
            if src.startswith('re:'):
                pattern = re.compile(src[3:].strip())
                rules.append((pattern, dst, True))
            else:
                escaped = re.sub(r'[еёЕЁ]', lambda m: '[ЕЁ]' if m.group(0).isupper() else '[её]', re.escape(src))
                pattern = re.compile(rf'(?<!\w){escaped}(?!\w)', re.IGNORECASE)
                rules.append((pattern, dst, False))
        return cls(rules)

    def apply(self, text):
        for pattern, dst, is_regex in self.rules:
            if is_regex:
                text = pattern.sub(dst, text)
            else:
                text = pattern.sub(lambda m, d=dst: _keep_capital(m.group(0), d), text)
        return text

    def __len__(self):
        return len(self.rules)


def _keep_capital(src, dst):
    if src[:1].isupper() and dst[:1].islower():
        return dst[:1].upper() + dst[1:]
    return dst


_CJK = ('zh', 'ja', 'ko')


def normalize_generic(text, lang, is_heading=False):
    """Light cleanup for languages without a dedicated normalizer (the TTS front-end handles numbers)."""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\.{3,}', '…', text)
    text = re.sub(r'([!?])\1+', r'\1', text)
    text = re.sub(r'[\[\]{}]', '', text)
    text = re.sub(r'[*_#|~^=\\<>]+', ' ', text)
    if lang not in _CJK:
        text = re.sub(r'(?<=\S)[ \t]*([—–])[ \t]*(?=\S)', r' \1 ', text)
        text = re.sub(r'[ \t]+([,.;:!?…»)])', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    if text and lang not in _CJK and re.search(r'[\w»"”’)]$', text) and not re.search(r'[.!?…:;][»"”’)]*$', text):
        text += '.'
    elif text and lang in _CJK and not re.search(r'[。！？!?…]$', text):
        text += '。'
    return text


def normalize_text(text, lang, is_heading=False, lexicon=None, enabled=True, keep_stress=False):
    lang = (lang or 'en')[:2].lower()
    if lexicon:
        text = lexicon.apply(text)
    if not enabled:
        return text.strip()
    if lang == 'ru':
        from audiblez.normalize_ru import normalize_ru
        return normalize_ru(text, is_heading=is_heading, keep_stress=keep_stress)
    return normalize_generic(text, lang, is_heading=is_heading)
