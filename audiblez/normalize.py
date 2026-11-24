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
                if re.search(_NO_SPACE_SCRIPT, src):  # Japanese/Chinese have no spaces: match anywhere
                    pattern = re.compile(escaped, re.IGNORECASE)
                else:
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


_CJK = ('zh', 'ja')  # languages written without spaces; Korean uses spaces and Western punctuation
_NO_SPACE_SCRIPT = r'[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]'
_DECORATIONS = r'[♪♫♬☆★♡♥❤✨※◆◇■□●○◎△▲▽▼→←↑↓⇒]'


def _fullwidth_alnum_to_ascii(text):
    """'２０２６年', 'ＡＩ' -> '2026年', 'AI' (full-width punctuation such as ！？「」 is kept)."""
    return re.sub(r'[\uff10-\uff19\uff21-\uff3a\uff41-\uff5a]', lambda m: unicodedata.normalize('NFKC', m.group(0)),
                  text)


def _halfwidth_katakana_to_fullwidth(text):
    return re.sub(r'[\uff61-\uff9f]+', lambda m: unicodedata.normalize('NFKC', m.group(0)), text)


def _ends_sentence(text, terminals):
    return bool(re.search(rf'[{terminals}][」』）)\]"”’»]*$', text))


def normalize_ja(text, is_heading=False):
    from audiblez.normalize_ja import normalize_heading_words_ja, normalize_numbers_ja
    text = unicodedata.normalize('NFC', text)
    text = _halfwidth_katakana_to_fullwidth(_fullwidth_alnum_to_ascii(text))
    if is_heading:
        text = normalize_heading_words_ja(text)
    text = normalize_numbers_ja(text.replace('％', '%'))  # numbers + counters: risky readings -> hiragana
    text = re.sub(r'(?:\.{3,}|・{3,}|…+)', '……', text)
    text = re.sub(r'[―—–]{1,}', '……', text)  # 「それは――」: a hesitation / trailing off
    text = re.sub(r'(?<=[\u3040-\u30ffー])[〜～~]+', 'ー', text)  # 「あ〜」 -> 「あー」 (elongation)
    text = re.sub(r'[〜～~]', '', text)
    text = re.sub(r'([！？!?])\1+', r'\1', text)  # 「！！」 -> 「！」 (「！？」 is kept)
    text = re.sub(r'[【】〔〕［］\[\]{}《》〈〉<>＜＞]', '', text)  # 【鑑定】 -> 鑑定
    text = re.sub(_DECORATIONS, '', text)
    text = re.sub(r'[*＊#＃|｜^=＝_]+', '', text)
    if is_heading:  # 「第一章　旅立ち」 -> 「第一章、旅立ち」: a pause between the number and the title
        text = re.sub(r'[ \t\u3000]+', '、', text.strip())
    text = re.sub(r'(?<=[\u3000-\u30ff\u3400-\u9fff\uff01-\uff5e])[ \t\u3000]+(?=[\u3000-\u30ff\u3400-\u9fff])', '',
                  text)  # no spaces inside Japanese text
    text = re.sub(r'\s{2,}', ' ', text).strip()
    if text and not _ends_sentence(text, '。！？!?…'):
        # also after a closing quote (「…見るって」。): the final 。 is the model's cue to finish the sentence; without
        # it the ending of such a line was clipped in our tests (caught by the Whisper check, but a retry costs time)
        text += '。'
    return text


def normalize_ko(text, is_heading=False):
    from audiblez.normalize_ko import normalize_numbers_ko
    text = unicodedata.normalize('NFC', text)
    text = _fullwidth_alnum_to_ascii(text)
    text = text.replace('％', '%')
    text = normalize_numbers_ko(text, heading=is_heading)  # digits are read unreliably: spell them in Hangul
    text = text.replace('。', '.').replace('、', ',').replace('！', '!').replace('？', '?').replace('％', '%')
    text = re.sub(r'(?:\.{3,}|…+|·{3,})', '…', text)
    text = re.sub(r'([!?])\1+', r'\1', text)
    text = re.sub(r'(?<=[\uac00-\ud7af])[〜～~]+', '', text)
    text = re.sub(r'[【】〔〕［］\[\]{}《》〈〉<>＜＞]', '', text)
    text = re.sub(_DECORATIONS, '', text)
    text = re.sub(r'[*＊#＃|^=_]+', ' ', text)
    text = re.sub(r'(?<=\S)[ \t]*([—–―])[ \t]*(?=\S)', r' \1 ', text)
    text = re.sub(r'[ \t]+([,.;:!?…)])', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    if text and re.search(r'[\w"”’)」』]$', text) and not _ends_sentence(text, '.!?…'):
        text += '.'
    return text


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
    elif text and lang in _CJK and not _ends_sentence(text, '。！？!?…'):
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
    if lang == 'ja':
        return normalize_ja(text, is_heading=is_heading)
    if lang == 'ko':
        return normalize_ko(text, is_heading=is_heading)
    return normalize_generic(text, lang, is_heading=is_heading)
