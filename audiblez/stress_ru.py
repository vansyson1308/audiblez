# -*- coding: utf-8 -*-
"""
Russian word stress for Qwen3-TTS.

The base Qwen3-TTS model reads Russian stress from its own judgement and gets homographs (за́мок / замо́к,
все / всё, сто́ит / стои́т) and rare words wrong; it follows a written stress mark only part of the time. The
fine-tune siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf (Apache-2.0) follows the combining acute U+0301 placed after
the stressed vowel. RUAccent (MIT) places those marks automatically, resolving homographs from the context and
restoring "ё".

Recipe of the model card: lowercase Title-case words before RUAccent (it stresses capitalized words worse), then
convert its "+" before the vowel into U+0301 after it. Marks already present in the text (from the e-book or from
a --lexicon rule such as "Германн = Ге́рманн") are kept.
"""
import hashlib
import json
import re
from pathlib import Path

STRESS_MODEL = 'siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf'
ACUTE = '́'
GRAVE = '̀'
VOWELS = 'аеёиоуыэюяАЕЁИОУЫЭЮЯ'
RUACCENT_OPTIONS = dict(omograph_model_size='turbo3.1', use_dictionary=True)
CACHE_VERSION = '1'

_WORD_RE = re.compile(rf'[А-ЯЁа-яё+{ACUTE}]+')


def marks_to_plus(text):
    """'за́мок' -> 'з+амок' (RUAccent notation); grave accents (secondary stress) are dropped."""
    text = text.replace(GRAVE, '')
    return re.sub(rf'([{VOWELS}]){ACUTE}', r'+\1', text)


def plus_to_acute(text):
    """'з+амок' -> 'за́мок'; any other '+' is removed."""
    text = re.sub(rf'\+([{VOWELS}])', lambda m: m.group(1) + ACUTE, text)
    return text.replace('+', '')


def protect_plus_signs(text):
    """A '+' that is not a stress mark (not followed by a vowel, or after a digit / Latin letter) reads 'плюс'."""
    return re.sub(rf'\+(?![{VOWELS}])|(?<=[0-9A-Za-z])\+', ' плюс ', text)


def lower_title_case(text):
    """'Какая сегодня погода' -> 'какая сегодня погода' (ALL-CAPS words and single letters are kept)."""
    def repl(m):
        w = m.group(0)
        letters = w.replace('+', '').replace(ACUTE, '')
        if len(letters) > 1 and letters[0].isupper() and letters[1:].islower():
            return w.lower()
        return w
    return _WORD_RE.sub(repl, text)


def restore_case(original, processed):
    """
    Copy the letter case of `original` onto `processed` (= RUAccent output for the lowercased original: the same
    letters with '+' marks inserted and some 'е' turned into 'ё'). None if the two cannot be aligned.
    """
    out, i = [], 0
    for ch in processed:
        if ch == '+' and not (i < len(original) and original[i] == '+'):
            out.append(ch)
            continue
        if i >= len(original):
            return None
        o = original[i]
        same = ch.lower() == o.lower() or (o.lower() == 'е' and ch.lower() == 'ё')
        if not same:
            return None
        out.append(ch.upper() if o.isupper() else ch.lower() if o.islower() else ch)
        i += 1
    return ''.join(out) if i == len(original) else None


class Stressor:
    """Adds stress marks to normalized Russian text, with a persistent cache (RUAccent takes ~0.1 s per sentence)."""

    def __init__(self, cache_path=None, accentizer=None):
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding='utf-8'))
            except Exception:
                self.cache = {}
        self._accentizer = accentizer
        self._dirty = False

    @property
    def accentizer(self):
        if self._accentizer is None:
            try:
                from ruaccent import RUAccent
            except ImportError:
                raise SystemExit('--stress ruaccent needs RUAccent: pip install ruaccent')
            print('Loading RUAccent (stress marks and ё)...')
            acc = RUAccent()
            acc.load(**RUACCENT_OPTIONS)
            self._accentizer = acc
        return self._accentizer

    def _key(self, text):
        return hashlib.sha1(f'{CACHE_VERSION}|{RUACCENT_OPTIONS}|{text}'.encode('utf-8')).hexdigest()

    def __call__(self, text):
        key = self._key(text)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        src = protect_plus_signs(marks_to_plus(text))
        src = re.sub(r'\s{2,}', ' ', src)
        lowered = lower_title_case(src)
        processed = self.accentizer.process_all(lowered)
        result = restore_case(src, processed)
        if result is None:  # should not happen; stress the text as written rather than lose the case
            result = self.accentizer.process_all(src)
        out = plus_to_acute(result)
        self.cache[key] = out
        self._dirty = True
        return out

    def save(self):
        if self.cache_path and self._dirty:
            tmp = self.cache_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(self.cache, ensure_ascii=False), encoding='utf-8')
            tmp.replace(self.cache_path)
            self._dirty = False


def strip_marks(text):
    return text.replace(ACUTE, '').replace(GRAVE, '')
