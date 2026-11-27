# -*- coding: utf-8 -*-
"""
Subtitles (.srt) of the narration with the text as written: YouTube captions show 高円寺, not the こうえんじ the
narrator was given, and 1812, not "тысяча восемьсот двенадцатом".

- Every chunk keeps the span of the original paragraph it was made from (`source_spans`): the normalized paragraph
  is aligned with the original one character by character, and chunk boundaries fall on punctuation, which the
  normalization keeps, so the spans are exact.
- A chunk is shown as short cues (a sentence, or a clause of a long sentence) timed in proportion to the spoken
  length of each cue, then moved to the nearest pause in the audio (`chunk_cues`).
"""
import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

import numpy as np

from audiblez.text import count_letters, split_sentences

CJK_LANGS = ('ja', 'zh', 'ko')
MAX_CUE_CHARS = {'ja': 28, 'zh': 24, 'ko': 34}  # one comfortable line of captions; other languages: 84
MIN_CUE_SECONDS = 0.7
LINGER = 0.4  # a cue stays on screen a little into the pause that follows it
SHORT_SENTENCE = 7  # characters

Cue = Tuple[float, float, str]


class PositionMap:
    """Maps character positions between two versions of a text (here: original <-> normalized)."""

    def __init__(self, a: str, b: str):
        self.ops = SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
        self.len_a, self.len_b = len(a), len(b)

    def _map(self, pos, side, forward):
        for tag, i1, i2, j1, j2 in self.ops:
            s1, s2, t1, t2 = (i1, i2, j1, j2) if forward else (j1, j2, i1, i2)
            inside = (s1 <= pos < s2) if side == 'start' else (s1 < pos <= s2)
            if not inside:
                continue
            if tag == 'equal':
                return t1 + (pos - s1)
            if pos == s1:
                return t1
            if pos == s2:
                return t2
            return t1 + round((pos - s1) * (t2 - t1) / (s2 - s1))
        return (self.len_b if forward else self.len_a) if pos > 0 else 0

    def a_to_b(self, pos, side='start'):
        return self._map(pos, side, True)

    def b_to_a(self, pos, side='start'):
        return self._map(pos, side, False)


def _locate(text: str, parts: List[str]) -> Optional[List[Tuple[int, int]]]:
    """(start, end) of each part in `text`, in order, ignoring whitespace (chunks are stripped and re-joined)."""
    idx = [i for i, ch in enumerate(text) if not ch.isspace()]
    flat = ''.join(text[i] for i in idx)
    spans, cur = [], 0
    for p in parts:
        pf = ''.join(ch for ch in p if not ch.isspace())
        if not pf:
            spans.append((idx[cur - 1] + 1 if cur else 0,) * 2)
            continue
        k = flat.find(pf, cur)
        if k < 0:
            return None
        spans.append((idx[k], idx[k + len(pf) - 1] + 1))
        cur = k + len(pf)
    return spans


def source_spans(original: str, normalized: str, chunks: List[str]) -> List[Optional[str]]:
    """The original text of each chunk of `normalized` (None where it cannot be found: the chunk is shown as read)."""
    spans = _locate(normalized, chunks)
    if spans is None:
        return [None] * len(chunks)
    pm = PositionMap(original, normalized)
    out = []
    for n0, n1 in spans:
        o0, o1 = pm.b_to_a(n0, 'start'), pm.b_to_a(n1, 'end')
        text = original[o0:o1].strip() if o1 > o0 else ''
        out.append(text or None)
    return out


_CLAUSE_RE = re.compile(r'[、，,;；:：—–]+\s*')


def cue_spans(text: str, lang: str, max_chars: Optional[int] = None) -> List[Tuple[int, int]]:
    """Positions of the cues of a chunk: its sentences, long sentences cut after commas into lines of max_chars."""
    lang = (lang or 'en')[:2].lower()
    max_chars = max_chars or MAX_CUE_CHARS.get(lang, 84)
    sentences = split_sentences(text, lang)
    spans = []
    for s0, s1 in _locate(text, sentences) or [(0, len(text))]:
        # a very short sentence (「そうか。, "No.") is shown with the next one, or with its first clause when the
        # whole would not fit on a line: alone, it would flash for a fraction of a second
        if spans and len(text[spans[-1][0]:spans[-1][1]].strip()) < SHORT_SENTENCE and not text[spans[-1][1]:s0].strip():
            first = _CLAUSE_RE.search(text, s0, s1)
            first_end = first.end() if first and first.end() < s1 else s1
            if first_end - spans[-1][0] <= max_chars:
                spans[-1] = (spans[-1][0], s1)
                continue
        spans.append((s0, s1))
    out = []
    for s0, s1 in spans:
        if s1 - s0 <= max_chars:
            out.append((s0, s1))
            continue
        cuts = [s0 + m.end() for m in _CLAUSE_RE.finditer(text[s0:s1]) if s0 + m.end() < s1]
        start, last_cut = s0, None
        for c in cuts + [s1]:
            if c - start > max_chars and last_cut is not None and last_cut > start:
                out.append((start, last_cut))
                start = last_cut
            last_cut = c
        out.append((start, s1))
    # a tiny tail (a few characters after the last comma) reads better with the line before it
    merged = []
    for s0, s1 in out:
        if merged and s1 - s0 < 6 and merged[-1][1] == s0 and s1 - merged[-1][0] <= max_chars * 1.25:
            merged[-1] = (merged[-1][0], s1)
        else:
            merged.append((s0, s1))
    return merged


def _spoken_weight(text):
    """Spoken length of a piece of normalized text: letters, plus the pauses of its punctuation."""
    return (count_letters(text) + 2.0 * len(re.findall(r'[。．.!?！？…]', text))
            + 1.0 * len(re.findall(r'[、，,;；:：—]', text)) + 0.5)


def snap_to_pauses(wav, sr, estimates, window=0.6, min_gap=MIN_CUE_SECONDS):
    """Each estimated boundary (seconds into `wav`) moved to the quietest moment within +-window (an audible pause)."""
    wav = np.asarray(wav, dtype=np.float32)
    hop = max(1, sr // 100)  # 10 ms frames
    n = len(wav) // hop
    if n < 3 or not estimates:
        return list(estimates)
    energy = (wav[:n * hop].reshape(n, hop) ** 2).mean(axis=1)
    k = 15  # 150 ms: a pause, not the gap between two syllables
    smooth = np.convolve(energy, np.ones(k) / k, mode='same')
    out, prev = [], 0.0
    duration = len(wav) / sr
    for i, t in enumerate(estimates):
        nxt_limit = estimates[i + 1] if i + 1 < len(estimates) else duration
        lo = max(t - window, prev + min_gap)
        hi = min(t + window, nxt_limit + window - min_gap, duration - min_gap)
        if hi <= lo:
            snapped = min(max(t, prev + min_gap), duration - min_gap) if duration - min_gap > prev + min_gap else t
        else:
            f0, f1 = int(lo * 100), max(int(lo * 100) + 1, int(hi * 100))
            seg = smooth[f0:min(f1, n)]
            if len(seg) == 0:
                snapped = t
            else:
                # quietest frame; among equally quiet frames, the one closest to the estimate
                floor = seg.min()
                quiet = np.flatnonzero(seg <= floor * 1.05 + 1e-12) + f0
                snapped = float(quiet[np.argmin(np.abs(quiet / 100 - t))]) / 100
        out.append(snapped)
        prev = snapped
    return out


def chunk_cues(display: str, spoken: str, wav, sr, start: float, end: float, lang: str) -> List[Cue]:
    """
    Cues of one chunk whose audio spans [start, end] seconds of the chapter. `display` is the written text, `spoken`
    the text that was read (to weigh the duration of each cue).
    """
    display = (display or spoken or '').strip()
    if not display:
        return []
    spans = cue_spans(display, lang)
    if len(spans) == 1:
        return [(start, end, display[spans[0][0]:spans[0][1]].strip())]
    pm = PositionMap(display, spoken or display)
    weights = []
    for s0, s1 in spans:
        n0, n1 = pm.a_to_b(s0, 'start'), pm.a_to_b(s1, 'end')
        weights.append(_spoken_weight((spoken or display)[n0:n1]) if n1 > n0 else 1.0)
    total = sum(weights)
    duration = end - start
    estimates, acc = [], 0.0
    for w in weights[:-1]:
        acc += w
        estimates.append(duration * acc / total)
    if wav is not None and len(wav):
        estimates = snap_to_pauses(wav, sr, estimates)
    bounds = [0.0] + [min(max(b, 0.0), duration) for b in estimates] + [duration]
    return [(start + bounds[i], start + bounds[i + 1], display[s0:s1].strip())
            for i, (s0, s1) in enumerate(spans) if display[s0:s1].strip()]


def finish_cues(cues: List[Cue], linger=LINGER) -> List[Cue]:
    """Cues in time order, each kept up to `linger` seconds into the following pause (never over the next cue)."""
    cues = sorted((c for c in cues if c[2]), key=lambda c: c[0])
    out = []
    for i, (a, b, text) in enumerate(cues):
        nxt = cues[i + 1][0] if i + 1 < len(cues) else b + linger
        out.append((a, max(b, min(b + linger, nxt)), text))
    return out


def srt_time(seconds):
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def write_srt(path, cues: List[Cue]):
    lines = []
    for i, (a, b, text) in enumerate(finish_cues(cues), start=1):
        lines += [str(i), f'{srt_time(a)} --> {srt_time(b)}', text, '']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path


def cues_for_episode(chapter_cues, pieces, speed=1.0) -> List[Cue]:
    """Cues of one episode (a list of export.Piece) in its own timeline, from per-chapter cues."""
    out, elapsed = [], 0.0
    for p in pieces:
        for a, b, text in chapter_cues[p.chapter_index] or []:
            if p.start <= a < p.end:
                out.append(((elapsed + a - p.start) / speed, (elapsed + min(b, p.end) - p.start) / speed, text))
        elapsed += p.end - p.start
    return out
