# -*- coding: utf-8 -*-
"""
Audio post-processing of synthesized chunks and automatic quality checks.

- Trim the silence the model leaves at the edges of each chunk, so that pauses are fully controlled by us.
- Shorten abnormal silences inside a chunk (dead air).
- Short fades at the edges (no clicks when chunks are joined).
- Quality checks to catch the typical failures of neural TTS on long-form text: skipped words (too fast / too
  short), hallucinated or looping speech (too slow / too long / hit the token limit), silence, clipping,
  and optionally an ASR (Whisper) comparison of what was said with what should have been said.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from audiblez.text import count_letters

FRAME_MS = 10


def _frame_db(wav, sr, frame_ms=FRAME_MS):
    hop = max(1, int(sr * frame_ms / 1000))
    n = len(wav) // hop
    if n == 0:
        return np.zeros(0), hop
    frames = wav[:n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12), hop


def _silence_threshold_db(db):
    """Adaptive threshold: 40 dB under the loud parts of the chunk, never above -40 dBFS nor below -60 dBFS."""
    if len(db) == 0:
        return -50.0
    loud = np.percentile(db, 95)
    return float(np.clip(loud - 40.0, -60.0, -40.0))


def trim_silence(wav, sr, pad_start=0.05, pad_end=0.10):
    """Remove leading/trailing silence, keeping a small margin so soft consonants and breaths are not cut."""
    if len(wav) == 0:
        return wav
    db, hop = _frame_db(wav, sr)
    if len(db) == 0:
        return wav
    voiced = np.where(db > _silence_threshold_db(db))[0]
    if len(voiced) == 0:
        return wav[:0]
    start = max(0, voiced[0] * hop - int(pad_start * sr))
    end = min(len(wav), (voiced[-1] + 1) * hop + int(pad_end * sr))
    return wav[start:end]


def limit_internal_silences(wav, sr, max_silence=1.0, keep=0.6):
    """Shorten any silence longer than max_silence seconds inside a chunk down to `keep` seconds."""
    db, hop = _frame_db(wav, sr)
    if len(db) == 0:
        return wav
    silent = db <= _silence_threshold_db(db)
    max_frames = int(max_silence * 1000 / FRAME_MS)
    keep_frames = int(keep * 1000 / FRAME_MS)
    pieces, last, i, n = [], 0, 0, len(silent)
    while i < n:
        if silent[i]:
            j = i
            while j < n and silent[j]:
                j += 1
            if j - i > max_frames and i > 0 and j < n:
                half = keep_frames // 2
                cut_from, cut_to = (i + half) * hop, (j - (keep_frames - half)) * hop
                pieces.append(wav[last:cut_from])
                last = cut_to
            i = j
        else:
            i += 1
    if not pieces:
        return wav
    pieces.append(wav[last:])
    return crossfade_concat(pieces, sr, fade_ms=8)


def fade_edges(wav, sr, fade_in_ms=6, fade_out_ms=12):
    wav = np.array(wav, dtype=np.float32, copy=True)
    fi, fo = min(len(wav), int(sr * fade_in_ms / 1000)), min(len(wav), int(sr * fade_out_ms / 1000))
    if fi > 0:
        wav[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        wav[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return wav


def crossfade_concat(pieces, sr, fade_ms=8):
    pieces = [p for p in pieces if len(p)]
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    out = np.asarray(pieces[0], dtype=np.float32)
    f = int(sr * fade_ms / 1000)
    for p in pieces[1:]:
        p = np.asarray(p, dtype=np.float32)
        k = min(f, len(out), len(p))
        if k > 0:
            ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
            mixed = out[-k:] * (1 - ramp) + p[:k] * ramp
            out = np.concatenate([out[:-k], mixed, p[k:]])
        else:
            out = np.concatenate([out, p])
    return out


def clean_chunk(wav, sr):
    """Standard processing applied to every synthesized chunk before it is assembled."""
    wav = np.nan_to_num(np.asarray(wav, dtype=np.float32))
    wav = wav - float(np.mean(wav)) if len(wav) else wav  # remove DC offset
    wav = trim_silence(wav, sr)
    wav = limit_internal_silences(wav, sr)
    return fade_edges(wav, sr)


def silence(seconds, sr):
    return np.zeros(int(round(max(0.0, seconds) * sr)), dtype=np.float32)


# ---------------------------------------------------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class QAResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    duration: float = 0.0
    letters_per_sec: float = 0.0
    cer: Optional[float] = None
    transcript: Optional[str] = None

    @property
    def score(self):
        """Lower is better: used to keep the best take when all retries fail."""
        s = 0.0 if self.ok else 1.0
        if self.cer is not None:
            s += self.cer
        if self.letters_per_sec:
            s += abs(np.log(max(self.letters_per_sec, 0.1) / 12.0)) * 0.2
        return s


@dataclass
class QAConfig:
    min_rate: float = 5.0   # letters per second below which the take is suspicious (hallucination / looping)
    max_rate: float = 25.0  # letters per second above which words were probably skipped
    max_cer: float = 0.25   # maximum character error rate against the ASR transcript
    max_retries: int = 3


def analyze_chunk(wav, sr, text, qa: QAConfig, raw_wav=None, hit_token_limit=False) -> QAResult:
    reasons = []
    duration = len(wav) / sr if sr else 0.0
    letters = count_letters(text)
    rate = letters / duration if duration > 0 else 0.0
    if duration < 0.15:
        reasons.append('no speech')
    elif letters >= 12:
        if rate < qa.min_rate:
            reasons.append(f'too slow ({rate:.1f} letters/s): possible hallucination or long pauses')
        if rate > qa.max_rate:
            reasons.append(f'too fast ({rate:.1f} letters/s): possible skipped words')
    else:
        if duration > 2.0 + letters * 0.35:
            reasons.append(f'too long for a short text ({duration:.1f}s)')
    if hit_token_limit:
        reasons.append('hit the generation length limit (runaway generation)')
    src = raw_wav if raw_wav is not None else wav
    if len(src):
        peak = float(np.max(np.abs(src)))
        if peak >= 0.999 and float(np.mean(np.abs(src) >= 0.999)) > 0.001:
            reasons.append('clipping')
    return QAResult(ok=not reasons, reasons=reasons, duration=duration, letters_per_sec=rate)


def _normalize_for_cer(text):
    text = text.lower().replace('ё', 'е')
    text = re.sub(r'[^\w\s]|_', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def char_error_rate(reference, hypothesis):
    ref, hyp = _normalize_for_cer(reference), _normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


class AsrVerifier:
    """Transcribes synthesized chunks with faster-whisper and measures the character error rate."""

    def __init__(self, language, model_size=None, device='auto'):
        from faster_whisper import WhisperModel
        import torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_size is None:
            model_size = 'large-v3-turbo' if device == 'cuda' else 'small'
        compute_type = 'float16' if device == 'cuda' else 'int8'
        print(f'Loading Whisper {model_size} on {device} for automatic quality control...')
        self.model = WhisperModel(model_size, device=device if device in ('cuda', 'cpu') else 'cpu',
                                  compute_type=compute_type)
        self.language = (language or 'en')[:2]

    def transcribe(self, wav, sr):
        if sr != 16000:
            import librosa
            wav = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=16000)
        segments, _ = self.model.transcribe(np.asarray(wav, dtype=np.float32), language=self.language, beam_size=5,
                                            condition_on_previous_text=False, vad_filter=False)
        return ' '.join(s.text.strip() for s in segments).strip()

    def check(self, wav, sr, text, qa: QAConfig, result: QAResult):
        if count_letters(text) < 20:  # Whisper itself is unreliable on very short clips
            return result
        transcript = self.transcribe(wav, sr)
        # Whisper writes "в 1812 году" while our text says "в тысяча восемьсот двенадцатом году": normalize both
        from audiblez.normalize import normalize_text
        try:
            heard = normalize_text(transcript, self.language)
        except Exception:
            heard = transcript
        cer = char_error_rate(text, heard)
        result.cer, result.transcript = cer, transcript
        if cer > qa.max_cer:
            result.ok = False
            result.reasons.append(f'ASR mismatch (CER {cer:.0%}): heard "{transcript[:120]}"')
        return result
