# -*- coding: utf-8 -*-
"""
Second opinion on a finished narration (--review): every chunk is transcribed again with a stronger Whisper, the
transcript and the text are compared as readings (Japanese: kana from pyopenjtalk, after the same pronunciation
dictionary), and the doubtful chunks get alternative takes; the take that Whisper hears best replaces the cached one.

Why: the check made during synthesis uses a small Whisper on CPU, with a tolerance high enough not to regenerate a
chunk for every name it spells differently. In a 27-minute Japanese story it let through a take that skipped a whole
sentence (「生意気なこと言うな。」: 20% of the chunk) and a dozen misread words, all found by large-v3-turbo.

Takes are reproducible (the seed depends on the text): regenerating a chunk with the same seed gives the same audio.
Alternative takes use other seeds, are checked like any take, and are compared with the current one by the same
stronger Whisper; on a tie the current take stays. A chunk gets its alternative takes once (recorded in
review.json), so running the command again reviews only what changed.
"""
import difflib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile

from audiblez.audio import _normalize_for_cer, levenshtein, strip_whisper_hallucinations, without_long_vowels
from audiblez.normalize import normalize_text

DEFAULT_MODEL = 'large-v3-turbo'
DEFAULT_THRESHOLD = 0.03  # reading error rate from which a chunk gets alternative takes
DEFAULT_RETAKES = 4
SEED_STEP = 7_777_777  # alternative takes use seeds the synthesis never uses


def compare(text, heard, language, lexicon=None):
    """(error rate, [(kind, expected, heard)]) between the text that was read and a transcript, as readings."""
    try:
        heard_text = normalize_text(heard, language, lexicon=lexicon)  # "1812" / 高円寺 written like the text
    except Exception:
        heard_text = heard
    ref, hyp = _normalize_for_cer(text, language), _normalize_for_cer(heard_text, language)
    if language == 'ja':  # vowel length is what Whisper hears least reliably (じゃけえ -> じゃけ, ねえ -> ね)
        ref, hyp = without_long_vowels(ref), without_long_vowels(hyp)
    diffs = [(tag, ref[a0:a1], hyp[b0:b1]) for tag, a0, a1, b0, b1 in
             difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes() if tag != 'equal']
    return levenshtein(ref, hyp) / max(1, len(ref)), diffs


def _stamp(path):
    st = Path(path).stat()
    return [st.st_size, st.st_mtime_ns]


class Review:
    def __init__(self, work_dir, cache, language, lexicon=None, model=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD,
                 retakes=DEFAULT_RETAKES, device='auto', seed=0, log=print, transcriber_factory=None):
        self.work_dir = Path(work_dir)
        self.cache = cache
        self.language = (language or 'en')[:2]
        self.lexicon = lexicon
        self.model = model or DEFAULT_MODEL
        self.threshold = threshold
        self.retakes = max(0, int(retakes))
        self.device = device
        self.seed = seed
        self.log = log
        self.state_path = self.work_dir / 'review.json'
        self.state = json.loads(self.state_path.read_text(encoding='utf-8')) if self.state_path.is_file() else {}
        self._factory = transcriber_factory

    # -- transcription -------------------------------------------------------------------------------------------
    def _transcriber(self):
        if self._factory is not None:
            return self._factory()
        from audiblez.audio import AsrVerifier
        return AsrVerifier(self.language, model_size=self.model, device=self.device, log=self.log)

    def _score(self, asr, path, text):
        wav, sr = soundfile.read(str(path), dtype='float32')
        heard = strip_whisper_hallucinations(asr.transcribe(wav, sr), text)
        cer, diffs = compare(text, heard, self.language, self.lexicon)
        return dict(cer=round(cer, 4), heard=heard, diffs=diffs, seconds=round(len(wav) / sr, 2))

    def _save(self):
        tmp = self.state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding='utf-8')
        tmp.replace(self.state_path)

    def key(self, text):
        from audiblez.core import chunk_key
        return chunk_key(self.cache.fingerprint, text)

    # -- main entry ------------------------------------------------------------------------------------------------
    def run(self, labelled_texts, make_synth=None):
        """
        labelled_texts: [(chapter label, text)] of the book. make_synth(): a fresh (engine, Synthesizer) for the
        alternative takes (None: report only). Returns a summary dict.
        """
        from audiblez.core import free_accelerator_memory
        seen, items = set(), []
        for label, text in labelled_texts:
            if text not in seen:
                seen.add(text)
                items.append((label, text))
        todo = [(label, t) for label, t in items
                if self.state.get(self.key(t), {}).get('stamp') != _stamp(self.cache.path(t))]
        for label, t in items:  # heard before: compared again (the dictionary may have changed), not transcribed
            r = self.state.get(self.key(t))
            if r and r.get('stamp') == _stamp(self.cache.path(t)) and 'heard' in r:
                cer, r['diffs'] = compare(t, r['heard'], self.language, self.lexicon)
                r['cer'] = round(cer, 4)
        t0 = time.time()
        if todo:
            self.log(f'\nReview: transcribing {len(todo)} chunks with Whisper {self.model}...')
            asr = self._transcriber()
            for n, (label, t) in enumerate(todo, start=1):
                k = self.key(t)
                prev = self.state.get(k, {})
                self.state[k] = dict(self._score(asr, self.cache.path(t), t), text=t, chapter=label,
                                     stamp=_stamp(self.cache.path(t)), retaken=prev.get('retaken', False))
                if n % 20 == 0 or n == len(todo):
                    self._save()
                    self.log(f'  {n}/{len(todo)} chunks ({time.time() - t0:.0f}s)')
            del asr
            free_accelerator_memory()
        suspects = [(label, t) for label, t in items
                    if self.state[self.key(t)]['cer'] >= self.threshold and not self.state[self.key(t)].get('retaken')]
        replaced = []
        if suspects and self.retakes and make_synth is not None:
            replaced = self._retake(suspects, make_synth)
        self._clear_overruled_failures(items)
        self._save()
        return self._report(items, replaced)

    def _clear_overruled_failures(self, items):
        """
        A chunk that failed only the transcription checks of the small Whisper (mismatch, missing ending) and that
        the stronger one hears well is not a failure: it leaves the QA report. Other failures (voice, speed...) stay.
        """
        from audiblez.core import append_failed, load_failed
        failed = load_failed(self.cache.cache_dir)
        for _, t in items:
            k = self.key(t)
            entry = failed.get(k)
            if not entry or self.state[k]['cer'] >= self.threshold:
                continue
            reasons = entry.get('reasons') or []
            if reasons and all(r.startswith(('ASR mismatch', 'end of the sentence missing')) for r in reasons):
                append_failed(self.cache.cache_dir / 'qa_failed.jsonl', dict(key=k, ok=True))

    def _retake(self, suspects, make_synth):
        from audiblez.core import free_accelerator_memory
        takes_dir = self.work_dir / 'review_takes'
        takes_dir.mkdir(exist_ok=True)
        self.log(f'Review: {len(suspects)} doubtful chunks (reading error rate >= {self.threshold:.0%}): '
                 f'{self.retakes} alternative takes each...')
        engine, synth = make_synth()
        if not getattr(engine, 'stochastic', False):
            self.log('Review: this engine always gives the same take; no alternatives to compare.')
            return []
        batch = max(1, min(self.retakes, getattr(synth.settings, 'batch_size', 0) or 1))
        takes = {}
        for i, (label, t) in enumerate(suspects, start=1):
            k = self.key(t)
            files, attempt = [], 0
            while len(files) < self.retakes and attempt < 2 * self.retakes:
                n = min(batch, self.retakes - len(files))
                seed = self.seed + int(k[:8], 16) + SEED_STEP * (attempt + 1)
                attempt += n
                for wav in engine.synthesize([t] * n, seed=seed):
                    cleaned, qa = synth.evaluate(wav, t)
                    if qa.ok and len(cleaned):
                        f = takes_dir / f'{k}_{len(files)}.flac'
                        soundfile.write(str(f), cleaned, engine.sample_rate, subtype='PCM_16', format='FLAC')
                        files.append(f)
            takes[k] = files
            self.log(f'  {i}/{len(suspects)}: {len(files)} takes for "{t[:40]}…"')
        del engine, synth
        free_accelerator_memory()
        asr = self._transcriber()
        replaced = []
        for label, t in suspects:
            k = self.key(t)
            current = self.state[k]
            best, best_file = current, None
            for f in takes.get(k, []):
                s = self._score(asr, f, t)
                if s['cer'] < best['cer']:
                    best, best_file = s, f
            if best_file is not None:
                dst = self.cache.path(t)
                tmp = dst.with_suffix('.tmp.flac')
                shutil.copyfile(best_file, tmp)
                tmp.replace(dst)
                replaced.append((t, current['cer'], best['cer']))
                from audiblez.core import append_failed
                append_failed(self.cache.cache_dir / 'qa_failed.jsonl', dict(key=k, ok=True))
                self.state[k] = dict(best, text=t, chapter=label, stamp=_stamp(dst), retaken=True,
                                     before=dict(cer=current['cer'], heard=current['heard']))
            else:
                self.state[k]['retaken'] = True
        del asr
        free_accelerator_memory()
        shutil.rmtree(takes_dir, ignore_errors=True)
        return replaced

    def _report(self, items, replaced):
        rows = [self.state[self.key(t)] for _, t in items]
        cers = [r['cer'] for r in rows]
        lines = [f'Review with Whisper {self.model}: {len(rows)} chunks, mean reading error rate '
                 f'{np.mean(cers) if cers else 0:.2%}, {sum(c == 0 for c in cers)} heard exactly.',
                 'Differences can be the narrator (a misread word: add a --lexicon rule, or delete the chunk from the '
                 'cache to regenerate it) or Whisper (a name written with other kanji, a dialect form).', '']
        if replaced:
            lines.append(f'{len(replaced)} chunks replaced by a better take:')
            lines += [f'  {before:.1%} -> {after:.1%}  {t[:70]}' for t, before, after in replaced]
            lines.append('')
        for r in sorted(rows, key=lambda r: -r['cer']):
            if not r['diffs']:
                continue
            lines.append(f'[{r["chapter"]}] {r["cer"]:.1%} ({r["seconds"]}s) {self.key(r["text"])}.flac')
            lines.append(f'  text : {r["text"]}')
            lines.append(f'  heard: {r["heard"]}')
            lines.append('  ' + '  '.join(f'{a or "∅"}→{b or "∅"}' for _, a, b in r['diffs']))
            lines.append('')
        path = self.work_dir / 'review.txt'
        path.write_text('\n'.join(lines), encoding='utf-8')
        doubtful = sum(c >= self.threshold for c in cers)
        self.log(f'Review: mean reading error rate {np.mean(cers) if cers else 0:.2%}; {len(replaced)} chunks replaced '
                 f'by a better take; {doubtful} still at {self.threshold:.0%} or more. Details: {path}')
        return dict(chunks=len(rows), mean_cer=float(np.mean(cers)) if cers else 0.0, replaced=replaced,
                    doubtful=doubtful, report=path)
