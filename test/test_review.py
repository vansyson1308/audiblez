"""--review: a stronger Whisper hears every chunk again; doubtful chunks get alternative takes, the best one is kept."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import audiblez.core as core
from audiblez.review import compare
from audiblez.text import count_letters

SR = 24000


class TakeEngine:
    """
    Fake narrator: every take is a tone of its own frequency, so that the fake Whisper below knows which take it
    hears. The first take of a text containing 'Пропуск' skips the first third of the text.
    """
    name = 'fake'
    sample_rate = SR
    native_speed = False
    stochastic = True
    default_max_chars = 300
    language = 'ru'

    def __init__(self, registry):
        self.registry = registry  # take id -> what a listener hears
        self.calls = []

    def fingerprint(self):
        return 'fake-review-v1'

    def describe(self):
        return 'fake engine'

    def synthesize(self, texts, seed=0):
        self.calls.append((list(texts), seed))
        out = []
        for t in texts:
            first = sum(t in c for c, _ in self.calls) == 1 and texts.index(t) == 0
            heard = t[len(t) // 3:] if 'Пропуск' in t and first else t
            take = len(self.registry)
            self.registry[take] = heard
            n = int(count_letters(heard) / 12.0 * SR)
            freq = 300 + 10 * take
            speech = (0.2 * np.sin(2 * np.pi * freq * np.arange(n) / SR)).astype(np.float32)
            out.append(np.concatenate([np.zeros(int(0.3 * SR), np.float32), speech, np.zeros(int(0.4 * SR), np.float32)]))
        return out


class FakeWhisper:
    def __init__(self, registry):
        self.registry = registry
        self.calls = 0

    def transcribe(self, wav, sr):
        self.calls += 1
        spectrum = np.abs(np.fft.rfft(wav))
        freq = np.argmax(spectrum) * sr / len(wav)
        return self.registry[int(round((freq - 300) / 10))]


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.registry = {}
        self.engine = TakeEngine(self.registry)
        self.whisper = FakeWhisper(self.registry)
        self._orig = core.create_engine
        core.create_engine = lambda choice, settings: self.engine
        import audiblez.review as review
        self._orig_tr = review.Review._transcriber
        review.Review._transcriber = lambda _self: self.whisper
        book = self.tmp / 'story.txt'
        book.write_text('---\ntitle: Повесть\nlanguage: ru\n---\n'
                        'Первый абзац рассказа, который читается без ошибок от начала до конца.\n\n'
                        'Пропуск: в этом абзаце первая треть текста теряется при первом прочтении.\n\n'
                        'Последний абзац тоже читается правильно и спокойно.\n', encoding='utf-8')
        self.book = book

    def tearDown(self):
        import audiblez.review as review
        core.create_engine = self._orig
        review.Review._transcriber = self._orig_tr
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_compare(self):
        cer, diffs = compare('Он пришёл в тысяча восемьсот двенадцатом году.', 'Он пришёл в 1812 году', 'ru')
        self.assertEqual((cer, diffs), (0.0, []))
        from audiblez.audio import japanese_reading
        if japanese_reading('佐江') == japanese_reading('さえ'):  # with a Japanese reading dictionary installed
            cer, diffs = compare('姉のさえの居場所だった。', '姉の佐江の居場所だった', 'ja')  # same reading
            self.assertEqual(cer, 0.0)
        from audiblez.review import canonical_kana
        self.assertEqual(canonical_kana('シネー'), canonical_kana('シネエ'))
        self.assertEqual(canonical_kana('コーゾー'), 'コオゾオ')
        self.assertEqual(canonical_kana('ツヅク'), 'ツズク')
        cer, diffs = compare('生意気なこと言うな。お前みたいなのが。', 'お前みたいなのが', 'ja')
        self.assertGreater(cer, 0.3)
        self.assertEqual(diffs[0][0], 'delete')

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg not installed')
    def test_skipped_words_are_replaced_by_a_better_take(self):
        out = self.tmp / 'out'
        settings = core.Settings(formats=('m4b',), asr='off', review=True, retakes=2, batch_size=2)
        res = core.main(self.book, output_folder=out, settings=settings)
        work = out / 'story.audiblez'
        state = json.loads((work / 'review.json').read_text(encoding='utf-8'))
        skipped = [v for v in state.values() if v['text'].startswith('Пропуск')][0]
        self.assertTrue(skipped['retaken'])
        self.assertEqual(skipped['cer'], 0.0)  # an alternative take reads the whole paragraph
        self.assertGreater(skipped['before']['cer'], 0.2)
        others = [v for v in state.values() if not v['text'].startswith('Пропуск')]
        self.assertTrue(all(v['cer'] == 0.0 and not v['retaken'] for v in others))
        # the cache holds the better take: Whisper hears the whole text
        text = skipped['text']
        cache = core.CacheView(work, 'fake', 'fake-review-v1')
        wav, sr = sf.read(str(cache.path(text)), dtype='float32')
        self.assertEqual(self.whisper.transcribe(wav, sr), text)
        report = (work / 'review.txt').read_text(encoding='utf-8')
        self.assertIn('1 chunks replaced by a better take', report)
        self.assertTrue(any(p.name.endswith('.m4b') for p in res['files']))
        # second run: nothing to transcribe or regenerate
        calls, heard = len(self.engine.calls), self.whisper.calls
        core.main(self.book, output_folder=out, settings=settings)
        self.assertEqual((len(self.engine.calls), self.whisper.calls), (calls, heard))
        self.assertFalse((work / 'review_takes').exists())

    def test_failures_overruled_by_the_stronger_whisper_leave_the_report(self):
        from audiblez.core import append_failed, chunk_key, load_failed
        from audiblez.review import Review
        cache = core.CacheView(self.tmp / 'w', 'fake', 'fp')
        cache.cache_dir.mkdir(parents=True)
        texts = ['「腹はふくれんけえ」。', '別の文です。', '三つ目の文です。']
        reasons = [['end of the sentence missing: heard "…ふくれんけい"'], ['speaker similarity 0.05'],
                   ['ASR mismatch (CER 40%): heard "…"']]
        for t, r in zip(texts, reasons):
            append_failed(cache.cache_dir / 'qa_failed.jsonl', dict(key=chunk_key('fp', t), reasons=r))
        review = Review(self.tmp / 'w', cache, 'ja')
        review.state = {chunk_key('fp', t): dict(cer=c) for t, c in zip(texts, (0.0, 0.0, 0.2))}
        review._clear_overruled_failures([('1', t) for t in texts])
        left = load_failed(cache.cache_dir)
        self.assertNotIn(chunk_key('fp', texts[0]), left)  # heard well by the stronger Whisper
        self.assertIn(chunk_key('fp', texts[1]), left)  # not a transcription problem
        self.assertIn(chunk_key('fp', texts[2]), left)  # still doubtful


if __name__ == '__main__':
    unittest.main()
