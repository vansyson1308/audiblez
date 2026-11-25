"""QA history across runs and workers, and --time-limit (fake engine: no model download)."""
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import audiblez.core as core

sys.path.insert(0, str(Path(__file__).parent))
from test_pipeline import make_epub  # noqa: E402

FACTORY = 'fake_engine:create_engine'


class QaHistoryTest(unittest.TestCase):
    def test_latest_record_wins(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'qa_failed.jsonl'
            lines = [dict(key='a', text='один', t=100.0), dict(key='b', text='два', t=100.0),
                     dict(key='a', ok=True, t=200.0),  # a later take of "a" passed
                     dict(key='c', text='три')]  # records of older versions have no timestamp
            path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in lines), encoding='utf-8')
            self.assertEqual(set(core.load_failed(d)), {'b', 'c'})
            core.append_failed(path, dict(key='b', ok=True))
            self.assertEqual(set(core.load_failed(d)), {'c'})


class RunsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.epub = self.tmp / 'book.epub'
        make_epub(self.epub)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_book(self, **kw):
        settings = core.Settings(formats=('m4b',), asr='off', engine_factory=FACTORY, **kw)
        return core.main(self.epub, output_folder=self.tmp / 'out', settings=settings, max_chapters=2)

    def test_failed_chunk_is_reported_on_cached_runs_until_it_passes(self):
        res = self.run_book(max_retries=0)  # the hallucinated first take of "Сбой..." is kept
        self.assertEqual([r['text'][:4] for r in res['qa_failures']], ['Сбой'])
        res2 = self.run_book(max_retries=0)  # everything from the cache: still reported
        self.assertEqual([r['text'][:4] for r in res2['qa_failures']], ['Сбой'])
        report = (self.tmp / 'out' / 'book.audiblez' / 'qa_report.txt').read_text(encoding='utf-8')
        self.assertIn('Сбой', report)
        res3 = self.run_book(max_retries=1, retry_failed=True)  # regenerated: the second take passes
        self.assertEqual(res3['qa_failures'], [])
        self.assertEqual(self.run_book(max_retries=0)['qa_failures'], [])  # and it stays fixed

    def test_time_limit(self):
        res = self.run_book(time_limit=-1.0)  # already expired: nothing is synthesized, nothing exported
        self.assertTrue(res['incomplete'])
        self.assertEqual(res['files'], [])
        res2 = self.run_book()  # the same command later completes the book
        self.assertFalse(res2.get('incomplete'))

    def test_time_limit_with_parallel_workers(self):
        t0 = time.time()
        res = self.run_book(time_limit=-1.0, gpus='cpu,cpu', batch_size=2)
        self.assertTrue(res['incomplete'])
        self.assertLess(time.time() - t0, 120)


class EmptyTakeTest(unittest.TestCase):
    def test_take_without_speech_never_breaks_the_book(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            story = d / 'story.txt'
            story.write_text('Первый абзац рассказа, совсем обычный.\n\nТишина в эфире.\n\nПоследний абзац.\n',
                             encoding='utf-8')
            settings = core.Settings(formats=('m4b',), asr='off', engine_factory=FACTORY, max_retries=1)
            res = core.main(story, output_folder=d / 'out', settings=settings)
            self.assertTrue(res['files'])  # the book is finished, the chunk is listed in the QA report
            self.assertEqual([r['text'] for r in res['qa_failures']], ['Тишина в эфире.'])
            flacs = list((d / 'out' / 'story.audiblez' / 'cache').glob('*/*.flac'))
            self.assertEqual(len(flacs), 3)
            self.assertTrue(all(f.stat().st_size > 0 for f in flacs))
            # empty files left in the cache by older versions are synthesized again instead of crashing the run
            for f in flacs:
                f.write_bytes(b'')
            res2 = core.main(story, output_folder=d / 'out', settings=settings)
            self.assertTrue(res2['files'])
            self.assertTrue(all(f.stat().st_size > 0 for f in flacs))


class AsrUnitsTest(unittest.TestCase):
    def test_unit_symbols_written_by_whisper(self):
        """Whisper writes "на 15%" for "на пятнадцать": not a clipped ending. A clipped ending is still one."""
        import numpy as np

        from audiblez.audio import AsrVerifier, QAConfig, QAResult
        verifier = object.__new__(AsrVerifier)  # no Whisper model: the transcript is given
        verifier.language = 'ru'
        text = 'Если мы упустим этого покупателя, следующий появится только весной, а цена будет ниже процентов на ' \
               'пятнадцать.'
        cases = [(text, 'Если мы упустим этого покупателя, следующий появится только весной, а цена будет ниже '
                        'процентов на 15%.'),
                 ('Цена на дом за год упала на пятнадцать процентов.', 'Цена на дом за год упала на 15%.')]
        for reference, transcript in cases:
            verifier.transcribe = lambda wav, sr, t=transcript: t
            result = verifier.check(np.zeros(24000, np.float32), 24000, reference, QAConfig(), QAResult(ok=True))
            self.assertTrue(result.ok, result.reasons)
        verifier.transcribe = lambda wav, sr: ('Если мы упустим этого покупателя, следующий появится только весной, '
                                               'а цена будет ниже процентов')
        result = verifier.check(np.zeros(24000, np.float32), 24000, text, QAConfig(), QAResult(ok=True))
        self.assertFalse(result.ok)


class EndingCheckTest(unittest.TestCase):
    """Real transcripts of the first story's render (Whisper small) and what they mean."""
    REDUCED = [  # the ending is there, written as pronounced or misspelled: not a clipped take
        ('Синий венчик огня задрожал, и от него сразу стало чуть менее зя́бко.',
         'Синий венчик огня задрожал, и от него сразу стала чуть менее зябка.'),
        ('Город у нас маленький, на набережной каждый второй знако́мый.',
         'город у нас маленький, на набердной каждой второй знакомой.'),
        ('принесла в дом и опустилась на колени перед чугунной дверцей голла́ндки.',
         'принесла в дом и опустилась на колени перед чугунной дверцей Галантки.'),
        ('и ты войдёшь на крыльцо, стряхивая воду с плаща́».',
         'и ты войдёшь на крыльцо, стряхивая воду с площадь.'),
        ('и сама диктовала, кому и сколько нужно отдать за покос травы.',
         'и сама диктовала, кому и сколько нужно отдать запак островы.'),
    ]
    CLIPPED = [  # the model stopped before the end
        ('Даже в последние месяцы она сама диктовала, кому и сколько нужно отдать за покос травы.',
         'Даже в последние месяцы она сама диктовала, кому и сколько.'),
        ('Принимать какую-то новую партию речных судов, я в железяках не разбираюсь. Командировка на четыре месяца.',
         'Принимать какую-то новую партию речных судов, я в железяках не разбираюсь.'),
        ('Она поставила чайник на плиту и долго смотрела в окно на тёмную реку.',
         'Она поставила чайник на плиту и долго смотрела в окно.'),
    ]

    def test_reduced_endings_are_heard_and_clipped_ones_are_not(self):
        from audiblez.audio import ending_is_missing
        for text, heard in self.REDUCED:
            self.assertFalse(ending_is_missing(text, heard, 'ru'), heard)
        for text, heard in self.CLIPPED:
            self.assertTrue(ending_is_missing(text, heard, 'ru'), heard)
        self.assertFalse(ending_is_missing('Он ушёл домой.', 'Он ушёл домой.', 'ru'))
        self.assertTrue(ending_is_missing('彼は帰ってこなかった。雨が降り始めた。', '彼は帰ってこなかった。', 'ja'))  # unchanged


class WhisperCreditsTest(unittest.TestCase):
    def test_subtitle_credits_are_not_speech(self):
        from audiblez.audio import strip_whisper_hallucinations
        text = 'Ты уехал ровно год назад, а в саду опять пахнет антоновкой.'
        for heard in ('Ты уехал ровно год назад, а в саду опять пахнет антоновкой. субтитров Н.Новикова',
                      'Ты уехал ровно год назад, а в саду опять пахнет антоновкой. Редактор субтитров А.Синецкая '
                      'Корректор А.Егорова',
                      'Ты уехал ровно год назад, а в саду опять пахнет антоновкой. Субтитры сделал DimaTorzok'):
            self.assertEqual(strip_whisper_hallucinations(heard, text).rstrip('. '), text.rstrip('.'))
        said = 'Она включила фильм с субтитрами и долго смотрела в экран.'  # the word is in the text: kept
        self.assertEqual(strip_whisper_hallucinations(said, said), said)
        early = 'Субтитры сделал DimaTorzok. Ты уехал ровно год назад, а в саду опять пахнет антоновкой.'
        self.assertIn('антоновкой', strip_whisper_hallucinations(early, text))  # not at the end: never cut


class TokenFloorTest(unittest.TestCase):
    def test_short_sentences_get_a_token_floor(self):
        """qwen-tts hard-codes min_new_tokens=2: the engine raises it so a short sentence is not cut after a syllable."""
        from types import SimpleNamespace

        import numpy as np

        from audiblez.engines import QwenEngine, find_voice_profile
        seen = []

        class Talker:
            def generate(self, **kwargs):
                seen.append(kwargs['min_new_tokens'])

        class Model:  # the layout of qwen_tts.Qwen3TTSModel
            device = 'cpu'

            def __init__(self):
                self.model = SimpleNamespace(talker=Talker())

            def create_voice_clone_prompt(self, **kwargs):
                return []

            def generate_voice_clone(self, text, **kwargs):
                self.model.talker.generate(min_new_tokens=2)
                return [np.zeros(2400, np.float32) for _ in text], 24000

        engine = QwenEngine(mode='clone', language='ru', voice_profile=find_voice_profile('ru_female_narrator'),
                            model=Model())
        engine.synthesize(['Приятного прослушивания.'])
        self.assertEqual(seen, [11])  # 22 letters at twice 12 letters/s = 0.92 s = 11 frames of 12.5 Hz
        engine.synthesize(['Да.', 'Приятного прослушивания.'])
        self.assertEqual(seen[-1], 2)  # a batch shares the floor of its shortest text
        long_text = 'Вера кивнула и долго смотрела на тёмную воду. ' * 6
        self.assertLess(engine.token_floor(long_text) / 12.5, len(long_text) / 12.0 / 1.5)  # far below a real reading


if __name__ == '__main__':
    unittest.main()
