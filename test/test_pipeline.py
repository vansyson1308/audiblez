"""End-to-end test of the audiobook pipeline with a fake TTS engine (no model download needed)."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ebooklib import epub

import audiblez.core as core
from audiblez.text import count_letters

SR = 24000


class FakeEngine:
    """Stochastic fake TTS: 12 letters/second of 'speech'. Texts containing 'Сбой' hallucinate on the first try."""
    name = 'fake'
    sample_rate = SR
    native_speed = False
    stochastic = True
    default_max_chars = 300
    language = 'ru'

    def __init__(self):
        self.calls = []

    def fingerprint(self):
        return 'fake-engine-v1'

    def describe(self):
        return 'fake engine'

    def synthesize(self, texts, seed=0):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            seconds = count_letters(t) / 12.0
            if 'Сбой' in t and sum('Сбой' in x for call in self.calls for x in call) == 1:
                seconds *= 6  # hallucination on first attempt
            n = int(seconds * SR)
            rng = np.random.default_rng(abs(hash(t)) % 2 ** 32)
            speech = (0.2 * np.sin(2 * np.pi * 180 * np.arange(n) / SR) * (0.6 + 0.4 * rng.random(n))).astype(np.float32)
            out.append(np.concatenate([np.zeros(int(0.3 * SR), np.float32), speech, np.zeros(int(0.4 * SR), np.float32)]))
        return out


def make_epub(path):
    book = epub.EpubBook()
    book.set_identifier('test-ru-1')
    book.set_title('Тестовая повесть')
    book.set_language('ru')
    book.add_author('Иван Иванов')
    chapters = []
    body = ''
    for n, roman in enumerate(['I', 'II', 'III'], start=1):
        paras = ''.join(f'<p>Это абзац номер {k} главы {n}. В 1812 году он ещё не знал, что будет дальше.</p>'
                        for k in range(1, 6))
        body += f'<div style="text-align:center">{roman}</div>{paras}'
        if n == 2:
            body += '<p>Сбой проверки качества здесь.</p><p>* * *</p><p>После разрыва сцены.</p>'
    c = epub.EpubHtml(title='Повесть', file_name='text.xhtml', lang='ru')
    c.content = f'<html><body><p>Эпиграф к повести.</p>{body}<h2>Примечания</h2><ol><li>Сноска.</li></ol></body></html>'
    chapters.append(c)
    about = epub.EpubHtml(title='Об издании', file_name='about.xhtml', lang='ru')
    about.content = '<html><body><h1>Об этом электронном издании</h1><p>Лицензия и прочее.</p></body></html>'
    for item in [c, about]:
        book.add_item(item)
    book.toc = (epub.Link('text.xhtml', 'Повесть', 'text'), epub.Link('about.xhtml', 'Об издании', 'about'))
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ['nav', c, about]
    epub.write_epub(str(path), book)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.epub = self.tmp / 'Тестовая.повесть.epub'
        make_epub(self.epub)
        self.engine = FakeEngine()
        self._orig = core.create_engine
        core.create_engine = lambda choice, settings: self.engine

    def tearDown(self):
        core.create_engine = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_plans_chapters(self):
        res = core.main(self.epub, output_folder=self.tmp / 'out', settings=core.Settings(dry_run=True))
        chapters = res['chapters']
        self.assertEqual([c.title for c in chapters], ['Глава I', 'Глава II', 'Глава III'])
        texts = [s.text for s in chapters[0].segments]
        self.assertEqual(texts[:4], ['Тестовая повесть.', 'Иван Иванов.', 'Эпиграф к повести.', 'Глава первая.'])
        self.assertIn('В тысяча восемьсот двенадцатом году он ещё не знал, что будет дальше.', ' '.join(texts))
        all_text = ' '.join(s.text for c in chapters for s in c.segments)
        self.assertNotIn('Сноска', all_text)
        self.assertNotIn('Лицензия', all_text)
        # scene break pause
        seg = [s for s in chapters[1].segments if s.text.startswith('Сбой')][0]
        self.assertGreaterEqual(seg.pause_after, core.PAUSES['scene_break'])
        self.assertEqual(self.engine.calls, [])

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg not installed')
    def test_full_run_with_retry_cache_and_episodes(self):
        out = self.tmp / 'out'
        settings = core.Settings(formats=('m4b', 'mp4'), episode_hours=40 / 3600, asr='off', batch_size=3)
        res = core.main(self.epub, output_folder=out, settings=settings)
        names = sorted(p.name for p in res['files'])
        self.assertTrue(any(n.endswith('_part01.m4b') for n in names), names)
        self.assertTrue(any(n.endswith('_part02.mp4') for n in names), names)
        self.assertTrue(all(n.startswith('Тестовая.повесть') for n in names), names)
        # the hallucinated chunk was regenerated
        attempts = [t for call in self.engine.calls for t in call if t.startswith('Сбой')]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(res['qa_failures'], [])
        calls_before = len(self.engine.calls)
        # second run: everything comes from the cache
        res2 = core.main(self.epub, output_folder=out, settings=settings)
        self.assertEqual(len(self.engine.calls), calls_before)
        m4b = [p for p in res2['files'] if p.name.endswith('_part01.m4b')][0]
        info = json.loads(subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_chapters',
                                          str(m4b)], capture_output=True, text=True).stdout)
        self.assertGreaterEqual(len(info['chapters']), 1)
        self.assertEqual(info['chapters'][0]['tags']['title'], 'Глава I')
        script = (out / 'Тестовая.повесть.audiblez' / 'script.txt').read_text(encoding='utf-8')
        self.assertIn('Глава вторая.', script)

    def test_single_file_book_named_index_is_kept(self):
        from types import SimpleNamespace
        import ebooklib
        docs = []
        for name, size in [('cover.xhtml', 50), ('index.html', 200000), ('about.xhtml', 900), ('toc.xhtml', 3000)]:
            d = SimpleNamespace(get_name=lambda n=name: n, get_type=lambda: ebooklib.ITEM_DOCUMENT,
                                extracted_text='x' * size)
            docs.append(d)
        self.assertEqual([d.get_name() for d in core.find_good_chapters(docs)], ['index.html'])

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg not installed')
    def test_parallel_workers(self):
        """Two worker processes (on CPU here, one per GPU on Kaggle) share the work and the cache."""
        from audiblez.parallel import resolve_devices
        self.assertEqual(resolve_devices('0,1', 'qwen'), ['cuda:0', 'cuda:1'])
        self.assertIsNone(resolve_devices('off', 'qwen'))
        self.assertIsNone(resolve_devices('0,1', 'kokoro'))
        out = self.tmp / 'out_parallel'
        settings = core.Settings(formats=('m4b',), asr='off', gpus='cpu,cpu', batch_size=2,
                                 engine_factory='fake_engine:create_engine')
        res = core.main(self.epub, output_folder=out, settings=settings)
        self.assertTrue(any(p.name.endswith('.m4b') for p in res['files']))
        self.assertEqual(res['qa_failures'], [])  # the hallucinated chunk was retried inside a worker
        # every chunk is in the shared cache: a second run synthesizes nothing (no worker started)
        res2 = core.main(self.epub, output_folder=out, settings=settings)
        self.assertEqual(len(res2['chapters']), 3)

    def test_engine_fingerprint_matches_engine(self):
        """The cache key computed without loading a model equals the one of the real engine."""
        from audiblez.engines import QwenEngine, find_voice_profile

        class DummyModel:
            device = 'cpu'

            def create_voice_clone_prompt(self, **kwargs):
                return []

        settings = core.Settings(voice_profile='ru_male_narrator', temperature=0.7)
        choice = core.resolve_engine_choice(settings, 'ru')
        engine = QwenEngine(mode='clone', language='ru', voice_profile=find_voice_profile('ru_male_narrator'),
                            model=DummyModel(), temperature=0.7)
        self.assertEqual(core.engine_fingerprint(choice, settings), (engine.name, engine.fingerprint()))
        settings = core.Settings(speaker='Ono_Anna', instruct='優しく', language='ja')
        choice = core.resolve_engine_choice(settings, 'ja')
        engine = QwenEngine(mode='custom', language='ja', speaker='Ono_Anna', instruct='優しく', model=DummyModel())
        self.assertEqual(core.engine_fingerprint(choice, settings), (engine.name, engine.fingerprint()))

    def test_chapter_selection(self):
        res = core.main(self.epub, output_folder=self.tmp / 'out',
                        settings=core.Settings(dry_run=True, chapters='2-3'))
        self.assertEqual([c.title for c in res['chapters']], ['Глава II', 'Глава III'])
        self.assertTrue(res['chapters'][0].segments[0].text.startswith('Глава вторая'))


if __name__ == '__main__':
    unittest.main()
