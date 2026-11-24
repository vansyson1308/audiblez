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
        # the ru-stress fine-tune clones from the short reference of the profile (cut in a pause), the base model
        # from the whole one; the cache key follows
        from audiblez.stress_ru import STRESS_MODEL
        prompts = []

        class RecordingModel(DummyModel):
            def create_voice_clone_prompt(self, **kwargs):
                prompts.append(kwargs)
                return []

        settings = core.Settings(voice_profile='ru_female_narrator', stress='ruaccent')
        choice = core.resolve_engine_choice(settings, 'ru')
        settings = core.with_stress_model(settings, choice)
        self.assertEqual(settings.qwen_model, STRESS_MODEL)
        profile = find_voice_profile('ru_female_narrator')
        engine = QwenEngine(mode='clone', language='ru', voice_profile=profile, model_id=STRESS_MODEL,
                            model=RecordingModel())
        self.assertEqual(core.engine_fingerprint(choice, settings), (engine.name, engine.fingerprint()))
        audio, sr = prompts[-1]['ref_audio']
        self.assertAlmostEqual(len(audio) / sr, 5.58, places=2)
        self.assertEqual(prompts[-1]['ref_text'], engine.ref_text)
        self.assertEqual(engine.ref_text, 'Тишина стояла такая, что было слышно, как тикают часы в соседней комнате.')
        base = QwenEngine(mode='clone', language='ru', voice_profile=profile, model=RecordingModel())
        self.assertEqual(base.ref_text, profile.text)
        self.assertAlmostEqual(len(prompts[-1]['ref_audio'][0]) / sr, 13.04, places=1)
        self.assertNotEqual(base.fingerprint(), engine.fingerprint().replace(STRESS_MODEL, base.model_id))
        settings = core.Settings(speaker='Ono_Anna', instruct='優しく', language='ja')
        choice = core.resolve_engine_choice(settings, 'ja')
        engine = QwenEngine(mode='custom', language='ja', speaker='Ono_Anna', instruct='優しく', model=DummyModel())
        self.assertEqual(core.engine_fingerprint(choice, settings), (engine.name, engine.fingerprint()))

    def test_short_references_are_cut_in_a_pause(self):
        """The short reference of a bundled narrator ends in the silence after its first sentence."""
        from audiblez.engines import find_voice_profile, list_bundled_voices, load_reference_audio
        checked = 0
        for name in list_bundled_voices():
            profile = find_voice_profile(name)
            if profile.short_reference is None:
                continue
            seconds, text = profile.short_reference
            self.assertTrue(profile.text.startswith(text), name)
            wav, sr = load_reference_audio(profile.audio_path)
            peak = np.sqrt(np.mean(wav ** 2))
            around = wav[int((seconds - 0.08) * sr):int((seconds + 0.08) * sr)]
            self.assertLess(20 * np.log10(np.sqrt(np.mean(around ** 2)) / peak + 1e-9), -35, name)
            checked += 1
        self.assertGreaterEqual(checked, 2)

    def test_benchmark(self):
        """--benchmark times the configurations of this device and runs the quality checks on every take."""
        from audiblez.bench import acceptable, run_benchmark
        settings = core.Settings(asr='off', device='cpu', engine_factory='fake_engine:create_engine')
        logs = []
        res = run_benchmark(self.epub, settings, output_folder=self.tmp / 'out', n_chunks=4, log=logs.append)
        self.assertEqual((res['dtype'], res['backend'], res['batch_size']), ('float32', 'qwen-tts', 1))
        self.assertEqual(res['failures'], 0)
        self.assertTrue(any(line.startswith('\nFastest: --backend qwen-tts --batch-size 1 --dtype float32')
                            for line in logs), logs)
        # a precision whose takes fail the checks is never recommended, even when it is faster
        results = [dict(speed=1.0, failures=0), dict(speed=2.0, failures=1), dict(speed=3.0, failures=4)]
        self.assertEqual(max(acceptable(results), key=lambda r: r['speed'])['speed'], 2.0)

    def test_chapter_selection(self):
        res = core.main(self.epub, output_folder=self.tmp / 'out',
                        settings=core.Settings(dry_run=True, chapters='2-3'))
        self.assertEqual([c.title for c in res['chapters']], ['Глава II', 'Глава III'])
        self.assertTrue(res['chapters'][0].segments[0].text.startswith('Глава вторая'))


class ShortReferenceTest(unittest.TestCase):
    """audiblez-voice finds the short reference of a new voice: a pause after a sentence, with the cut in 4-8 s."""
    TEXT = 'Раз два три. Четыре пять шесть, семь восемь. Девять десять одиннадцать.'
    # tone bursts, one per word; the first sentence ends too early (1.70 s), the comma pause 4.20-4.80 s does not end
    # a sentence, the second sentence ends at 6.05 s and is followed by an inhale at INHALE before the third
    WORDS = [(' Раз', 0.10, 0.60), (' два', 0.65, 1.15), (' три.', 1.20, 1.70),
             (' Четыре', 2.30, 2.90), (' пять', 2.95, 3.55), (' шесть,', 3.60, 4.20),
             (' семь', 4.80, 5.40), (' восемь.', 5.45, 6.05),
             (' Девять', 7.60, 8.20), (' десять', 8.25, 8.85), (' одиннадцать.', 8.90, 9.70)]
    INHALE = (6.55, 6.85)
    # Whisper's times are approximate: it hears "восемь." end at 6.25 s and "Девять" start at 7.30 s
    HEARD = [(w, 7.30, e) if w == ' Девять' else (w, s, 6.25) if w == ' восемь.' else (w, s, e) for w, s, e in WORDS]

    def reference(self, inhale_db=-40.0, floor_db=-60.0):
        """The synthetic reference; inhale and noise floor in dB relative to the RMS of the bursts."""
        rng = np.random.default_rng(1)
        wav = np.zeros(int(9.95 * SR))
        for k, (_, start, end) in enumerate(self.WORDS):
            t = np.arange(int(start * SR), int(end * SR)) / SR
            fade = np.clip(np.minimum(t - start, end - t) / 0.01, 0, 1)
            wav[int(start * SR):int(start * SR) + len(t)] = 0.3 * np.sin(2 * np.pi * (150 + 20 * k) * t) * fade
        rms = np.sqrt(np.mean(wav ** 2))
        wav += rng.normal(0, rms * 10 ** (floor_db / 20), len(wav))
        a, b = (int(x * SR) for x in self.INHALE)
        wav[a:b] += rng.normal(0, rms * 10 ** (inhale_db / 20), b - a)
        return wav.astype(np.float32)

    @staticmethod
    def level(wav, seconds):
        """dB of the audio within ±80 ms of a cut, relative to the RMS of the clip (as for the bundled voices)."""
        around = wav[int((seconds - 0.08) * SR):int((seconds + 0.08) * SR)]
        return 20 * np.log10(np.sqrt(np.mean(around ** 2)) / np.sqrt(np.mean(wav ** 2)))

    def test_cut_in_the_pause_before_the_inhale(self):
        from audiblez.voice_tool import find_short_reference
        # an inhale louder than the threshold, and one quieter (it does not interrupt the quiet stretch)
        for inhale_db in (-25.0, -40.0):
            wav = self.reference(inhale_db=inhale_db)
            short, why = find_short_reference(wav, SR, self.TEXT, self.HEARD, 'ru')
            self.assertIsNone(why)
            self.assertEqual(short['text'], 'Раз два три. Четыре пять шесть, семь восемь.')
            # ±80 ms around the cut lie in the silence after "восемь." and before the inhale
            self.assertGreaterEqual(short['seconds'], 6.05 + 0.08, inhale_db)
            self.assertLessEqual(short['seconds'], self.INHALE[0] - 0.08, inhale_db)
            self.assertLess(self.level(wav, short['seconds']), -35)

    def test_profile_declares_the_short_reference(self):
        from audiblez.engines import VoiceProfile, load_reference_audio
        from audiblez.voice_tool import _write_profile
        with tempfile.TemporaryDirectory() as tmp:
            notes = []
            folder = _write_profile(Path(tmp) / 'voice', self.reference(), SR, self.TEXT + '\n', 'ru',
                                    dict(source='test'), words=self.HEARD, log=notes.append)
            profile = VoiceProfile.load(folder)
            seconds, text = profile.short_reference
            self.assertTrue(profile.text.startswith(text))
            wav, sr = load_reference_audio(profile.audio_path)
            self.assertEqual(sr, SR)
            self.assertLess(self.level(wav, seconds), -35)
            self.assertIn(f'the first {seconds:.2f} s', notes[0])
            # without Whisper's word times: no short reference, and nothing to say about it
            notes.clear()
            folder = _write_profile(Path(tmp) / 'no_asr', self.reference(), SR, self.TEXT, 'ru', {}, log=notes.append)
            self.assertIsNone(VoiceProfile.load(folder).short_reference)
            self.assertEqual(notes, [])

    def test_no_short_reference_without_a_clear_pause(self):
        from audiblez.engines import VoiceProfile
        from audiblez.voice_tool import _write_profile, find_short_reference
        # a noisy recording: no pause is 35 dB quieter than the speech
        noisy = self.reference(floor_db=-25.0)
        short, why = find_short_reference(noisy, SR, self.TEXT, self.HEARD, 'ru')
        self.assertIsNone(short)
        self.assertIn('35 dB quieter', why)
        with tempfile.TemporaryDirectory() as tmp:
            notes = []
            folder = _write_profile(Path(tmp) / 'voice', noisy, SR, self.TEXT, 'ru', {}, words=self.HEARD,
                                    log=notes.append)
            self.assertIsNone(VoiceProfile.load(folder).short_reference)
            self.assertNotIn('short_reference', json.loads((folder / 'profile.json').read_text(encoding='utf-8')))
            self.assertEqual(len(notes), 1)
            self.assertIn('no short reference', notes[0])
        # Whisper did not hear the end of the sentence ("восем"): the cut could fall anywhere, so there is none
        misheard = [(' восем', s, e) if w == ' восемь.' else (w, s, e) for w, s, e in self.HEARD]
        short, why = find_short_reference(self.reference(), SR, self.TEXT, misheard, 'ru')
        self.assertIsNone(short)
        self.assertIn('did not hear the end of "восемь."', why)


if __name__ == '__main__':
    unittest.main()
