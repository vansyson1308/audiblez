"""Parallel GPU workers (with a fake engine: real worker subprocesses, no GPU or model needed)."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import audiblez.core as core
import audiblez.parallel as parallel
from audiblez.text import count_letters

sys.path.insert(0, str(Path(__file__).parent))
from test_pipeline import FakeEngine, make_epub  # noqa: E402


class GpuFakeEngine(FakeEngine):
    device = 'cuda'

    def synthesize(self, texts, seed=0):
        texts = list(texts)
        out = super().synthesize(texts, seed)
        for i, t in enumerate(texts):
            if 'Сбой' in t:  # always hallucinates: never passes the checks
                out[i] = __import__('numpy').concatenate([out[i]] * 6)
        return out


def make_fake_engine(choice, settings):
    return GpuFakeEngine()


class SplitTest(unittest.TestCase):
    def test_balanced_and_complete(self):
        todo = {('слово ' * n).strip(): f'ch{n % 3}' for n in range(1, 60)}
        shards = parallel.split_balanced(todo, 2)
        self.assertEqual(sum(len(s) for s in shards), len(todo))
        self.assertEqual(set(shards[0]) | set(shards[1]), set(todo))
        loads = [sum(count_letters(t) for t in s) for s in shards]
        self.assertLess(abs(loads[0] - loads[1]), 0.05 * sum(loads))
        self.assertEqual(shards, parallel.split_balanced(dict(reversed(list(todo.items()))), 2))  # deterministic

    def test_plan_workers(self):
        orig = parallel.visible_gpus
        parallel.visible_gpus = lambda: ['0', '1']
        try:
            s = core.Settings()
            self.assertEqual(parallel.plan_workers(s, GpuFakeEngine(), 100), 2)
            self.assertEqual(parallel.plan_workers(s, GpuFakeEngine(), 10), 1)  # not worth a second model copy
            self.assertEqual(parallel.plan_workers(core.Settings(gpus='1'), GpuFakeEngine(), 100), 1)
            self.assertEqual(parallel.plan_workers(s, FakeEngine(), 100), 1)  # not on a GPU
        finally:
            parallel.visible_gpus = orig


class QaHistoryTest(unittest.TestCase):
    def test_latest_record_wins_across_files(self):
        import json
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            engine = FakeEngine()
            synth = core.Synthesizer(engine, Path(d), core.Settings(), SimpleNamespace())
            k1, k2 = synth.key('один'), synth.key('два')
            (synth.cache_dir / 'qa_failed.w0.jsonl').write_text(  # a worker failed both chunks yesterday
                json.dumps(dict(key=k1, text='один', t=100.0)) + '\n' + json.dumps(dict(key=k2, text='два', t=100.0))
                + '\n', encoding='utf-8')
            (synth.cache_dir / 'qa_failed.jsonl').write_text(  # today a single-GPU retry fixed the first one
                json.dumps(dict(key=k1, ok=True, t=200.0)) + '\n', encoding='utf-8')
            synth.reload_failed()
            self.assertEqual(set(synth.failed), {k2})


class WorkersTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.epub = self.tmp / 'book.epub'
        make_epub(self.epub)
        self.saved = (core.create_engine, parallel.visible_gpus, parallel.plan_workers, dict(os.environ))
        self.main_engine = GpuFakeEngine()
        core.create_engine = lambda choice, settings: self.main_engine
        parallel.visible_gpus = lambda: ['0', '1']
        parallel.plan_workers = lambda settings, engine, n: 2
        os.environ['AUDIBLEZ_WORKER_ENGINE'] = 'test_parallel:make_fake_engine'
        os.environ['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).parent), str(Path(__file__).parent.parent),
                                                    os.environ.get('PYTHONPATH', '')])

    def tearDown(self):
        core.create_engine, parallel.visible_gpus, parallel.plan_workers, env = self.saved
        os.environ.clear()
        os.environ.update(env)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_workers_fill_the_cache_and_qa_report_persists(self):
        settings = core.Settings(formats=('m4b',), asr='off', max_retries=1)
        res = core.main(self.epub, output_folder=self.tmp / 'out', settings=settings, max_chapters=2)
        self.assertEqual(self.main_engine.calls, [])  # everything was synthesized by the two worker processes
        work = self.tmp / 'out' / 'book.audiblez'
        logs = sorted((work / 'jobs').glob('worker_*.log'))
        self.assertEqual(len(logs), 2)
        self.assertTrue(all('Worker done' in p.read_text(encoding='utf-8') for p in logs))
        failed_files = sorted(p.name for p in next((work / 'cache').iterdir()).glob('qa_failed*.jsonl'))
        self.assertTrue(failed_files and all(n.startswith('qa_failed.w') for n in failed_files), failed_files)
        self.assertEqual(len(res['qa_failures']), 1)
        self.assertTrue(res['qa_failures'][0]['text'].startswith('Сбой'))
        # second run: all cached, nothing synthesized, the failed chunk is still reported
        res2 = core.main(self.epub, output_folder=self.tmp / 'out', settings=settings, max_chapters=2)
        self.assertEqual(self.main_engine.calls, [])
        self.assertEqual([r['text'] for r in res2['qa_failures']], [r['text'] for r in res['qa_failures']])
        self.assertIn('Сбой', (work / 'qa_report.txt').read_text(encoding='utf-8'))

    def test_crashed_worker_is_covered_by_the_main_process(self):
        os.environ['AUDIBLEZ_WORKER_ENGINE'] = 'no_such_module:nothing'  # both workers crash at start
        settings = core.Settings(formats=('m4b',), asr='off', max_retries=1)
        res = core.main(self.epub, output_folder=self.tmp / 'out', settings=settings, max_chapters=1)
        self.assertTrue(self.main_engine.calls)  # the main process did the work
        self.assertFalse(res.get('incomplete'))
        self.assertTrue(res['files'] or not shutil.which('ffmpeg'))

    def test_time_limit_stops_cleanly(self):
        settings = core.Settings(formats=('m4b',), asr='off', time_limit=-1.0)  # already expired
        res = core.main(self.epub, output_folder=self.tmp / 'out', settings=settings, max_chapters=1)
        self.assertTrue(res['incomplete'])
        self.assertEqual(res['files'], [])


if __name__ == '__main__':
    unittest.main()
