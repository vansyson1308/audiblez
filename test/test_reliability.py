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


if __name__ == '__main__':
    unittest.main()
