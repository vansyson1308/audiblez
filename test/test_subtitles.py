"""Subtitles: the written text of every chunk, split into short cues timed to the narration."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audiblez.normalize import Lexicon, normalize_text
from audiblez.subtitles import (PositionMap, chunk_cues, cue_spans, cues_for_episode, finish_cues, snap_to_pauses,
                                source_spans, srt_time, write_srt)
from audiblez.text import chunk_paragraph

SR = 16000


def lexicon(rules):
    path = Path(tempfile.mkdtemp()) / 'yomi.txt'
    path.write_text(rules, encoding='utf-8')
    return Lexicon.load(path)


class SourceSpansTest(unittest.TestCase):
    def test_japanese_readings_are_shown_as_written(self):
        lex = lexicon('慎二 = しんじ\n冴 = さえ\n')
        orig = ('十二歳だった慎二にとって、この窓口は姉の冴の居場所だった。切符を裏返すと、ちびた鉛筆で連絡船の絵が'
                '描かれている。煙突から上がる細い煙、船腹を打つ波の線まで添えられていた。')
        spoken = normalize_text(orig, 'ja', lexicon=lex)
        self.assertIn('しんじ', spoken)
        chunks = [c.text for c in chunk_paragraph(spoken, 'ja', 50)]
        self.assertGreater(len(chunks), 1)
        shown = source_spans(orig, spoken, chunks)
        self.assertEqual(''.join(shown), orig)  # every character once, in order
        self.assertTrue(shown[0].startswith('十二歳だった慎二にとって'))

    def test_russian_numbers_are_shown_as_digits(self):
        orig = 'В 1812 году он ещё не знал, что будет дальше. Потом всё изменилось.'
        spoken = normalize_text(orig, 'ru')
        self.assertNotIn('1812', spoken)
        chunks = [c.text for c in chunk_paragraph(spoken, 'ru', 80)]
        shown = source_spans(orig, spoken, chunks)
        self.assertEqual(shown[0], 'В 1812 году он ещё не знал, что будет дальше.')
        self.assertEqual(shown[-1], 'Потом всё изменилось.')

    def test_unknown_chunk_falls_back(self):
        self.assertEqual(source_spans('abc.', 'abc.', ['xyz.']), [None])

    def test_position_map(self):
        pm = PositionMap('姉の冴の居場所。', '姉のさえの居場所。')
        self.assertEqual(pm.b_to_a(0), 0)
        self.assertEqual(pm.b_to_a(len('姉のさえの居場所。'), 'end'), len('姉の冴の居場所。'))
        self.assertEqual(pm.a_to_b(3), 4)  # after 冴 -> after さえ


class CueTest(unittest.TestCase):
    def test_sentences_and_long_sentences(self):
        text = '短い文だ。煙突から上がる細い煙、船腹を打つ波の線、ときには桟橋で竿を垂れる小さな少年の後ろ姿まで添えられていた。'
        spans = cue_spans(text, 'ja')
        parts = [text[a:b] for a, b in spans]
        self.assertTrue(parts[0].startswith('短い文だ。煙突から'), parts)  # a short sentence is not shown alone
        self.assertEqual(''.join(parts), text)
        self.assertTrue(all(len(p) <= 34 for p in parts), parts)
        self.assertEqual(len(parts), 2)

    def test_short_sentence_joins_the_next(self):
        text = '「そうか。今日は、この待合所もとうとう壊すんじゃと。わしも最後に見届けておこう思うて出てきたんよ」'
        parts = [text[a:b] for a, b in cue_spans(text, 'ja')]
        self.assertEqual(parts[0], '「そうか。今日は、この待合所もとうとう壊すんじゃと。')
        self.assertEqual(''.join(parts), text)
        # too long together: the short sentence goes with the first clause of the next one
        text = '「ええ。冴ちゃんが亡くなる少し前まで、あそこで働いとったけえね。'
        self.assertEqual([text[a:b] for a, b in cue_spans(text, 'ja')],
                         ['「ええ。冴ちゃんが亡くなる少し前まで、', 'あそこで働いとったけえね。'])
        # no clause short enough: it stays alone
        text = '「ええ。冴ちゃんが亡くなる少し前まであそこでずっと働いとったけえね。'
        self.assertEqual([text[a:b] for a, b in cue_spans(text, 'ja')][0], '「ええ。')

    def test_tiny_tail_is_merged(self):
        text = 'Он долго молчал, глядя в окно на мокрую улицу и редких прохожих под зонтами, а потом сказал: да.'
        parts = [text[a:b] for a, b in cue_spans(text, 'ru', max_chars=60)]
        self.assertTrue(parts[-1].endswith('сказал: да.'), parts)

    def test_snap_to_pauses(self):
        wav = 0.3 * np.sin(np.arange(5 * SR) * 2 * np.pi * 200 / SR).astype(np.float32)
        wav[int(2.0 * SR):int(2.4 * SR)] = 0  # the pause after the first sentence
        snapped = snap_to_pauses(wav, SR, [1.7])
        self.assertTrue(2.0 <= snapped[0] <= 2.4, snapped)
        # no pause nearby: the estimate is kept (within the window)
        self.assertAlmostEqual(snap_to_pauses(wav, SR, [4.2])[0], 4.2, delta=0.61)

    def test_chunk_cues_follow_the_pause(self):
        display = '姉は黙っていた。やがてゆっくりと振り返った。'
        wav = 0.3 * np.sin(np.arange(6 * SR) * 2 * np.pi * 200 / SR).astype(np.float32)
        wav[int(2.6 * SR):int(3.0 * SR)] = 0
        cues = chunk_cues(display, display, wav, SR, 10.0, 16.0, 'ja')
        self.assertEqual([c[2] for c in cues], ['姉は黙っていた。', 'やがてゆっくりと振り返った。'])
        self.assertEqual(cues[0][0], 10.0)
        self.assertTrue(12.6 <= cues[0][1] <= 13.0, cues)
        self.assertEqual(cues[0][1], cues[1][0])
        self.assertEqual(cues[1][1], 16.0)

    def test_srt(self):
        self.assertEqual(srt_time(3725.5), '01:02:05,500')
        cues = finish_cues([(0.0, 1.0, 'a'), (1.1, 2.0, 'b'), (5.0, 6.0, 'c')])
        self.assertEqual(cues[0][1], 1.1)  # lingers up to the next cue
        self.assertEqual(cues[1][1], 2.4)
        path = Path(tempfile.mkdtemp()) / 'x.srt'
        write_srt(path, [(0.0, 1.0, '一'), (1.5, 2.5, '二')])
        self.assertEqual(path.read_text(encoding='utf-8').splitlines()[:3], ['1', '00:00:00,000 --> 00:00:01,400', '一'])

    def test_episode_timeline(self):
        from audiblez.export import Piece
        chapter_cues = [[(1.0, 2.0, 'a'), (50.0, 51.0, 'b')], [(3.0, 4.0, 'c')]]
        pieces = [Piece(0, 'x', 40.0, 60.0), Piece(1, 'y', 0.0, 10.0)]
        self.assertEqual(cues_for_episode(chapter_cues, pieces), [(10.0, 11.0, 'b'), (23.0, 24.0, 'c')])
        self.assertEqual(cues_for_episode(chapter_cues, pieces, speed=2.0)[0], (5.0, 5.5, 'b'))


if __name__ == '__main__':
    unittest.main()
