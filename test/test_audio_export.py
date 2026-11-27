import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from audiblez.audio import (QAConfig, active_level_db, analyze_chunk, char_error_rate, clean_chunk, level_chunks,
                            limit_internal_silences, trim_silence)
from audiblez.export import (ChapterAudio, OutputOptions, export_audiobook, format_timestamp, plan_episodes,
                             write_ffmetadata, youtube_chapters_text)

SR = 24000


def tone(seconds, freq=220.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def sil(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


class AudioTest(unittest.TestCase):
    def test_trim_silence_keeps_speech(self):
        wav = np.concatenate([sil(0.8), tone(1.0), sil(1.2)])
        out = trim_silence(wav, SR)
        self.assertAlmostEqual(len(out) / SR, 1.0 + 0.05 + 0.10, delta=0.03)

    def test_internal_silence_limited(self):
        wav = np.concatenate([tone(0.5), sil(3.0), tone(0.5)])
        out = limit_internal_silences(wav, SR, max_silence=1.0, keep=0.6)
        self.assertLess(len(out) / SR, 1.0 + 0.7)
        self.assertGreater(len(out) / SR, 1.0 + 0.5)

    def test_short_pauses_untouched(self):
        wav = np.concatenate([tone(0.5), sil(0.4), tone(0.5)])
        self.assertEqual(len(limit_internal_silences(wav, SR)), len(wav))

    def test_clean_chunk_no_click(self):
        out = clean_chunk(np.concatenate([sil(0.3), tone(1.0, amp=0.5), sil(0.3)]), SR)
        self.assertLess(abs(out[0]), 1e-3)
        self.assertLess(abs(out[-1]), 1e-3)

    def test_qa_flags(self):
        qa = QAConfig()
        text = 'Однажды играли в карты у конногвардейца Нарумова.'  # 42 letters
        self.assertTrue(analyze_chunk(tone(3.5), SR, text, qa).ok)
        self.assertFalse(analyze_chunk(tone(15.0), SR, text, qa).ok)  # hallucination / looping
        self.assertFalse(analyze_chunk(tone(0.8), SR, text, qa).ok)  # skipped words
        self.assertFalse(analyze_chunk(sil(0.1), SR, text, qa).ok)
        self.assertFalse(analyze_chunk(tone(3.5), SR, text, qa, hit_token_limit=True).ok)
        self.assertTrue(analyze_chunk(tone(0.6), SR, '— Как?', qa).ok)
        self.assertFalse(analyze_chunk(tone(6.0), SR, '— Как?', qa).ok)

    def test_level_chunks(self):
        chunks = [tone(1.0, amp=0.1), tone(1.0, amp=0.11), tone(1.0, amp=0.095), tone(1.0, amp=0.2), tone(1.0, amp=0.05)]
        levels = [active_level_db(c, SR) for c in chunks]
        out = [active_level_db(c, SR) for c in level_chunks(chunks, SR)]
        ref = sorted(levels)[2]
        self.assertAlmostEqual(out[0], levels[0], places=3)  # within the deadband: untouched
        self.assertAlmostEqual(out[1], levels[1], places=3)
        self.assertLess(out[3], levels[3])  # +6 dB exclamation reduced...
        self.assertGreater(out[3], ref + 1.5)  # ...but still louder than the narration
        self.assertGreater(out[4], levels[4])  # quiet chunk raised

    def test_room_tone(self):
        from audiblez.audio import RoomTone, estimate_noise_floor
        rng = np.random.default_rng(1)
        noisy = [np.concatenate([tone(1.0), (10 ** (-66 / 20) * rng.standard_normal(int(0.5 * SR))).astype(np.float32),
                                 tone(1.0)]) for _ in range(3)]
        level = estimate_noise_floor(noisy, SR)
        self.assertAlmostEqual(level, -66.0, delta=2.0)
        self.assertIsNone(estimate_noise_floor([tone(2.0)], SR))  # no pause to measure: digital silence is kept
        rt = RoomTone(level, SR, seed=3)
        gap = rt(0.75)
        self.assertEqual(len(gap), int(0.75 * SR))
        rms = 20 * np.log10(np.sqrt(np.mean(gap[SR // 50:-SR // 50] ** 2)))
        self.assertAlmostEqual(rms, level, delta=2.0)
        self.assertEqual(gap[0], 0.0)  # faded in: no click
        self.assertEqual(len(rt(30.0)), 30 * SR)  # longer than the noise buffer
        self.assertFalse(np.array_equal(rt(0.5), rt(0.5)))  # not the same noise in every pause

    def test_cer(self):
        self.assertEqual(char_error_rate('Всё ещё идёт!', 'все еще идет'), 0.0)
        self.assertGreater(char_error_rate('Что ты сделал', 'Кто ты сделал'), 0.0)
        self.assertGreater(char_error_rate('Что ты сделал, Сурин?', 'Что'), 0.5)


class EpisodesTest(unittest.TestCase):
    def chapters(self, durations, splits=True):
        return [ChapterAudio(f'Ch {i + 1}', Path(f'c{i}.wav'), d,
                             list(np.arange(30.0, d, 30.0)) if splits else []) for i, d in enumerate(durations)]

    def test_no_episodes(self):
        eps = plan_episodes(self.chapters([100, 200]), None)
        self.assertEqual(len(eps), 1)
        self.assertEqual(len(eps[0]), 2)

    def test_balanced_episodes(self):
        eps = plan_episodes(self.chapters([1800] * 17), 2.0)  # 8.5 h -> 5 parts of ~1.7 h
        self.assertEqual(len(eps), 5)
        lengths = [sum(p.duration for p in ep) for ep in eps]
        self.assertTrue(all(l <= 2.0 * 3600 * 1.15 for l in lengths), lengths)
        self.assertEqual(sum(len(ep) for ep in eps), 17)
        self.assertLess(max(lengths) - min(lengths), 1801)

    def test_long_chapter_is_split_at_paragraphs(self):
        eps = plan_episodes(self.chapters([5 * 3600]), 2.0)
        self.assertEqual(len(eps), 3)
        pieces = [p for ep in eps for p in ep]
        self.assertAlmostEqual(pieces[0].start, 0.0)
        self.assertAlmostEqual(pieces[-1].end, 5 * 3600)
        for a, b in zip(pieces, pieces[1:]):
            self.assertAlmostEqual(a.end, b.start)
            self.assertAlmostEqual(a.end % 30.0, 0.0)  # cut at a paragraph boundary

    def test_timestamps(self):
        self.assertEqual(format_timestamp(3725.4, True), '1:02:05')
        self.assertEqual(format_timestamp(65, False), '01:05')
        text, warnings = youtube_chapters_text([('A', 0, 20), ('B', 20, 40)], 40)
        self.assertEqual(text, '00:00 A\n00:20 B')
        self.assertTrue(warnings)

    def test_ffmetadata_escaping(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'meta.txt'
            write_ffmetadata(p, 'A=B; #1', 'X\\Y', [('Глава = 1', 0.0, 1.5)])
            content = p.read_text(encoding='utf-8')
            self.assertIn('title=A\\=B\\; \\#1', content)
            self.assertIn('artist=X\\\\Y', content)
            self.assertIn('START=0\nEND=1500\ntitle=Глава \\= 1', content)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg not installed')
class ExportTest(unittest.TestCase):
    def test_video_background(self):
        """A 16:9 cover fills the frame; other covers are shown whole over a darkened, blurred copy."""
        from audiblez.export import make_video_background

        def corner_pixel(png):
            raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(png), '-vf', 'crop=1:1:5:5', '-f',
                                  'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
            return tuple(raw[:3])

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, size in (('wide', '1280x720'), ('square', '600x600')):
                subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'lavfi', '-i', f'color=c=0xc8c8c8:s={size}',
                                '-frames:v', '1', str(d / f'{name}.png')], check=True)
                (d / name).mkdir()
            wide = corner_pixel(make_video_background(d / 'wide.png', 'T', 'A', d / 'wide'))
            square = corner_pixel(make_video_background(d / 'square.png', 'T', 'A', d / 'square'))
            self.assertTrue(all(abs(c - 200) <= 3 for c in wide), wide)  # full frame, unchanged
            self.assertTrue(all(c < 190 for c in square), square)  # darkened blurred border

    def test_export_m4b_mp4_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            chapters = []
            for i in range(4):
                wav = np.concatenate([sil(0.5), tone(4.0, 200 + 40 * i, amp=0.05), sil(1.0)])
                path = d / f'ch{i}.wav'
                sf.write(path, wav, SR, subtype='PCM_16')
                chapters.append(ChapterAudio(f'Глава {i + 1}', path, len(wav) / SR, [2.5]))
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=0x336699:s=320x480',
                            '-frames:v', '1', str(d / 'cover.png')], check=True)
            opts = OutputOptions(formats=('m4b', 'mp4'), episode_hours=12 / 3600, speed=1.0)
            files = export_audiobook(chapters, d / 'Книга.v2', 'Книга', 'Автор', (d / 'cover.png').read_bytes(), opts,
                                     language='ru', log=lambda *a: None)
            names = sorted(f.name for f in files)
            self.assertIn('Книга.v2_part01.m4b', names)
            self.assertIn('Книга.v2_part02.mp4', names)
            self.assertIn('Книга.v2_part01_youtube.txt', names)
            self.assertIn('Книга.v2_parts.txt', names)
            m4b = d / 'Книга.v2_part01.m4b'
            info = json.loads(subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_chapters',
                                              '-show_streams', '-show_format', str(m4b)],
                                             capture_output=True, text=True).stdout)
            self.assertEqual([c['tags']['title'] for c in info['chapters']], ['Глава 1', 'Глава 2'])
            self.assertTrue(any(s['codec_type'] == 'video' for s in info['streams']))  # embedded cover
            self.assertEqual(info['format']['tags']['title'], 'Книга — Часть 1')
            mp4 = d / 'Книга.v2_part02.mp4'
            info = json.loads(subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
                                              '-show_format', str(mp4)], capture_output=True, text=True).stdout)
            video = [s for s in info['streams'] if s['codec_type'] == 'video'][0]
            self.assertEqual((video['width'], video['height']), (1920, 1080))
            audio = [s for s in info['streams'] if s['codec_type'] == 'audio'][0]
            self.assertEqual(int(audio['sample_rate']), 48000)
            self.assertAlmostEqual(float(info['format']['duration']), 2 * 5.5, delta=0.6)
            # loudness of the master is normalized to the target
            out = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', str(m4b), '-af', 'ebur128', '-f',
                                  'null', '-'], capture_output=True, text=True).stderr
            lufs = float(re.findall(r'I:\s+(-?\d+\.\d) LUFS', out)[-1])
            self.assertAlmostEqual(lufs, -16.0, delta=1.5)
            yt = (d / 'Книга.v2_part01_youtube.txt').read_text(encoding='utf-8')
            self.assertIn('00:00 Глава 1', yt)

    def test_video_with_chapter_cards_end_screen_and_ambience(self):
        """Chapter titles over the picture, a dimmed end card after the narration, a quiet looped ambience."""
        def pixel(mp4, t, x, y):
            raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-ss', str(t), '-i', str(mp4), '-frames:v', '1',
                                  '-vf', f'crop=2:2:{x}:{y},format=rgb24', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                 capture_output=True, check=True).stdout
            return tuple(raw[:3])

        def level(wav, a, b):
            data, sr = sf.read(str(wav))
            seg = data[int(a * sr):int(b * sr)]
            return 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            chapters = []
            for i in range(2):
                wav = np.concatenate([sil(0.5), tone(9.0, 200 + 40 * i, amp=0.05), sil(3.0)])
                path = d / f'ch{i}.wav'
                sf.write(path, wav, SR, subtype='PCM_16')
                chapters.append(ChapterAudio(f'第{i + 1}章', path, len(wav) / SR, []))
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=0x6080a0:s=1920x1080',
                            '-frames:v', '1', str(d / 'cover.png')], check=True)
            amb = d / 'sea.wav'  # 3 s of noise, looped under 25 s of narration
            sf.write(amb, (0.1 * np.random.default_rng(0).standard_normal(3 * SR)).astype(np.float32), SR)
            opts = OutputOptions(formats=('mp4',), cover_path=d / 'cover.png', chapter_cards=True, video_tail=6.0,
                                 ambience=amb, ambience_level=-20.0, keep_master=True)
            export_audiobook(chapters, d / 'book', 'Book', '', None, opts, language='ja', log=lambda *a: None)
            mp4 = d / 'book.mp4'
            duration = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
                                             'csv=p=0', str(mp4)], capture_output=True, text=True).stdout)
            self.assertAlmostEqual(duration, 25.0 + 6.0, delta=0.6)
            top, bottom = pixel(mp4, 5, 960, 20), pixel(mp4, 5, 20, 1068)
            self.assertTrue(all(abs(c - v) <= 6 for c, v in zip(top, (0x60, 0x80, 0xa0))), top)
            self.assertTrue(all(c < v - 25 for c, v in zip(bottom, (0x60, 0x80, 0xa0))), bottom)  # card shade
            end = pixel(mp4, duration - 1.0, 960, 20)
            self.assertTrue(all(c < v - 15 for c, v in zip(end, (0x60, 0x80, 0xa0))), end)  # dimmed end card
            master = d / 'book.master.wav'
            pause, speech = level(master, 10.0, 12.0), level(master, 3.0, 8.0)
            self.assertTrue(-60 < pause < speech - 12, (pause, speech))  # ambience audible in the pauses, quiet
            self.assertLess(level(master, 0.0, 0.3), pause)  # faded in


if __name__ == '__main__':
    unittest.main()
