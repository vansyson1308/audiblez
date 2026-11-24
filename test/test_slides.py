"""Several pictures in the YouTube video (--slides)."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import audiblez.core as core
from audiblez.export import ChapterAudio, OutputOptions, Piece, export_audiobook
from audiblez.slides import MIN_GAP, load_slides, place_slides, slides_for_episode

SR = 24000
STORY = '''Дождь в Костроме шёл третий день. Вера стояла на крыльце материнского дома.

Здесь всё дышало Галиной Ивановной. В углу стояло старое чёрное пианино.

Льняная ткань распалась. Внутри лежала плотная, ровная пачка пожелтевших почтовых конвертов.

Утром дождь наконец кончился. Небо над Волгой очистилось.
'''


def color_png(path, color):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                    f'color=c={color}:s=1344x756', '-frames:v', '1', str(path)], check=True)


class SlidesPlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        for name in ('3.jpg', '1.jpg', '2.jpg', '10.jpg'):
            (self.tmp / name).write_bytes(b'fake')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load(self):
        self.assertEqual([s.image.name for s in load_slides(self.tmp)], ['1.jpg', '2.jpg', '3.jpg', '10.jpg'])
        spec = self.tmp / 'slides.txt'
        spec.write_text('# image | cue\n3.jpg\n\n1.jpg | Здесь всё дышало\n2.jpg|  Внутри лежала  \n', encoding='utf-8')
        slides = load_slides(spec)
        self.assertEqual([(s.image.name, s.cue) for s in slides],
                         [('3.jpg', None), ('1.jpg', 'Здесь всё дышало'), ('2.jpg', 'Внутри лежала')])
        spec.write_text('missing.jpg | x\n', encoding='utf-8')
        with self.assertRaises(SystemExit):
            load_slides(spec)

    def test_cues_are_checked_when_planning(self):
        story = self.tmp / 'story.txt'
        story.write_text(STORY, encoding='utf-8')
        spec = self.tmp / 'slides.txt'
        spec.write_text('3.jpg\n1.jpg | здесь ВСЕ дышало\n2.jpg | Внутри лежала плотная\n10.jpg\n', encoding='utf-8')
        res = core.main(story, output_folder=self.tmp / 'out', settings=core.Settings(dry_run=True, slides=str(spec)))
        segs = res['chapters'][0].segments
        self.assertTrue(segs[0].text.startswith('Дождь'))
        spec.write_text('3.jpg\n1.jpg | этого текста нет в рассказе\n', encoding='utf-8')
        with self.assertRaises(SystemExit):  # a typo must not silently drop a picture
            core.main(story, output_folder=self.tmp / 'out', settings=core.Settings(dry_run=True, slides=str(spec)))
        spec.write_text('3.jpg\n2.jpg | Внутри лежала\n1.jpg | Здесь всё дышало\n', encoding='utf-8')
        with self.assertRaises(SystemExit):  # cues must be in reading order
            core.main(story, output_folder=self.tmp / 'out', settings=core.Settings(dry_run=True, slides=str(spec)))

    def test_place(self):
        from audiblez.slides import Slide
        a, b, c, d = (self.tmp / n for n in ('1.jpg', '2.jpg', '3.jpg', '10.jpg'))
        times = [[(0.5, 100.0), (100.6, 200.0)], [(202.0, 300.0), (300.4, 400.0)]]  # 2 chapters x 2 segments
        slides = [Slide(a), Slide(b, 'x', 0, 1), Slide(c), Slide(d, 'y', 1, 1)]
        out = place_slides(slides, times, 401.0, log=lambda m: None)
        self.assertEqual([p.name for p, _ in out], ['1.jpg', '2.jpg', '3.jpg', '10.jpg'])
        self.assertAlmostEqual(out[1][1], 100.3)  # in the middle of the pause before the cue
        self.assertAlmostEqual(out[3][1], 300.2)
        self.assertAlmostEqual(out[2][1], 200.25)  # no cue: halfway between its neighbours
        close = place_slides([Slide(a), Slide(b, 'x', 0, 0)], times, 401.0, log=lambda m: None)
        self.assertEqual(len(close), 1)  # 0.25 s after the first slide: not shown
        self.assertGreater(MIN_GAP, 5)

    def test_episodes(self):
        a, b, c = (self.tmp / n for n in ('1.jpg', '2.jpg', '3.jpg'))
        slides = [(a, 0.0), (b, 150.0), (c, 260.0)]
        durations = [200.0, 200.0]
        ep1 = [Piece(0, 'A', 0.0, 200.0)]
        ep2 = [Piece(1, 'B', 0.0, 200.0)]
        self.assertEqual(slides_for_episode(slides, durations, ep1), [(a, 0.0), (b, 150.0)])
        self.assertEqual(slides_for_episode(slides, durations, ep2), [(b, 0.0), (c, 60.0)])  # b still on screen
        self.assertEqual(slides_for_episode(slides, durations, ep2, speed=2.0), [(b, 0.0), (c, 30.0)])


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg not installed')
class SlideshowVideoTest(unittest.TestCase):
    def test_video_follows_the_slides(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, color in (('red.png', 'red'), ('green.png', 'green'), ('blue.png', 'blue')):
                color_png(d / name, color)
            t = np.arange(int(36 * SR)) / SR
            wav = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            sf.write(d / 'ch.wav', wav, SR, subtype='PCM_16')
            opts = OutputOptions(formats=('mp4',), slides=[(d / 'red.png', 0.0), (d / 'green.png', 12.0),
                                                           (d / 'blue.png', 24.0)])
            files = export_audiobook([ChapterAudio('Глава 1', d / 'ch.wav', 36.0, [])], d / 'Видео', 'Видео', '',
                                     b'', opts, language='ru', log=lambda *a: None)
            mp4 = d / 'Видео.mp4'
            self.assertIn(mp4, files)
            info = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,width,height:'
                                   'format=duration', '-of', 'csv=p=0', str(mp4)], capture_output=True,
                                  text=True).stdout
            self.assertIn('1920,1080', info)
            self.assertAlmostEqual(float(info.strip().splitlines()[-1]), 36.0, delta=0.3)
            expected = {5: 'red', 11.0: 'red', 18: 'green', 30: 'blue', 35.5: 'blue'}
            for at, color in expected.items():
                raw = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', str(at), '-i', str(mp4),
                                      '-frames:v', '1', '-vf', 'scale=4:4', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                     capture_output=True).stdout
                rgb = np.frombuffer(raw, np.uint8).reshape(-1, 3).mean(0)
                self.assertEqual(['red', 'green', 'blue'][int(np.argmax(rgb))], color, f'{at}s: {rgb}')
                self.assertGreater(rgb.max(), 100)
            mid = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', '12.0', '-i', str(mp4),
                                  '-frames:v', '1', '-vf', 'scale=4:4', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                 capture_output=True).stdout
            rgb = np.frombuffer(mid, np.uint8).reshape(-1, 3).mean(0)
            self.assertTrue(rgb[0] > 30 and rgb[1] > 30, f'crossfade at 12 s: {rgb}')  # red and green mixed


if __name__ == '__main__':
    unittest.main()
