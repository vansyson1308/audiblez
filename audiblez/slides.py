# -*- coding: utf-8 -*-
"""
Several pictures in the YouTube video, each shown from a chosen moment of the story, with a soft crossfade.

--slides takes a folder of images (shown in name order, evenly spaced) or a text file, one slide per line:

    # image                 | the slide starts at the paragraph that contains this text (optional)
    3.jpg
    1.jpg                   | Здесь всё дышало Галиной Ивановной
    2.jpg                   | Внутри лежала плотная, ровная пачка
    3.jpg                   | Утром дождь наконец кончился

Image paths are relative to the text file. The first slide starts at 0:00; a slide without a cue is placed halfway
between its neighbours. Cues are matched against the text that is read (case, ё/е, stress marks and punctuation are
ignored), in order, and checked when the book is planned (also with --dry-run), before anything is synthesized.

The video is rendered at a low frame rate (still pictures) from images decoded once, so a 2-hour episode takes a few
minutes to encode: see encode_slideshow.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
CROSSFADE = 2.5  # seconds
FPS = 5  # enough for a smooth crossfade of still pictures, cheap to encode
MIN_GAP = 3 * CROSSFADE + 1.0  # slides closer than this are not shown


@dataclass
class Slide:
    image: Path
    cue: Optional[str] = None
    chapter: Optional[int] = None  # position (0-based) of the planned chapter where the cue was found
    segment: Optional[int] = None  # index of the segment (chunk) in that chapter
    time: Optional[float] = None  # start in the audiobook timeline, seconds


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path.name)]


def load_slides(path) -> List[Slide]:
    path = Path(path).expanduser()
    if path.is_dir():
        images = sorted((p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS), key=_natural_key)
        if not images:
            raise SystemExit(f'--slides: no image ({", ".join(IMAGE_EXTENSIONS)}) in {path}')
        return [Slide(p) for p in images]
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [Slide(path)]
    if not path.is_file():
        raise SystemExit(f'--slides: {path} not found')
    slides = []
    for lineno, raw in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        image, _, cue = line.partition('|')
        image_path = (path.parent / image.strip()).expanduser()
        if not image_path.is_file():
            raise SystemExit(f'{path}:{lineno}: image not found: {image_path}')
        slides.append(Slide(image_path, cue.strip() or None))
    if not slides:
        raise SystemExit(f'--slides: no slide in {path}')
    return slides


def _norm(text):
    text = text.replace('́', '').replace('̀', '').lower().replace('ё', 'е')
    text = re.sub(r'[^\w\s]|_', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def resolve_cues(slides: List[Slide], chapters, strict=True, log=print):
    """
    Find the segment where each cued slide starts (in order). Unknown cues stop the run when `strict` (a typo would
    otherwise silently drop a picture), or are dropped with a warning (partial renders: --chapters...).
    """
    positions, flat, cursor = [], '', 0
    for ci, c in enumerate(chapters):
        for si, seg in enumerate(c.segments):
            positions.append((len(flat), ci, si))
            flat += _norm(seg.text) + ' '
    kept, missing = [], []
    for k, slide in enumerate(slides):
        if k == 0 or not slide.cue:
            kept.append(slide)
            continue
        needle = _norm(slide.cue)
        idx = flat.find(needle, cursor) if needle else -1
        if idx < 0:
            missing.append(slide)
            continue
        cursor = idx + len(needle)
        _, slide.chapter, slide.segment = max(p for p in positions if p[0] <= idx)
        kept.append(slide)
    if missing:
        msg = '\n'.join(f'  {s.image.name} | {s.cue}' for s in missing)
        if strict:
            raise SystemExit('--slides: these cues were not found in the text (in this order), check them against '
                             f'script.txt:\n{msg}')
        log(f'\033[93m--slides: cues not in the selected chapters, those slides are not shown:\n{msg}\033[0m')
    if slides and slides[0].cue:
        log(f'Note: the first slide ({slides[0].image.name}) is shown from the start; its cue is not used.')
    return kept


def place_slides(slides: List[Slide], segment_times, total, log=print):
    """
    Start time of every slide. segment_times[ci][si] = (start, end) in seconds of each segment in the audiobook
    timeline; a cued slide starts in the middle of the pause before its segment. Returns [(image, time)].
    """
    times = []
    for k, s in enumerate(slides):
        if k == 0:
            times.append(0.0)
        elif s.chapter is not None:
            start, _ = segment_times[s.chapter][s.segment]
            prev_end = (segment_times[s.chapter][s.segment - 1][1] if s.segment > 0
                        else (segment_times[s.chapter - 1][-1][1] if s.chapter > 0 and segment_times[s.chapter - 1]
                              else start))
            times.append((prev_end + start) / 2)
        else:
            times.append(None)
    # slides without a cue: evenly between the known neighbours
    k = 0
    while k < len(times):
        if times[k] is None:
            j = k
            while j < len(times) and times[j] is None:
                j += 1
            lo = times[k - 1]
            hi = times[j] if j < len(times) else total
            for n, i in enumerate(range(k, j), start=1):
                times[i] = lo + (hi - lo) * n / (j - k + 1)
            k = j
        k += 1
    out = []
    for s, t in zip(slides, times):
        if out and (t - out[-1][1] < MIN_GAP or total - t < MIN_GAP):
            log(f'\033[93m--slides: {s.image.name} would start {t:.0f}s, too close to another slide or to the end: '
                f'not shown.\033[0m')
            continue
        out.append((s.image, float(t)))
    return out


def slides_for_episode(slides, chapter_durations, pieces, speed=1.0):
    """
    The slides of one episode (a list of export.Piece), with times in the episode's output timeline: the slide
    active when the episode starts is shown from 0:00.
    """
    offsets, t = [], 0.0
    for d in chapter_durations:
        offsets.append(t)
        t += d
    ep_start = offsets[pieces[0].chapter_index] + pieces[0].start
    out, before = [], None
    for image, st in slides:
        if st <= ep_start + 1e-6:
            before = image
            continue
        elapsed = 0.0
        for p in pieces:
            g0 = offsets[p.chapter_index] + p.start
            g1 = offsets[p.chapter_index] + p.end
            if g0 <= st < g1:
                out.append((image, (elapsed + st - g0) / speed))
                break
            elapsed += p.end - p.start
    if before is not None:
        out.insert(0, (before, 0.0))
    elif out:
        out[0] = (out[0][0], 0.0)
    return out


def render_slide(image: Path, out_path: Path, run):
    """1920x1080 picture: full screen for a 16:9 image, otherwise the picture over a blurred copy of itself."""
    from audiblez.export import _image_size
    w, h = _image_size(image)
    if w and h and abs(w / h - 16 / 9) < 0.06:
        graph = 'scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,crop=1920:1080,format=rgb24'
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', image, '-vf', graph, '-frames:v', '1',
             out_path], capture=True)
    else:
        graph = ('[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=40:4,'
                 'eq=brightness=-0.12[bg];[0:v]scale=1840:1000:force_original_aspect_ratio=decrease:flags=lanczos[fg];'
                 '[bg][fg]overlay=(W-w)/2:(H-h)/2,format=rgb24')
        run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', image, '-filter_complex', graph,
             '-frames:v', '1', out_path], capture=True)
    return out_path


def slideshow_graph(times, duration, fps=FPS, fade=CROSSFADE):
    """filter_complex for inputs 0..n-1 (one picture each) shown from times[k], with crossfades. Output [vout]."""
    n = len(times)
    parts = []
    for k in range(n):
        start = times[k] - (fade / 2 if k else 0.0)
        end = times[k + 1] + fade / 2 if k + 1 < n else duration + 1.0
        parts.append(f'[{k}:v]format=yuv420p,loop=loop=-1:size=1,fps={fps},trim=duration={end - start:.3f},'
                     f'settb=AVTB,setpts=PTS-STARTPTS[v{k}]')
    last = 'v0'
    for k in range(1, n):
        out = 'vout' if k == n - 1 else f'x{k}'
        parts.append(f'[{last}][v{k}]xfade=transition=fade:duration={fade}:offset={times[k] - fade / 2:.3f}[{out}]')
        last = out
    if n == 1:
        parts[-1] = parts[-1][:-len('[v0]')] + '[vout]'
    return ';'.join(parts)
