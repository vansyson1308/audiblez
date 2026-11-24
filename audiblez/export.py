# -*- coding: utf-8 -*-
"""
Mastering and packaging of the audiobook with ffmpeg.

- Episode planning: group chapters into parts of about N hours (long chapters are split at paragraph boundaries).
- Mastering: high-pass (rumble/DC), optional tempo change, two-pass EBU R128 loudness normalization, 48 kHz.
- Outputs: .m4b audiobook (AAC, real chapter titles, cover), .mp4 video for YouTube (still image + AAC),
  YouTube chapter timestamps (description text).
"""
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf


@dataclass
class ChapterAudio:
    title: str
    path: Path
    duration: float  # seconds (source tempo)
    split_points: List[float] = field(default_factory=list)  # paragraph boundaries (seconds), candidates for splits


@dataclass
class Piece:
    chapter_index: int
    title: str
    start: float
    end: float

    @property
    def duration(self):
        return self.end - self.start


@dataclass
class OutputOptions:
    formats: tuple = ('m4b',)  # any of: m4b, mp4, wav
    loudness: float = -16.0  # integrated LUFS target
    true_peak: float = -1.5  # dBTP ceiling
    sample_rate: int = 48000
    m4b_bitrate: str = '128k'
    mp4_audio_bitrate: str = '192k'
    speed: float = 1.0  # tempo change applied at mastering (engines without native speed control)
    episode_hours: Optional[float] = None
    cover_path: Optional[Path] = None
    keep_master: bool = False
    slides: Optional[list] = None  # [(image path, start seconds in the audiobook timeline)] for the video (--slides)


def _with_ext(base: Path, ext):
    """'Dir/Name.With.Dots' + '.m4b' (Path.with_suffix would replace '.Dots')."""
    return base.with_name(base.name + ext)


def has_ffmpeg():
    return shutil.which('ffmpeg') is not None and shutil.which('ffprobe') is not None


def _run(cmd, capture=False, cwd=None):
    proc = subprocess.run([str(c) for c in cmd], capture_output=capture, text=True, cwd=cwd,
                          encoding='utf-8', errors='replace')
    if proc.returncode != 0:
        tail = (proc.stderr or '')[-2000:] if capture else ''
        raise RuntimeError(f'ffmpeg failed ({proc.returncode}): {" ".join(map(str, cmd[:6]))}...\n{tail}')
    return proc


@dataclass
class _FFmpegCaps:
    soxr: bool
    libx264: bool


_CAPS = None


def ffmpeg_caps():
    global _CAPS
    if _CAPS is None:
        def works(args):
            try:
                p = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', *args], capture_output=True,
                                   text=True, timeout=60)
                return p.returncode == 0
            except Exception:
                return False
        soxr = works(['-f', 'lavfi', '-i', 'anullsrc=r=24000:cl=mono', '-t', '0.1',
                      '-af', 'aresample=48000:resampler=soxr', '-f', 'null', '-'])
        x264 = works(['-f', 'lavfi', '-i', 'color=c=black:s=64x64:r=1', '-t', '1', '-c:v', 'libx264',
                      '-f', 'null', '-'])
        _CAPS = _FFmpegCaps(soxr=soxr, libx264=x264)
    return _CAPS


def probe_duration(path):
    args = ['ffprobe', '-i', str(path), '-show_entries', 'format=duration', '-v', 'quiet', '-of',
            'default=noprint_wrappers=1:nokey=1']
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


# ---------------------------------------------------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------------------------------------------------

def _split_long_chapter(index, ch: ChapterAudio, max_seconds) -> List[Piece]:
    if ch.duration <= max_seconds * 1.15 or not ch.split_points:
        return [Piece(index, ch.title, 0.0, ch.duration)]
    parts = math.ceil(ch.duration / max_seconds)
    cuts, prev = [], 0.0
    for k in range(1, parts):
        ideal = ch.duration * k / parts
        candidates = [p for p in ch.split_points if prev + 60 < p < ch.duration - 60]
        if not candidates:
            break
        cut = min(candidates, key=lambda p: abs(p - ideal))
        if cut > prev:
            cuts.append(cut)
            prev = cut
    bounds = [0.0] + cuts + [ch.duration]
    return [Piece(index, ch.title, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def plan_episodes(chapters: List[ChapterAudio], episode_hours=None) -> List[List[Piece]]:
    """Group chapters into episodes of roughly equal length, each at most about `episode_hours` long."""
    if not episode_hours:
        return [[Piece(i, c.title, 0.0, c.duration) for i, c in enumerate(chapters)]]
    target = episode_hours * 3600
    pieces = [p for i, c in enumerate(chapters) for p in _split_long_chapter(i, c, target)]
    total = sum(p.duration for p in pieces)
    n_episodes = min(len(pieces), max(1, math.ceil(total / (target * 1.05))))
    if n_episodes == 1:
        return [pieces]
    ends = np.cumsum([p.duration for p in pieces])
    episodes, start_idx = [], 0
    for e in range(1, n_episodes):
        ideal = total * e / n_episodes
        remaining_needed = n_episodes - e  # pieces must remain for the next episodes
        best = None
        for j in range(start_idx, len(pieces) - remaining_needed):
            if best is None or abs(ends[j] - ideal) < abs(ends[best] - ideal):
                best = j
        if best is None:
            break
        episodes.append(pieces[start_idx:best + 1])
        start_idx = best + 1
    episodes.append(pieces[start_idx:])
    return [ep for ep in episodes if ep]


# ---------------------------------------------------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------------------------------------------------

def _ffmeta_escape(s):
    return re.sub(r'([=;#\\\n])', r'\\\1', str(s or ''))


def write_ffmetadata(path, title, artist, chapters, album=None, comment=None):
    """chapters: list of (title, start_seconds, end_seconds) in output time."""
    lines = [';FFMETADATA1', f'title={_ffmeta_escape(title)}', f'artist={_ffmeta_escape(artist)}',
             f'album_artist={_ffmeta_escape(artist)}', f'album={_ffmeta_escape(album or title)}', 'genre=Audiobook']
    if comment:
        lines.append(f'comment={_ffmeta_escape(comment)}')
    lines.append('')
    for t, start, end in chapters:
        lines += ['[CHAPTER]', 'TIMEBASE=1/1000', f'START={int(round(start * 1000))}', f'END={int(round(end * 1000))}',
                  f'title={_ffmeta_escape(t)}', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def format_timestamp(seconds, with_hours):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}:{m:02d}:{s:02d}' if with_hours else f'{m:02d}:{s:02d}'


def youtube_chapters_text(chapters, total_duration):
    with_hours = total_duration >= 3600
    lines = [f'{format_timestamp(start, with_hours)} {t}' for t, start, _ in chapters]
    warnings = []
    if len(chapters) < 3:
        warnings.append('YouTube needs at least 3 timestamps to show chapters.')
    if any(end - start < 10 for _, start, end in chapters):
        warnings.append('YouTube needs every chapter to last at least 10 seconds.')
    return '\n'.join(lines), warnings


# ---------------------------------------------------------------------------------------------------------------------
# Audio assembly and mastering
# ---------------------------------------------------------------------------------------------------------------------

def write_pieces_wav(chapters: List[ChapterAudio], pieces: List[Piece], out_path, gap_seconds=0.0):
    """Concatenate the given chapter ranges into one WAV file (streaming, sample-accurate)."""
    sr = sf.info(str(chapters[pieces[0].chapter_index].path)).samplerate
    gap = np.zeros(int(gap_seconds * sr), dtype=np.float32)
    with sf.SoundFile(str(out_path), 'w', samplerate=sr, channels=1, subtype='PCM_24') as out:
        for k, p in enumerate(pieces):
            with sf.SoundFile(str(chapters[p.chapter_index].path)) as f:
                start, end = int(round(p.start * f.samplerate)), int(round(p.end * f.samplerate))
                f.seek(min(start, f.frames))
                remaining = max(0, min(end, f.frames) - start)
                while remaining > 0:
                    block = f.read(min(remaining, 1 << 18), dtype='float32', always_2d=True)
                    if len(block) == 0:
                        break
                    out.write(block.mean(axis=1))
                    remaining -= len(block)
            if k < len(pieces) - 1 and len(gap):
                out.write(gap)
    return sr


def _pre_filters(speed):
    filters = ['highpass=f=50:poles=2']
    if abs(speed - 1.0) > 1e-3:
        s = speed
        while s > 2.0:
            filters.append('atempo=2.0')
            s /= 2.0
        while s < 0.5:
            filters.append('atempo=0.5')
            s /= 0.5
        filters.append(f'atempo={s:.4f}')
    return filters


def measure_loudness(in_wav, filters=()):
    """Integrated loudness (LUFS) and true peak (dBTP) with ffmpeg's EBU R128 meter."""
    af = ','.join(list(filters) + ['ebur128=peak=true:framelog=quiet'])
    p = _run(['ffmpeg', '-hide_banner', '-nostats', '-i', in_wav, '-af', af, '-f', 'null', '-'], capture=True)
    summary = p.stderr[p.stderr.rfind('Summary:'):]
    i = re.search(r'I:\s+(-?[\d.]+|-inf) LUFS', summary)
    tp = re.search(r'True peak:\s+Peak:\s+(-?[\d.]+|-inf) dBFS', summary, re.S)
    if not i or not tp:
        raise RuntimeError('Could not measure loudness:\n' + p.stderr[-1500:])
    to_f = lambda v: float('-inf') if v == '-inf' else float(v)  # noqa: E731
    return to_f(i.group(1)), to_f(tp.group(1))


def master_audio(in_wav, out_wav, opts: OutputOptions):
    """
    Mastering that keeps the natural dynamics of speech: resampling to opts.sample_rate, a linear gain to reach
    opts.loudness LUFS, and a transparent look-ahead limiter that only touches the rare peaks above the true-peak
    ceiling. The gain is measured and corrected iteratively, so the result is within ±0.2 LU of the target.
    """
    resample = f'aresample={opts.sample_rate}' + (':resampler=soxr:precision=28' if ffmpeg_caps().soxr else '')
    pre = _pre_filters(opts.speed) + [resample]
    input_i, input_tp = measure_loudness(in_wav, pre)
    if not math.isfinite(input_i):
        raise RuntimeError(f'{in_wav} is silent')
    gain = opts.loudness - input_i
    ceiling = opts.true_peak - 0.3  # sample-peak ceiling, a little margin for inter-sample peaks

    def chain(g, c):
        return pre + [f'volume={g:.3f}dB', f'alimiter=limit={10 ** (c / 20):.5f}:attack=2:release=80:level=false']

    out_i, out_tp = input_i, input_tp
    for _ in range(4):
        out_i, out_tp = measure_loudness(in_wav, chain(gain, ceiling))
        delta_i, delta_tp = opts.loudness - out_i, out_tp - opts.true_peak
        if abs(delta_i) <= 0.2 and delta_tp <= 0.05:
            break
        gain += delta_i
        if delta_tp > 0.05:
            ceiling -= delta_tp + 0.1
    _run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', in_wav, '-af', ','.join(chain(gain, ceiling)),
          '-c:a', 'pcm_s24le', out_wav], capture=True)
    return dict(input_i=input_i, output_i=out_i, output_tp=out_tp, gain=gain,
                mode=f'linear gain {gain:+.1f} dB + peak limiter')


def prepare_cover(cover_bytes_or_path, work_dir) -> Optional[Path]:
    """Return a JPEG/PNG cover file usable by ffmpeg, or None."""
    if not cover_bytes_or_path:
        return None
    work_dir = Path(work_dir)
    if isinstance(cover_bytes_or_path, (str, Path)):
        src = Path(cover_bytes_or_path)
        data = src.read_bytes()
    else:
        data = bytes(cover_bytes_or_path)
    if data[:3] == b'\xff\xd8\xff':
        path = work_dir / 'cover.jpg'
        path.write_bytes(data)
        return path
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        path = work_dir / 'cover.png'
        path.write_bytes(data)
        return path
    raw = work_dir / 'cover_src'
    raw.write_bytes(data)
    path = work_dir / 'cover.jpg'
    try:
        _run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', raw, '-frames:v', '1', '-q:v', '2', path],
             capture=True)
        return path
    except RuntimeError:
        print('Warning: could not convert the cover image, it will not be embedded.')
        return None


def encode_m4b(master_wav, out_path, meta_path, cover: Optional[Path], opts: OutputOptions):
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-stats', '-y', '-i', master_wav, '-i', meta_path]
    if cover:
        cmd += ['-i', cover]
    cmd += ['-map', '0:a', '-map_metadata', '1', '-map_chapters', '1']
    if cover:
        cmd += ['-map', '2:v', '-c:v', 'copy', '-disposition:v:0', 'attached_pic']
    cmd += ['-c:a', 'aac', '-b:a', opts.m4b_bitrate, '-ac', '1', '-movflags', '+faststart', '-f', 'mp4', out_path]
    _run(cmd)


def make_video_background(cover: Optional[Path], title, author, work_dir) -> Path:
    """1920x1080 still image: the cover over a blurred copy of itself, or a dark title card."""
    work_dir = Path(work_dir)
    out = work_dir / 'video_background.png'
    if cover:
        graph = ('[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=40:4,'
                 'eq=brightness=-0.12[bg];[0:v]scale=1840:1000:force_original_aspect_ratio=decrease[fg];'
                 '[bg][fg]overlay=(W-w)/2:(H-h)/2,format=rgb24')
        try:
            _run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', cover, '-filter_complex', graph,
                  '-frames:v', '1', out], capture=True)
            return out
        except RuntimeError:
            pass
    (work_dir / 'title.txt').write_text(title or '', encoding='utf-8')
    (work_dir / 'author.txt').write_text(author or '', encoding='utf-8')
    text_graph = ("drawtext=textfile=title.txt:fontcolor=white:fontsize=72:x=(w-text_w)/2:y=(h-text_h)/2-60,"
                  "drawtext=textfile=author.txt:fontcolor=0xbbbbbb:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2+60")
    for vf in (text_graph, None):
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
               'color=c=0x15171c:s=1920x1080:r=1', '-frames:v', '1']
        if vf:
            cmd += ['-vf', vf]
        try:
            _run(cmd + [out.name], capture=True, cwd=work_dir)
            return out
        except RuntimeError:
            continue
    raise RuntimeError('Could not create the video background image')


def encode_mp4(master_wav, out_path, image: Path, meta_path, duration, opts: OutputOptions):
    caps = ffmpeg_caps()
    video = (['-c:v', 'libx264', '-preset', 'medium', '-tune', 'stillimage', '-crf', '18'] if caps.libx264
             else ['-c:v', 'mpeg4', '-q:v', '3'])
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-stats', '-y', '-loop', '1', '-framerate', '1',
           '-i', image, '-i', master_wav, '-i', meta_path, '-map', '0:v', '-map', '1:a', '-map_metadata', '2',
           '-map_chapters', '2', *video, '-pix_fmt', 'yuv420p', '-r', '1', '-g', '30',
           '-c:a', 'aac', '-b:a', opts.mp4_audio_bitrate, '-ac', '2', '-ar', str(opts.sample_rate),
           '-t', f'{duration:.3f}', '-movflags', '+faststart', out_path]
    _run(cmd)


def _image_size(path):
    """(width, height) of an image with ffprobe, (0, 0) if unknown."""
    try:
        p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                            '-of', 'csv=p=0:s=x', str(path)], capture_output=True, text=True, timeout=60)
        w, h = p.stdout.strip().split('x')[:2]
        return int(w), int(h)
    except Exception:
        return 0, 0


def encode_slideshow(master_wav, out_path, slides, meta_path, duration, opts: OutputOptions, work_dir):
    """
    .mp4 whose picture changes at the given times with a crossfade. slides: [(image, start seconds)], the first at 0.
    The pictures are decoded once and repeated (the loop filter) at a low frame rate: a 2 h episode encodes in minutes.
    """
    from audiblez.slides import FPS, render_slide, slideshow_graph
    work_dir = Path(work_dir)
    rendered = {}
    for image, _ in slides:
        if image not in rendered:
            rendered[image] = render_slide(image, work_dir / f'slide_{len(rendered):02d}.png', _run)
    if len(slides) == 1:
        return encode_mp4(master_wav, out_path, rendered[slides[0][0]], meta_path, duration, opts)
    caps = ffmpeg_caps()
    video = (['-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'stillimage', '-crf', '20'] if caps.libx264
             else ['-c:v', 'mpeg4', '-q:v', '3'])
    n = len(slides)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-stats', '-y']
    for image, _ in slides:
        cmd += ['-i', rendered[image]]
    cmd += ['-i', master_wav, '-i', meta_path,
            '-filter_complex', slideshow_graph([t for _, t in slides], duration),
            '-map', '[vout]', '-map', f'{n}:a', '-map_metadata', str(n + 1), '-map_chapters', str(n + 1),
            *video, '-pix_fmt', 'yuv420p', '-r', str(FPS), '-force_key_frames', 'expr:gte(t,n_forced*30)',
            '-c:a', 'aac', '-b:a', opts.mp4_audio_bitrate, '-ac', '2', '-ar', str(opts.sample_rate),
            '-t', f'{duration:.3f}', '-movflags', '+faststart', out_path]
    _run(cmd)


def export_audiobook(chapters: List[ChapterAudio], out_base: Path, title, author, cover_bytes, opts: OutputOptions,
                     language='en', part_label=None, log=print):
    """Master and encode the audiobook. Returns the list of written files."""
    out_base = Path(out_base)
    written = []
    episodes = plan_episodes(chapters, opts.episode_hours)
    cont = {'ru': 'продолжение', 'de': 'Fortsetzung', 'fr': 'suite', 'es': 'continuación', 'it': 'continua',
            'pt': 'continuação'}.get(language[:2], 'continued')
    part_word = part_label or {'ru': 'Часть', 'de': 'Teil', 'fr': 'Partie', 'es': 'Parte', 'it': 'Parte',
                               'pt': 'Parte'}.get(language[:2], 'Part')
    with tempfile.TemporaryDirectory(prefix='audiblez_') as tmp:
        tmp = Path(tmp)
        cover_src = opts.cover_path if opts.cover_path else cover_bytes
        cover = prepare_cover(cover_src, tmp) if cover_src else None
        background = None
        summary = []
        for e, pieces in enumerate(episodes, start=1):
            multi = len(episodes) > 1
            base = out_base.with_name(out_base.name + (f'_part{e:02d}' if multi else ''))
            ep_title = f'{title} — {part_word} {e}' if multi and title else (title or out_base.name)
            log(f'\n=== Mastering {base.name} ({sum(p.duration for p in pieces) / opts.speed / 3600:.2f} h) ===')
            raw = tmp / f'raw_{e:02d}.wav'
            write_pieces_wav(chapters, pieces, raw)
            master = (_with_ext(base, '.master.wav') if opts.keep_master or 'wav' in opts.formats
                      else tmp / f'master_{e:02d}.wav')
            stats = master_audio(raw, master, opts)
            raw.unlink()
            log(f'Loudness: {stats["input_i"]:.1f} LUFS -> {stats["output_i"]:.1f} LUFS, true peak '
                f'{stats["output_tp"]:.1f} dBTP ({stats["mode"]})')
            meta_chapters, t, seen = [], 0.0, set()
            for p in pieces:
                d = p.duration / opts.speed
                name = p.title if p.chapter_index not in seen else f'{p.title} ({cont})'
                seen.add(p.chapter_index)
                meta_chapters.append((name, t, t + d))
                t += d
            total = t
            meta_path = tmp / f'meta_{e:02d}.txt'
            write_ffmetadata(meta_path, ep_title, author, meta_chapters, album=title)
            if 'm4b' in opts.formats:
                path = _with_ext(base, '.m4b')
                log(f'Encoding {path.name}...')
                encode_m4b(master, path, meta_path, cover, opts)
                written.append(path)
            if 'mp4' in opts.formats:
                path = _with_ext(base, '.mp4')
                ep_slides = None
                if opts.slides:
                    from audiblez.slides import slides_for_episode
                    ep_slides = slides_for_episode(opts.slides, [c.duration for c in chapters], pieces, opts.speed)
                if ep_slides:
                    shown = ', '.join(f'{format_timestamp(t, total >= 3600)} {Path(im).name}' for im, t in ep_slides)
                    log(f'Encoding {path.name} (YouTube video, {len(ep_slides)} slides: {shown})...')
                    encode_slideshow(master, path, ep_slides, meta_path, total, opts, tmp)
                else:
                    if background is None:
                        background = make_video_background(cover, title, author, tmp)
                    log(f'Encoding {path.name} (YouTube video)...')
                    encode_mp4(master, path, background, meta_path, total, opts)
                written.append(path)
                text, warnings = youtube_chapters_text(meta_chapters, total)
                yt = base.with_name(base.name + '_youtube.txt')
                header = f'{ep_title}\n{author}\n\n' if author else f'{ep_title}\n\n'
                yt.write_text(header + text + '\n', encoding='utf-8')
                written.append(yt)
                for w in warnings:
                    log(f'Note for {yt.name}: {w}')
            if 'wav' in opts.formats:
                written.append(master)
            summary.append((base.name, total, meta_chapters))
    if len(episodes) > 1:
        lines = []
        for name, total, chs in summary:
            lines.append(f'{name}: {format_timestamp(total, True)}  ({len(chs)} chapters: {chs[0][0]} … {chs[-1][0]})')
        idx = out_base.with_name(out_base.name + '_parts.txt')
        idx.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        written.append(idx)
    return written
