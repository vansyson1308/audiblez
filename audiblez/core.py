#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M and Qwen3-TTS models for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk
import hashlib
import json
import platform
import re
import subprocess
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from types import SimpleNamespace
from typing import List, Optional

import ebooklib
import numpy as np
import soundfile
from ebooklib import epub
from tabulate import tabulate

from audiblez import text as textmod
from audiblez.audio import (QAConfig, RoomTone, analyze_chunk, clean_chunk, estimate_noise_floor, level_chunks,
                            silence)
from audiblez.normalize import Lexicon, normalize_text
from audiblez.text import BREAK, HEADING, Section, blocks_to_text, chunk_paragraph, html_to_blocks

sample_rate = 24000

# Pauses (seconds) inserted between synthesized chunks; scaled by Settings.pause_scale
PAUSES = dict(clause=0.18, sentence=0.35, paragraph=0.75, heading=1.1, before_heading=1.4, scene_break=1.8,
              chapter_lead=0.5, chapter_tail=1.6, title=0.9)

KOKORO_LANGUAGE_CODES = {'en': 'a', 'es': 'e', 'fr': 'f', 'hi': 'h', 'it': 'i', 'ja': 'j', 'pt': 'p', 'zh': 'z'}
DEFAULT_QWEN_VOICES = {'ru': 'ru_male_narrator'}  # bundled voice profiles (audiblez/voices/<name>)


@dataclass
class Settings:
    # engine
    engine: str = 'auto'  # auto | kokoro | qwen
    voice: Optional[str] = None  # Kokoro voice, e.g. af_sky
    language: Optional[str] = None  # ISO code (ru, en...); default: from the e-book metadata
    speed: float = 1.0
    device: str = 'auto'  # auto | cuda | mps | cpu
    dtype: str = 'auto'
    # Qwen3-TTS
    qwen_mode: Optional[str] = None  # clone | custom | design (default: inferred from the options)
    qwen_model: Optional[str] = None
    voice_profile: Optional[str] = None  # folder with reference.wav + reference.txt, or a bundled voice name
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    x_vector_only: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    seed: int = 0
    batch_size: int = 0  # 0 = auto (4 on CUDA, 1 otherwise)
    max_chunk_chars: Optional[int] = None
    # quality control
    qa: bool = True
    asr: str = 'auto'  # auto | on | off
    asr_model: Optional[str] = None
    max_retries: int = 3
    retry_failed: bool = False
    # text
    normalize: bool = True
    lexicon: Optional[str] = None
    keep_stress: bool = False
    announce_title: bool = True
    pause_scale: float = 1.0
    room_tone: bool = True  # pauses filled with the background noise of the voice instead of digital silence
    # output
    formats: tuple = ('m4b',)
    loudness: float = -16.0
    true_peak: float = -1.5
    episode_hours: Optional[float] = None
    cover: Optional[str] = None
    keep_master: bool = False
    dry_run: bool = False
    chapters: Optional[str] = None  # e.g. "1-3,5"
    # speed
    qwen_backend: str = 'auto'  # auto | standard | fast (faster-qwen3-tts CUDA graphs)
    gpus: str = 'auto'  # auto (every visible CUDA GPU) or a number of GPUs to use in parallel
    time_limit: Optional[float] = None  # hours: stop synthesizing (cleanly, resumable) after this time
    # Russian stress marks
    stress: str = 'off'  # off | ruaccent


@dataclass
class Segment:
    text: str
    pause_after: float
    kind: str = 'para'
    para_end: bool = True


@dataclass
class PlannedChapter:
    index: int  # 1-based position in the audiobook
    title: str
    segments: List[Segment]
    source_index: int  # index of the EPUB document (used by the GUI)


# ---------------------------------------------------------------------------------------------------------------------
# EPUB reading
# ---------------------------------------------------------------------------------------------------------------------

def read_book(file_path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # ebooklib deprecation chatter
        book = epub.read_epub(str(file_path), options={'ignore_ncx': False})
    meta_title = book.get_metadata('DC', 'title')
    title = meta_title[0][0].strip() if meta_title else ''
    meta_creator = book.get_metadata('DC', 'creator')
    creator = meta_creator[0][0].strip() if meta_creator else ''
    creator = re.sub(r'^\s*(?:автор|авторы|author|by)\s*[:\-—]?\s+', '', creator, flags=re.I)  # "автор Пушкин"
    meta_lang = book.get_metadata('DC', 'language')
    language = meta_lang[0][0].strip().lower()[:2] if meta_lang and meta_lang[0][0] else None
    return book, title, creator, language


def find_cover(book):
    def is_image(item):
        return item is not None and item.media_type.startswith('image/')

    for item in book.get_items_of_type(ebooklib.ITEM_COVER):
        if is_image(item):
            return item
    # https://idpf.org/forum/topic-715
    for meta in book.get_metadata('OPF', 'cover'):
        if is_image(item := book.get_item_with_id(meta[1].get('content', ''))):
            return item
    if is_image(item := book.get_item_with_id('cover')):
        return item
    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        if 'cover' in item.get_name().lower() and is_image(item):
            return item
    return None


def _spine_documents(book):
    """EPUB documents in reading order (spine), falling back to manifest order."""
    docs, seen = [], set()
    for entry in book.spine:
        idref = entry[0] if isinstance(entry, (tuple, list)) else entry
        item = book.get_item_with_id(idref)
        if item is not None and item.get_type() == ebooklib.ITEM_DOCUMENT and item.id not in seen:
            docs.append(item)
            seen.add(item.id)
    if not docs:
        docs = [i for i in book.get_items() if i.get_type() == ebooklib.ITEM_DOCUMENT]
    return docs


def toc_titles(book):
    """Map document href (without #fragment) -> first TOC title pointing to it."""
    titles = {}

    def walk(entries):
        for entry in entries:
            if isinstance(entry, (tuple, list)):
                head, children = entry[0], entry[1] if len(entry) > 1 else []
                walk([head])
                walk(children)
                continue
            href, title = getattr(entry, 'href', None), getattr(entry, 'title', None)
            if href and title:
                titles.setdefault(href.split('#')[0], title.strip())

    try:
        walk(book.toc)
    except Exception:
        pass
    return titles


def find_document_chapters_and_extract_texts(book):
    """Returns every document of the book (reading order), enriched with .blocks and .extracted_text."""
    document_chapters = []
    toc = toc_titles(book)
    for chapter in _spine_documents(book):
        try:
            blocks = html_to_blocks(chapter.get_content())
        except Exception:
            traceback.print_exc()
            blocks = []
        chapter.blocks = blocks
        chapter.extracted_text = blocks_to_text(blocks)
        chapter._text_signature = chapter.extracted_text
        chapter.toc_title = toc.get(chapter.get_name())
        document_chapters.append(chapter)
    for i, c in enumerate(document_chapters):
        c.chapter_index = i  # this is used in the UI to identify chapters
    return document_chapters


_NON_CONTENT_RE = re.compile(  # never the story
    r'(^|[/_\-.])(nav|toc|contents?|cover|copyright|colophon|titlepage|halftitle|bibliograph\w*|glossary|'
    r'footnotes?|endnotes?|notes|acknowledge?ments?)([/_\-.\d]|$)', re.I)
_MAYBE_NON_CONTENT_RE = re.compile(  # usually not the story, unless it is the main (largest) document
    r'(^|[/_\-.])(title|about|imprint|index|wrap\d*)([/_\-.\d]|$)', re.I)


def is_chapter(c):
    name = c.get_name().lower()
    has_min_len = len(c.extracted_text) > 100
    title_looks_like_chapter = bool(
        'chapter' in name
        or re.search(r'part_?\d{1,3}', name)
        or re.search(r'split_?\d{1,3}', name)
        or re.search(r'ch_?\d{1,3}', name)
        or re.search(r'chap_?\d{1,3}', name)
    )
    return has_min_len and title_looks_like_chapter


def is_non_content(c, largest=0):
    if isinstance(c, epub.EpubNav) or _NON_CONTENT_RE.search(c.get_name()):
        return True
    size = len(getattr(c, 'extracted_text', ''))
    return bool(_MAYBE_NON_CONTENT_RE.search(c.get_name())) and (size < 5000 or size < 0.5 * largest)


def chapter_beginning_one_liner(c, chars=20):
    s = c.extracted_text[:chars].strip().replace('\n', ' ').replace('\r', ' ')
    return s + '…' if len(s) > 0 else ''


def find_good_chapters(document_chapters):
    chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT and is_chapter(c)]
    if len(chapters) == 0:
        print('Not easy to recognize the chapters, defaulting to all non-empty documents.')
        largest = max((len(c.extracted_text) for c in document_chapters), default=0)
        chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT
                    and len(c.extracted_text) > 10 and not is_non_content(c, largest)]
        if not chapters:
            chapters = [c for c in document_chapters if len(c.extracted_text) > 10]
    return chapters


def print_selected_chapters(document_chapters, chapters):
    ok = 'X' if platform.system() == 'Windows' else '✅'
    print(tabulate([
        [i, c.get_name(), len(c.extracted_text), ok if c in chapters else '', chapter_beginning_one_liner(c)]
        for i, c in enumerate(document_chapters, start=1)
    ], headers=['#', 'Chapter', 'Text Length', 'Selected', 'First words']))


def pick_chapters(chapters):
    from pick import pick
    # Display the document name, the length and first 50 characters of the text
    chapters_by_names = {
        f'{c.get_name()}\t({len(c.extracted_text)} chars)\t[{chapter_beginning_one_liner(c, 50)}]': c
        for c in chapters}
    title = 'Select which chapters to read in the audiobook'
    ret = pick(list(chapters_by_names.keys()), title, multiselect=True, min_selection_count=1)
    selected_chapters_out_of_order = [chapters_by_names[r[0]] for r in ret]
    return [c for c in chapters if c in selected_chapters_out_of_order]


def parse_selection(spec, count):
    """'1-3,5,8-' -> sorted 1-based indices within 1..count."""
    selected = set()
    for part in (spec or '').replace(' ', '').split(','):
        if not part:
            continue
        m = re.fullmatch(r'(\d*)-(\d*)', part)
        if m:
            a = int(m.group(1)) if m.group(1) else 1
            b = int(m.group(2)) if m.group(2) else count
            selected.update(range(a, b + 1))
        elif part.isdigit():
            selected.add(int(part))
        else:
            raise ValueError(f'Invalid chapter selection "{spec}" (expected e.g. "1-3,5")')
    return sorted(i for i in selected if 1 <= i <= count)


def display_title(title, index, language):
    t = (title or '').strip()
    if not t or re.fullmatch(r'[IVXLCDMХІ]+\.?|\d{1,4}\.?', t):
        word = {'ru': 'Глава', 'de': 'Kapitel', 'fr': 'Chapitre', 'es': 'Capítulo', 'it': 'Capitolo',
                'pt': 'Capítulo'}.get(language, 'Chapter')
        return f'{word} {t.rstrip(".") or index}'
    return t


def document_sections(chapter, min_chars=300):
    """Split one EPUB document into audiobook chapters (a document may contain several chapters)."""
    if getattr(chapter, '_text_signature', None) != chapter.extracted_text or not hasattr(chapter, 'blocks'):
        blocks = textmod.text_to_blocks(chapter.extracted_text)  # text edited in the GUI
    else:
        blocks = chapter.blocks
    return textmod.split_sections(blocks, default_title=getattr(chapter, 'toc_title', None), min_chars=min_chars)


# ---------------------------------------------------------------------------------------------------------------------
# Planning: blocks -> normalized TTS chunks with pauses
# ---------------------------------------------------------------------------------------------------------------------

def plan_section(section: Section, language, max_chars, settings: Settings, lexicon=None) -> List[Segment]:
    scale = settings.pause_scale
    segments: List[Segment] = []
    blocks = section.blocks
    for bi, b in enumerate(blocks):
        nxt = next((x for x in blocks[bi + 1:] if x.kind != BREAK), None)
        if b.kind == BREAK:
            if segments:
                segments[-1].pause_after = max(segments[-1].pause_after, PAUSES['scene_break'] * scale)
            continue
        is_heading = b.kind == HEADING
        text = normalize_text(b.text, language, is_heading=is_heading, lexicon=lexicon, enabled=settings.normalize,
                              keep_stress=settings.keep_stress)
        if not text or not re.search(r'\w', text):
            continue
        chunks = chunk_paragraph(text, language, max_chars)
        for ci, c in enumerate(chunks):
            last = ci == len(chunks) - 1
            if not last:
                pause = PAUSES['sentence'] if c.sentence_end else PAUSES['clause']
            else:
                pause = PAUSES['heading'] if is_heading else PAUSES['paragraph']
                if nxt is not None and nxt.kind == HEADING and not is_heading:
                    pause = max(pause, PAUSES['before_heading'])
            segments.append(Segment(c.text, pause * scale, b.kind, para_end=last))
    return segments


def intro_segments(title, creator, language, max_chars, settings, lexicon=None):
    segs = []
    for t in (title, creator):
        if t:
            text = normalize_text(t, language, is_heading=False, lexicon=lexicon, enabled=settings.normalize)
            for c in chunk_paragraph(text, language, max_chars):
                segs.append(Segment(c.text, PAUSES['title'] * settings.pause_scale, 'heading'))
    if segs:
        segs[-1].pause_after = PAUSES['before_heading'] * settings.pause_scale
    return segs


# ---------------------------------------------------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------------------------------------------------

def resolve_engine_choice(settings: Settings, book_language):
    """Decide engine, language and (for Qwen) the mode, without loading any model."""
    from audiblez.engines import KOKORO_LANGUAGES, QWEN_LANGUAGES
    from audiblez.voices import voices as kokoro_voices
    qwen_opts = any([settings.voice_profile, settings.ref_audio, settings.speaker, settings.instruct,
                     settings.qwen_mode, settings.qwen_model])
    engine = settings.engine
    language = (settings.language or '').lower()[:2] or None
    if engine == 'auto':
        if qwen_opts:
            engine = 'qwen'
        elif settings.voice:
            engine = 'kokoro'
        else:
            lang = language or book_language or 'en'
            engine = 'kokoro' if lang in KOKORO_LANGUAGE_CODES and _kokoro_installed() else 'qwen'
    if engine == 'kokoro':
        voice = settings.voice
        if not voice:
            code = KOKORO_LANGUAGE_CODES.get(language or book_language or 'en', 'a')
            voice = 'af_sky' if code == 'a' else kokoro_voices[code][0]
        language = KOKORO_LANGUAGES.get(voice[0], 'en')
        return dict(engine='kokoro', voice=voice, language=language)
    language = language or book_language or 'en'
    if language not in QWEN_LANGUAGES:
        raise SystemExit(f'Language "{language}" is not supported by Qwen3-TTS ({", ".join(sorted(QWEN_LANGUAGES))}).'
                         f' Use --language to override the e-book language.')
    mode = settings.qwen_mode
    if not mode:
        if settings.voice_profile or settings.ref_audio:
            mode = 'clone'
        elif settings.speaker:
            mode = 'custom'
        elif settings.instruct:
            mode = 'design'
        else:
            mode = 'clone'
    profile = settings.voice_profile
    if mode == 'clone' and not profile and not settings.ref_audio:
        profile = DEFAULT_QWEN_VOICES.get(language)
        if not profile:
            raise SystemExit(f'No voice for "{language}". Create one with "audiblez-voice design" and pass '
                             f'--voice-profile, or pass --ref-audio/--ref-text, or --speaker.')
    choice = dict(engine='qwen', mode=mode, language=language, voice_profile=profile)
    if settings.stress == 'ruaccent' and language == 'ru' and not settings.qwen_model and mode == 'clone':
        from audiblez.stress_ru import STRESS_MODEL
        choice['model'] = STRESS_MODEL  # a Qwen3-TTS fine-tune that follows the stress marks
    return choice


def make_stressor(settings: Settings, choice, work_dir=None):
    """Russian stress marks (--stress ruaccent), or None."""
    if settings.stress in (None, '', 'off'):
        return None
    if settings.stress != 'ruaccent':
        raise SystemExit(f'Unknown --stress mode "{settings.stress}" (expected off or ruaccent)')
    if choice['language'] != 'ru' or choice['engine'] != 'qwen':
        print('\033[93m--stress ruaccent only applies to Russian with Qwen3-TTS: ignored.\033[0m')
        return None
    from audiblez.stress_ru import STRESS_MODEL, Stressor
    model = settings.qwen_model or choice.get('model')
    if model != STRESS_MODEL:
        print(f'\033[93mNote: {model or "the base model"} may not follow stress marks; {STRESS_MODEL} is the '
              f'model trained to read them.\033[0m')
    return Stressor(Path(work_dir) / 'stress_cache.json' if work_dir else None)


def _kokoro_installed():
    import importlib.util
    return importlib.util.find_spec('kokoro') is not None


def create_engine(choice, settings: Settings):
    if choice['engine'] == 'kokoro':
        if not _kokoro_installed():
            raise SystemExit('Kokoro is not installed (pip install kokoro; it needs Python 3.10-3.12). '
                             'Use --engine qwen with a --voice-profile instead.')
        from audiblez.engines import KokoroEngine
        return KokoroEngine(choice['voice'], speed=settings.speed, device=settings.device)
    from audiblez.engines import QwenEngine, VoiceProfile, find_voice_profile
    profile = None
    if choice['mode'] == 'clone':
        if settings.ref_audio:
            ref_text = settings.ref_text or ''
            if ref_text.startswith('@'):
                ref_text = Path(ref_text[1:]).read_text(encoding='utf-8')
            if not ref_text.strip() and not settings.x_vector_only:
                raise SystemExit('--ref-text is required with --ref-audio (exact transcript of the recording).')
            profile = VoiceProfile(settings.ref_audio, ref_text or '-', choice['language'])
        else:
            profile = find_voice_profile(choice['voice_profile'])
        if profile.language and profile.language[:2] != choice['language']:
            print(f'\033[93mNote: voice profile language is "{profile.language}", book language is '
                  f'"{choice["language"]}". Cross-lingual cloning works but a native reference sounds better.\033[0m')
    engine_class = QwenEngine
    if qwen_backend(settings) == 'fast':
        from audiblez.engines import FastQwenEngine
        engine_class = FastQwenEngine
    return engine_class(mode=choice['mode'], language=choice['language'],
                        model_id=settings.qwen_model or choice.get('model'), device=settings.device,
                        dtype=settings.dtype, voice_profile=profile, speaker=settings.speaker,
                        instruct=settings.instruct, x_vector_only=settings.x_vector_only,
                        temperature=settings.temperature, top_p=settings.top_p, top_k=settings.top_k,
                        repetition_penalty=settings.repetition_penalty)


def qwen_backend(settings: Settings):
    """'fast' (faster-qwen3-tts CUDA graphs) when requested, or by default when it is installed and a GPU is used."""
    from audiblez.engines import has_fast_backend, resolve_device
    wanted = (settings.qwen_backend or 'auto').lower()
    if wanted not in ('auto', 'standard', 'fast'):
        raise SystemExit(f'--qwen-backend must be auto, standard or fast (not "{settings.qwen_backend}")')
    if wanted == 'auto':
        return 'fast' if has_fast_backend() and resolve_device(settings.device).startswith('cuda') else 'standard'
    if wanted == 'fast' and not has_fast_backend():
        raise SystemExit('--qwen-backend fast needs faster-qwen3-tts (in its own environment, see the guide): '
                         'pip install faster-qwen3-tts')
    return wanted


# ---------------------------------------------------------------------------------------------------------------------
# Synthesis with cache, quality control and retries
# ---------------------------------------------------------------------------------------------------------------------

class Synthesizer:
    def __init__(self, engine, work_dir: Path, settings: Settings, stats, verifier=None, post_event=None,
                 worker_id=None):
        self.engine = engine
        self.settings = settings
        self.stats = stats
        self.verifier = verifier
        self.post_event = post_event
        self.fingerprint = engine.fingerprint()
        fp_hash = hashlib.sha1(self.fingerprint.encode('utf-8')).hexdigest()[:12]
        self.cache_dir = Path(work_dir) / 'cache' / f'{engine.name}-{fp_hash}'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / 'engine.json').write_text(json.dumps({'fingerprint': self.fingerprint}, ensure_ascii=False),
                                                    encoding='utf-8')
        speed = settings.speed if engine.native_speed else 1.0  # tempo changes of other engines are applied later
        self.qa_cfg = QAConfig(max_retries=settings.max_retries if engine.stochastic else 0,
                               min_rate=QAConfig.min_rate * speed, max_rate=QAConfig.max_rate * speed)
        # Chunks that never passed the checks, persisted across runs (and across parallel workers: one file each)
        self.failed_file = self.cache_dir / (f'qa_failed.{worker_id}.jsonl' if worker_id else 'qa_failed.jsonl')
        self.failed = {}
        self.reload_failed()
        self.retry_keys = set(self.failed) if settings.retry_failed else set()
        self.report = []  # chunks of this book that did not pass the quality checks
        self._reported = set()

    def reload_failed(self):
        """
        Chunks whose latest take did not pass the checks, from the records of every run and every parallel worker
        (one file each). For each chunk the most recent record wins: a failure, or {"key", "ok": true} when a
        later take passed.
        """
        latest = {}
        for path in sorted(self.cache_dir.glob('qa_failed*.jsonl')):
            for seq, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                key = entry.get('key')
                if not key:
                    continue
                rank = (float(entry.get('t', 0)), seq)
                if key not in latest or rank >= latest[key][0]:
                    latest[key] = (rank, entry)
        self.failed = {k: e for k, (_, e) in latest.items() if not e.get('ok')}

    def key(self, text):
        return hashlib.sha1((self.fingerprint + '\n' + text).encode('utf-8')).hexdigest()

    def cache_path(self, text):
        return self.cache_dir / f'{self.key(text)}.flac'

    def needs_synthesis(self, text):
        return not self.cache_path(text).exists() or self.key(text) in self.retry_keys

    def _cached(self, text):
        if self.needs_synthesis(text):
            return None
        try:
            wav, sr = soundfile.read(str(self.cache_path(text)), dtype='float32')
            return wav
        except Exception:
            return None

    def batch_size(self):
        return int(self.settings.batch_size or 0) or (
            4 if self.engine.stochastic and str(getattr(self.engine, 'device', '')).startswith('cuda') else 1)

    def pending_texts(self, segments_by_label):
        """{text: chapter label} of the chunks that still have to be synthesized (each distinct text once)."""
        todo = {}
        for label, segments in segments_by_label:
            for seg in segments:
                if seg.text not in todo and self.needs_synthesis(seg.text):
                    todo[seg.text] = label
        return todo

    def synthesize_texts(self, todo: dict, deadline=None):
        """
        Synthesize the given {text: label} chunks into the cache, longest first so that the chunks of a batch have
        similar lengths (short ones do not wait for long ones). Returns False if the deadline stopped the work.
        """
        texts = sorted(todo, key=lambda t: (-len(t), t))
        if texts and hasattr(self.engine, 'prepare'):
            self.engine.prepare()  # load the model once, outside the retry loop of synthesis errors
        bs = self.batch_size()
        for start in range(0, len(texts), bs):
            if deadline is not None and time.time() >= deadline:
                return False
            batch = texts[start:start + bs]
            self._render_batch(batch, [todo[t] for t in batch])
        return True

    def render(self, segments: List[Segment], label=''):
        """Audio of every segment (from the cache, synthesizing what is missing)."""
        results = [None] * len(segments)
        todo = []
        for i, seg in enumerate(segments):
            wav = self._cached(seg.text)
            if wav is not None:
                results[i] = wav
            else:
                todo.append(i)
        if todo:
            self.synthesize_texts({segments[i].text: label for i in todo})
            for i in todo:
                results[i] = self._cached(segments[i].text)
                if results[i] is None:  # unreadable cache file
                    raise RuntimeError(f'Could not synthesize "{segments[i].text[:80]}"')
        for seg in segments:
            entry = self.failed.get(self.key(seg.text))
            if entry is not None and entry['key'] not in self._reported:
                self._reported.add(entry['key'])
                self.report.append(dict(entry, chapter=label))
        return results

    def _render_batch(self, texts, labels):
        sr = self.engine.sample_rate
        best = {}
        pending = list(range(len(texts)))
        attempt = 0
        while pending:
            batch_texts = [texts[k] for k in pending]
            seed = self.settings.seed + int(self.key(batch_texts[0])[:8], 16) + attempt * 1_000_003
            t0 = time.time()
            try:
                raw = self.engine.synthesize(batch_texts, seed=seed)
            except Exception as e:
                if attempt >= self.qa_cfg.max_retries:
                    raise
                print(f'\033[93mSynthesis error ({e}), retrying...\033[0m')
                attempt += 1
                continue
            elapsed = time.time() - t0
            still = []
            for k, wav in zip(pending, raw):
                text = texts[k]
                cleaned = clean_chunk(wav, sr)
                hit = bool(getattr(self.engine, 'hit_token_limit', None) and self.engine.hit_token_limit(wav, text))
                qa = analyze_chunk(cleaned, sr, text, self.qa_cfg, raw_wav=wav, hit_token_limit=hit)
                if self.settings.qa and self.verifier is not None and qa.ok:
                    try:
                        self.verifier.check(cleaned, sr, text, self.qa_cfg, qa)
                    except Exception as e:
                        print(f'ASR check failed: {e}')
                if not self.settings.qa:
                    qa.ok, qa.reasons = True, []
                if k not in best or qa.score < best[k][0]:
                    best[k] = (qa.score, cleaned, qa)
                if not qa.ok and attempt < self.qa_cfg.max_retries:
                    still.append(k)
                    print(f'\033[93m  ↻ retry {attempt + 1}/{self.qa_cfg.max_retries}: {"; ".join(qa.reasons)}'
                          f'\n    "{text[:100]}"\033[0m')
            self.stats.synth_seconds += elapsed
            self.stats.synth_chars += sum(len(texts[k]) for k in pending)
            self.stats.audio_seconds = getattr(self.stats, 'audio_seconds', 0.0) + sum(
                len(np.asarray(w).reshape(-1)) for w in raw) / sr
            pending = still
            attempt += 1
        out = {}
        for k, (score, wav, qa) in best.items():
            text = texts[k]
            key = self.key(text)
            path = self.cache_path(text)
            tmp = path.with_name(path.stem + '.part.flac')
            soundfile.write(str(tmp), wav, sr, subtype='PCM_16', format='FLAC')
            tmp.replace(path)  # atomic: an interrupted run never leaves a truncated chunk in the cache
            self.retry_keys.discard(key)
            if not qa.ok:
                entry = dict(key=key, file=path.name, chapter=labels[k], text=text, reasons=qa.reasons,
                             transcript=qa.transcript)
                self.failed[key] = entry
                self._append_failed(entry)
            elif key in self.failed:
                del self.failed[key]
                self._append_failed(dict(key=key, ok=True))
            out[k] = wav
            self._progress(len(text))
        return out

    def _append_failed(self, entry):
        entry = dict(entry, t=round(time.time(), 3))
        with open(self.failed_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _progress(self, chars):
        s = self.stats
        s.processed_chars += chars
        s.progress = min(100, s.processed_chars * 100 // max(1, s.total_chars))
        rate = s.synth_chars / s.synth_seconds if s.synth_seconds > 0 else s.chars_per_sec
        s.chars_per_sec = rate
        s.eta = strfdelta(max(0, s.total_chars - s.processed_chars) / max(rate, 1e-6))
        if self.post_event:
            self.post_event('CORE_PROGRESS', stats=s)
        speed = getattr(s, 'audio_seconds', 0.0) / s.synth_seconds if s.synth_seconds > 0 else 0.0
        print(f'Progress: {s.progress}%  |  {rate:.0f} chars/s  |  {speed:.2f}x real time  |  ETA {s.eta}')


def assemble_chapter(segments: List[Segment], audios, sr, scale=1.0, room_tone=None):
    """Chunks and pauses of a chapter; pauses are digital silence, or `room_tone(seconds)` (see audio.RoomTone)."""
    gap = room_tone or (lambda seconds: silence(seconds, sr))
    lead = PAUSES['chapter_lead'] * scale
    parts, t, split_points = [gap(lead)], lead, []
    for i, (seg, wav) in enumerate(zip(segments, audios)):
        parts.append(np.asarray(wav, dtype=np.float32))
        t += len(wav) / sr
        pause = seg.pause_after
        if i == len(segments) - 1:
            pause = max(pause, PAUSES['chapter_tail'] * scale)
        parts.append(gap(pause))
        if seg.para_end:
            split_points.append(t + pause / 2)
        t += pause
    return np.concatenate(parts), split_points


# ---------------------------------------------------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------------------------------------------------

def main(file_path, voice=None, pick_manually=False, speed=1.0, output_folder='.',
         max_chapters=None, max_sentences=None, selected_chapters=None, post_event=None, settings=None):
    settings = settings or Settings()
    if voice and not settings.voice:
        settings.voice = voice
    if speed and settings.speed == 1.0:
        settings.speed = float(speed)
    if post_event:
        post_event('CORE_STARTED')
    file_path = Path(file_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    stem = file_path.name[:-len('.fb2.zip')] if file_path.name.lower().endswith('.fb2.zip') else file_path.stem

    from audiblez.sources import is_supported, read_source
    if is_supported(file_path):  # story script (.txt, .md), .fb2, .docx
        source = read_source(file_path)
        title, creator, book_language, cover_image = source.title, source.creator, source.language, source.cover
        document_chapters = source.documents
        default_chapters = document_chapters  # no navigation/copyright documents to filter out
        print(f'Read {file_path.name}: {sum(len(d.extracted_text) for d in document_chapters):,} characters'
              + (f', title "{title}"' if title else '') + (f', language {book_language}' if book_language else ''))
    else:
        book, title, creator, book_language = read_book(file_path)
        cover_item = find_cover(book)
        cover_image = cover_item.get_content() if cover_item else b''
        if cover_item:
            print(f'Found cover image {cover_item.file_name} in {cover_item.media_type} format')
        document_chapters = find_document_chapters_and_extract_texts(book)
        default_chapters = None
    if not selected_chapters:
        if pick_manually:
            selected_chapters = pick_chapters(document_chapters)
        else:
            selected_chapters = default_chapters or find_good_chapters(document_chapters)
    print_selected_chapters(document_chapters, selected_chapters)

    choice = resolve_engine_choice(settings, book_language)
    language = choice['language']
    work_dir = output_folder / f'{stem}.audiblez'
    work_dir.mkdir(parents=True, exist_ok=True)
    stressor = make_stressor(settings, choice, work_dir)
    if stressor is not None:
        settings.keep_stress = True  # stress marks of the e-book and of the lexicon are kept and completed
    lexicon = Lexicon.load(settings.lexicon) if settings.lexicon else None
    if lexicon:
        print(f'Loaded {len(lexicon)} pronunciation rules from {settings.lexicon}')
    from audiblez.engines import KokoroEngine, QwenEngine
    max_chars = settings.max_chunk_chars or (QwenEngine if choice['engine'] == 'qwen' else KokoroEngine).default_max_chars

    # Audiobook chapters: documents split at chapter headings
    chapters: List[PlannedChapter] = []
    for doc in selected_chapters:
        for section in document_sections(doc):
            segs = plan_section(section, language, max_chars, settings, lexicon)
            if not segs:
                continue
            title_i = display_title(section.title, len(chapters) + 1, language)
            chapters.append(PlannedChapter(len(chapters) + 1, title_i, segs, getattr(doc, 'chapter_index', 0)))
    if settings.chapters:
        keep = set(parse_selection(settings.chapters, len(chapters)))
        chapters = [c for c in chapters if c.index in keep]
    if max_chapters:
        chapters = chapters[:max_chapters]
    if max_sentences:
        for c in chapters:
            c.segments = c.segments[:max_sentences]
    if not chapters:
        raise SystemExit('No text to synthesize (no chapter selected or chapters are empty).')
    if settings.announce_title and chapters[0].index == 1:
        chapters[0].segments = intro_segments(title, creator, language, max_chars, settings, lexicon) + \
                               chapters[0].segments

    if stressor is not None:
        t0 = time.time()
        for c in chapters:
            for seg in c.segments:
                seg.text = stressor(seg.text)
        stressor.save()
        print(f'Stress marks placed by RUAccent ({time.time() - t0:.0f}s). Check them in script.txt; fix a word '
              f'with a --lexicon rule such as "замок = замо́к".')
    total_chars = sum(len(s.text) for c in chapters for s in c.segments)
    est_hours = sum(textmod.count_letters(s.text) for c in chapters for s in c.segments) / 13.0 / 3600 / settings.speed
    print(tabulate([[c.index, c.title, len(c.segments), sum(len(s.text) for s in c.segments),
                     c.segments[0].text[:50] + '…'] for c in chapters],
                   headers=['#', 'Audiobook chapter', 'Chunks', 'Characters', 'Starts with']))
    print(f'Language: {language} | engine: {choice["engine"]} | {len(chapters)} chapters | {total_chars:,} characters'
          f' | about {est_hours:.1f} h of audio')
    script_path = work_dir / 'script.txt'
    write_script(script_path, title, creator, chapters)
    print(f'Normalized text to be read (check pronunciation issues here): {script_path}')
    if settings.dry_run:
        print('Dry run: nothing synthesized.')
        return dict(chapters=chapters, script=script_path)

    from audiblez.export import has_ffmpeg
    if not has_ffmpeg():
        print('\033[91mffmpeg not found. Please install ffmpeg to create m4b/mp4 files.\033[0m')

    engine = create_engine(choice, settings)  # the model itself is only loaded if something must be synthesized
    print('Voice:', engine.describe())
    verifier = make_verifier(settings, engine, language)
    stats = SimpleNamespace(total_chars=0, processed_chars=0, synth_chars=0, synth_seconds=0.0, audio_seconds=0.0,
                            chars_per_sec=400 if _cuda() else 20, progress=0, eta='?')
    synth = Synthesizer(engine, work_dir, settings, stats, verifier=verifier, post_event=post_event)
    by_chapter = [(c.title, c.segments) for c in chapters]
    todo = synth.pending_texts(by_chapter)
    stats.total_chars = sum(len(t) for t in todo)
    distinct = len({s.text for c in chapters for s in c.segments})
    print(f'{len(todo)} of {distinct} chunks to synthesize ({stats.total_chars:,} characters); '
          f'{distinct - len(todo)} already in the cache')
    deadline = time.time() + settings.time_limit * 3600 if settings.time_limit else None
    print('Started at:', time.strftime('%H:%M:%S'))
    t_synth = time.time()
    complete = True
    if todo:
        from audiblez.parallel import plan_workers, run_workers
        workers = plan_workers(settings, engine, len(todo))
        if workers > 1:
            run_workers(workers, todo, choice, settings, work_dir, deadline)
            synth.reload_failed()
            synth.retry_keys = {k for k in synth.retry_keys  # chunks regenerated by the workers are done
                                if not (synth.cache_dir / f'{k}.flac').exists()
                                or (synth.cache_dir / f'{k}.flac').stat().st_mtime < t_synth}
            todo = synth.pending_texts(by_chapter)  # whatever a worker could not do (crash, time limit)
            if todo:
                print(f'{len(todo)} chunks left after the parallel workers: synthesizing them here.')
        if todo:
            complete = synth.synthesize_texts(todo, deadline)
    if stats.synth_seconds > 0:
        print(f'Synthesis: {time.time() - t_synth:.0f}s, {stats.audio_seconds / 60:.1f} min of audio generated '
              f'({stats.audio_seconds / stats.synth_seconds:.2f}x real time)')
    if not complete:
        left = len(synth.pending_texts(by_chapter))
        print(f'\033[93mTime limit reached: {left} chunks are still to synthesize. Run the same command again to '
              f'continue (everything done so far is in the cache {synth.cache_dir}).\033[0m')
        return dict(chapters=chapters, files=[], work_dir=work_dir, qa_failures=synth.report, incomplete=True)

    sr = engine.sample_rate
    chapter_audio = []
    from audiblez.export import ChapterAudio
    chapters_dir = work_dir / 'chapters'
    chapters_dir.mkdir(exist_ok=True)
    for c in chapters:
        if post_event:
            post_event('CORE_CHAPTER_STARTED', chapter_index=c.source_index)
        audios = level_chunks(synth.render(c.segments, label=c.title), sr)
        tone_db = estimate_noise_floor(audios, sr) if settings.room_tone else None
        tone = RoomTone(tone_db, sr, seed=c.index) if tone_db is not None else None
        wav, split_points = assemble_chapter(c.segments, audios, sr, settings.pause_scale, room_tone=tone)
        path = chapters_dir / f'chapter_{c.index:03d}.wav'
        soundfile.write(str(path), wav, sr, subtype='PCM_16')
        chapter_audio.append(ChapterAudio(c.title, path, len(wav) / sr, split_points))
        print(f'Chapter {c.index} ({c.title}): {len(c.segments)} chunks, {len(wav) / sr / 60:.1f} min of audio')
        if post_event:
            post_event('CORE_CHAPTER_FINISHED', chapter_index=c.source_index)

    write_qa_report(work_dir / 'qa_report.txt', synth.report)
    if synth.report:
        print(f'\033[93m{len(synth.report)} chunks did not pass the quality checks after retries. '
              f'Listen to them (list in {work_dir / "qa_report.txt"}); re-run with --retry-failed to regenerate.\033[0m')

    if has_ffmpeg():
        from audiblez.export import OutputOptions, export_audiobook
        opts = OutputOptions(formats=tuple(settings.formats), loudness=settings.loudness, true_peak=settings.true_peak,
                             speed=1.0 if engine.native_speed else settings.speed, episode_hours=settings.episode_hours,
                             cover_path=Path(settings.cover) if settings.cover else None,
                             keep_master=settings.keep_master)
        written = export_audiobook(chapter_audio, output_folder / stem, title, creator, cover_image, opts,
                                   language=language)
        print('\nCreated:')
        for w in written:
            print('  ', w)
        print(f'Intermediate files (cache, chapter wavs, script, QA report) are in {work_dir}; '
              f'keep them to resume or regenerate parts quickly.')
        if post_event:
            post_event('CORE_FINISHED')
        return dict(chapters=chapters, files=written, work_dir=work_dir, qa_failures=synth.report)
    return dict(chapters=chapters, files=[], work_dir=work_dir, qa_failures=synth.report)


def make_verifier(settings: Settings, engine, language):
    """Whisper checker for stochastic engines (loaded lazily: nothing is loaded if every chunk is cached)."""
    if not (settings.qa and engine.stochastic and settings.asr != 'off'):
        return None
    try:
        from audiblez.audio import AsrVerifier
        return AsrVerifier(language, model_size=settings.asr_model, device=settings.device)
    except ImportError:
        if settings.asr == 'on':
            raise SystemExit('ASR quality control needs faster-whisper: pip install faster-whisper')
        print('Tip: pip install faster-whisper to enable automatic Whisper checks of every sentence.')
        return None


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def write_script(path, title, creator, chapters: List[PlannedChapter]):
    lines = ['# ' + ' — '.join(t for t in (title, creator) if t), '']
    for c in chapters:
        lines.append(f'=== {c.index}. {c.title} ===')
        for s in c.segments:
            lines.append(f'{s.text}    [pause {s.pause_after:.2f}s]')
        lines.append('')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def write_qa_report(path, report):
    if not report:
        Path(path).write_text('All chunks passed the automatic quality checks.\n', encoding='utf-8')
        return
    lines = ['Chunks that did not pass the automatic checks (best take kept). Listen to them in the cache folder;',
             'delete a file and re-run (or use --retry-failed) to regenerate it.', '']
    for r in report:
        lines += [f'[{r["chapter"]}] {r["file"]}', f'  text: {r["text"]}', f'  problem: {"; ".join(r["reasons"])}']
        if r.get('transcript'):
            lines.append(f'  heard: {r["transcript"]}')
        lines.append('')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


# ---------------------------------------------------------------------------------------------------------------------
# Small utilities kept for the GUI and scripts
# ---------------------------------------------------------------------------------------------------------------------

_PREVIEW_ENGINES = {}


def preview_audio(text, voice=None, speed=1.0, settings=None, language=None):
    """Synthesize a short text (GUI preview). Returns (float32 audio, sample_rate)."""
    settings = settings or Settings(voice=voice, speed=speed, language=language)
    choice = resolve_engine_choice(settings, language)
    key = (json.dumps(choice, sort_keys=True), settings.speed, settings.device, settings.speaker, settings.instruct)
    engine = _PREVIEW_ENGINES.get(key)
    if engine is None:
        _PREVIEW_ENGINES.clear()  # keep a single model in memory
        engine = _PREVIEW_ENGINES[key] = create_engine(choice, settings)
    from audiblez.engines import KokoroEngine, QwenEngine
    max_chars = settings.max_chunk_chars or (QwenEngine if choice['engine'] == 'qwen' else KokoroEngine).default_max_chars
    stressor = make_stressor(settings, choice)
    segments = []
    for block in textmod.text_to_blocks(text):
        t = normalize_text(block.text, choice['language'], is_heading=block.kind == HEADING,
                           keep_stress=stressor is not None)
        segments += [Segment(stressor(c.text) if stressor else c.text, PAUSES['sentence'])
                     for c in chunk_paragraph(t, choice['language'], max_chars)]
    audios = [clean_chunk(w, engine.sample_rate) for w in engine.synthesize([s.text for s in segments])] if segments else []
    wav, _ = assemble_chapter(segments, audios, engine.sample_rate) if segments else (np.zeros(0, np.float32), [])
    return wav, engine.sample_rate


def gen_text(text, voice='af_heart', output_file='text.wav', speed=1, play=False):
    wav, sr = preview_audio(text, voice=voice, speed=speed)
    soundfile.write(output_file, wav, sr)
    if play:
        subprocess.run(['ffplay', '-autoexit', '-nodisp', output_file])


def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s'):
    remainder = int(tdelta)
    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for fld in possible_fields:
        if fld in desired_fields and fld in constants:
            values[fld], remainder = divmod(remainder, constants[fld])
    return f.format(fmt, **values)
