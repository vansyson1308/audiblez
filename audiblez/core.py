#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M and Qwen3-TTS models for high-quality text-to-speech synthesis.
# by Claudio Santini 2025 - https://claudio.uk
import dataclasses
import hashlib
import importlib
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
DEFAULT_QWEN_VOICES = {'ru': 'ru_male_narrator', 'ja': 'ja_male_narrator',
                       'ko': 'ko_male_narrator'}  # bundled voice profiles (audiblez/voices/<name>)


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
    backend: str = 'auto'  # Qwen3-TTS backend: auto | qwen-tts | faster (CUDA graphs, needs faster-qwen3-tts)
    gpus: str = 'auto'  # parallel workers: auto (all GPUs when there are 2+), off, or a list like "0,1"
    engine_factory: Optional[str] = None  # "module:function" returning an engine (tests, plugins)
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
    stress: str = 'off'  # off | ruaccent (Russian: automatic stress marks + a model that follows them)
    announce_title: bool = True
    pause_scale: float = 1.0
    room_tone: bool = True  # pauses filled with the background noise of the voice instead of digital silence
    # output
    formats: tuple = ('m4b',)
    loudness: float = -16.0
    true_peak: float = -1.5
    episode_hours: Optional[float] = None
    cover: Optional[str] = None
    slides: Optional[str] = None  # pictures of the video: a folder, or a text file "image | cue" (see slides.py)
    keep_master: bool = False
    dry_run: bool = False
    chapters: Optional[str] = None  # e.g. "1-3,5"
    time_limit: Optional[float] = None  # hours: stop synthesizing (cleanly, resumable) after this time
    deadline: Optional[float] = None  # internal: time.time() at which synthesis stops (from time_limit)


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


def find_document_chapters_and_extract_texts(book, ruby='base'):
    """Returns every document of the book (reading order), enriched with .blocks and .extracted_text."""
    document_chapters = []
    toc = toc_titles(book)
    for chapter in _spine_documents(book):
        try:
            blocks = html_to_blocks(chapter.get_content(), ruby=ruby)
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
        n = t.rstrip('.') or index
        if language in ('ja', 'zh'):
            return f'第{n}章'
        if language == 'ko':
            return f'제{n}장'
        word = {'ru': 'Глава', 'de': 'Kapitel', 'fr': 'Chapitre', 'es': 'Capítulo', 'it': 'Capitolo',
                'pt': 'Capítulo'}.get(language, 'Chapter')
        return f'{word} {n}'
    return t


def document_sections(chapter, min_chars=300):
    """min_chars: sections with less text are merged with the next one (about 25 s of speech)."""
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
            # Qwen3-TTS (expressive, cloned narrator) when audiblez ships a narrator voice for the language,
            # Kokoro (fast) for its other languages
            engine = 'qwen' if (lang in DEFAULT_QWEN_VOICES or lang not in KOKORO_LANGUAGE_CODES
                                or not _kokoro_installed()) else 'kokoro'
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
    return dict(engine='qwen', mode=mode, language=language, voice_profile=profile)


def with_stress_model(settings: Settings, choice):
    """--stress ruaccent: read with the Qwen3-TTS fine-tune that follows the stress marks (unless --qwen-model)."""
    if (settings.stress == 'ruaccent' and not settings.qwen_model and choice.get('engine') == 'qwen'
            and choice.get('language') == 'ru' and choice.get('mode') == 'clone'):
        from audiblez.stress_ru import STRESS_MODEL
        return dataclasses.replace(settings, qwen_model=STRESS_MODEL)
    return settings


def make_stressor(settings: Settings, choice, work_dir=None):
    """Russian stress marks (--stress ruaccent), or None."""
    if settings.stress in (None, '', 'off'):
        return None
    if settings.stress != 'ruaccent':
        raise SystemExit(f'Unknown --stress mode "{settings.stress}" (expected off or ruaccent)')
    if choice.get('language') != 'ru' or choice.get('engine') != 'qwen':
        print('\033[93m--stress ruaccent only applies to Russian with Qwen3-TTS: ignored.\033[0m')
        return None
    from audiblez.stress_ru import STRESS_MODEL, Stressor
    model = with_stress_model(settings, choice).qwen_model
    if model != STRESS_MODEL:
        print(f'\033[93mNote: {model or "the base model"} may not follow stress marks; {STRESS_MODEL} is the '
              f'model trained to read them.\033[0m')
    return Stressor(Path(work_dir) / 'stress_cache.json' if work_dir else None)


def _voice_profile_for(choice, settings: Settings):
    from audiblez.engines import VoiceProfile, find_voice_profile
    if choice.get('mode') != 'clone':
        return None
    if settings.ref_audio:
        ref_text = settings.ref_text or ''
        if ref_text.startswith('@'):
            ref_text = Path(ref_text[1:]).read_text(encoding='utf-8')
        if not ref_text.strip() and not settings.x_vector_only:
            raise SystemExit('--ref-text is required with --ref-audio (exact transcript of the recording).')
        return VoiceProfile(settings.ref_audio, ref_text or '-', choice['language'])
    return find_voice_profile(choice['voice_profile'])


def _qwen_gen_kwargs(settings: Settings):
    return {k: v for k, v in dict(temperature=settings.temperature, top_p=settings.top_p, top_k=settings.top_k,
                                  repetition_penalty=settings.repetition_penalty).items() if v is not None}


def _load_factory(settings: Settings):
    module, _, attr = settings.engine_factory.partition(':')
    return getattr(importlib.import_module(module), attr or 'create_engine')


def _kokoro_installed():
    import importlib.util
    return importlib.util.find_spec('kokoro') is not None


def create_engine(choice, settings: Settings):
    if settings.engine_factory:
        return _load_factory(settings)(choice, settings)
    if choice['engine'] == 'kokoro':
        if not _kokoro_installed():
            raise SystemExit('Kokoro is not installed (pip install kokoro; it needs Python 3.10-3.12). '
                             'Use --engine qwen with a --voice-profile instead.')
        from audiblez.engines import KokoroEngine
        return KokoroEngine(choice['voice'], speed=settings.speed, device=settings.device)
    from audiblez.engines import QwenEngine
    profile = _voice_profile_for(choice, settings)
    if profile is not None and profile.language and profile.language[:2] != choice['language']:
        print(f'\033[93mNote: voice profile language is "{profile.language}", book language is '
              f'"{choice["language"]}". Cross-lingual cloning works but a native reference sounds better.\033[0m')
    return QwenEngine(mode=choice['mode'], language=choice['language'], model_id=settings.qwen_model,
                      device=settings.device, dtype=settings.dtype, voice_profile=profile, speaker=settings.speaker,
                      instruct=settings.instruct, x_vector_only=settings.x_vector_only, backend=settings.backend,
                      **_qwen_gen_kwargs(settings))


def engine_fingerprint(choice, settings: Settings):
    """(engine name, fingerprint) of the engine create_engine() would build, without loading any model."""
    if settings.engine_factory:
        factory_module = importlib.import_module(settings.engine_factory.partition(':')[0])
        if hasattr(factory_module, 'fingerprint'):
            return factory_module.fingerprint(choice, settings)
        engine = create_engine(choice, settings)
        return engine.name, engine.fingerprint()
    if choice['engine'] == 'kokoro':
        return 'kokoro', f'kokoro-82M|{choice["voice"]}|{float(settings.speed)}'
    from audiblez.engines import QWEN_DEFAULT_MODELS, qwen_fingerprint
    model_id = settings.qwen_model or QWEN_DEFAULT_MODELS[choice['mode']]
    return 'qwen', qwen_fingerprint(model_id, choice['mode'], choice['language'], settings.speaker,
                                    settings.instruct or '', settings.x_vector_only, _qwen_gen_kwargs(settings),
                                    _voice_profile_for(choice, settings))


def make_verifier(engine, language, settings: Settings, log=print):
    """Whisper-based check of every chunk (stochastic engines only), or None."""
    if not (settings.qa and engine.stochastic and settings.asr != 'off'):
        return None
    try:
        from audiblez.audio import AsrVerifier
        return AsrVerifier(language, model_size=settings.asr_model, device=getattr(engine, 'device', settings.device),
                           log=log)
    except ImportError:
        if settings.asr == 'on':
            raise SystemExit('ASR quality control needs faster-whisper: pip install faster-whisper')
        log('Tip: pip install faster-whisper to enable automatic Whisper checks of every sentence.')
        return None


def cache_file_ok(path):
    """A cached chunk exists and is not empty (an empty file was left by older versions for takes without speech)."""
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def load_failed(cache_dir):
    """
    {key: entry} of the chunks whose latest take did not pass the quality checks, from qa_failed.jsonl (appended by
    every run and every parallel worker). The most recent record of a chunk wins: a failure, or {"key", "ok": true}
    written when a later take passed (e.g. with --retry-failed).
    """
    latest = {}
    for path in sorted(Path(cache_dir).glob('qa_failed*.jsonl')):
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
    return {k: e for k, (_, e) in latest.items() if not e.get('ok')}


def append_failed(path, entry):
    with open(path, 'a', encoding='utf-8') as f:  # one short line per write: safe with parallel workers
        f.write(json.dumps(dict(entry, t=round(time.time(), 3)), ensure_ascii=False) + '\n')


def time_is_up(settings):
    return bool(settings.deadline) and time.time() >= settings.deadline


class CacheView:
    """Read access to the synthesis cache without loading a model."""

    def __init__(self, work_dir, engine_name, fingerprint):
        self.fingerprint = fingerprint
        fp_hash = hashlib.sha1(fingerprint.encode('utf-8')).hexdigest()[:12]
        self.cache_dir = Path(work_dir) / 'cache' / f'{engine_name}-{fp_hash}'
        self.failed = load_failed(self.cache_dir)
        self.failed_keys = set(self.failed)

    def report(self, labelled_texts):
        """QA report entries of the given (chapter label, text) chunks whose latest take failed the checks."""
        self.failed = load_failed(self.cache_dir)
        out, seen = [], set()
        for label, text in labelled_texts:
            key = chunk_key(self.fingerprint, text)
            if key in self.failed and key not in seen:
                seen.add(key)
                out.append(dict(self.failed[key], chapter=label))
        return out

    def path(self, text):
        return self.cache_dir / f'{chunk_key(self.fingerprint, text)}.flac'

    def is_cached(self, text, retry_failed=False, since=None):
        """With retry_failed, a failed chunk counts as missing unless it was regenerated after `since`."""
        path = self.path(text)
        if not cache_file_ok(path):
            return False
        if retry_failed and chunk_key(self.fingerprint, text) in self.failed_keys:
            return since is not None and path.stat().st_mtime >= since
        return True

    def load(self, text):
        wav, _ = soundfile.read(str(self.path(text)), dtype='float32')
        return wav


# ---------------------------------------------------------------------------------------------------------------------
# Synthesis with cache, quality control and retries
# ---------------------------------------------------------------------------------------------------------------------

class Synthesizer:
    """
    Synthesizes chunks with quality control and retries, and stores them in a cache shared by all workers:
    a chunk is identified by the engine fingerprint (model, voice, settings) and its text.
    """

    def __init__(self, engine, work_dir: Path, settings: Settings, stats=None, verifier=None, post_event=None,
                 language=None, log=print, progress=True):
        self.engine = engine
        self.show_progress = progress
        self.settings = settings
        self.stats = stats or new_stats(0)
        self.verifier = verifier
        self.post_event = post_event
        self.log = log
        self.fingerprint = engine.fingerprint()
        self.cache_dir = cache_dir_for(work_dir, engine)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / 'engine.json').write_text(json.dumps({'fingerprint': self.fingerprint}, ensure_ascii=False),
                                                    encoding='utf-8')
        speed = settings.speed if engine.native_speed else 1.0  # tempo changes of other engines are applied later
        rate = textmod.language_profile(language or getattr(engine, 'language', 'en'))['rate']
        self.qa_cfg = QAConfig(max_retries=settings.max_retries if engine.stochastic else 0,
                               min_rate=0.4 * rate * speed, max_rate=2.0 * rate * speed)
        self.failed_file = self.cache_dir / 'qa_failed.jsonl'
        self.failed_keys = set(load_failed(self.cache_dir))
        self.report = []  # chunks that never passed the quality checks

    def key(self, text):
        return chunk_key(self.fingerprint, text)

    def cache_path(self, text):
        return self.cache_dir / f'{self.key(text)}.flac'

    def is_cached(self, text):
        if not cache_file_ok(self.cache_path(text)):
            return False
        return not (self.settings.retry_failed and self.key(text) in self.failed_keys)

    def load(self, text):
        wav, _ = soundfile.read(str(self.cache_path(text)), dtype='float32')
        return wav

    @property
    def batch_size(self):
        if self.settings.batch_size:
            return max(1, int(self.settings.batch_size))
        on_cuda = str(getattr(self.engine, 'device', '')).startswith('cuda')
        return 4 if self.engine.stochastic and on_cuda and getattr(self.engine, 'supports_batch', True) else 1

    def missing(self, texts):
        seen, out = set(), []
        for t in texts:
            if t not in seen and not self.is_cached(t):
                seen.add(t)
                out.append(t)
        return out

    def synthesize(self, texts, label=''):
        """Synthesize (and cache) the given texts, skipping the ones already cached. False if stopped by --time-limit."""
        todo = self.missing(texts)
        bs = self.batch_size
        if bs > 1:  # similar lengths in a batch: short chunks do not wait for long ones
            todo.sort(key=len, reverse=True)
        for start in range(0, len(todo), bs):
            if time_is_up(self.settings):
                return False
            self._render_batch(todo[start:start + bs], label)
        return True

    def render(self, segments: List[Segment], label=''):
        """Audio of every segment (synthesizing the missing ones)."""
        texts = [seg.text for seg in segments]
        cached = sum(len(t) for t in texts if self.is_cached(t))
        if cached:
            self._progress(cached, cached=True)
        self.synthesize(texts, label)
        return [self.load(t) for t in texts]

    def _render_batch(self, texts, label):
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
                self.log(f'\033[93mSynthesis error ({e}), retrying...\033[0m')
                attempt += 1
                continue
            elapsed = time.time() - t0
            still = []
            for k, wav in zip(pending, raw):
                text = texts[k]
                cleaned, qa = self.evaluate(wav, text)
                rank = (len(cleaned) == 0, qa.score)  # a take with some speech beats an empty one
                if k not in best or rank < best[k][0]:
                    best[k] = (rank, cleaned, qa)
                if not qa.ok and attempt < self.qa_cfg.max_retries:
                    still.append(k)
                    self.log(f'\033[93m  ↻ retry {attempt + 1}/{self.qa_cfg.max_retries}: {"; ".join(qa.reasons)}'
                             f'\n    "{text[:100]}"\033[0m')
            self.stats.synth_seconds += elapsed
            self.stats.synth_chars += sum(len(texts[k]) for k in pending)
            self.stats.synth_audio += sum(len(w) for w in raw) / sr
            pending = still
            attempt += 1
        for k, (_, wav, qa) in best.items():
            text = texts[k]
            if len(wav) == 0:  # no take had any speech: a short silence keeps the book going (listed in the report)
                wav = np.zeros(int(0.2 * sr), np.float32)
            tmp = self.cache_path(text).with_suffix('.tmp.flac')
            soundfile.write(str(tmp), wav, sr, subtype='PCM_16', format='FLAC')
            tmp.replace(self.cache_path(text))  # atomic: a crash never leaves a truncated chunk in the cache
            if not qa.ok:
                entry = dict(key=self.key(text), file=self.cache_path(text).name, chapter=label, text=text,
                             reasons=qa.reasons, transcript=qa.transcript)
                self.report.append(entry)
                append_failed(self.failed_file, entry)
                self.failed_keys.add(self.key(text))
            elif self.key(text) in self.failed_keys:
                self.failed_keys.discard(self.key(text))
                append_failed(self.failed_file, dict(key=self.key(text), ok=True))  # a later take passed
            self._progress(len(text), cached=False)

    def evaluate(self, wav, text):
        """Clean a raw take and run every quality check on it. Returns (cleaned audio, QAResult)."""
        sr = self.engine.sample_rate
        wav = np.asarray(wav, dtype=np.float32)
        finite = bool(np.isfinite(wav).all())
        if not finite:  # numeric overflow (e.g. float16 on some GPUs)
            wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        cleaned = clean_chunk(wav, sr)
        hit = bool(getattr(self.engine, 'hit_token_limit', None) and self.engine.hit_token_limit(wav, text))
        qa = analyze_chunk(cleaned, sr, text, self.qa_cfg, raw_wav=wav, hit_token_limit=hit)
        if not finite:
            qa.ok = False
            qa.reasons.insert(0, 'invalid samples (NaN/inf: numeric overflow)')
        if self.settings.qa and qa.ok:
            self._check_voice(cleaned, sr, qa)
        if self.settings.qa and self.verifier is not None and qa.ok:
            try:
                self.verifier.check(cleaned, sr, text, self.qa_cfg, qa)
            except Exception as e:
                self.log(f'ASR check failed: {e}')
        if not self.settings.qa:
            qa.ok, qa.reasons = True, []
        return cleaned, qa

    def _check_voice(self, wav, sr, qa):
        """Speaker-similarity gate: the take must sound like the reference narrator."""
        fn = getattr(self.engine, 'speaker_similarity', None)
        if fn is None:
            return
        try:
            sim = fn(wav)
        except Exception as e:
            self.log(f'Speaker similarity check failed: {e}')
            return
        if sim is None:
            return
        qa.similarity = sim
        if sim < self.qa_cfg.min_similarity:
            qa.ok = False
            qa.reasons.append(f'voice differs from the narrator (similarity {sim:.2f})')

    def _progress(self, chars, cached):
        if not self.show_progress:
            return
        s = self.stats
        s.processed_chars += chars
        s.progress = min(100, s.processed_chars * 100 // max(1, s.total_chars))
        rate = s.synth_chars / s.synth_seconds if s.synth_seconds > 0 else s.chars_per_sec
        s.chars_per_sec = rate
        s.eta = strfdelta((s.total_chars - s.processed_chars) / max(rate, 1e-6))
        if self.post_event:
            self.post_event('CORE_PROGRESS', stats=s)
        if not cached:
            speed = f' | {s.synth_audio / s.synth_seconds:.2f}x real time' if s.synth_seconds > 0 else ''
            self.log(f'Progress: {s.progress}%  |  {rate:.{0 if rate >= 10 else 1}f} chars/s{speed}  |  ETA {s.eta}')


def new_stats(total_chars):
    return SimpleNamespace(total_chars=total_chars, processed_chars=0, synth_chars=0, synth_seconds=0.0,
                           synth_audio=0.0, chars_per_sec=400 if _cuda() else 20, progress=0, eta='?')


def chunk_key(fingerprint, text):
    return hashlib.sha1((fingerprint + '\n' + text).encode('utf-8')).hexdigest()


def cache_dir_for(work_dir, engine):
    fp_hash = hashlib.sha1(engine.fingerprint().encode('utf-8')).hexdigest()[:12]
    return Path(work_dir) / 'cache' / f'{engine.name}-{fp_hash}'


def assemble_chapter(segments: List[Segment], audios, sr, scale=1.0, room_tone=None, timings=None):
    """
    Chunks and pauses of a chapter; pauses are digital silence, or `room_tone(seconds)` (see audio.RoomTone).
    `timings`, if a list, receives the (start, end) seconds of every segment in the chapter.
    """
    gap = room_tone or (lambda seconds: silence(seconds, sr))
    lead = PAUSES['chapter_lead'] * scale
    parts, t, split_points = [gap(lead)], lead, []
    for i, (seg, wav) in enumerate(zip(segments, audios)):
        parts.append(np.asarray(wav, dtype=np.float32))
        if timings is not None:
            timings.append((t, t + len(wav) / sr))
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

def load_source(file_path):
    """EPUB, FB2, DOCX, .txt/.md file or folder -> title, author, language, cover bytes, documents."""
    path = Path(file_path)
    if path.is_file() and path.suffix.lower() == '.epub':
        book, title, creator, language = read_book(path)
        cover_item = find_cover(book)
        if cover_item:
            print(f'Found cover image {cover_item.file_name} in {cover_item.media_type} format')
        docs = find_document_chapters_and_extract_texts(book, ruby='reading' if language == 'ja' else 'base')
        return SimpleNamespace(kind='epub', title=title, creator=creator, language=language,
                               cover=cover_item.get_content() if cover_item else b'', documents=docs)
    from audiblez.formats import is_docx, is_fb2, read_docx, read_fb2
    if path.is_file() and is_fb2(path):
        return read_fb2(path)
    if path.is_file() and is_docx(path):
        return read_docx(path)
    from audiblez.textinput import TEXT_EXTENSIONS, read_text_book
    if path.is_dir() or (path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS):
        meta, docs = read_text_book(path)
        cover = Path(meta['cover']).read_bytes() if meta.get('cover') else b''
        return SimpleNamespace(kind='text', title=meta['title'], creator=meta.get('author', ''),
                               language=meta.get('language'), cover=cover, documents=docs,
                               announce=not meta.get('title_from_file'))
    if not path.exists():
        raise SystemExit(f'File not found: {path}')
    raise SystemExit(f'Unsupported input {path}: use an .epub, .fb2, .docx, .txt or .md file, or a folder of '
                     f'.txt/.md files')


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
    stem = file_path.stem if file_path.is_file() else file_path.name
    if file_path.name.lower().endswith('.fb2.zip'):
        stem = file_path.name[:-len('.fb2.zip')]

    source = load_source(file_path)
    title, creator, book_language, cover_image = source.title, source.creator, source.language, source.cover
    document_chapters = source.documents
    if not selected_chapters:
        if pick_manually:
            selected_chapters = pick_chapters(document_chapters)
        elif source.kind in ('text', 'fb2'):
            selected_chapters = list(document_chapters)  # own files, FB2 bodies: read everything
        else:
            selected_chapters = find_good_chapters(document_chapters)
    print_selected_chapters(document_chapters, selected_chapters)

    choice = resolve_engine_choice(settings, book_language)
    language = choice['language']
    work_dir = output_folder / f'{stem}.audiblez'
    work_dir.mkdir(parents=True, exist_ok=True)
    stressor = make_stressor(settings, choice, work_dir)
    if stressor is not None:  # marks of the e-book and of the lexicon are kept, RUAccent adds the others
        settings = dataclasses.replace(with_stress_model(settings, choice), keep_stress=True)
    lexicon = Lexicon.load(settings.lexicon) if settings.lexicon else None
    if lexicon:
        print(f'Loaded {len(lexicon)} pronunciation rules from {settings.lexicon}')
    profile = textmod.language_profile(language)
    max_chars = settings.max_chunk_chars or (profile['max_chars'] if choice['engine'] == 'qwen'
                                             else profile['kokoro_max_chars'])

    # Audiobook chapters: documents split at chapter headings
    chapters: List[PlannedChapter] = []
    for doc in selected_chapters:
        # EPUBs: merge tiny sections (title pages such as "Part One"); own texts: every heading is a chapter
        min_chars = int(25 * profile['rate']) if source.kind in ('epub', 'fb2') else 1
        for section in document_sections(doc, min_chars=min_chars):
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
    if settings.announce_title and chapters[0].index == 1 and getattr(source, 'announce', True):
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
    slides = None
    if settings.slides:
        from audiblez.slides import load_slides, resolve_cues
        partial = bool(settings.chapters or max_chapters or max_sentences)
        slides = resolve_cues(load_slides(settings.slides), chapters, strict=not partial)
        print('Video slides:')
        for sl in slides:
            where = ('from 0:00' if sl.chapter is None and sl is slides[0] else
                     f'from chapter {chapters[sl.chapter].index}: '
                     f'"{chapters[sl.chapter].segments[sl.segment].text[:60]}…"'
                     if sl.chapter is not None else 'evenly between the cued slides around it')
            print(f'  {sl.image.name:24s} {where}')
    all_texts = [(c.title, s.text) for c in chapters for s in c.segments]
    total_chars = sum(len(t) for _, t in all_texts)
    est_hours = sum(textmod.count_letters(t) for _, t in all_texts) / profile['rate'] / 3600 / settings.speed
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

    # 1. Synthesize what is not in the cache yet (several GPUs in parallel when available)
    engine_name, fingerprint = engine_fingerprint(choice, settings)
    cache = CacheView(work_dir, engine_name, fingerprint)

    # with --retry-failed, failed chunks written after this are done (2 s margin: coarse file system clocks)
    t_start = time.time() - 2

    def missing_jobs():
        jobs, seen = {}, set()
        for label, t in all_texts:
            if t not in seen and not cache.is_cached(t, settings.retry_failed, since=t_start):
                seen.add(t)
                jobs.setdefault(label, []).append(t)
        return jobs

    if settings.time_limit:
        settings = dataclasses.replace(settings, deadline=time.time() + settings.time_limit * 3600)
    jobs = missing_jobs()
    missing_chars = sum(len(t) for texts in jobs.values() for t in texts)
    stats = new_stats(total_chars)
    stats.processed_chars = total_chars - missing_chars
    if jobs:
        print(f'{sum(len(v) for v in jobs.values())} chunks to synthesize ({missing_chars:,} characters), '
              f'{len(all_texts) - sum(len(v) for v in jobs.values())} already in the cache.')
    print('Started at:', time.strftime('%H:%M:%S'))
    report = []
    from audiblez.parallel import resolve_devices, synthesize_parallel
    devices = resolve_devices(settings.gpus, choice['engine'], settings.device) if jobs else None
    if devices and len(devices) > 1:
        report += synthesize_parallel(devices, choice, settings, work_dir, language, list(jobs.items()),
                                      total_chars, already_done=stats.processed_chars, post_event=post_event,
                                      stats=stats)
        jobs = missing_jobs()
        if jobs:
            print(f'{sum(len(v) for v in jobs.values())} chunks still missing after the parallel run: '
                  f'synthesizing them here.')
    elif devices:
        settings = dataclasses.replace(settings, device=devices[0])
    engine = None
    if jobs:
        if devices and len(devices) > 1:
            settings = dataclasses.replace(settings, device=devices[0])
        engine = create_engine(choice, settings)
        print('Voice:', engine.describe())
        if (engine.name, engine.fingerprint()) != (engine_name, fingerprint):  # the engine is the source of truth
            cache = CacheView(work_dir, engine.name, engine.fingerprint())
        verifier = make_verifier(engine, language, settings)
        synth = Synthesizer(engine, work_dir, settings, stats, verifier=verifier, post_event=post_event,
                            language=language)
        for label, texts in jobs.items():
            if not synth.synthesize(texts, label):
                break
        report += synth.report
    still = missing_jobs()
    report = cache.report(all_texts)  # every chunk of the book whose latest take failed, this run or before
    if still and time_is_up(settings):
        write_qa_report(work_dir / 'qa_report.txt', report)
        print(f'\033[93mTime limit reached: {sum(len(v) for v in still.values())} chunks are still to synthesize. '
              f'Run the same command again to continue (everything done so far is in the cache).\033[0m')
        return dict(chapters=chapters, files=[], work_dir=work_dir, qa_failures=report, incomplete=True)
    if still:
        raise SystemExit(f'{sum(len(v) for v in still.values())} chunks could not be synthesized (see the errors '
                         f'above). Run the same command again to resume.')

    # 2. Assemble the chapters from the cache
    sr = 24000
    chapter_audio = []
    from audiblez.export import ChapterAudio
    chapters_dir = work_dir / 'chapters'
    chapters_dir.mkdir(exist_ok=True)
    chapter_timings = []
    for c in chapters:
        if post_event:
            post_event('CORE_CHAPTER_STARTED', chapter_index=c.source_index)
        audios = level_chunks([cache.load(s.text) for s in c.segments], sr)
        tone_db = estimate_noise_floor(audios, sr) if settings.room_tone else None
        tone = RoomTone(tone_db, sr, seed=c.index) if tone_db is not None else None
        timings = []
        wav, split_points = assemble_chapter(c.segments, audios, sr, settings.pause_scale, room_tone=tone,
                                             timings=timings)
        chapter_timings.append(timings)
        path = chapters_dir / f'chapter_{c.index:03d}.wav'
        soundfile.write(str(path), wav, sr, subtype='PCM_16')
        chapter_audio.append(ChapterAudio(c.title, path, len(wav) / sr, split_points))
        print(f'Chapter {c.index} ({c.title}): {len(wav) / sr / 60:.1f} min of audio')
        if post_event:
            post_event('CORE_CHAPTER_FINISHED', chapter_index=c.source_index)

    write_qa_report(work_dir / 'qa_report.txt', report)
    if report:
        print(f'\033[93m{len(report)} chunks did not pass the quality checks after retries. '
              f'Listen to them (list in {work_dir / "qa_report.txt"}); re-run with --retry-failed to regenerate.\033[0m')

    if has_ffmpeg():
        from audiblez.export import OutputOptions, export_audiobook
        native_speed = choice['engine'] == 'kokoro'
        slide_times = None
        if slides:
            from audiblez.slides import place_slides
            offsets, t = [], 0.0
            for ch in chapter_audio:
                offsets.append(t)
                t += ch.duration
            segment_times = [[(off + a, off + b) for a, b in tm] for off, tm in zip(offsets, chapter_timings)]
            slide_times = place_slides(slides, segment_times, t)
        cover_path = Path(settings.cover) if settings.cover else None
        if cover_path is None and not cover_image and slides:
            cover_path = slides[0].image  # the m4b artwork
        opts = OutputOptions(formats=tuple(settings.formats), loudness=settings.loudness, true_peak=settings.true_peak,
                             speed=1.0 if native_speed else settings.speed, episode_hours=settings.episode_hours,
                             cover_path=cover_path, keep_master=settings.keep_master, slides=slide_times)
        written = export_audiobook(chapter_audio, output_folder / stem, title, creator, cover_image, opts,
                                   language=language)
        print('\nCreated:')
        for w in written:
            print('  ', w)
        print(f'Intermediate files (cache, chapter wavs, script, QA report) are in {work_dir}; '
              f'keep them to resume or regenerate parts quickly.')
        if post_event:
            post_event('CORE_FINISHED')
        return dict(chapters=chapters, files=written, work_dir=work_dir, qa_failures=report)
    return dict(chapters=chapters, files=[], work_dir=work_dir, qa_failures=report)


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def write_script(path, title, creator, chapters: List[PlannedChapter]):
    lines = [f'# {title} — {creator}', '']
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
    stressor = make_stressor(settings, choice)
    settings = with_stress_model(settings, choice)
    key = (json.dumps(choice, sort_keys=True), settings.speed, settings.device, settings.speaker, settings.instruct,
           settings.qwen_model)
    engine = _PREVIEW_ENGINES.get(key)
    if engine is None:
        _PREVIEW_ENGINES.clear()  # keep a single model in memory
        engine = _PREVIEW_ENGINES[key] = create_engine(choice, settings)
    from audiblez.engines import KokoroEngine, QwenEngine
    max_chars = settings.max_chunk_chars or (QwenEngine if choice['engine'] == 'qwen' else KokoroEngine).default_max_chars
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
