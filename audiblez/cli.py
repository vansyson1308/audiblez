# -*- coding: utf-8 -*-
import argparse
import sys

from audiblez.voices import voices, available_voices_str

EPILOG = '''examples:
  audiblez book.epub -v af_sky                                   English, Kokoro voice
  audiblez book_ru.epub --voice-profile ru_male_narrator --youtube --episode-hours 2
                                                                 Russian, Qwen3-TTS cloned narrator, YouTube videos
                                                                 of about 2 hours each
  audiblez book_ru.epub --dry-run                                check the normalized text (script.txt) first
  audiblez story_ja.md --youtube --episode-hours 2               your own Japanese story (Markdown, # = chapter)
  audiblez chapters_ko/ --voice-profile ko_male_narrator         a folder of .txt files, one per chapter
  audiblez story_ja.md --benchmark                               measure the fastest settings on this GPU
  audiblez book_ru.epub --chapters 1 --voice-profile voices/narrator/candidate_02
                                                                 quick test on the first chapter
  audiblez-voice design --preset ru_male_narrator                create candidate voices to choose from

to run GUI just run:
  audiblez-ui

available Kokoro voices:
''' + available_voices_str


def build_parser():
    parser = argparse.ArgumentParser(prog='audiblez', epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Generate audiobooks from e-books (epub to m4b / YouTube mp4).')
    parser.add_argument('epub_file_path', metavar='BOOK',
                        help='An .epub / .fb2 / .docx, a .txt / .md file, or a folder of .txt / .md files '
                             '(one per chapter)')
    parser.add_argument('-o', '--output', default='.', help='Output folder for the audiobook and temporary files',
                        metavar='FOLDER')
    parser.add_argument('-p', '--pick', default=False, action='store_true',
                        help='Interactively select which chapters to read in the audiobook')
    parser.add_argument('--chapters', metavar='LIST', help='Only these audiobook chapters, e.g. "1-3,5" (see the table)')
    parser.add_argument('--dry-run', action='store_true', help='Only extract and normalize the text (script.txt)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Measure the speed of every backend / batch size on this machine and recommend one')

    g = parser.add_argument_group('voice and engine')
    g.add_argument('--engine', choices=['auto', 'kokoro', 'qwen'], default='auto',
                   help='TTS engine (default: Kokoro for its languages, Qwen3-TTS otherwise, e.g. Russian)')
    g.add_argument('-v', '--voice', default=None, help=f'Kokoro voice: {", ".join(voices)} (see list below)')
    g.add_argument('-l', '--language', default=None, help='Language code (ru, en, de...). Default: from the e-book')
    g.add_argument('-s', '--speed', default=1.0, type=float, help='Speed from 0.5 to 2.0 (default 1.0)')
    g.add_argument('--voice-profile', metavar='DIR|NAME',
                   help='Qwen3-TTS cloned voice: folder made by audiblez-voice, or a bundled voice (ru_male_narrator)')
    g.add_argument('--ref-audio', metavar='WAV', help='Qwen3-TTS: clone this recording (6-15 s of clean speech)')
    g.add_argument('--ref-text', metavar='TEXT', help='Exact transcript of --ref-audio (or @file.txt)')
    g.add_argument('--speaker', help='Qwen3-TTS built-in speaker (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, '
                                    'Ono_Anna, Sohee). They have an accent outside their native language')
    g.add_argument('--instruct', help='Qwen3-TTS style instruction (with --speaker on 1.7B) or voice description '
                                      '(voice design; the voice can drift between sentences: prefer audiblez-voice)')
    g.add_argument('--qwen-mode', choices=['clone', 'custom', 'design'], help='Force the Qwen3-TTS mode')
    g.add_argument('--qwen-model', help='Qwen3-TTS model id or local folder (default: the 1.7B model for the mode)')
    g.add_argument('--x-vector-only', action='store_true',
                   help='Clone from the speaker embedding only (no transcript needed, lower similarity)')
    g.add_argument('-c', '--cuda', default=False, action='store_true', help='Use GPU via CUDA (same as --device cuda)')
    g.add_argument('--device', default='auto', help='auto (default), cuda, mps or cpu')
    g.add_argument('--dtype', default='auto', help='auto (default), bfloat16, float16, float32')
    g.add_argument('--batch-size', type=int, default=0, help='Chunks synthesized together (default: 4 on a CUDA GPU, '
                                                          '1 on CPU; 8 can be faster on large GPUs)')
    g.add_argument('--temperature', type=float, help='Qwen3-TTS sampling temperature (default 0.9; lower = steadier)')
    g.add_argument('--top-p', type=float)
    g.add_argument('--top-k', type=int)
    g.add_argument('--repetition-penalty', type=float)
    g.add_argument('--seed', type=int, default=0, help='Base random seed (results are reproducible)')
    g.add_argument('--backend', choices=['auto', 'qwen-tts', 'faster'], default='auto',
                   help='Qwen3-TTS backend: faster = CUDA graphs (pip install faster-qwen3-tts); auto uses it when '
                        'installed on a CUDA GPU and falls back to qwen-tts on any error')
    g.add_argument('--time-limit', type=float, metavar='HOURS',
                   help='Stop synthesizing after this many hours, cleanly (run again to continue), e.g. 11.5 on Kaggle')
    g.add_argument('--gpus', default='auto',
                   help='Parallel workers: auto (one per GPU when there are 2+, e.g. Kaggle 2xT4), off, or "0,1"')

    g = parser.add_argument_group('text')
    g.add_argument('--lexicon', metavar='FILE', help='Pronunciation dictionary: lines "word = replacement"')
    g.add_argument('--no-normalize', action='store_true', help='Do not normalize numbers/abbreviations/ё')
    g.add_argument('--keep-stress-marks', action='store_true', help='Keep combining stress marks (U+0301)')
    g.add_argument('--stress', choices=['off', 'ruaccent'], default='off',
                   help='Russian: mark the stressed vowel of every word with RUAccent (homographs, names, ё) and read '
                        'with a Qwen3-TTS fine-tune that follows the marks (siriusfreak/qwen3-tts-12hz-1.7b-ru-'
                        'stress-cf unless --qwen-model is given). pip install "audiblez[stress]"')
    g.add_argument('--no-title', action='store_true', help='Do not read the book title and author at the start')
    g.add_argument('--max-chunk-chars', type=int, help='Maximum characters per synthesized chunk (default: 300 for '
                                                       'Qwen3-TTS, 400 for Kokoro)')
    g.add_argument('--pause-scale', type=float, default=1.0, help='Multiply all pauses (e.g. 1.2 for a calmer pace)')
    g.add_argument('--no-room-tone', action='store_true',
                   help='Digital silence in the pauses (default: the soft background noise of the voice, like a real '
                        'recording, so that the background does not switch off between sentences)')

    g = parser.add_argument_group('quality control')
    g.add_argument('--no-qa', action='store_true', help='Disable automatic checks and retries')
    g.add_argument('--asr', choices=['auto', 'on', 'off'], default='auto',
                   help='Check every chunk with Whisper and regenerate mismatches (auto: if faster-whisper is installed)')
    g.add_argument('--asr-model', help='Whisper model (default: large-v3-turbo on GPU, small on CPU)')
    g.add_argument('--max-retries', type=int, default=3, help='Regenerations of a chunk that fails the checks')
    g.add_argument('--retry-failed', action='store_true', help='Regenerate the chunks listed in the QA report')

    g = parser.add_argument_group('output')
    g.add_argument('--youtube', action='store_true', help='Also create an .mp4 video (cover image) and chapter '
                                                          'timestamps for the YouTube description')
    g.add_argument('--formats', default=None, help='Comma separated: m4b, mp4, wav (default m4b; m4b,mp4 with --youtube)')
    g.add_argument('--episode-hours', type=float, help='Split the output in parts of about this many hours, e.g. 2')
    g.add_argument('--loudness', type=float, default=-16.0, help='Target loudness in LUFS (default -16)')
    g.add_argument('--cover', metavar='IMAGE', help='Cover image for the m4b and the video (default: from the e-book)')
    g.add_argument('--slides', metavar='FILE|FOLDER',
                   help='Pictures of the video: a folder of images (evenly spaced), or a text file with one line per '
                        'picture "image.jpg | text where it starts" (crossfades; checked with --dry-run)')
    g.add_argument('--keep-master', action='store_true', help='Keep the mastered WAV file(s)')
    return parser


def settings_from_args(args):
    from audiblez.core import Settings
    formats = args.formats.split(',') if args.formats else (['m4b', 'mp4'] if args.youtube else ['m4b'])
    formats = tuple(f.strip().lower() for f in formats if f.strip())
    bad = [f for f in formats if f not in ('m4b', 'mp4', 'wav')]
    if bad:
        sys.exit(f'Unknown output format(s): {", ".join(bad)}')
    device = 'cuda' if args.cuda else args.device
    return Settings(
        engine=args.engine, voice=args.voice, language=args.language, speed=args.speed, device=device,
        dtype=args.dtype, qwen_mode=args.qwen_mode, qwen_model=args.qwen_model, voice_profile=args.voice_profile,
        ref_audio=args.ref_audio, ref_text=args.ref_text, speaker=args.speaker, instruct=args.instruct,
        x_vector_only=args.x_vector_only, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        repetition_penalty=args.repetition_penalty, seed=args.seed, batch_size=args.batch_size,
        backend=args.backend, gpus=args.gpus, time_limit=args.time_limit,
        max_chunk_chars=args.max_chunk_chars, qa=not args.no_qa, asr=args.asr, asr_model=args.asr_model,
        max_retries=args.max_retries, retry_failed=args.retry_failed, normalize=not args.no_normalize,
        lexicon=args.lexicon, keep_stress=args.keep_stress_marks, stress=args.stress,
        announce_title=not args.no_title,
        pause_scale=args.pause_scale, room_tone=not args.no_room_tone, formats=formats, loudness=args.loudness,
        episode_hours=args.episode_hours,
        cover=args.cover, slides=args.slides, keep_master=args.keep_master, dry_run=args.dry_run,
        chapters=args.chapters)


def cli_main(argv=None):
    from audiblez import ensure_utf8_stdio
    ensure_utf8_stdio()
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 0:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args(argv)
    if not 0.5 <= args.speed <= 2.0:
        parser.error('--speed must be between 0.5 and 2.0')
    settings = settings_from_args(args)

    if settings.device in ('cuda', 'auto') and not settings.dry_run:
        import torch
        if torch.cuda.is_available():
            print('CUDA GPU available:', torch.cuda.get_device_name(0))
        elif settings.device == 'cuda':
            print('CUDA GPU not available. Defaulting to CPU')
            settings.device = 'cpu'

    if args.benchmark:
        from audiblez.bench import run_benchmark
        run_benchmark(args.epub_file_path, settings, args.output)
        return
    from audiblez.core import main
    main(args.epub_file_path, args.voice, args.pick, args.speed, args.output, settings=settings)


if __name__ == '__main__':
    cli_main()
