# -*- coding: utf-8 -*-
"""
audiblez-voice: create and test the narrator voices used by the Qwen3-TTS engine.

Recommended workflow for a consistent, expressive narrator over a whole book:
  1. design:  generate a few candidate voices from a text description (Qwen3-TTS VoiceDesign),
  2. listen to the candidates and keep the one you like (each candidate is a ready-to-use voice profile),
  3. use it for the whole book: audiblez book.epub --voice-profile voices/narrator/candidate_02

You can also create a profile from your own clean recording (6-15 seconds) with `create`.
Only clone voices you have the right to use.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PRESETS = {
    'ru_male_narrator': dict(
        language='ru',
        instruct=('A deep, warm baritone male voice, about 45 years old, native Russian speaker. Professional '
                  'audiobook narrator: calm, expressive and natural intonation, clear diction, moderate pace, '
                  'no foreign accent.'),
        text=('Тишина стояла такая, что было слышно, как тикают часы в соседней комнате. — Ты правда думаешь, что '
              'он вернётся? — спросила она едва слышно. Я ничего не ответил: за окном уже светало.')),
    'ru_female_narrator': dict(
        language='ru',
        instruct=('A warm, gentle female voice, about 35 years old, native Russian speaker. Professional audiobook '
                  'narrator: soft, emotionally expressive storytelling intonation, clear diction, moderate pace, '
                  'no foreign accent.'),
        text=('Тишина стояла такая, что было слышно, как тикают часы в соседней комнате. — Ты правда думаешь, что '
              'он вернётся? — спросила она едва слышно. Я ничего не ответил: за окном уже светало.')),
    'ru_male_storyteller': dict(
        language='ru',
        instruct=('An older male voice, about 65 years old, native Russian speaker, slightly husky and very warm. '
                  'A wise storyteller reading a novel aloud: unhurried, intimate, rich in emotion, with natural '
                  'pauses, clear diction, no foreign accent.'),
        text=('Тишина стояла такая, что было слышно, как тикают часы в соседней комнате. — Ты правда думаешь, что '
              'он вернётся? — спросила она едва слышно. Я ничего не ответил: за окном уже светало.')),
    'ja_male_narrator': dict(
        language='ja',
        instruct=('A deep, warm male voice, about 40 years old, native Japanese speaker with standard Tokyo '
                  'pronunciation. Professional audiobook narrator (rōdoku): expressive, natural intonation, clear '
                  'diction, natural conversational pace, not slow.'),
        text='静かな夜だった。「本当に、彼は帰ってくると思う？」彼女は小さな声で尋ねた。僕は何も答えられなかった。'),
    'ja_female_narrator': dict(
        language='ja',
        instruct=('A warm, gentle female voice, about 30 years old, native Japanese speaker with standard Tokyo '
                  'pronunciation. Professional audiobook narrator: emotionally expressive storytelling, clear '
                  'diction, natural conversational pace, not slow.'),
        text='静かな夜だった。「本当に、彼は帰ってくると思う？」彼女は小さな声で尋ねた。僕は何も答えられなかった。'),
    'ko_male_narrator': dict(
        language='ko',
        instruct=('A deep, warm male voice, about 40 years old, native Korean speaker with standard Seoul '
                  'pronunciation. Professional audiobook narrator: calm, expressive, natural intonation, clear '
                  'diction, moderate pace.'),
        text=('방 안은 너무 조용해서 옆방의 시계 소리까지 들렸다. "정말 그가 돌아올 거라고 생각해?" 그녀가 작은 '
              '목소리로 물었다. 나는 아무 대답도 하지 못했다. 창밖은 이미 밝아 오고 있었다.')),
    'ko_female_narrator': dict(
        language='ko',
        instruct=('A warm, gentle female voice, about 30 years old, native Korean speaker with standard Seoul '
                  'pronunciation. Professional audiobook narrator: soft, emotionally expressive storytelling, clear '
                  'diction, moderate pace.'),
        text=('방 안은 너무 조용해서 옆방의 시계 소리까지 들렸다. "정말 그가 돌아올 거라고 생각해?" 그녀가 작은 '
              '목소리로 물었다. 나는 아무 대답도 하지 못했다. 창밖은 이미 밝아 오고 있었다.')),
    'en_male_narrator': dict(
        language='en',
        instruct=('A deep, warm baritone male voice, about 45 years old. Professional audiobook narrator: calm, '
                  'expressive, natural intonation, clear diction, moderate pace.'),
        text=('The house was so quiet that you could hear the clock ticking in the next room. "Do you really think '
              'he will come back?" she whispered. I did not answer: outside, the sky was already growing pale.')),
    'en_female_narrator': dict(
        language='en',
        instruct=('A warm, gentle female voice, about 35 years old. Professional audiobook narrator: soft, '
                  'emotionally expressive storytelling, clear diction, moderate pace.'),
        text=('The house was so quiet that you could hear the clock ticking in the next room. "Do you really think '
              'he will come back?" she whispered. I did not answer: outside, the sky was already growing pale.')),
}

PREVIEW_TEXTS = {
    'ja': ('その夜、駅のホームには誰もいなかった。「遅かったじゃない」と、背後から声がした。振り返ると、三年前に別れた'
           'はずの彼女が、少し困ったように笑っていた。'),
    'ko': ('그날 밤, 역 승강장에는 아무도 없었다. "늦었네." 뒤에서 누군가 말했다. 돌아보니 삼 년 전에 헤어진 그녀가 '
           '조금 곤란한 듯 웃고 있었다.'),
    'ru': ('Однажды играли в карты у конногвардейца Нарумова. Долгая зимняя ночь прошла незаметно; сели ужинать в '
           'пятом часу утра. — Что ты сделал, Сурин? — спросил хозяин. — Проиграл, по обыкновению!'),
    'en': ('It was a bright cold day in April, and the clocks were striking thirteen. "Are you coming?" she asked. '
           'He hesitated for a moment, then followed her into the dark corridor.'),
}


def _write_profile(folder, wav, sr, text, language, meta):
    import soundfile as sf
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(str(folder / 'reference.flac'), wav, sr, subtype='PCM_16', format='FLAC')
    (folder / 'reference.txt').write_text(text.strip() + '\n', encoding='utf-8')
    meta = dict(meta, audio='reference.flac', text=text.strip(), language=language,
                duration=round(len(wav) / sr, 2))
    (folder / 'profile.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return folder


def prepare_reference(wav, sr):
    """Clean a reference clip: mono float32, DC removed, edges trimmed, peak normalized to -1 dBFS."""
    from audiblez.audio import trim_silence, fade_edges
    wav = np.asarray(wav, dtype=np.float32)
    wav = wav - float(np.mean(wav))
    wav = trim_silence(wav, sr, pad_start=0.08, pad_end=0.25)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 0:
        wav = wav * (0.89 / peak)
    return fade_edges(wav, sr, 5, 30)


def voice_stats(wav, sr, text):
    """Median pitch (Hz) and speaking rate (letters/s) of a clip: helps to pick a candidate matching the description."""
    import librosa
    from audiblez.text import count_letters
    f0, _, _ = librosa.pyin(np.asarray(wav, dtype=np.float32), fmin=60, fmax=400, sr=sr, frame_length=2048)
    f0 = f0[~np.isnan(f0)]
    pitch = float(np.median(f0)) if len(f0) else float('nan')
    return pitch, count_letters(text) / max(len(wav) / sr, 1e-6)


def _expected_gender(instruct):
    t = (instruct or '').lower()
    male = any(w in t for w in ('male', 'man', 'мужск', 'baritone', 'bass', 'баритон', 'бас'))
    female = any(w in t for w in ('female', 'woman', 'женск', 'girl', 'девуш'))
    if 'female' in t or 'женск' in t:
        return 'female'
    return 'male' if male and not female else ('female' if female else None)


def _asr(language, device):
    try:
        from audiblez.audio import AsrVerifier
        return AsrVerifier(language, device=device)
    except ImportError:
        return None


def cmd_design(args):
    from audiblez.engines import load_qwen_model, seed_everything, QWEN_DEFAULT_MODELS
    from audiblez.audio import char_error_rate
    from audiblez.normalize import normalize_text
    preset = PRESETS.get(args.preset) if args.preset else None
    if args.preset and not preset:
        sys.exit(f'Unknown preset {args.preset}. Available: {", ".join(PRESETS)}')
    language = args.language or (preset or {}).get('language') or 'ru'
    instruct = args.instruct or (preset or {}).get('instruct')
    text = args.text or (preset or {}).get('text') or PRESETS['ru_male_narrator']['text']
    if not instruct:
        sys.exit('Pass --preset or --instruct "description of the voice"')
    text = normalize_text(text, language)
    out = Path(args.output) / (args.name or args.preset or 'voice')
    model = load_qwen_model(args.model or QWEN_DEFAULT_MODELS['design'], args.device, args.dtype)
    from audiblez.engines import QWEN_LANGUAGES
    asr = None if args.no_asr else _asr(language, args.device)
    results = []
    for i in range(1, args.count + 1):
        seed = args.seed + i
        seed_everything(seed)
        t0 = time.time()
        wavs, sr = model.generate_voice_design(text=text, language=QWEN_LANGUAGES[language], instruct=instruct,
                                               max_new_tokens=1200)
        wav = prepare_reference(wavs[0], sr)
        folder = _write_profile(out / f'candidate_{i:02d}', wav, sr, text, language,
                                dict(source='qwen3-tts voice design', instruct=instruct, seed=seed))
        info = f'{len(wav) / sr:.1f}s, generated in {time.time() - t0:.0f}s'
        mismatch = False
        try:
            pitch, rate = voice_stats(wav, sr, text)
            info += f', pitch {pitch:.0f} Hz, {rate:.1f} letters/s'
            expected = _expected_gender(instruct)
            mismatch = (expected == 'male' and pitch > 165) or (expected == 'female' and pitch < 150)
            if mismatch:
                info += f' \033[93m(pitch does not sound {expected}: probably discard)\033[0m'
        except Exception:
            pass
        cer = None
        if asr is not None:
            transcript = asr.transcribe(wav, sr)
            cer = char_error_rate(text, transcript)
            info += f', ASR error {cer:.0%}'
        from audiblez.text import language_profile
        try:
            rate_dev = abs(float(np.log(max(rate, 0.1) / language_profile(language)['rate'])))
        except Exception:
            rate_dev = 0.0
        too_long = len(wav) / sr > 16.0  # cloning is most reliable with references under ~15 s
        results.append((folder, cer, mismatch, too_long, rate_dev))
        print(f'  {folder}  ({info})')
    print('\nListen to the candidates (reference.flac in each folder), then use the one you prefer:')
    best = min(results, key=lambda r: (r[2], (r[1] or 0) > 0.10, r[3], r[4] + (r[1] or 0)))[0]
    print(f'  audiblez book.epub --voice-profile {best}')
    print('Tips: typical pitch is 85-155 Hz for male voices and 165-255 Hz for female voices; a deep baritone is '
          'around 90-120 Hz. Discard candidates with a high ASR error, a wrong pitch or odd intonation, and run '
          'design again with another --seed or a refined --instruct for more options.')


def cmd_create(args):
    from audiblez.engines import load_reference_audio
    text = args.text
    if text.startswith('@'):
        text = Path(text[1:]).read_text(encoding='utf-8')
    wav, sr = load_reference_audio(args.audio)
    wav = prepare_reference(wav, sr)
    duration = len(wav) / sr
    if duration < 3:
        sys.exit(f'Reference too short ({duration:.1f}s): use 6-15 seconds of clean speech.')
    if duration > 15:
        print(f'\033[93mWarning: reference is {duration:.1f}s. 6-15 s works best; longer references make cloning '
              f'less reliable. Consider trimming it (and its transcript).\033[0m')
    folder = _write_profile(args.output, wav, sr, text, args.language, dict(source=str(args.audio)))
    if not args.no_asr:
        asr = _asr(args.language, args.device)
        if asr is not None:
            from audiblez.audio import char_error_rate
            transcript = asr.transcribe(wav, sr)
            cer = char_error_rate(text, transcript)
            print(f'Transcript check: {cer:.0%} difference. Heard: "{transcript}"')
            if cer > 0.15:
                print('\033[93mThe transcript does not match the audio well. The text must be EXACTLY what is said '
                      'in the recording, otherwise cloning quality drops.\033[0m')
    print(f'Voice profile created: {folder}\nUse it with: audiblez book.epub --voice-profile {folder}')


def cmd_preview(args):
    import soundfile as sf
    from audiblez.core import Settings, preview_audio
    language = args.language
    settings = Settings(engine='qwen', language=language, voice_profile=args.voice_profile, speaker=args.speaker,
                        instruct=args.instruct, device=args.device, dtype=args.dtype, qwen_model=args.model)
    text = args.text or PREVIEW_TEXTS.get(language, PREVIEW_TEXTS['en'])
    t0 = time.time()
    wav, sr = preview_audio(text, settings=settings, language=language)
    sf.write(args.output, wav, sr)
    print(f'Wrote {args.output} ({len(wav) / sr:.1f}s of audio in {time.time() - t0:.0f}s)')


def cmd_list(args):
    from audiblez.engines import list_bundled_voices, QWEN_SPEAKERS
    print('Presets for "audiblez-voice design --preset NAME":')
    for name, p in PRESETS.items():
        print(f'  {name:22s} [{p["language"]}] {p["instruct"][:90]}…')
    print('\nVoices bundled with audiblez (use with --voice-profile NAME):')
    for v in list_bundled_voices():
        print('  ', v)
    print('\nQwen3-TTS built-in speakers (--speaker NAME, best in their native language):', ', '.join(QWEN_SPEAKERS))


def main(argv=None):
    from audiblez import ensure_utf8_stdio
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(prog='audiblez-voice', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    def common(p):
        p.add_argument('--device', default='auto', help='auto, cuda, mps or cpu')
        p.add_argument('--dtype', default='auto', help='auto, bfloat16, float16 or float32')
        p.add_argument('--model', default=None, help='Qwen3-TTS model id or local folder')

    p = sub.add_parser('design', help='generate candidate voices from a description')
    p.add_argument('--preset', help=f'one of: {", ".join(PRESETS)}')
    p.add_argument('--instruct', help='voice description, e.g. "A deep warm male voice, 50 years old, native Russian"')
    p.add_argument('--text', help='text spoken in the reference (10-15 s, varied intonation)')
    p.add_argument('--language', default=None, help='language code (default: from preset, else ru)')
    p.add_argument('-n', '--count', type=int, default=4, help='number of candidates (default 4)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name', help='folder name (default: preset name)')
    p.add_argument('-o', '--output', default='voices', help='parent folder (default ./voices)')
    p.add_argument('--no-asr', action='store_true', help='do not check the candidates with Whisper')
    common(p)
    p.set_defaults(func=cmd_design)

    p = sub.add_parser('create', help='create a voice profile from your own recording')
    p.add_argument('--audio', required=True, help='clean recording, 6-15 seconds, one speaker, no music')
    p.add_argument('--text', required=True, help='exact transcript of the recording (or @file.txt)')
    p.add_argument('--language', default='ru')
    p.add_argument('-o', '--output', required=True, help='profile folder to create')
    p.add_argument('--no-asr', action='store_true')
    p.add_argument('--device', default='auto')
    p.set_defaults(func=cmd_create)

    p = sub.add_parser('preview', help='read a sample text with a voice')
    p.add_argument('--voice-profile', help='profile folder or bundled voice name')
    p.add_argument('--speaker', help='Qwen built-in speaker (instead of a profile)')
    p.add_argument('--instruct', help='style instruction (custom/design modes)')
    p.add_argument('--text', help='text to read (default: a sample paragraph)')
    p.add_argument('--language', default='ru')
    p.add_argument('-o', '--output', default='preview.wav')
    common(p)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser('list', help='list presets, bundled voices and built-in speakers')
    p.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
