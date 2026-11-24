# -*- coding: utf-8 -*-
"""
audiblez-bench: measure what is fastest on THIS GPU before rendering a whole book.

Published speed figures for Qwen3-TTS vary a lot with the GPU, the precision and the runtime, and the fastest
setting is worthless if it degrades the audio (half precision can produce noise or silence on some GPUs). For each
combination of backend (standard qwen-tts / faster-qwen3-tts CUDA graphs), dtype and batch size, this synthesizes
the same Russian story excerpt with the voice of the book and reports:

  - speed in "x real time" (seconds of audio per second of computation; > 1 is faster than real time),
  - how many chunks pass the automatic checks (rate, length, silence, clipping and, with faster-whisper, the
    Whisper character error rate),
  - peak GPU memory.

It then recommends the fastest setting whose quality is as good as the best one, and writes the results as JSON
(the Kaggle notebook reads it to pick its options).

    audiblez-bench --voice-profile ru_male_narrator
    audiblez-bench --backends standard,fast --dtypes float32,float16 --batch-sizes 1,4,8 -o bench.json
"""
import argparse
import gc
import json
import sys
import time

import numpy as np

SAMPLE_RU = [
    'Татьяна стояла у окна и смотрела, как по стеклу стекают капли дождя.',
    '— Ты опять задержался, — тихо сказала она, не оборачиваясь. — Я ждала тебя к ужину.',
    'Муж молча снял пальто, повесил его на крючок и прошёл на кухню, будто не услышал.',
    'Двадцать четыре года брака научили её понимать без слов: что-то изменилось, и изменилось навсегда.',
    '— Работа, — буркнул он наконец. — Ты же знаешь, какой сейчас конец квартала.',
    'Она знала другое. Вчера в кармане его пиджака она нашла чек из ресторана, где они никогда не бывали вместе.',
    'Утром позвонила свекровь и, как всегда, начала с упрёков, словно это Татьяна была во всём виновата.',
    'Но в этот раз она не стала оправдываться. Она положила трубку, достала с антресолей чемодан и начала '
    'собирать вещи.',
]


def _parse_list(value, cast=str):
    return [cast(v.strip()) for v in str(value).split(',') if v.strip()]


def _default_dtypes(device):
    import torch
    if not device.startswith('cuda'):
        return ['float32']
    major = torch.cuda.get_device_capability()[0]
    # Ampere and newer: native bf16. Turing (T4) and older: fp16 tensor cores, bf16 only emulated.
    return ['bfloat16', 'float32'] if major >= 8 else ['float32', 'float16']


def run_config(engine, texts, batch_size, qa_cfg, verifier, seed=0):
    from audiblez.audio import analyze_chunk, clean_chunk
    sr = engine.sample_rate
    # warm-up (CUDA graph capture, cuDNN autotuning, lazy loading of the model and of the voice prompt)
    engine.synthesize(texts[:1], seed=seed + 999)
    _sync()
    t0 = time.time()
    wavs = []
    for i in range(0, len(texts), batch_size):
        wavs += engine.synthesize(texts[i:i + batch_size], seed=seed + i)
    _sync()
    elapsed = time.time() - t0
    audio = sum(len(w) for w in wavs) / sr
    passed, cers, problems = 0, [], []
    for text, wav in zip(texts, wavs):
        wav = np.asarray(wav, dtype=np.float32)
        if not np.all(np.isfinite(wav)):
            problems.append('NaN/inf in the audio')
            continue
        cleaned = clean_chunk(wav, sr)
        qa = analyze_chunk(cleaned, sr, text, qa_cfg, raw_wav=wav, hit_token_limit=engine.hit_token_limit(wav, text))
        if verifier is not None and qa.ok:
            verifier.check(cleaned, sr, text, qa_cfg, qa)
        if qa.cer is not None:
            cers.append(qa.cer)
        passed += qa.ok
        problems += qa.reasons
    return dict(seconds=round(elapsed, 2), audio_seconds=round(audio, 2),
                x_realtime=round(audio / elapsed, 3) if elapsed > 0 else 0.0, passed=passed, total=len(texts),
                mean_cer=round(float(np.mean(cers)), 4) if cers else None, problems=problems[:5],
                peak_memory_gb=round(_peak_memory() / 2 ** 30, 2))


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _peak_memory():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated()
    except Exception:
        pass
    return 0


def _reset_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def recommend(results):
    """The fastest configuration whose pass rate is within one chunk of the best and that produced no NaN."""
    ok = [r for r in results if r.get('error') is None and not any('NaN' in p for p in r.get('problems', []))]
    if not ok:
        return None
    best_pass = max(r['passed'] for r in ok)
    good = [r for r in ok if r['passed'] >= best_pass - 1]
    best_cer = min((r['mean_cer'] for r in good if r['mean_cer'] is not None), default=None)
    if best_cer is not None:  # a faster setting must not be clearly less intelligible
        good = [r for r in good if r['mean_cer'] is None or r['mean_cer'] <= best_cer + 0.03]
    return max(good, key=lambda r: r['x_realtime'])


def main(argv=None):
    """Console entry point: exit code 0 when a usable configuration was found, 1 otherwise."""
    return 0 if run(argv) is not None else 1


def run(argv=None):
    """Run the benchmark; returns the recommended configuration (dict) or None."""
    from audiblez import ensure_utf8_stdio
    ensure_utf8_stdio()
    p = argparse.ArgumentParser(prog='audiblez-bench', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--voice-profile', default='ru_male_narrator', help='voice to clone (folder or bundled name)')
    p.add_argument('--model', help='Qwen3-TTS model id (default: the 1.7B Base, or the stress model with --stress)')
    p.add_argument('--stress', choices=['off', 'ruaccent'], default='off', help='benchmark with stress marks')
    p.add_argument('--backends', default='auto', help='comma separated: standard, fast (default: those installed)')
    p.add_argument('--dtypes', default='auto', help='comma separated: float32, float16, bfloat16 (default: '
                                                    'depends on the GPU)')
    p.add_argument('--batch-sizes', default='1,4,8', help='batch sizes for the standard backend (default 1,4,8)')
    p.add_argument('--chunks', type=int, default=8, help='number of sample chunks per configuration (default 8)')
    p.add_argument('--text', help='UTF-8 text file to use instead of the built-in Russian sample (one chunk per line)')
    p.add_argument('--device', default='auto')
    p.add_argument('--asr', choices=['auto', 'on', 'off'], default='auto', help='Whisper check of each chunk')
    p.add_argument('--asr-model', default=None)
    p.add_argument('-o', '--output', default='bench.json', help='JSON results (default bench.json)')
    args = p.parse_args(argv)

    from audiblez.audio import QAConfig
    from audiblez.core import Settings, make_stressor, resolve_engine_choice
    from audiblez.engines import FastQwenEngine, QwenEngine, find_voice_profile, has_fast_backend, resolve_device
    from audiblez.normalize import normalize_text
    device = resolve_device(args.device)
    lines = SAMPLE_RU if not args.text else [line.strip() for line in open(args.text, encoding='utf-8')
                                             if line.strip()]
    texts = [normalize_text(t, 'ru') for t in (lines * (1 + args.chunks // max(1, len(lines))))[:args.chunks]]
    settings = Settings(engine='qwen', language='ru', voice_profile=args.voice_profile, qwen_model=args.model,
                        stress=args.stress)
    choice = resolve_engine_choice(settings, 'ru')
    stressor = make_stressor(settings, choice)
    if stressor is not None:
        texts = [stressor(t) for t in texts]
    model_id = args.model or choice.get('model')
    profile = find_voice_profile(args.voice_profile)

    backends = _parse_list(args.backends)
    if backends == ['auto']:
        backends = ['standard'] + (['fast'] if has_fast_backend() and device.startswith('cuda') else [])
    dtypes = _default_dtypes(device) if args.dtypes == 'auto' else _parse_list(args.dtypes)
    batch_sizes = _parse_list(args.batch_sizes, int) if device.startswith('cuda') else [1]

    verifier = None
    if args.asr != 'off':
        try:
            from audiblez.audio import AsrVerifier
            verifier = AsrVerifier('ru', model_size=args.asr_model, device=args.device)
        except ImportError:
            if args.asr == 'on':
                sys.exit('pip install faster-whisper')
    gpu = ''
    if device.startswith('cuda'):
        import torch
        gpu = torch.cuda.get_device_name(0)
    print(f'Device: {device} {gpu} | {len(texts)} chunks per configuration | backends {backends} | dtypes {dtypes}'
          f' | batch sizes {batch_sizes}')
    results = []
    for backend in backends:
        for dtype in dtypes:
            _reset_memory()
            cls = FastQwenEngine if backend == 'fast' else QwenEngine
            engine = None
            try:
                engine = cls(mode='clone', language='ru', model_id=model_id, device=device, dtype=dtype,
                             voice_profile=profile)
                for bs in (batch_sizes if backend == 'standard' else [1]):
                    label = f'{backend:8s} {dtype:9s} batch {bs}'
                    print(f'--- {label} ...', flush=True)
                    try:
                        r = run_config(engine, texts, bs, QAConfig(), verifier)
                        r.update(backend=backend, dtype=dtype, batch_size=bs, error=None)
                    except Exception as e:  # e.g. NaN probabilities in half precision, out of memory
                        r = dict(backend=backend, dtype=dtype, batch_size=bs, error=f'{type(e).__name__}: {e}'[:300])
                        _reset_memory()
                    results.append(r)
                    _print_row(r)
            except Exception as e:
                r = dict(backend=backend, dtype=dtype, batch_size=None, error=f'{type(e).__name__}: {e}'[:300])
                results.append(r)
                _print_row(r)
            del engine
    best = recommend(results)
    flags = None
    if best:
        flags = f'--qwen-backend {best["backend"]} --dtype {best["dtype"]}' + (
            f' --batch-size {best["batch_size"]}' if best['backend'] == 'standard' else '')
        print(f'\nRecommended on this GPU: {flags}  ({best["x_realtime"]}x real time per GPU, '
              f'{best["passed"]}/{best["total"]} chunks passed the checks)')
    else:
        print('\nNo configuration produced usable audio: check the errors above.')
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dict(device=device, gpu=gpu, results=results, recommended=best, flags=flags), f, ensure_ascii=False,
                  indent=2)
    print(f'Results written to {args.output}')
    return best


def _print_row(r):
    if r.get('error'):
        print(f'    {r["backend"]} {r["dtype"]} batch {r["batch_size"]}: ERROR {r["error"]}')
        return
    cer = f', CER {r["mean_cer"]:.1%}' if r.get('mean_cer') is not None else ''
    print(f'    {r["x_realtime"]:.2f}x real time ({r["audio_seconds"]:.0f}s of audio in {r["seconds"]:.0f}s), '
          f'{r["passed"]}/{r["total"]} passed{cer}, peak GPU memory {r["peak_memory_gb"]} GB'
          + (f' | {"; ".join(r["problems"][:2])}' if r['problems'] else ''))


if __name__ == '__main__':
    sys.exit(main())
