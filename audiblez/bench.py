# -*- coding: utf-8 -*-
"""
Measure the real synthesis speed of the available configurations on this machine, with chunks of the book itself,
and recommend the fastest one. Published speed numbers rarely match a given GPU (and none exist for Qwen3-TTS
voice cloning on a T4), so we measure instead of guessing.
"""
import dataclasses
import gc
import time

from tabulate import tabulate


def _free_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _time_config(engine, texts, batch_size, use_fast):
    """Seconds of audio produced per second of compute for `texts` (after a warm-up generation)."""
    saved_fast = engine.fast
    if not use_fast:
        engine.fast = None
    try:
        engine.synthesize(texts[:1], seed=1)  # warm-up: CUDA graphs capture, kernels, caches
        audio, t0 = 0.0, time.time()
        for i in range(0, len(texts), batch_size):
            wavs = engine.synthesize(texts[i:i + batch_size], seed=100 + i)
            audio += sum(len(w) for w in wavs) / engine.sample_rate
        elapsed = time.time() - t0
    finally:
        engine.fast = saved_fast
    return audio / max(elapsed, 1e-6), audio, elapsed


def run_benchmark(file_path, settings, output_folder='.', n_chunks=8, log=print):
    from audiblez import core
    from audiblez.engines import faster_backend_available, resolve_device
    plan = core.main(file_path, output_folder=output_folder, settings=dataclasses.replace(settings, dry_run=True))
    segments = [s for c in plan['chapters'] for s in c.segments]
    texts = [s.text for s in segments if len(s.text) >= 40][:n_chunks] or [s.text for s in segments][:n_chunks]
    if not texts:
        raise SystemExit('No text to benchmark')
    language = core.resolve_engine_choice(settings, core.load_source(file_path).language)['language']
    choice = core.resolve_engine_choice(settings, language)
    if choice['engine'] != 'qwen':
        raise SystemExit('The benchmark compares Qwen3-TTS backends; Kokoro has a single configuration.')
    device = resolve_device(settings.device)
    on_cuda = device.startswith('cuda')
    rows, results = [], []
    backends = ['faster', 'qwen-tts'] if (on_cuda and faster_backend_available()) else ['qwen-tts']
    log(f'Benchmark on {device} with {len(texts)} chunks of the book ({sum(map(len, texts))} characters)')
    engine = core.create_engine(choice, dataclasses.replace(settings, backend=backends[0], device=device))
    try:
        for backend in backends:
            use_fast = backend == 'faster'
            if use_fast and engine.fast is None:
                continue
            for bs in ([1] if use_fast or not on_cuda else [1, 4, 8]):
                try:
                    speed, audio, elapsed = _time_config(engine, texts, bs, use_fast)
                except Exception as e:  # e.g. out of memory with a large batch
                    log(f'  {backend} batch {bs}: failed ({type(e).__name__}: {e})')
                    _free_cuda()
                    continue
                results.append((speed, backend, bs))
                rows.append([backend, bs, f'{audio:.1f}', f'{elapsed:.1f}', f'{speed:.2f}x'])
                log(f'  {backend} batch {bs}: {speed:.2f}x real time')
    finally:
        del engine
        _free_cuda()
    if not results:
        raise SystemExit('Every configuration failed')
    print(tabulate(rows, headers=['Backend', 'Batch', 'Audio (s)', 'Compute (s)', 'Speed']))
    speed, backend, bs = max(results)
    n_gpu = 1
    try:
        import torch
        n_gpu = torch.cuda.device_count() if on_cuda else 1
    except Exception:
        pass
    total = speed * max(1, n_gpu)
    log(f'\nFastest: --backend {backend} --batch-size {bs} ({speed:.2f}x real time per device'
        + (f', about {total:.2f}x with the {n_gpu} GPUs in parallel' if n_gpu > 1 else '') + ').')
    log(f'A 2-hour episode would take about {2 / total:.1f} h of synthesis (+ Whisper checks and retries).')
    return dict(backend=backend, batch_size=bs, speed=speed, results=results)
