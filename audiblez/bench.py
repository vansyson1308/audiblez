# -*- coding: utf-8 -*-
"""
Measure the real synthesis speed of the available configurations on this machine, with chunks of the book itself,
and recommend the fastest one that keeps the quality. Published speed numbers rarely match a given GPU, so we
measure instead of guessing.

1. Precision: on GPUs without native bfloat16 (T4 on Kaggle / Colab, V100, RTX 20xx) float32, bfloat16 (emulated)
   and float16 are all possible; the fastest is not known in advance and float16 can overflow in this model. Each
   one is timed and every take goes through the same quality checks as a real render (speaking rate, silence,
   NaN, voice similarity, Whisper): a precision that damages the audio is never recommended.
2. Backend and batch size, with the chosen precision: faster-qwen3-tts (CUDA graphs, one sentence at a time) and
   qwen-tts with batches of 1, 4 or 8 sentences.
"""
import dataclasses
import gc
import shutil
import tempfile
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
    """(seconds of audio per second of compute, audio seconds, compute seconds, takes) after a warm-up generation."""
    saved_fast = getattr(engine, 'fast', None)
    if not use_fast and saved_fast is not None:
        engine.fast = None
    try:
        engine.synthesize(texts[:1], seed=1)  # warm-up: CUDA graphs capture, kernels, caches
        audio, takes, t0 = 0.0, [], time.time()
        for i in range(0, len(texts), batch_size):
            wavs = engine.synthesize(texts[i:i + batch_size], seed=100 + i)
            takes += wavs
            audio += sum(len(w) for w in wavs) / engine.sample_rate
        elapsed = time.time() - t0
    finally:
        if saved_fast is not None:
            engine.fast = saved_fast
    return audio / max(elapsed, 1e-6), audio, elapsed, takes


def dtypes_to_try(dtype, device):
    """Precisions worth measuring on this device."""
    if dtype and dtype != 'auto':
        return [dtype]
    if not device.startswith('cuda'):
        return ['float32']
    import torch
    if torch.cuda.get_device_capability(torch.device(device))[0] >= 8:
        return ['bfloat16']  # native bfloat16 (Ampere and newer): as fast as float16, and safe
    return ['float32', 'bfloat16', 'float16']


def acceptable(results):
    """Configurations whose quality matches the best one (one random failure more is tolerated)."""
    best = min(r['failures'] for r in results)
    return [r for r in results if r['failures'] <= best + 1]


def run_benchmark(file_path, settings, output_folder='.', n_chunks=6, log=print):
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
    has_fast = on_cuda and faster_backend_available() and settings.backend in ('auto', 'faster')
    dtypes = dtypes_to_try(settings.dtype, device)
    log(f'Benchmark on {device} with {len(texts)} chunks of the book ({sum(map(len, texts))} characters); '
        f'precision{"s" if len(dtypes) > 1 else ""}: {", ".join(dtypes)}')
    work_dir = tempfile.mkdtemp(prefix='audiblez-bench-')
    results, verifier = [], None

    def measure(dtype, configs):
        """Load the engine in `dtype` and time each (backend, batch size) of `configs`."""
        nonlocal verifier
        backend = 'auto' if any(b == 'faster' for b, _ in configs) else 'qwen-tts'
        try:
            engine = core.create_engine(choice, dataclasses.replace(settings, backend=backend, device=device,
                                                                    dtype=dtype))
        except Exception as e:  # e.g. out of memory in float32 on a small GPU
            log(f'  {dtype}: the model could not be loaded ({type(e).__name__}: {e})')
            _free_cuda()
            return
        synth = None
        try:
            if verifier is None:
                verifier = core.make_verifier(engine, language, settings, log=log)
            synth = core.Synthesizer(engine, work_dir, settings, verifier=verifier, language=language,
                                     log=lambda m: None, progress=False)
            if getattr(engine, 'fast', None) is None and any(b == 'faster' for b, _ in configs):
                log('  faster-qwen3-tts is not usable here: measuring the standard backend instead')
                configs = [('qwen-tts', 4 if on_cuda else 1) if b == 'faster' else (b, bs) for b, bs in configs]
            for backend, bs in configs:
                use_fast = backend == 'faster'
                try:
                    speed, audio, elapsed, takes = _time_config(engine, texts, bs, use_fast)
                except Exception as e:  # e.g. out of memory with a large batch
                    log(f'  {dtype} {backend} batch {bs}: failed ({type(e).__name__}: {e})')
                    _free_cuda()
                    continue
                checks = [synth.evaluate(w, t)[1] for w, t in zip(takes, texts)]
                failures = sum(not q.ok for q in checks)
                cers = [q.cer for q in checks if q.cer is not None]
                r = dict(dtype=dtype, backend=backend, batch_size=bs, speed=speed, audio=audio, elapsed=elapsed,
                         failures=failures, cer=sum(cers) / len(cers) if cers else None,
                         reasons=sorted({x for q in checks for x in q.reasons})[:3])
                results.append(r)
                log(f'  {dtype} {backend} batch {bs}: {speed:.2f}x real time, {failures}/{len(texts)} takes failed '
                    f'the quality checks' + (f' ({"; ".join(r["reasons"])})' if failures else ''))
        finally:
            synth = engine = None
            _free_cuda()

    try:
        # 1. precision, with the default backend
        default = [('faster', 1)] if has_fast else [('qwen-tts', 4 if on_cuda else 1)]
        for dtype in dtypes:
            measure(dtype, default)
        if not results:
            raise SystemExit('Every configuration failed')
        dtype = max(acceptable(results), key=lambda r: r['speed'])['dtype']
        # 2. backend and batch size with that precision
        done = {(r['backend'], r['batch_size']) for r in results}
        others = [('qwen-tts', bs) for bs in ([1, 4, 8] if on_cuda else []) if ('qwen-tts', bs) not in done]
        if others:
            measure(dtype, others)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    rows = [[r['dtype'], r['backend'], r['batch_size'], f'{r["audio"]:.1f}', f'{r["elapsed"]:.1f}',
             f'{r["speed"]:.2f}x', f'{r["failures"]}/{len(texts)}', '' if r['cer'] is None else f'{r["cer"]:.1%}']
            for r in results]
    print(tabulate(rows, headers=['Precision', 'Backend', 'Batch', 'Audio (s)', 'Compute (s)', 'Speed', 'QA failed',
                                  'Whisper CER']))
    best = max(acceptable(results), key=lambda r: r['speed'])
    rejected = [r for r in results if r not in acceptable(results)]
    if rejected:
        log('Not recommended (quality checks failed more often): '
            + ', '.join(f'{r["dtype"]} {r["backend"]} batch {r["batch_size"]}' for r in rejected))
    n_gpu = 1
    try:
        import torch
        n_gpu = torch.cuda.device_count() if on_cuda else 1
    except Exception:
        pass
    total = best['speed'] * max(1, n_gpu)
    log(f'\nFastest: --backend {best["backend"]} --batch-size {best["batch_size"]} --dtype {best["dtype"]} '
        f'({best["speed"]:.2f}x real time per device'
        + (f', about {total:.2f}x with the {n_gpu} GPUs in parallel' if n_gpu > 1 else '') + ').')
    log(f'A 2-hour episode would take about {2 / total:.1f} h of synthesis (+ Whisper checks and retries).')
    return dict(best, results=results)
