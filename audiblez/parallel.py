# -*- coding: utf-8 -*-
"""
Parallel synthesis on several devices (e.g. the two T4 GPUs of a Kaggle notebook): one worker process per device,
each with its own model, pulling batches of chunks from a shared queue and writing them to the shared cache.

Workers are started with the 'spawn' method (required for CUDA). If a worker cannot start or crashes, the other
ones keep going, and whatever is still missing at the end is synthesized by the caller.
"""
import dataclasses
import multiprocessing as mp
import os
import queue
import time
import traceback

from audiblez.text import count_letters


def resolve_devices(gpus, engine_name, device='auto'):
    """
    '--gpus' value -> list of devices for parallel workers, or None for a single in-process engine.
    'auto': every visible CUDA GPU when there are at least two (Qwen3-TTS only).
    """
    if engine_name != 'qwen':
        return None
    spec = str(gpus or 'auto').strip().lower()
    if spec in ('off', 'none', 'single', 'no', 'false'):
        return None
    if spec == 'auto':
        if device not in ('auto', 'cuda'):
            return None
        try:
            import torch
            n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            n = 0
        return [f'cuda:{i}' for i in range(n)] if n > 1 else None
    if spec == 'all':
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return [f'cuda:{i}' for i in range(n)] or None
    devices = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        devices.append(f'cuda:{part}' if part.isdigit() else part)
    return devices or None


def _worker(wid, device, n_workers, choice, settings, work_dir, language, task_q, result_q):
    """Worker process: load the engine on `device`, then synthesize batches until the sentinel (None)."""
    try:
        import torch
        if device.startswith('cpu'):
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // n_workers))
        elif device.startswith('cuda'):
            torch.cuda.set_device(torch.device(device))
        from audiblez import core
        settings = dataclasses.replace(settings, device=device)
        engine = core.create_engine(choice, settings)
        verifier = core.make_verifier(engine, language, settings, log=lambda m: None)
        synth = core.Synthesizer(engine, work_dir, settings, verifier=verifier, language=language,
                                 log=lambda m: result_q.put(('log', wid, m)), progress=False)
        result_q.put(('ready', wid, engine.describe()))
    except BaseException as e:  # noqa: B036 - report every failure to the parent
        result_q.put(('error', wid, f'{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}'))
        return
    while True:
        item = task_q.get()
        if item is None:
            break
        label, texts = item
        before = len(synth.report)
        s0 = (synth.stats.synth_seconds, synth.stats.synth_audio)
        try:
            synth.synthesize(texts, label)
        except BaseException as e:  # noqa: B036
            result_q.put(('failed', wid, dict(error=f'{type(e).__name__}: {e}', texts=texts,
                                                 trace=traceback.format_exc()[-2000:])))
            break  # a CUDA error usually leaves the process unusable: let the others (and the caller) finish
        result_q.put(('done', wid, dict(chars=sum(len(t) for t in texts), letters=sum(map(count_letters, texts)),
                                           seconds=synth.stats.synth_seconds - s0[0],
                                           audio=synth.stats.synth_audio - s0[1],
                                           failures=synth.report[before:])))
    result_q.put(('exit', wid, None))


def synthesize_parallel(devices, choice, settings, work_dir, language, jobs, total_chars, already_done=0,
                        log=print, post_event=None, stats=None):
    """
    jobs: list of (label, texts) still to synthesize. Returns the QA report entries of chunks that failed.
    """
    from audiblez.core import strfdelta
    bs = int(settings.batch_size or 0) or 4
    tasks = []
    for label, texts in jobs:
        ordered = sorted(dict.fromkeys(texts), key=len, reverse=True)
        tasks += [(label, ordered[i:i + bs]) for i in range(0, len(ordered), bs)]
    tasks.sort(key=lambda t: -sum(len(x) for x in t[1]))  # longest batches first: better load balance at the end
    if not tasks:
        return []
    ctx = mp.get_context('spawn')
    task_q, result_q = ctx.Queue(), ctx.Queue()
    for t in tasks:
        task_q.put(t)
    for _ in devices:
        task_q.put(None)
    # Worker ids are unique even when a device hosts several workers ("--gpus 0,0" on a large GPU)
    procs = {f'{d}#{i}': ctx.Process(target=_worker, args=(f'{d}#{i}', d, len(devices), choice, settings,
                                                           str(work_dir), language, task_q, result_q), daemon=True)
             for i, d in enumerate(devices)}
    log(f'Starting {len(devices)} parallel workers on {", ".join(devices)} ({len(tasks)} batches of up to {bs} chunks)')
    for p in procs.values():
        p.start()
    finished, report = set(), []
    done_chars, synth_audio, t_start = already_done, 0.0, time.time()
    while len(finished) < len(procs):
        try:
            kind, device, payload = result_q.get(timeout=10)
        except queue.Empty:
            for d, p in procs.items():
                if d not in finished and not p.is_alive():
                    log(f'\033[91mWorker {d} stopped unexpectedly (exit code {p.exitcode}).\033[0m')
                    finished.add(d)
            continue
        if kind == 'ready':
            log(f'Worker {device} ready: {payload}')
        elif kind == 'log':
            log(f'[{device}] {payload}')
        elif kind == 'error':
            log(f'\033[91mWorker {device} could not start: {payload}\033[0m')
            finished.add(device)
        elif kind == 'failed':
            log(f'\033[91mWorker {device} failed: {payload["error"]}\n{payload["trace"]}\033[0m')
        elif kind == 'exit':
            finished.add(device)
        elif kind == 'done':
            report += payload['failures']
            done_chars += payload['chars']
            synth_audio += payload['audio']
            elapsed = time.time() - t_start
            rate = (done_chars - already_done) / max(elapsed, 1e-6)
            eta = strfdelta((total_chars - done_chars) / max(rate, 1e-6))
            pct = min(100, done_chars * 100 // max(1, total_chars))
            speed = synth_audio / max(elapsed, 1e-6)
            log(f'Progress: {pct}%  |  {rate:.{0 if rate >= 10 else 1}f} chars/s  |  {speed:.2f}x real time '
                f'(all workers)  |  ETA {eta}')
            if stats is not None:
                stats.processed_chars, stats.progress, stats.eta = done_chars, pct, eta
                if post_event:
                    post_event('CORE_PROGRESS', stats=stats)
    for p in procs.values():
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    return report
