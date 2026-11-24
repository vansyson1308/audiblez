# -*- coding: utf-8 -*-
"""
Parallel synthesis on several GPUs (e.g. Kaggle's free "GPU T4 x2").

The chunks still to synthesize are split into balanced shards (by number of letters, longest first). Each shard is
synthesized by its own worker process, pinned to one GPU with CUDA_VISIBLE_DEVICES, which loads its own copy of the
models and runs the usual synthesis + quality checks + retries, writing into the same cache folder as a single-GPU
run. The coordinator then assembles the chapters from the cache. Since everything goes through the cache, a crashed
worker only costs its unfinished chunks (the coordinator synthesizes them itself), and an interrupted run resumes.

    python -m audiblez.parallel JOB.json     (started by audiblez itself)
"""
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

MIN_CHUNKS_PER_WORKER = 8  # below this, loading one more copy of the models is not worth it


def visible_gpus():
    """Physical ids of the CUDA GPUs visible to this process (honours CUDA_VISIBLE_DEVICES)."""
    try:
        import torch
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return []
    env = os.environ.get('CUDA_VISIBLE_DEVICES')
    if env is not None and env.strip():
        ids = [x.strip() for x in env.split(',') if x.strip()]
        return ids[:count]
    return [str(i) for i in range(count)]


def plan_workers(settings, engine, n_chunks):
    """Number of parallel GPU workers to use (1 = synthesize in this process)."""
    if not getattr(engine, 'stochastic', False) or not str(getattr(engine, 'device', '')).startswith('cuda'):
        return 1
    if str(getattr(engine, 'device', '')) not in ('cuda', 'cuda:0'):
        return 1  # an explicit device such as cuda:1 was requested
    wanted = str(settings.gpus or 'auto').strip().lower()
    gpus = visible_gpus()
    if wanted == 'auto':
        n = len(gpus)
    else:
        try:
            n = int(wanted)
        except ValueError:
            raise SystemExit(f'--gpus must be "auto" or a number, not "{settings.gpus}"')
        if n > len(gpus):
            print(f'\033[93m--gpus {n}: only {len(gpus)} CUDA GPU(s) visible, using {len(gpus)}.\033[0m')
            n = len(gpus)
    return max(1, min(n, n_chunks // MIN_CHUNKS_PER_WORKER))


def split_balanced(todo: dict, n):
    """Split {text: label} into n shards of about the same total length (longest-processing-time-first)."""
    from audiblez.text import count_letters
    shards = [dict() for _ in range(n)]
    loads = [0] * n
    for text in sorted(todo, key=lambda t: (-count_letters(t), t)):
        i = min(range(n), key=lambda k: (loads[k], k))
        shards[i][text] = todo[text]
        loads[i] += count_letters(text) + 20  # + a constant: every chunk also has a fixed cost
    return shards


def _settings_dict(settings):
    d = asdict(settings)
    d['formats'] = list(d.get('formats') or ())
    return d


def write_jobs(gpus, todo: dict, choice, settings, work_dir, deadline=None):
    """One job file per GPU with its balanced share of the {text: label} chunks. Returns [(gpu, job_path)]."""
    jobs_dir = Path(work_dir) / 'jobs'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    shards = split_balanced(todo, len(gpus))
    print(f'Parallel synthesis on {len(gpus)} GPUs: ' +
          ', '.join(f'GPU {g}: {len(s)} chunks' for g, s in zip(gpus, shards)))
    jobs = []
    for i, (gpu, shard) in enumerate(zip(gpus, shards)):
        job = dict(worker=i, gpu=gpu, choice=choice, settings=_settings_dict(settings), work_dir=str(work_dir),
                   deadline=deadline, todo=[[t, label] for t, label in shard.items()])
        job_path = jobs_dir / f'worker_{i}.json'
        job_path.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
        jobs.append((gpu, job_path))
    return jobs


def run_workers(n, todo: dict, choice, settings, work_dir, deadline=None):
    """Synthesize the {text: label} chunks with n worker processes, one per GPU. Blocks until they all exit."""
    jobs_dir = Path(work_dir) / 'jobs'
    procs = []
    for i, (gpu, job_path) in enumerate(write_jobs(visible_gpus()[:n], todo, choice, settings, work_dir, deadline)):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        proc = subprocess.Popen([sys.executable, '-m', 'audiblez.parallel', str(job_path)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                errors='replace', bufsize=1)
        log = open(jobs_dir / f'worker_{i}.log', 'a', encoding='utf-8')
        thread = threading.Thread(target=_relay, args=(proc, f'[GPU {gpu}] ', log), daemon=True)
        thread.start()
        procs.append((gpu, proc, thread, log))
    failed = []
    try:
        for gpu, proc, thread, log in procs:
            code = proc.wait()
            thread.join()
            log.close()
            if code != 0:
                failed.append(gpu)
    except KeyboardInterrupt:
        for _, proc, _, _ in procs:
            proc.terminate()
        raise
    if failed:
        print(f'\033[93mWorker(s) on GPU {", ".join(failed)} stopped with an error (log in {jobs_dir}); their '
              f'remaining chunks are synthesized by the main process.\033[0m')
    return not failed


def _relay(proc, prefix, log):
    for line in proc.stdout:
        line = line.rstrip('\n')
        print(prefix + line, flush=True)
        log.write(line + '\n')


def worker_main(job_path):
    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    from types import SimpleNamespace
    from audiblez.core import Settings, Synthesizer, create_engine, make_verifier
    fields = Settings.__dataclass_fields__
    settings = Settings(**{k: v for k, v in job['settings'].items() if k in fields})
    settings.formats = tuple(settings.formats)
    settings.gpus = '1'
    if str(settings.device).startswith('cuda') or settings.device == 'auto':
        settings.device = 'cuda'  # the GPU given by CUDA_VISIBLE_DEVICES
    todo = {t: label for t, label in job['todo']}
    factory = os.environ.get('AUDIBLEZ_WORKER_ENGINE')  # "module:function(choice, settings)", used by the tests
    if factory:
        import importlib
        module, _, func = factory.partition(':')
        engine = getattr(importlib.import_module(module), func)(job['choice'], settings)
    else:
        engine = create_engine(job['choice'], settings)
    verifier = make_verifier(settings, engine, job['choice']['language'])
    stats = SimpleNamespace(total_chars=sum(len(t) for t in todo), processed_chars=0, synth_chars=0,
                            synth_seconds=0.0, audio_seconds=0.0, chars_per_sec=400, progress=0, eta='?')
    synth = Synthesizer(engine, Path(job['work_dir']), settings, stats, verifier=verifier,
                        worker_id=f'w{job["worker"]}')
    t0 = time.time()
    complete = synth.synthesize_texts(todo, job.get('deadline'))
    speed = stats.audio_seconds / stats.synth_seconds if stats.synth_seconds else 0.0
    print(f'Worker done in {time.time() - t0:.0f}s: {stats.audio_seconds / 60:.1f} min of audio, {speed:.2f}x real '
          f'time' + ('' if complete else ' (stopped by the time limit)'))


if __name__ == '__main__':
    from audiblez import ensure_utf8_stdio
    ensure_utf8_stdio()
    worker_main(sys.argv[1])
