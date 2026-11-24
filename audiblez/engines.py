# -*- coding: utf-8 -*-
"""
Text-to-speech engines.

- KokoroEngine: Kokoro-82M (fast, 9 languages, no Russian).
- QwenEngine:   Qwen3-TTS (Apache-2.0, 10 languages including Russian, expressive, voice cloning / voice design).

Every engine exposes the same interface: `synthesize(list_of_texts, seed) -> list_of_float32_waveforms`.
"""
import glob
import hashlib
import json
import os
import platform
import subprocess
import traceback
from pathlib import Path

import numpy as np

QWEN_LANGUAGES = {'ru': 'russian', 'en': 'english', 'zh': 'chinese', 'ja': 'japanese', 'ko': 'korean', 'de': 'german',
                  'fr': 'french', 'pt': 'portuguese', 'es': 'spanish', 'it': 'italian'}
QWEN_DEFAULT_MODELS = {
    'clone': 'Qwen/Qwen3-TTS-12Hz-1.7B-Base',
    'custom': 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
    'design': 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
}
QWEN_SPEAKERS = ['Vivian', 'Serena', 'Uncle_Fu', 'Dylan', 'Eric', 'Ryan', 'Aiden', 'Ono_Anna', 'Sohee']
QWEN_FRAME_RATE = 12.5  # codec frames per second (24000 Hz / 1920 samples)

KOKORO_LANGUAGES = {'a': 'en', 'b': 'en', 'e': 'es', 'f': 'fr', 'h': 'hi', 'i': 'it', 'j': 'ja', 'p': 'pt', 'z': 'zh'}


def resolve_device(device='auto'):
    import torch
    if device and device != 'auto':
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def resolve_dtype(dtype, device):
    import torch
    if dtype and dtype != 'auto':
        return {'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16, 'float16': torch.float16, 'fp16': torch.float16,
                'float32': torch.float32, 'fp32': torch.float32}[dtype]
    # Native bf16 only (Ampere / RTX 30xx and newer). is_bf16_supported() also reports *emulated* bf16 on older GPUs
    # such as the T4 (Kaggle, Colab), which is slow there.
    if device.startswith('cuda') and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    # float16 can overflow in this model (NaN -> noise); float32 is the safe default on older GPUs (e.g. T4),
    # Apple MPS and CPU. Force --dtype float16 to trade safety for speed/VRAM.
    return torch.float32


def seed_everything(seed):
    import torch
    seed = int(seed) % (2 ** 31)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Engine:
    name = 'base'
    sample_rate = 24000
    native_speed = False  # True when the engine handles the speaking speed itself
    stochastic = False  # True when re-generating can give a different (better) result
    default_max_chars = 300
    language = 'en'

    def fingerprint(self) -> str:
        raise NotImplementedError

    def synthesize(self, texts, seed=0):
        raise NotImplementedError

    def describe(self):
        return self.name


# ---------------------------------------------------------------------------------------------------------------------
# Kokoro
# ---------------------------------------------------------------------------------------------------------------------

def set_espeak_library():
    """Find the espeak-ng library (needed by Kokoro's non-English phonemizers)."""
    try:
        library = None
        if os.environ.get('ESPEAK_LIBRARY'):
            library = os.environ['ESPEAK_LIBRARY']
        elif platform.system() == 'Darwin':
            try:
                cellar = Path(subprocess.check_output(['brew', '--cellar'], text=True).strip())
                library = next(iter(glob.glob(str(cellar / 'espeak-ng' / '*' / 'lib' / '*.dylib'))), None)
            except (subprocess.CalledProcessError, FileNotFoundError):
                library = None
            if not library:
                library = next(iter(glob.glob('/opt/homebrew/lib/libespeak-ng*.dylib') +
                                    glob.glob('/usr/local/lib/libespeak-ng*.dylib')), None)
        elif platform.system() == 'Linux':
            candidates = (glob.glob('/usr/lib/*/libespeak-ng.so*') + glob.glob('/usr/lib/libespeak-ng.so*') +
                          glob.glob('/usr/local/lib/libespeak-ng.so*') + glob.glob('/usr/lib64/libespeak-ng.so*'))
            library = next(iter(sorted(candidates)), None)
        elif platform.system() == 'Windows':
            library = next(iter(glob.glob('C:\\Program Files*\\eSpeak NG\\libespeak-ng.dll')), None)
        if not library:
            raise RuntimeError('espeak-ng library not found')
        print('Using espeak library:', library)
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(library)
    except Exception:
        traceback.print_exc()
        print('Error finding espeak-ng library. Probably you have not installed espeak-ng.')
        print('On Mac: brew install espeak-ng  |  On Linux: sudo apt install espeak-ng  |  '
              'or set the ESPEAK_LIBRARY environment variable.')


class KokoroEngine(Engine):
    name = 'kokoro'
    native_speed = True
    default_max_chars = 400

    def __init__(self, voice, speed=1.0, device='auto'):
        from kokoro import KPipeline
        self.voice = voice
        self.speed = float(speed)
        self.lang_code = voice[0]
        self.language = KOKORO_LANGUAGES.get(self.lang_code, 'en')
        if self.lang_code not in 'ab':
            set_espeak_library()
        dev = resolve_device(device)
        self.pipeline = KPipeline(lang_code=self.lang_code, repo_id='hexgrad/Kokoro-82M',
                                  device=dev if dev in ('cuda', 'cpu', 'mps') else None)

    def fingerprint(self):
        return f'kokoro-82M|{self.voice}|{self.speed}'

    def synthesize(self, texts, seed=0):
        out = []
        for text in texts:
            segments = []
            for result in self.pipeline(text, voice=self.voice, speed=self.speed, split_pattern=None):
                audio = getattr(result, 'audio', None)
                if audio is None:
                    continue
                if hasattr(audio, 'detach'):
                    audio = audio.detach().cpu().numpy()
                segments.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            out.append(np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32))
        return out

    def describe(self):
        return f'Kokoro-82M, voice {self.voice}'


# ---------------------------------------------------------------------------------------------------------------------
# Qwen3-TTS
# ---------------------------------------------------------------------------------------------------------------------

class VoiceProfile:
    """A reusable cloned voice: a short clean reference recording (ideally 6-15 s) and its exact transcript."""

    def __init__(self, audio_path, text, language=None, meta=None, path=None):
        self.audio_path = Path(audio_path)
        self.text = text.strip()
        self.language = language
        self.meta = meta or {}
        self.path = path

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_dir():
            meta = {}
            if (path / 'profile.json').exists():
                meta = json.loads((path / 'profile.json').read_text(encoding='utf-8'))
            audio = path / meta['audio'] if meta.get('audio') else next(
                (path / n for n in ('reference.flac', 'reference.wav', 'reference.mp3') if (path / n).exists()),
                path / 'reference.wav')
            text_file = path / 'reference.txt'
            text = meta.get('text') or (text_file.read_text(encoding='utf-8') if text_file.exists() else '')
            if not audio.exists():
                raise FileNotFoundError(f'Voice profile {path}: missing reference audio {audio.name}')
            if not text.strip():
                raise ValueError(f'Voice profile {path}: missing reference.txt (exact transcript of the reference)')
            return cls(audio, text, meta.get('language'), meta, path)
        raise FileNotFoundError(f'Voice profile folder not found: {path}')

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'reference.wav'
        if self.audio_path.resolve() != target.resolve():
            import shutil
            shutil.copyfile(self.audio_path, target)
        (folder / 'reference.txt').write_text(self.text + '\n', encoding='utf-8')
        meta = dict(self.meta, audio='reference.wav', text=self.text, language=self.language)
        (folder / 'profile.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        self.audio_path, self.path = target, folder
        return folder


def bundled_voices_dir():
    return Path(__file__).resolve().parent / 'voices'


def find_voice_profile(name_or_path):
    """Accept a folder path or the name of a voice bundled with audiblez (audiblez/voices/<name>)."""
    p = Path(name_or_path).expanduser()
    if p.is_dir():
        return VoiceProfile.load(p)
    bundled = bundled_voices_dir() / str(name_or_path)
    if bundled.is_dir():
        return VoiceProfile.load(bundled)
    raise FileNotFoundError(f'Voice profile "{name_or_path}" not found (neither a folder nor a bundled voice). '
                            f'Bundled voices: {", ".join(list_bundled_voices()) or "none"}')


def list_bundled_voices():
    d = bundled_voices_dir()
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if (p / 'reference.txt').exists() or (p / 'profile.json').exists())


def load_qwen_model(model_id, device='auto', dtype='auto'):
    import logging
    logging.getLogger('sox').setLevel(logging.ERROR)  # the optional SoX binary is not used by audiblez
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        raise SystemExit('Qwen3-TTS is not installed. Install it with:  pip install "audiblez[qwen,asr]"  '
                         '(or "audiblez[fast,asr]" on an NVIDIA GPU)') from e
    try:
        import transformers
        transformers.logging.set_verbosity_error()  # silences the per-sentence "Setting pad_token_id" notice
    except Exception:
        pass
    device = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, device)
    kwargs = dict(device_map=device if device != 'cuda' else 'cuda:0', dtype=torch_dtype)
    if device.startswith('cuda'):
        import importlib.util
        has_flash = importlib.util.find_spec('flash_attn') is not None
        kwargs['attn_implementation'] = 'flash_attention_2' if has_flash else 'sdpa'
    print(f'Loading {model_id} on {device} ({str(torch_dtype).replace("torch.", "")})...')
    return Qwen3TTSModel.from_pretrained(model_id, **kwargs)


def load_reference_audio(path, target_sr=24000):
    """Load a reference recording as mono float32 at target_sr."""
    import soundfile as sf
    try:
        audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
    except Exception:
        import librosa
        audio, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return np.asarray(audio, dtype=np.float32), sr


def qwen_fingerprint(model_id, mode, language, speaker=None, instruct='', x_vector_only=False, gen_kwargs=None,
                     voice_profile=None):
    """Identity of the generated audio (used as cache key). Computable without loading the model."""
    parts = dict(model=model_id, mode=mode, lang=(language or '')[:2].lower(), speaker=speaker, instruct=instruct or '',
                 xvec=bool(x_vector_only), gen=sorted((gen_kwargs or {}).items()))
    if voice_profile is not None:
        parts['ref'] = file_sha1(voice_profile.audio_path)
        parts['ref_text'] = voice_profile.text
    return 'qwen|' + json.dumps(parts, sort_keys=True, ensure_ascii=False)


def faster_backend_available():
    import importlib.util
    return importlib.util.find_spec('faster_qwen3_tts') is not None


def load_faster_model(model_id, device, dtype):
    """CUDA-graph accelerated Qwen3-TTS (github.com/andimarafioti/faster-qwen3-tts)."""
    from faster_qwen3_tts import FasterQwen3TTS
    import importlib.util
    torch_dtype = resolve_dtype(dtype, device)
    attn = 'flash_attention_2' if importlib.util.find_spec('flash_attn') is not None else 'sdpa'
    print(f'Loading {model_id} with faster-qwen3-tts (CUDA graphs) on {device} '
          f'({str(torch_dtype).replace("torch.", "")})...')
    return FasterQwen3TTS.from_pretrained(model_id, device=device, dtype=torch_dtype, attn_implementation=attn)


class QwenEngine(Engine):
    name = 'qwen'
    stochastic = True
    default_max_chars = 300

    def __init__(self, mode='clone', language='ru', model_id=None, device='auto', dtype='auto', voice_profile=None,
                 speaker=None, instruct=None, x_vector_only=False, temperature=None, top_p=None, top_k=None,
                 repetition_penalty=None, model=None, backend='auto'):
        if mode not in QWEN_DEFAULT_MODELS:
            raise ValueError(f'Unknown Qwen mode {mode}, expected one of {list(QWEN_DEFAULT_MODELS)}')
        self.mode = mode
        self.language = (language or 'ru')[:2].lower()
        if self.language not in QWEN_LANGUAGES:
            raise ValueError(f'Qwen3-TTS does not support language "{language}". '
                             f'Supported: {", ".join(sorted(QWEN_LANGUAGES))}')
        self.qwen_language = QWEN_LANGUAGES[self.language]
        self.model_id = model_id or QWEN_DEFAULT_MODELS[mode]
        self.voice_profile = voice_profile
        self.speaker = speaker
        self.instruct = instruct or ''
        self.x_vector_only = x_vector_only
        self.gen_kwargs = {k: v for k, v in dict(temperature=temperature, top_p=top_p, top_k=top_k,
                                                 repetition_penalty=repetition_penalty).items() if v is not None}
        if mode == 'clone' and voice_profile is None:
            raise ValueError('Qwen clone mode needs a voice profile (reference audio + transcript)')
        if mode == 'custom' and not speaker:
            raise ValueError(f'Qwen custom-voice mode needs a speaker: {", ".join(QWEN_SPEAKERS)}')
        if mode == 'design' and not self.instruct:
            raise ValueError('Qwen voice-design mode needs an --instruct voice description')
        self.device = resolve_device(device) if model is None else str(getattr(model, 'device', 'cpu'))
        self.fast = None
        if model is None and backend in ('auto', 'faster') and self.device.startswith('cuda'):
            if faster_backend_available():
                try:
                    self.fast = load_faster_model(self.model_id, self.device, dtype)
                except Exception as e:
                    if backend == 'faster':
                        raise
                    print(f'\033[93mfaster-qwen3-tts could not be loaded ({e}); using the standard backend.\033[0m')
            elif backend == 'faster':
                raise RuntimeError('--backend faster needs: pip install faster-qwen3-tts (see the README)')
        elif backend == 'faster' and model is None:
            raise RuntimeError('--backend faster needs a CUDA GPU')
        self.model = self.fast.model if self.fast is not None else (model or load_qwen_model(self.model_id, self.device,
                                                                                               dtype))
        self.supports_batch = self.fast is None  # the CUDA-graph path generates one stream at a time
        self.prompt = None
        self._ref_embedding = None
        self._cohort = None
        if mode == 'clone':
            ref_audio, ref_sr = load_reference_audio(voice_profile.audio_path)
            duration = len(ref_audio) / ref_sr
            if duration > 20:
                print(f'\033[93mWarning: the reference audio is {duration:.0f}s long. Cloning is most reliable with '
                      f'6-15 seconds of clean speech; long references increase the failure rate.\033[0m')
            self.prompt = self.model.create_voice_clone_prompt(
                ref_audio=(ref_audio, ref_sr), ref_text=voice_profile.text, x_vector_only_mode=x_vector_only)
            try:
                self._ref_embedding = self.prompt[0].ref_spk_embedding
            except Exception:
                self._ref_embedding = None

    @property
    def backend(self):
        return 'faster-qwen3-tts' if self.fast is not None else 'qwen-tts'

    def fingerprint(self):
        return qwen_fingerprint(self.model_id, self.mode, self.language, self.speaker, self.instruct,
                                self.x_vector_only, self.gen_kwargs, self.voice_profile)

    def token_budget(self, text):
        """Upper bound of codec tokens for a text: generous for slow dramatic reading, but stops runaway loops."""
        from audiblez.text import count_letters, language_profile
        slowest = 0.4 * language_profile(self.language)['rate']  # letters per second of a very slow reading
        seconds = count_letters(text) / slowest + 5.0
        return int(min(2048, max(96, seconds * QWEN_FRAME_RATE)))

    def synthesize(self, texts, seed=0):
        texts = list(texts)
        seed_everything(seed)
        if self.fast is not None:
            try:
                return self._synthesize_fast(texts)
            except Exception as e:  # never lose a long job because of the accelerated path
                print(f'\033[93mfaster-qwen3-tts failed ({type(e).__name__}: {e}); switching to the standard '
                      f'backend for the rest of the run.\033[0m')
                self.fast = None
                self.supports_batch = True
                seed_everything(seed)
        kwargs = dict(self.gen_kwargs, max_new_tokens=max(self.token_budget(t) for t in texts))
        languages = [self.qwen_language] * len(texts)
        if self.mode == 'clone':
            wavs, sr = self.model.generate_voice_clone(text=texts, language=languages, voice_clone_prompt=self.prompt,
                                                       **kwargs)
        elif self.mode == 'custom':
            wavs, sr = self.model.generate_custom_voice(text=texts, language=languages, speaker=self.speaker,
                                                        instruct=self.instruct or None, **kwargs)
        else:
            wavs, sr = self.model.generate_voice_design(text=texts, language=languages, instruct=self.instruct,
                                                        **kwargs)
        return self._to_output(wavs, sr)

    def _synthesize_fast(self, texts):
        out = []
        for text in texts:
            kwargs = dict(self.gen_kwargs, max_new_tokens=self.token_budget(text))
            if self.mode == 'clone':
                audio, sr = self.fast.generate_voice_clone(text=text, language=self.qwen_language,
                                                           ref_text=self.voice_profile.text,
                                                           voice_clone_prompt=self.prompt, **kwargs)
            elif self.mode == 'custom':
                audio, sr = self.fast.generate_custom_voice(text=text, speaker=self.speaker,
                                                            language=self.qwen_language,
                                                            instruct=self.instruct or None, **kwargs)
            else:
                audio, sr = self.fast.generate_voice_design(text=text, instruct=self.instruct,
                                                            language=self.qwen_language, **kwargs)
            parts = [np.asarray(a.detach().float().cpu().numpy() if hasattr(a, 'detach') else a,
                                dtype=np.float32).reshape(-1) for a in (audio if isinstance(audio, (list, tuple))
                                                                         else [audio])]
            out.append(self._to_output([np.concatenate(parts) if parts else np.zeros(0, np.float32)], sr)[0])
        return out

    def _to_output(self, wavs, sr):
        if sr != self.sample_rate:
            import librosa
            wavs = [librosa.resample(np.asarray(w, dtype=np.float32), orig_sr=sr, target_sr=self.sample_rate)
                    for w in wavs]
        return [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs]

    def _speaker_embedding(self, wav):
        import torch
        with torch.no_grad():
            emb = self.model.model.extract_speaker_embedding(audio=np.ascontiguousarray(wav, dtype=np.float32),
                                                             sr=24000)
        return emb.flatten().float()

    def _cohort_mean(self):
        """
        Mean speaker embedding of the bundled narrators (computed once). Raw cosine similarities of this speaker
        encoder are all high: 0.92-0.99 between *different* voices, 0.97-0.99 for correct takes. Subtracting the
        cohort mean makes them discriminative: on 48 Russian takes of ru_male_narrator the centered similarity is
        0.38-0.81 with the narrator, -0.67..-0.08 with the female voices and up to 0.47 with the other male voices;
        Japanese and Korean takes of ja_male_narrator / ko_male_narrator measured 0.41-0.81 and 0.45-0.77.
        """
        if self._cohort is None:
            import torch
            embs = []
            for name in list_bundled_voices():
                try:
                    wav, _ = load_reference_audio(find_voice_profile(name).audio_path)
                    embs.append(self._speaker_embedding(wav).cpu())
                except Exception:
                    continue
            self._cohort = torch.stack(embs).mean(0) if len(embs) >= 3 else False
        return None if self._cohort is False else self._cohort

    def speaker_similarity(self, wav):
        """
        Cohort-centered cosine similarity between the take and the reference narrator, with the model's own
        speaker encoder (see _cohort_mean). Catches takes where the clone drifted to another voice or gender.
        """
        if self._ref_embedding is None or len(wav) < self.sample_rate // 2:
            return None
        import torch
        mean = self._cohort_mean()
        if mean is None:
            return None
        emb = self._speaker_embedding(wav)
        ref = self._ref_embedding.flatten().float().to(emb.device)
        mean = mean.to(emb.device)
        return float(torch.nn.functional.cosine_similarity(emb - mean, ref - mean, dim=0))

    def hit_token_limit(self, wav, text):
        """True if the generation was most likely cut by max_new_tokens (runaway / looping output)."""
        return len(wav) / self.sample_rate >= self.token_budget(text) / QWEN_FRAME_RATE - 0.5

    def describe(self):
        who = {'clone': f'cloned voice {getattr(self.voice_profile, "path", None) or self.voice_profile.audio_path}',
               'custom': f'speaker {self.speaker}', 'design': f'designed voice "{self.instruct}"'}[self.mode]
        return f'{self.model_id} ({self.qwen_language}, {self.backend} on {self.device}), {who}'
