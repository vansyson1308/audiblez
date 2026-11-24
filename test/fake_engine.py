"""A fake stochastic TTS engine importable by worker processes (used with Settings.engine_factory)."""
import os

import numpy as np

from audiblez.text import count_letters, language_profile

SR = 24000


class FakeEngine:
    """'Speech' at the typical rate of the language. Texts containing 'Сбой'/'失敗' hallucinate on the first try."""
    name = 'fake'
    sample_rate = SR
    native_speed = False
    stochastic = True
    default_max_chars = 300
    supports_batch = True

    def __init__(self, language='ru', device='cpu'):
        self.language = language
        self.device = device
        self.calls = []
        self.seen = {}

    def fingerprint(self):
        return 'fake-engine-v1'

    def describe(self):
        return f'fake engine on {self.device} (pid {os.getpid()})'

    def synthesize(self, texts, seed=0):
        self.calls.append(list(texts))
        out = []
        rate = language_profile(self.language)['rate']
        for t in texts:
            self.seen[t] = self.seen.get(t, 0) + 1
            seconds = count_letters(t) / rate
            if ('Сбой' in t or '失敗' in t) and self.seen[t] == 1:
                seconds *= 6  # hallucination on the first attempt
            n = int(seconds * SR)
            rng = np.random.default_rng(abs(hash(t)) % 2 ** 32)
            speech = (0.2 * np.sin(2 * np.pi * 180 * np.arange(n) / SR) * (0.6 + 0.4 * rng.random(n))).astype(np.float32)
            out.append(np.concatenate([np.zeros(int(0.3 * SR), np.float32), speech, np.zeros(int(0.4 * SR), np.float32)]))
        return out


def create_engine(choice, settings):
    return FakeEngine(choice.get('language', 'ru'), settings.device)


def fingerprint(choice, settings):
    return 'fake', 'fake-engine-v1'
