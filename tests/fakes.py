"""Test doubles. FakeSpeechEngine makes speech-shaped audio (syllable-rate envelope on band-passed noise)
so the full audio pipeline can be exercised without downloading a neural TTS model."""
from __future__ import annotations

import hashlib

import numpy as np

from core.tts import TTSEngine


class FakeSpeechEngine(TTSEngine):
    name = "fake"
    sample_rate = 24000

    def __init__(self, native_tempo: float = 0.88, seconds_per_word: float = 0.12) -> None:
        self.native_tempo = native_tempo
        self.spw = seconds_per_word

    def synthesize(self, text: str) -> np.ndarray:
        words = max(1, len(text.split()))
        n = int(words * self.spw * self.sample_rate)
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        noise = np.convolve(rng.normal(0, 1, n), np.ones(6) / 6, mode="same")
        t = np.arange(n) / self.sample_rate
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t) ** 2
        return (0.3 * noise * env / (np.abs(noise).max() or 1)).astype(np.float32)
