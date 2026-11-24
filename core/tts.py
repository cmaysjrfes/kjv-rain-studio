"""Text-to-speech engines (Kokoro-82M, Piper, Chatterbox, ElevenLabs) behind one interface, with a disk cache."""
from __future__ import annotations

import hashlib
import io
import logging
import random
import time
import wave
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import requests

from core.config import Settings

log = logging.getLogger(__name__)


class TTSError(RuntimeError):
    pass


class TTSEngine(ABC):
    name: str = "tts"
    sample_rate: int = 24000
    #: tempo the engine applies natively; 1.0 means Agent 2 must apply `atempo` afterwards
    native_tempo: float = 1.0

    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Return mono float32 audio in [-1, 1] at `self.sample_rate`."""

    @property
    def cache_key(self) -> str:
        return f"{self.name}:{self.sample_rate}:{self.native_tempo}"


class KokoroEngine(TTSEngine):
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, voice: str = "am_michael", speed: float = 0.88) -> None:
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise TTSError("Kokoro not installed: pip install kokoro soundfile (and espeak-ng)") from exc
        self.voice = voice
        self.native_tempo = speed
        self.pipeline = KPipeline(lang_code="b" if voice.startswith("b") else "a")

    @property
    def cache_key(self) -> str:
        return f"kokoro:{self.voice}:{self.native_tempo}"

    def synthesize(self, text: str) -> np.ndarray:
        parts = []
        for _, _, audio in self.pipeline(text, voice=self.voice, speed=self.native_tempo):
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            parts.append(arr.astype(np.float32))
        if not parts:
            raise TTSError(f"Kokoro produced no audio for {text[:60]!r}")
        return np.concatenate(parts)


class PiperEngine(TTSEngine):
    name = "piper"

    def __init__(self, model_path: Path, speed: float = 0.88) -> None:
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise TTSError("Piper not installed: pip install piper-tts") from exc
        if not Path(model_path).exists():
            raise TTSError(f"Piper voice model not found: {model_path} (download .onnx + .onnx.json)")
        self.model_path = Path(model_path)
        self.voice = PiperVoice.load(str(model_path))
        self.sample_rate = int(self.voice.config.sample_rate)
        self.native_tempo = speed

    @property
    def cache_key(self) -> str:
        return f"piper:{self.model_path.name}:{self.native_tempo}"

    def synthesize(self, text: str) -> np.ndarray:
        length_scale = 1.0 / self.native_tempo
        try:  # piper-tts >= 1.3
            from piper import SynthesisConfig

            chunks = self.voice.synthesize(text, syn_config=SynthesisConfig(length_scale=length_scale))
            arrs = [c.audio_float_array.astype(np.float32) for c in chunks]
            if arrs:
                return np.concatenate(arrs)
        except ImportError:
            pass
        buf = io.BytesIO()  # piper-tts 1.2 API
        with wave.open(buf, "wb") as wf:
            self.voice.synthesize(text, wf, length_scale=length_scale)
        buf.seek(0)
        return read_wav(buf)[0]


class ChatterboxEngine(TTSEngine):
    name = "chatterbox"

    def __init__(self, voice_prompt: str = "", exaggeration: float = 0.3, cfg_weight: float = 0.3) -> None:
        try:
            import torch
            from chatterbox.tts import ChatterboxTTS
        except ImportError as exc:
            raise TTSError("Chatterbox not installed: pip install chatterbox-tts") from exc
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.sample_rate = int(self.model.sr)
        self.voice_prompt = voice_prompt or None
        self.exaggeration, self.cfg_weight = exaggeration, cfg_weight
        self.native_tempo = 1.0  # tempo applied by Agent 2 with atempo

    @property
    def cache_key(self) -> str:
        return f"chatterbox:{self.voice_prompt}:{self.exaggeration}:{self.cfg_weight}"

    def synthesize(self, text: str) -> np.ndarray:
        wav = self.model.generate(text, audio_prompt_path=self.voice_prompt, exaggeration=self.exaggeration,
                                  cfg_weight=self.cfg_weight)
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)


class ElevenLabsEngine(TTSEngine):
    name = "elevenlabs"
    sample_rate = 24000
    URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_24000"

    def __init__(self, api_key: str, voice_id: str, model_id: str = "eleven_multilingual_v2",
                 speed: float = 0.9) -> None:
        if not api_key or not voice_id:
            raise TTSError("ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID are required for the ElevenLabs engine")
        self.api_key, self.voice_id, self.model_id = api_key, voice_id, model_id
        self.native_tempo = speed
        self.session = requests.Session()

    @property
    def cache_key(self) -> str:
        return f"elevenlabs:{self.voice_id}:{self.model_id}:{self.native_tempo}"

    def synthesize(self, text: str) -> np.ndarray:
        body = {"text": text, "model_id": self.model_id,
                "voice_settings": {"stability": 0.75, "similarity_boost": 0.75, "style": 0.0,
                                   "use_speaker_boost": True, "speed": self.native_tempo}}
        for attempt in range(7):
            try:
                r = self.session.post(self.URL.format(voice=self.voice_id), json=body, timeout=120,
                                      headers={"xi-api-key": self.api_key, "accept": "audio/pcm"})
            except requests.RequestException as exc:
                if attempt == 6:
                    raise TTSError(f"ElevenLabs network error: {exc}") from exc
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            if r.status_code == 200:
                return np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 6:
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            raise TTSError(f"ElevenLabs HTTP {r.status_code}: {r.text[:300]}")
        raise TTSError("ElevenLabs retries exhausted")


class CachedTTS:
    """Wraps an engine with a content-addressed WAV cache so re-renders never re-synthesise."""

    def __init__(self, engine: TTSEngine, cache_dir: Path) -> None:
        self.engine = engine
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def sample_rate(self) -> int:
        return self.engine.sample_rate

    @property
    def native_tempo(self) -> float:
        return self.engine.native_tempo

    def synthesize(self, text: str) -> np.ndarray:
        key = hashlib.sha256(f"{self.engine.cache_key}|{text}".encode()).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.wav"
        if path.exists():
            audio, sr = read_wav(path)
            if sr == self.engine.sample_rate:
                return audio
        audio = self.engine.synthesize(text)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise TTSError(f"{self.engine.name} returned empty/invalid audio for {text[:60]!r}")
        peak = float(np.abs(audio).max())
        if peak > 1.0:
            audio = audio / peak
        path.parent.mkdir(parents=True, exist_ok=True)
        write_wav(path, audio, self.engine.sample_rate)
        return audio


# ------------------------------------------------------------------ wav helpers (stdlib only)
def write_wav(path: Path | io.BytesIO, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path) if isinstance(path, Path) else path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def read_wav(src: Path | io.BytesIO) -> tuple[np.ndarray, int]:
    with wave.open(str(src) if isinstance(src, Path) else src, "rb") as wf:
        sr, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise TTSError(f"Only 16-bit WAV supported, got {width * 8}-bit")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


EngineFactory = Callable[[Settings], TTSEngine]
ENGINE_REGISTRY: dict[str, EngineFactory] = {
    "kokoro": lambda s: KokoroEngine(s.kokoro_voice),
    "piper": lambda s: PiperEngine(s.piper_model_path),
    "chatterbox": lambda s: ChatterboxEngine(s.chatterbox_voice_prompt),
    "elevenlabs": lambda s: ElevenLabsEngine(s.elevenlabs_api_key, s.elevenlabs_voice_id),
}


def register_engine(name: str, factory: EngineFactory) -> None:
    ENGINE_REGISTRY[name] = factory


def make_engine(settings: Settings, name: str | None = None, **_: Any) -> TTSEngine:
    key = (name or settings.tts_engine).lower()
    if key not in ENGINE_REGISTRY:
        raise TTSError(f"Unknown TTS_ENGINE {key!r}; choose one of {sorted(ENGINE_REGISTRY)}")
    return ENGINE_REGISTRY[key](settings)
