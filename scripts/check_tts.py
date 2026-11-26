#!/usr/bin/env python3
"""Check the configured TTS engine: synthesise Psalm 23:1-3, save a WAV, measure speed, estimate episode time.

    python scripts/check_tts.py                  # uses TTS_ENGINE / KOKORO_VOICE from .env
    python scripts/check_tts.py --voice bm_george --out psalm23_george.wav
"""
from __future__ import annotations

import argparse
import dataclasses
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_settings  # noqa: E402
from core.tts import make_engine, write_wav  # noqa: E402

SAMPLE = ("The Lord is my shepherd; I shall not want. He maketh me to lie down in green pastures: "
          "he leadeth me beside the still waters. He restoreth my soul: he leadeth me in the paths of "
          "righteousness for his name's sake.")
NARRATION_SECONDS = 9000  # hours 0-2.5 of each episode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", help="override TTS_ENGINE")
    ap.add_argument("--voice", help="override KOKORO_VOICE")
    ap.add_argument("--out", default="tts_check.wav")
    args = ap.parse_args()
    s = get_settings()
    if args.voice:
        s = dataclasses.replace(s, kokoro_voice=args.voice)
    print(f"Python {platform.python_version()} on {platform.machine()} / {platform.system()}")
    try:
        import numpy
        import torch
        print(f"torch {torch.__version__}  numpy {numpy.__version__}  threads {torch.get_num_threads()}")
    except ImportError:
        pass
    t0 = time.time()
    engine = make_engine(s, args.engine)
    print(f"Loaded {engine.cache_key} in {time.time() - t0:.1f}s (first run downloads the model)")
    engine.synthesize("Warm up.")
    t1 = time.time()
    audio = engine.synthesize(SAMPLE)
    took = time.time() - t1
    secs = len(audio) / engine.sample_rate
    write_wav(Path(args.out), audio, engine.sample_rate)
    rtf = secs / took if took else float("inf")
    print(f"Synthesised {secs:.1f}s of speech in {took:.1f}s -> {rtf:.1f}x realtime. Saved {args.out}")
    # speech is ~75% of the narration window; pauses are inserted as silence (free)
    est = NARRATION_SECONDS * 0.75 / rtf / 60
    print(f"Estimated first-time narration synthesis per episode: ~{est:.0f} min "
          "(cached verses are reused on re-renders)")
    if rtf < 1.0:
        print("Slower than realtime: consider fewer/longer runs overnight, or ElevenLabs for production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
