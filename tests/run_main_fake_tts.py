"""Run main.py with the fake TTS engine (no neural model needed). Point DB_PATH / OUTPUT_DIR at a scratch dir:

    DB_PATH=/tmp/x/s.db OUTPUT_DIR=/tmp/x/out python -m tests.run_main_fake_tts --preview 120 --no-dashboard
"""
from __future__ import annotations

import os
import sys

os.environ["TTS_ENGINE"] = "fake"

from core.tts import register_engine  # noqa: E402
from tests.fakes import FakeSpeechEngine  # noqa: E402

register_engine("fake", lambda s: FakeSpeechEngine())

import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main.main(sys.argv[1:]))
