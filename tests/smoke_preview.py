"""Manual smoke test: run Agents 2-4 in preview + dry-run mode against a scratch DB/output dir.

    python -m tests.smoke_preview [seconds]
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

from core.config import get_settings
from core.logging_setup import setup_logging
from db.database import Database
from tests.fakes import FakeSpeechEngine


def main() -> int:
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    setup_logging("INFO")
    root = Path(tempfile.mkdtemp(prefix="kjv-smoke-"))
    base = get_settings()
    s = dataclasses.replace(base, output_dir=root / "output", db_path=root / "state.sqlite3", preview_seconds=seconds)
    s.output_dir.mkdir(parents=True)
    db = Database(s.db_path)
    from agents.agent2_audio import AudioEngineerAgent
    r2 = AudioEngineerAgent(s, db, engine=FakeSpeechEngine()).run()
    print(json.dumps(r2.summary, indent=2), r2.error)
    if not r2.ok:
        return 1
    try:
        from agents.agent3_visuals import VisualsAgent
    except ImportError:
        return 0
    r3 = VisualsAgent(s, db).run()
    print(json.dumps(r3.summary, indent=2), r3.error)
    if not r3.ok:
        return 1
    try:
        from agents.agent4_publisher import PublisherAgent
    except ImportError:
        return 0
    r4 = PublisherAgent(s, db, dry_run=True).run()
    print(json.dumps(r4.summary, indent=2), r4.error)
    print("artifacts in", root)
    return 0 if r4.ok else 1


if __name__ == "__main__":
    sys.exit(main())
