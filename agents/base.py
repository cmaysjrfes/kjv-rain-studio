"""Shared agent scaffolding: run bookkeeping, status reporting for the Rich dashboard."""
from __future__ import annotations

import logging
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.config import Settings
from db.database import Database

StatusCallback = Callable[[str, str, str], None]  # (agent_name, state, message)


@dataclass
class AgentResult:
    agent: str
    ok: bool
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0


class BaseAgent(ABC):
    name: str = "agent"

    def __init__(self, settings: Settings, db: Database, *, dry_run: bool = False,
                 status_cb: StatusCallback | None = None) -> None:
        self.settings = settings
        self.db = db
        self.dry_run = dry_run
        self.log = logging.getLogger(f"agents.{self.name}")
        self._status_cb = status_cb
        self._progress_marks: dict[str, int] = {}

    def status(self, message: str, state: str = "running") -> None:
        self.log.info(message)
        if self._status_cb:
            self._status_cb(self.name, state, message)

    def progress(self, label: str, frac: float) -> None:
        """High-frequency progress: always feeds the dashboard, logs only every 10%."""
        if self._status_cb:
            self._status_cb(self.name, "running", f"{label} {frac:.1%}")
        bucket = int(frac * 10)
        if bucket != self._progress_marks.get(label):
            self._progress_marks[label] = bucket
            self.log.info("%s %d%%", label, bucket * 10)

    @abstractmethod
    def execute(self) -> AgentResult: ...

    def run(self) -> AgentResult:
        run_id = self.db.start_run(self.name, "dry-run" if self.dry_run else "live")
        t0 = time.monotonic()
        try:
            result = self.execute()
        except Exception as exc:  # noqa: BLE001 - record any failure, then surface it
            tb = traceback.format_exc()
            self.db.finish_run(run_id, "failed", error=tb)
            self.status(f"failed: {exc}", "failed")
            return AgentResult(self.name, False, error=f"{type(exc).__name__}: {exc}",
                               seconds=time.monotonic() - t0)
        result.seconds = time.monotonic() - t0
        self.db.finish_run(run_id, "success" if result.ok else "failed",
                           summary={**result.summary, "warnings": result.warnings}, error=result.error)
        self.status("done" if result.ok else f"failed: {result.error}", "done" if result.ok else "failed")
        return result
