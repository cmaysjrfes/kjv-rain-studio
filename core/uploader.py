"""YouTube resumable upload protocol, implemented directly so it survives process restarts.

The session URI and byte offset are checkpointed in SQLite after every chunk. On restart
the uploader asks Google how many bytes it holds (``Content-Range: bytes */size``) and
continues from there - a 10 GB upload interrupted at 9 GB resumes at 9 GB. Sessions stay
valid for about a week; an expired session (404/410) starts a new one.
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests
from google.auth.transport.requests import AuthorizedSession

from db.database import Database, QuotaExceededError

log = logging.getLogger(__name__)

UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
RETRY_STATUS = {429, 500, 502, 503, 504}
CHUNK_UNIT = 256 * 1024


class UploadError(RuntimeError):
    pass


class ResumableUploader:
    def __init__(self, credentials: Any, db: Database, *, chunk_mb: int = 64, max_retries: int = 12,
                 agent: str = "agent4_publisher", session: Any = None) -> None:
        self.db = db
        self.session = session or AuthorizedSession(credentials)
        self.chunk = max(CHUNK_UNIT, (chunk_mb * 1024 * 1024 // CHUNK_UNIT) * CHUNK_UNIT)
        self.max_retries = max_retries
        self.agent = agent

    # ------------------------------------------------------------------ protocol
    def _start_session(self, body: dict[str, Any], size: int, mime: str, notify: bool = True) -> str:
        self.db.consume_quota("videos.insert", agent=self.agent, use_reserve=True)
        params = {"uploadType": "resumable", "notifySubscribers": "true" if notify else "false",
                  "part": ",".join(k for k in body if k in {"snippet", "status", "recordingDetails"})}
        for attempt in range(self.max_retries):
            try:
                r = self.session.post(UPLOAD_URL, params=params, json=body, timeout=60, headers={
                    "X-Upload-Content-Length": str(size), "X-Upload-Content-Type": mime})
            except requests.RequestException as exc:
                self._sleep(attempt, f"session start network error {exc}")
                continue
            if r.status_code == 200 and r.headers.get("Location"):
                return r.headers["Location"]
            reason = _reason(r)
            if r.status_code == 403 and reason in {"quotaExceeded", "uploadLimitExceeded", "dailyLimitExceeded"}:
                raise QuotaExceededError(f"YouTube refused upload: {reason}")
            if r.status_code in RETRY_STATUS:
                self._sleep(attempt, f"session start HTTP {r.status_code}")
                continue
            self.db.refund_quota("videos.insert", 1600, agent=self.agent)
            raise UploadError(f"Upload session rejected HTTP {r.status_code} {reason}: {r.text[:500]}")
        raise UploadError("Could not start upload session (retries exhausted)")

    def _query(self, uri: str, size: int) -> tuple[str, int | dict[str, Any]]:
        """-> ('offset', n) | ('done', resource) | ('expired', 0)."""
        for attempt in range(self.max_retries):
            try:
                r = self.session.put(uri, timeout=60, headers={"Content-Length": "0",
                                                               "Content-Range": f"bytes */{size}"})
            except requests.RequestException as exc:
                self._sleep(attempt, f"status query network error {exc}")
                continue
            if r.status_code in (200, 201):
                return "done", r.json()
            if r.status_code == 308:
                return "offset", _next_offset(r)
            if r.status_code in (404, 410):
                return "expired", 0
            if r.status_code in RETRY_STATUS:
                self._sleep(attempt, f"status query HTTP {r.status_code}")
                continue
            raise UploadError(f"Upload status query failed HTTP {r.status_code}: {r.text[:300]}")
        raise UploadError("Upload status query retries exhausted")

    def upload(self, upload_id: int, body: dict[str, Any], *, mime: str = "video/mp4", notify: bool = True,
               progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        row = self.db.get_upload(upload_id)
        if row is None:
            raise UploadError(f"upload row {upload_id} missing")
        path = Path(row["file_path"])
        size = path.stat().st_size
        uri: str | None = row.get("resumable_uri")
        offset = 0
        if uri:
            state, val = self._query(uri, size)
            if state == "done":
                return val  # type: ignore[return-value]
            if state == "offset":
                offset = int(val)  # type: ignore[arg-type]
                log.info("Resuming upload %s at %.1f%%", path.name, 100 * offset / size)
            else:
                uri = None
        if not uri:
            uri = self._start_session(body, size, mime, notify)
            offset = 0
        self.db.set_upload_status(upload_id, "uploading", resumable_uri=uri, bytes_uploaded=offset)

        restarts = 0
        failures = 0
        with path.open("rb") as fh:
            while True:
                fh.seek(offset)
                data = fh.read(self.chunk)
                end = offset + len(data) - 1
                try:
                    r = self.session.put(uri, data=data, timeout=600, headers={
                        "Content-Length": str(len(data)), "Content-Range": f"bytes {offset}-{end}/{size}"})
                except requests.RequestException as exc:
                    failures += 1
                    if failures > self.max_retries:
                        raise UploadError(f"Upload network failures exhausted: {exc}") from exc
                    self._sleep(failures, f"chunk network error {exc}")
                    state, val = self._query(uri, size)
                    if state == "done":
                        return val  # type: ignore[return-value]
                    if state == "expired":
                        raise UploadError("Upload session expired mid-transfer") from exc
                    offset = int(val)  # type: ignore[arg-type]
                    continue
                if r.status_code in (200, 201):
                    self.db.save_upload_progress(upload_id, size)
                    if progress:
                        progress(size, size)
                    return r.json()
                if r.status_code == 308:
                    offset = _next_offset(r)
                    failures = 0
                    self.db.save_upload_progress(upload_id, offset)
                    if progress:
                        progress(offset, size)
                    continue
                if r.status_code in (404, 410) and restarts < 1:
                    restarts += 1
                    log.warning("Upload session expired; starting a new session")
                    uri = self._start_session(body, size, mime, notify)
                    offset = 0
                    self.db.save_upload_progress(upload_id, 0, uri)
                    continue
                if r.status_code in RETRY_STATUS:
                    failures += 1
                    if failures > self.max_retries:
                        raise UploadError(f"Upload HTTP {r.status_code} retries exhausted")
                    self._sleep(failures, f"chunk HTTP {r.status_code}")
                    state, val = self._query(uri, size)
                    if state == "done":
                        return val  # type: ignore[return-value]
                    offset = int(val) if state == "offset" else 0  # type: ignore[arg-type]
                    continue
                raise UploadError(f"Upload failed HTTP {r.status_code} {_reason(r)}: {r.text[:500]}")

    @staticmethod
    def _sleep(attempt: int, why: str) -> None:
        delay = min(120.0, 2 ** min(attempt, 7)) + random.random()
        log.warning("%s; retrying in %.1fs", why, delay)
        time.sleep(delay)


def _next_offset(r: Any) -> int:
    rng = r.headers.get("Range")  # "bytes=0-12345"
    return int(rng.split("-")[-1]) + 1 if rng else 0


def _reason(r: Any) -> str:
    try:
        return json.loads(r.text)["error"]["errors"][0].get("reason", "")
    except Exception:  # noqa: BLE001
        return ""
