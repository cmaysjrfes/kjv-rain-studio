"""Quota-aware YouTube Data API v3 client (public reads via API key, writes via OAuth2)."""
from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable, Iterator
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from db.database import QUOTA_COSTS, Database, QuotaExceededError

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {500, 502, 503, 504}
_ISO_DUR = re.compile(r"P(?:(?P<d>\d+)D)?(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?)?$")


def parse_iso8601_duration(value: str | None) -> int:
    """'PT8H0M12S' -> 28812. Returns 0 for live/unknown durations."""
    if not value:
        return 0
    m = _ISO_DUR.match(value)
    if not m:
        return 0
    d, h, mi, s = (int(m.group(k) or 0) for k in ("d", "h", "m", "s"))
    return d * 86400 + h * 3600 + mi * 60 + s


def chunked(items: list[str], size: int = 50) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


class YouTubeDataClient:
    """Wraps googleapiclient with pre-charged quota accounting and exponential backoff."""

    def __init__(self, db: Database, *, api_key: str | None = None, credentials: Any = None,
                 agent: str = "unknown", max_retries: int = 6) -> None:
        if not api_key and credentials is None:
            raise ValueError("YouTubeDataClient needs an API key or OAuth credentials")
        self.db = db
        self.agent = agent
        self.max_retries = max_retries
        self.service = build("youtube", "v3", developerKey=api_key, credentials=credentials,
                             cache_discovery=False)

    def execute(self, endpoint: str, request_factory: Callable[[], Any], *, use_reserve: bool = False) -> dict[str, Any]:
        cost = QUOTA_COSTS.get(endpoint, 1)
        self.db.consume_quota(endpoint, cost, agent=self.agent, use_reserve=use_reserve)
        for attempt in range(self.max_retries + 1):
            try:
                return request_factory().execute()
            except HttpError as exc:
                status = int(getattr(exc.resp, "status", 0) or 0)
                reason = _http_reason(exc)
                if status == 403 and reason in {"quotaExceeded", "dailyLimitExceeded"}:
                    raise QuotaExceededError(f"Google reports quota exhausted on {endpoint}: {reason}") from exc
                if status in RETRYABLE_STATUS or (status == 403 and reason in {"rateLimitExceeded",
                                                                               "userRateLimitExceeded"}):
                    if attempt == self.max_retries:
                        raise
                    delay = min(64.0, 2 ** attempt) + random.random()
                    log.warning("%s HTTP %s (%s); retry %d in %.1fs", endpoint, status, reason, attempt + 1, delay)
                    time.sleep(delay)
                    continue
                raise
            except (ConnectionError, TimeoutError, OSError) as exc:
                if attempt == self.max_retries:
                    raise
                delay = min(64.0, 2 ** attempt) + random.random()
                log.warning("%s network error %s; retry in %.1fs", endpoint, exc, delay)
                time.sleep(delay)
        raise RuntimeError("unreachable")

    # ---------------------------------------------------------------- reads
    def search_videos(self, query: str, max_results: int = 50, *, duration: str = "long",
                      order: str = "viewCount") -> list[dict[str, Any]]:
        resp = self.execute("search.list", lambda: self.service.search().list(
            q=query, part="snippet", type="video", videoDuration=duration, order=order,
            maxResults=min(50, max_results), relevanceLanguage="en", safeSearch="none"))
        return resp.get("items", [])

    def videos(self, video_ids: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for batch in chunked(list(dict.fromkeys(video_ids))):
            resp = self.execute("videos.list", lambda b=batch: self.service.videos().list(
                part="snippet,contentDetails,statistics", id=",".join(b), maxResults=50))
            out.extend(resp.get("items", []))
        return out

    def channels(self, channel_ids: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for batch in chunked(list(dict.fromkeys(channel_ids))):
            resp = self.execute("channels.list", lambda b=batch: self.service.channels().list(
                part="snippet,statistics,brandingSettings,contentDetails", id=",".join(b), maxResults=50))
            out.extend(resp.get("items", []))
        return out

    def playlists(self, channel_id: str, max_results: int = 50) -> list[dict[str, Any]]:
        resp = self.execute("playlists.list", lambda: self.service.playlists().list(
            part="snippet,contentDetails", channelId=channel_id, maxResults=min(50, max_results)))
        return resp.get("items", [])

    def playlist_items(self, playlist_id: str, max_results: int = 50) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        token: str | None = None
        while len(items) < max_results:
            resp = self.execute("playlistItems.list", lambda t=token: self.service.playlistItems().list(
                part="contentDetails,snippet", playlistId=playlist_id, maxResults=50, pageToken=t))
            items.extend(resp.get("items", []))
            token = resp.get("nextPageToken")
            if not token:
                break
        return items[:max_results]


def _http_reason(exc: HttpError) -> str:
    try:
        import json

        body = json.loads(exc.content.decode("utf-8"))
        return body["error"]["errors"][0].get("reason", "")
    except Exception:  # noqa: BLE001 - best effort parse of Google error envelope
        return ""
