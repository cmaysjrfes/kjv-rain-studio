"""SQLite state, uniqueness and quota tracking for the 5-agent system.

Single-file, dependency-free persistence layer:

* **Competitor intelligence** - channels, videos and point-in-time metric snapshots.
* **Uniqueness history** - exact (SHA-256) and perceptual (64-bit dHash) fingerprints so
  no two uploads share an audio bed mix, visual loop, title, or thumbnail.
* **Episodes & render jobs** - the production pipeline state machine.
* **API quota usage** - YouTube Data API units, bucketed by the Pacific-time day YouTube
  uses for its quota reset, with a reserve that only uploads may dip into.
* **Uploads** - resumable-session URIs and byte offsets so an interrupted multi-GB upload
  resumes instead of restarting.
* **A/B tests & analytics** - thumbnail/title variants and the metrics that judge them.
* **Media files** - local file inventory for post-upload disk hygiene.

Connections are per-thread, WAL-journaled, and every write runs in a transaction.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
SCHEMA_VERSION = 1

# Official YouTube Data API v3 unit costs (https://developers.google.com/youtube/v3/determine_quota_cost)
QUOTA_COSTS: dict[str, int] = {
    "search.list": 100,
    "videos.list": 1,
    "channels.list": 1,
    "playlists.list": 1,
    "playlistItems.list": 1,
    "commentThreads.list": 1,
    "videos.insert": 1600,
    "videos.update": 50,
    "thumbnails.set": 50,
    "playlists.insert": 50,
    "playlists.update": 50,
    "playlistItems.insert": 50,
    "commentThreads.insert": 50,
    "channels.update": 50,
    "channelBanners.insert": 50,
    "watermarks.set": 50,
}

UPLOAD_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"uploading", "failed", "cancelled"},
    "uploading": {"uploading", "uploaded", "failed", "cancelled"},
    "uploaded": {"processing", "live", "failed"},
    "processing": {"processing", "live", "failed"},
    "live": {"live"},
    "failed": {"pending", "uploading", "cancelled"},
    "cancelled": set(),
}

EPISODE_STATUSES = (
    "planned", "audio_ready", "video_ready", "uploading", "published", "failed", "archived",
)


class QuotaExceededError(RuntimeError):
    """Raised when spending API units would breach the daily quota (or its reserve)."""


class DuplicateContentError(RuntimeError):
    """Raised when a fingerprint has already been used by an earlier episode."""


class InvalidTransitionError(RuntimeError):
    """Raised on an illegal upload status transition."""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def quota_day(now: datetime | None = None) -> str:
    """YouTube quotas reset at midnight Pacific Time."""
    now = now or datetime.now(timezone.utc)
    return now.astimezone(PACIFIC).date().isoformat()


def sha256_fingerprint(*parts: Any) -> str:
    """Stable content fingerprint of arbitrary JSON-serialisable parts."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, (bytes, bytearray)):
            h.update(bytes(part))
        else:
            h.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def sha256_file(path: str | Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_head_tail(path: str | Path, span: int = 64 * 1024 * 1024) -> str:
    """Fast fingerprint for multi-GB media: size + first and last `span` bytes."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(p, "rb") as fh:
        h.update(fh.read(span))
        if size > span:
            fh.seek(max(span, size - span))
            h.update(fh.read(span))
    return h.hexdigest()


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    agent         TEXT NOT NULL,
    mode          TEXT NOT NULL DEFAULT 'live',
    status        TEXT NOT NULL DEFAULT 'running',
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    summary_json  TEXT,
    error         TEXT
);

CREATE TABLE IF NOT EXISTS competitor_channels (
    channel_id        TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    handle            TEXT,
    description       TEXT,
    country           TEXT,
    keywords          TEXT,
    banner_url        TEXT,
    avatar_url        TEXT,
    subscriber_count  INTEGER,
    view_count        INTEGER,
    video_count       INTEGER,
    long_video_count  INTEGER DEFAULT 0,
    median_upload_gap_days REAL,
    score             REAL DEFAULT 0,
    source            TEXT NOT NULL DEFAULT 'api',
    first_seen        TEXT NOT NULL,
    last_seen         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS competitor_channel_snapshots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id        TEXT NOT NULL REFERENCES competitor_channels(channel_id) ON DELETE CASCADE,
    captured_at       TEXT NOT NULL,
    subscriber_count  INTEGER,
    view_count        INTEGER,
    video_count       INTEGER
);

CREATE TABLE IF NOT EXISTS competitor_videos (
    video_id       TEXT PRIMARY KEY,
    channel_id     TEXT NOT NULL,
    title          TEXT NOT NULL,
    description    TEXT,
    published_at   TEXT,
    duration_s     INTEGER,
    view_count     INTEGER,
    like_count     INTEGER,
    comment_count  INTEGER,
    thumbnail_url  TEXT,
    tags_json      TEXT,
    has_chapters   INTEGER DEFAULT 0,
    thumb_metrics_json TEXT,
    source         TEXT NOT NULL DEFAULT 'api',
    captured_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cv_channel ON competitor_videos(channel_id);

CREATE TABLE IF NOT EXISTS competitor_video_snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id       TEXT NOT NULL REFERENCES competitor_videos(video_id) ON DELETE CASCADE,
    captured_at    TEXT NOT NULL,
    view_count     INTEGER,
    like_count     INTEGER,
    comment_count  INTEGER
);

CREATE TABLE IF NOT EXISTS episodes (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    slug             TEXT NOT NULL UNIQUE,
    status           TEXT NOT NULL DEFAULT 'planned',
    scripture_book   TEXT,
    scripture_ref    TEXT,
    ambient_bed      TEXT,
    visual_scene     TEXT,
    title_template   TEXT,
    thumb_template   TEXT,
    title            TEXT,
    duration_s       INTEGER,
    metadata_json    TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);

CREATE TABLE IF NOT EXISTS uniqueness_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,
    fingerprint  TEXT NOT NULL,
    algo         TEXT NOT NULL DEFAULT 'sha256',
    episode_id   INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
    detail       TEXT,
    created_at   TEXT NOT NULL,
    UNIQUE(kind, fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_uniq_kind_algo ON uniqueness_history(kind, algo);

CREATE TABLE IF NOT EXISTS render_jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id   INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'queued',
    encoder      TEXT,
    input_json   TEXT,
    output_path  TEXT,
    progress     REAL DEFAULT 0,
    attempts     INTEGER DEFAULT 0,
    error        TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_render_status ON render_jobs(status);

CREATE TABLE IF NOT EXISTS api_quota_usage (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    quota_day  TEXT NOT NULL,
    api        TEXT NOT NULL DEFAULT 'youtube_data',
    endpoint   TEXT NOT NULL,
    units      INTEGER NOT NULL,
    agent      TEXT,
    ts         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quota_day ON api_quota_usage(api, quota_day);

CREATE TABLE IF NOT EXISTS uploads (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id         INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
    kind               TEXT NOT NULL DEFAULT 'long',
    file_path          TEXT NOT NULL,
    file_size          INTEGER,
    file_sha256        TEXT,
    status             TEXT NOT NULL DEFAULT 'pending',
    resumable_uri      TEXT,
    bytes_uploaded     INTEGER DEFAULT 0,
    youtube_video_id   TEXT UNIQUE,
    processing_status  TEXT,
    privacy            TEXT DEFAULT 'private',
    attempts           INTEGER DEFAULT 0,
    last_error         TEXT,
    metadata_json      TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status);

CREATE TABLE IF NOT EXISTS ab_tests (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id         INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
    youtube_video_id   TEXT NOT NULL,
    element            TEXT NOT NULL CHECK (element IN ('thumbnail', 'title')),
    status             TEXT NOT NULL DEFAULT 'running',
    rotate_after_hours INTEGER NOT NULL DEFAULT 72,
    winner_variant_id  INTEGER,
    started_at         TEXT NOT NULL,
    ended_at           TEXT
);

CREATE TABLE IF NOT EXISTS ab_variants (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id              INTEGER NOT NULL REFERENCES ab_tests(id) ON DELETE CASCADE,
    label                TEXT NOT NULL,
    payload              TEXT NOT NULL,
    is_active            INTEGER NOT NULL DEFAULT 0,
    active_from          TEXT,
    active_to            TEXT,
    impressions          INTEGER DEFAULT 0,
    views                INTEGER DEFAULT 0,
    ctr                  REAL,
    avg_view_duration_s  REAL,
    UNIQUE(test_id, label)
);

CREATE TABLE IF NOT EXISTS analytics_snapshots (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    youtube_video_id     TEXT NOT NULL,
    captured_at          TEXT NOT NULL,
    window_start         TEXT,
    window_end           TEXT,
    views                INTEGER,
    watch_minutes        REAL,
    avg_view_duration_s  REAL,
    impressions          INTEGER,
    ctr                  REAL,
    metrics_json         TEXT
);
CREATE INDEX IF NOT EXISTS idx_analytics_vid ON analytics_snapshots(youtube_video_id, captured_at);

CREATE TABLE IF NOT EXISTS media_files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id  INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
    path        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL,
    size_bytes  INTEGER,
    keep        INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    deleted_at  TEXT
);

CREATE TABLE IF NOT EXISTS kv_state (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class QuotaStatus:
    day: str
    used: int
    limit: int
    reserve: int

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def remaining_unreserved(self) -> int:
        return max(0, self.limit - self.reserve - self.used)


class Database:
    """Thread-safe SQLite facade. One connection per thread; WAL for concurrent readers."""

    def __init__(self, path: str | Path, daily_quota: int = 10_000, quota_reserve: int = 2_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_quota = daily_quota
        self.quota_reserve = quota_reserve
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._migrate()

    # ------------------------------------------------------------------ plumbing
    @property
    def conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=True)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Serialised write transaction (BEGIN IMMEDIATE avoids writer deadlocks)."""
        with self._write_lock:
            c = self.conn
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
            except BaseException:
                c.execute("ROLLBACK")
                raise
            else:
                c.execute("COMMIT")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _migrate(self) -> None:
        with self._write_lock:
            current = self.conn.execute("PRAGMA user_version").fetchone()[0]
            if current < SCHEMA_VERSION:
                self.conn.executescript(_SCHEMA)
                self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ agent runs
    def start_run(self, agent: str, mode: str = "live") -> int:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO agent_runs(agent, mode, started_at) VALUES (?,?,?)", (agent, mode, utcnow())
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str = "success", summary: dict[str, Any] | None = None,
                   error: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE agent_runs SET status=?, finished_at=?, summary_json=?, error=? WHERE id=?",
                (status, utcnow(), json.dumps(summary or {}, default=str), error, run_id),
            )

    # ------------------------------------------------------------------ quota
    def quota_status(self, api: str = "youtube_data") -> QuotaStatus:
        day = quota_day()
        row = self.conn.execute(
            "SELECT COALESCE(SUM(units),0) FROM api_quota_usage WHERE api=? AND quota_day=?", (api, day)
        ).fetchone()
        return QuotaStatus(day=day, used=int(row[0]), limit=self.daily_quota, reserve=self.quota_reserve)

    def can_spend(self, units: int, *, use_reserve: bool = False, api: str = "youtube_data") -> bool:
        st = self.quota_status(api)
        budget = st.remaining if use_reserve else st.remaining_unreserved
        return units <= budget

    def consume_quota(self, endpoint: str, units: int | None = None, *, agent: str | None = None,
                      use_reserve: bool = False, api: str = "youtube_data", enforce: bool = True) -> int:
        """Record API units *before* the call is made. Raises QuotaExceededError when enforced."""
        cost = QUOTA_COSTS.get(endpoint, 1) if units is None else units
        with self.tx() as c:
            day = quota_day()
            used = int(c.execute(
                "SELECT COALESCE(SUM(units),0) FROM api_quota_usage WHERE api=? AND quota_day=?", (api, day)
            ).fetchone()[0])
            ceiling = self.daily_quota if use_reserve else self.daily_quota - self.quota_reserve
            if enforce and used + cost > ceiling:
                raise QuotaExceededError(
                    f"{endpoint} needs {cost} units; {used}/{self.daily_quota} used today "
                    f"(reserve {self.quota_reserve}, use_reserve={use_reserve}). Resets at midnight PT."
                )
            c.execute(
                "INSERT INTO api_quota_usage(quota_day, api, endpoint, units, agent, ts) VALUES (?,?,?,?,?,?)",
                (day, api, endpoint, cost, agent, utcnow()),
            )
        return cost

    def refund_quota(self, endpoint: str, units: int, *, agent: str | None = None,
                     api: str = "youtube_data") -> None:
        """Negative entry for calls that were pre-charged but never sent (e.g. local validation failed)."""
        with self.tx() as c:
            c.execute(
                "INSERT INTO api_quota_usage(quota_day, api, endpoint, units, agent, ts) VALUES (?,?,?,?,?,?)",
                (quota_day(), api, f"refund:{endpoint}", -abs(units), agent, utcnow()),
            )

    def quota_breakdown(self, day: str | None = None, api: str = "youtube_data") -> list[dict[str, Any]]:
        return self.query(
            "SELECT endpoint, SUM(units) AS units, COUNT(*) AS calls FROM api_quota_usage "
            "WHERE api=? AND quota_day=? GROUP BY endpoint ORDER BY units DESC",
            (api, day or quota_day()),
        )

    # ------------------------------------------------------------------ competitors
    def upsert_competitor_channel(self, ch: dict[str, Any]) -> None:
        now = utcnow()
        with self.tx() as c:
            c.execute(
                """INSERT INTO competitor_channels(channel_id, title, handle, description, country, keywords,
                       banner_url, avatar_url, subscriber_count, view_count, video_count, long_video_count,
                       median_upload_gap_days, score, source, first_seen, last_seen)
                   VALUES (:channel_id,:title,:handle,:description,:country,:keywords,:banner_url,:avatar_url,
                       :subscriber_count,:view_count,:video_count,:long_video_count,:median_upload_gap_days,
                       :score,:source,:now,:now)
                   ON CONFLICT(channel_id) DO UPDATE SET
                       title=excluded.title, handle=excluded.handle, description=excluded.description,
                       country=excluded.country, keywords=excluded.keywords, banner_url=excluded.banner_url,
                       avatar_url=excluded.avatar_url, subscriber_count=excluded.subscriber_count,
                       view_count=excluded.view_count, video_count=excluded.video_count,
                       long_video_count=excluded.long_video_count,
                       median_upload_gap_days=excluded.median_upload_gap_days, score=excluded.score,
                       source=excluded.source, last_seen=excluded.last_seen""",
                {
                    "channel_id": ch["channel_id"], "title": ch.get("title", ""), "handle": ch.get("handle"),
                    "description": ch.get("description"), "country": ch.get("country"),
                    "keywords": ch.get("keywords"), "banner_url": ch.get("banner_url"),
                    "avatar_url": ch.get("avatar_url"), "subscriber_count": ch.get("subscriber_count"),
                    "view_count": ch.get("view_count"), "video_count": ch.get("video_count"),
                    "long_video_count": ch.get("long_video_count", 0),
                    "median_upload_gap_days": ch.get("median_upload_gap_days"), "score": ch.get("score", 0.0),
                    "source": ch.get("source", "api"), "now": now,
                },
            )
            c.execute(
                "INSERT INTO competitor_channel_snapshots(channel_id, captured_at, subscriber_count, view_count,"
                " video_count) VALUES (?,?,?,?,?)",
                (ch["channel_id"], now, ch.get("subscriber_count"), ch.get("view_count"), ch.get("video_count")),
            )

    def upsert_competitor_video(self, v: dict[str, Any]) -> None:
        now = utcnow()
        with self.tx() as c:
            c.execute(
                """INSERT INTO competitor_videos(video_id, channel_id, title, description, published_at,
                       duration_s, view_count, like_count, comment_count, thumbnail_url, tags_json,
                       has_chapters, thumb_metrics_json, source, captured_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(video_id) DO UPDATE SET
                       title=excluded.title, description=excluded.description, duration_s=excluded.duration_s,
                       view_count=excluded.view_count, like_count=excluded.like_count,
                       comment_count=excluded.comment_count, thumbnail_url=excluded.thumbnail_url,
                       tags_json=excluded.tags_json, has_chapters=excluded.has_chapters,
                       thumb_metrics_json=COALESCE(excluded.thumb_metrics_json, competitor_videos.thumb_metrics_json),
                       captured_at=excluded.captured_at""",
                (
                    v["video_id"], v["channel_id"], v.get("title", ""), v.get("description"),
                    v.get("published_at"), v.get("duration_s"), v.get("view_count"), v.get("like_count"),
                    v.get("comment_count"), v.get("thumbnail_url"), json.dumps(v.get("tags") or []),
                    int(bool(v.get("has_chapters"))),
                    json.dumps(v["thumb_metrics"]) if v.get("thumb_metrics") else None,
                    v.get("source", "api"), now,
                ),
            )
            c.execute(
                "INSERT INTO competitor_video_snapshots(video_id, captured_at, view_count, like_count, comment_count)"
                " VALUES (?,?,?,?,?)",
                (v["video_id"], now, v.get("view_count"), v.get("like_count"), v.get("comment_count")),
            )

    def top_competitor_channels(self, limit: int = 15) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM competitor_channels ORDER BY score DESC LIMIT ?", (limit,))

    def competitor_videos(self, min_duration_s: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM competitor_videos WHERE COALESCE(duration_s,0) >= ? ORDER BY view_count DESC LIMIT ?",
            (min_duration_s, limit),
        )

    # ------------------------------------------------------------------ episodes
    def create_episode(self, slug: str, **fields: Any) -> int:
        allowed = {"status", "scripture_book", "scripture_ref", "ambient_bed", "visual_scene",
                   "title_template", "thumb_template", "title", "duration_s"}
        unknown = set(fields) - allowed - {"metadata"}
        if unknown:
            raise ValueError(f"Unknown episode fields: {sorted(unknown)}")
        cols = {k: v for k, v in fields.items() if k in allowed}
        if "status" in cols and cols["status"] not in EPISODE_STATUSES:
            raise ValueError(f"Invalid episode status {cols['status']!r}")
        cols["metadata_json"] = json.dumps(fields.get("metadata") or {}, default=str)
        now = utcnow()
        names = ["slug", *cols.keys(), "created_at", "updated_at"]
        with self.tx() as c:
            cur = c.execute(
                f"INSERT INTO episodes({','.join(names)}) VALUES ({','.join('?' * len(names))})",
                (slug, *cols.values(), now, now),
            )
            return int(cur.lastrowid)

    def update_episode(self, episode_id: int, **fields: Any) -> None:
        if not fields:
            return
        if "status" in fields and fields["status"] not in EPISODE_STATUSES:
            raise ValueError(f"Invalid episode status {fields['status']!r}")
        if "metadata" in fields:
            existing = self.get_episode(episode_id) or {}
            merged = {**json.loads(existing.get("metadata_json") or "{}"), **fields.pop("metadata")}
            fields["metadata_json"] = json.dumps(merged, default=str)
        sets = ", ".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE episodes SET {sets}, updated_at=? WHERE id=?", (*fields.values(), utcnow(), episode_id))

    def get_episode(self, episode_id: int) -> dict[str, Any] | None:
        return self.query_one("SELECT * FROM episodes WHERE id=?", (episode_id,))

    def recent_titles(self, n: int = 50) -> list[str]:
        rows = self.conn.execute(
            "SELECT title FROM episodes WHERE title IS NOT NULL AND slug NOT LIKE 'preview-%' "
            "ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [r[0] for r in rows]

    def upload_for_episode(self, episode_id: int, kind: str = "long") -> dict[str, Any] | None:
        return self.query_one("SELECT * FROM uploads WHERE episode_id=? AND kind=? ORDER BY id DESC LIMIT 1",
                              (episode_id, kind))

    def uploads_by_kind(self, kind: str, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            return self.query("SELECT * FROM uploads WHERE kind=? AND status=? ORDER BY id", (kind, status))
        return self.query("SELECT * FROM uploads WHERE kind=? ORDER BY id", (kind,))

    def set_upload_metadata(self, upload_id: int, metadata: dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute("UPDATE uploads SET metadata_json=?, updated_at=? WHERE id=?",
                      (json.dumps(metadata, default=str), utcnow(), upload_id))

    def episodes_by_status(self, *statuses: str) -> list[dict[str, Any]]:
        marks = ",".join("?" * len(statuses))
        return self.query(f"SELECT * FROM episodes WHERE status IN ({marks}) ORDER BY id", statuses)

    def recent_episode_values(self, column: str, n: int) -> list[str]:
        """Most-recent-first values of a rotation column, for min-gap anti-repetition checks."""
        if column not in {"scripture_book", "scripture_ref", "ambient_bed", "visual_scene",
                          "title_template", "thumb_template"}:
            raise ValueError(f"Column {column!r} is not a rotation column")
        rows = self.conn.execute(
            f"SELECT {column} FROM episodes WHERE {column} IS NOT NULL AND status NOT IN ('failed','archived') "
            "AND slug NOT LIKE 'preview-%' ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [r[0] for r in rows]

    def used_scripture_refs(self) -> set[str]:
        rows = self.conn.execute(
            "SELECT scripture_ref FROM episodes WHERE scripture_ref IS NOT NULL AND status NOT IN "
            "('failed','archived') AND slug NOT LIKE 'preview-%'").fetchall()
        return {r[0] for r in rows}

    # ------------------------------------------------------------------ uniqueness
    def is_unique(self, kind: str, fingerprint: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM uniqueness_history WHERE kind=? AND fingerprint=?", (kind, fingerprint)
        ).fetchone() is None

    def find_similar_phash(self, kind: str, phash_hex: str, max_distance: int = 6) -> dict[str, Any] | None:
        """Nearest perceptual-hash neighbour within `max_distance` bits (Hamming), else None."""
        best: dict[str, Any] | None = None
        for row in self.conn.execute(
            "SELECT * FROM uniqueness_history WHERE kind=? AND algo='dhash64'", (kind,)
        ):
            d = hamming_hex(row["fingerprint"], phash_hex)
            if d <= max_distance and (best is None or d < best["distance"]):
                best = {**dict(row), "distance": d}
        return best

    def register_fingerprint(self, kind: str, fingerprint: str, *, episode_id: int | None = None,
                             algo: str = "sha256", detail: dict[str, Any] | str | None = None,
                             phash_max_distance: int = 6) -> None:
        """Record a fingerprint; raise DuplicateContentError if it (or a near-duplicate dHash) exists."""
        if algo == "dhash64":
            hit = self.find_similar_phash(kind, fingerprint, phash_max_distance)
            if hit:
                raise DuplicateContentError(
                    f"{kind} perceptually matches episode {hit['episode_id']} (distance {hit['distance']} bits)"
                )
        detail_s = detail if isinstance(detail, str) or detail is None else json.dumps(detail, default=str)
        try:
            with self.tx() as c:
                c.execute(
                    "INSERT INTO uniqueness_history(kind, fingerprint, algo, episode_id, detail, created_at)"
                    " VALUES (?,?,?,?,?,?)",
                    (kind, fingerprint, algo, episode_id, detail_s, utcnow()),
                )
        except sqlite3.IntegrityError as exc:
            prior = self.query_one(
                "SELECT episode_id, created_at FROM uniqueness_history WHERE kind=? AND fingerprint=?",
                (kind, fingerprint),
            )
            raise DuplicateContentError(
                f"{kind} fingerprint {fingerprint[:12]}... already used by episode "
                f"{prior and prior['episode_id']} at {prior and prior['created_at']}"
            ) from exc

    def release_fingerprints(self, episode_id: int) -> int:
        """Free fingerprints of an episode that failed before publishing so its parts can be reused."""
        with self.tx() as c:
            return c.execute("DELETE FROM uniqueness_history WHERE episode_id=?", (episode_id,)).rowcount

    # ------------------------------------------------------------------ render jobs
    def create_render_job(self, episode_id: int, stage: str, inputs: dict[str, Any] | None = None,
                          encoder: str | None = None) -> int:
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO render_jobs(episode_id, stage, encoder, input_json, created_at) VALUES (?,?,?,?,?)",
                (episode_id, stage, encoder, json.dumps(inputs or {}, default=str), utcnow()),
            )
            return int(cur.lastrowid)

    def update_render_job(self, job_id: int, *, status: str | None = None, progress: float | None = None,
                          output_path: str | None = None, error: str | None = None,
                          encoder: str | None = None) -> None:
        sets: list[str] = []
        vals: list[Any] = []
        if status is not None:
            sets.append("status=?")
            vals.append(status)
            if status == "running":
                sets += ["started_at=COALESCE(started_at, ?)", "attempts=attempts+1"]
                vals.append(utcnow())
            if status in {"done", "failed"}:
                sets.append("finished_at=?")
                vals.append(utcnow())
        for col, val in (("progress", progress), ("output_path", output_path), ("error", error),
                         ("encoder", encoder)):
            if val is not None:
                sets.append(f"{col}=?")
                vals.append(val)
        if not sets:
            return
        with self.tx() as c:
            c.execute(f"UPDATE render_jobs SET {', '.join(sets)} WHERE id=?", (*vals, job_id))

    def render_jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            return self.query("SELECT * FROM render_jobs WHERE status=? ORDER BY id", (status,))
        return self.query("SELECT * FROM render_jobs ORDER BY id DESC LIMIT 200")

    # ------------------------------------------------------------------ uploads
    def create_upload(self, file_path: str | Path, *, episode_id: int | None = None, kind: str = "long",
                      privacy: str = "private", metadata: dict[str, Any] | None = None,
                      file_sha256: str | None = None) -> int:
        p = Path(file_path)
        size = p.stat().st_size if p.exists() else None
        now = utcnow()
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO uploads(episode_id, kind, file_path, file_size, file_sha256, privacy, metadata_json,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (episode_id, kind, str(p), size, file_sha256, privacy, json.dumps(metadata or {}, default=str),
                 now, now),
            )
            return int(cur.lastrowid)

    def get_upload(self, upload_id: int) -> dict[str, Any] | None:
        return self.query_one("SELECT * FROM uploads WHERE id=?", (upload_id,))

    def set_upload_status(self, upload_id: int, status: str, **fields: Any) -> None:
        allowed = {"resumable_uri", "bytes_uploaded", "youtube_video_id", "processing_status", "last_error"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Unknown upload fields: {sorted(bad)}")
        with self.tx() as c:
            row = c.execute("SELECT status FROM uploads WHERE id=?", (upload_id,)).fetchone()
            if row is None:
                raise KeyError(f"upload {upload_id} not found")
            if status not in UPLOAD_TRANSITIONS.get(row["status"], set()):
                raise InvalidTransitionError(f"upload {upload_id}: {row['status']} -> {status} not allowed")
            sets = ["status=?", "updated_at=?"] + [f"{k}=?" for k in fields]
            vals: list[Any] = [status, utcnow(), *fields.values()]
            if status == "uploading" and row["status"] != "uploading":
                sets.append("attempts=attempts+1")
            c.execute(f"UPDATE uploads SET {', '.join(sets)} WHERE id=?", (*vals, upload_id))

    def save_upload_progress(self, upload_id: int, bytes_uploaded: int, resumable_uri: str | None = None) -> None:
        """Checkpoint called from inside the chunk loop; no status transition."""
        with self.tx() as c:
            if resumable_uri:
                c.execute("UPDATE uploads SET bytes_uploaded=?, resumable_uri=?, updated_at=? WHERE id=?",
                          (bytes_uploaded, resumable_uri, utcnow(), upload_id))
            else:
                c.execute("UPDATE uploads SET bytes_uploaded=?, updated_at=? WHERE id=?",
                          (bytes_uploaded, utcnow(), upload_id))

    def resumable_uploads(self) -> list[dict[str, Any]]:
        return self.query(
            "SELECT * FROM uploads WHERE status IN ('pending','uploading','failed') AND attempts < 10 ORDER BY id"
        )

    def uploads_awaiting_processing(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM uploads WHERE status IN ('uploaded','processing') ORDER BY id")

    # ------------------------------------------------------------------ A/B tests
    def create_ab_test(self, youtube_video_id: str, element: str, variants: dict[str, str], *,
                       episode_id: int | None = None, rotate_after_hours: int = 72) -> int:
        if len(variants) < 2:
            raise ValueError("An A/B test needs at least two variants")
        now = utcnow()
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO ab_tests(episode_id, youtube_video_id, element, rotate_after_hours, started_at)"
                " VALUES (?,?,?,?,?)",
                (episode_id, youtube_video_id, element, rotate_after_hours, now),
            )
            test_id = int(cur.lastrowid)
            for i, (label, payload) in enumerate(variants.items()):
                c.execute(
                    "INSERT INTO ab_variants(test_id, label, payload, is_active, active_from) VALUES (?,?,?,?,?)",
                    (test_id, label, payload, int(i == 0), now if i == 0 else None),
                )
            return test_id

    def ab_variants(self, test_id: int) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM ab_variants WHERE test_id=? ORDER BY id", (test_id,))

    def running_ab_tests(self) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM ab_tests WHERE status='running' ORDER BY id")

    def activate_variant(self, test_id: int, variant_id: int) -> None:
        now = utcnow()
        with self.tx() as c:
            c.execute("UPDATE ab_variants SET is_active=0, active_to=? WHERE test_id=? AND is_active=1",
                      (now, test_id))
            c.execute("UPDATE ab_variants SET is_active=1, active_from=?, active_to=NULL WHERE id=? AND test_id=?",
                      (now, variant_id, test_id))

    def record_variant_metrics(self, variant_id: int, *, impressions: int, views: int,
                               avg_view_duration_s: float | None = None, ctr: float | None = None) -> None:
        """`ctr` from YouTube reach reports overrides the views/impressions approximation."""
        if ctr is None:
            ctr = (views / impressions) if impressions else None
        with self.tx() as c:
            c.execute(
                "UPDATE ab_variants SET impressions=?, views=?, ctr=?, avg_view_duration_s=? WHERE id=?",
                (impressions, views, ctr, avg_view_duration_s, variant_id),
            )

    def conclude_ab_test(self, test_id: int, winner_variant_id: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE ab_tests SET status='concluded', winner_variant_id=?, ended_at=? WHERE id=?",
                      (winner_variant_id, utcnow(), test_id))

    # ------------------------------------------------------------------ analytics
    def add_analytics_snapshot(self, youtube_video_id: str, metrics: dict[str, Any], *,
                               window_start: str | None = None, window_end: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO analytics_snapshots(youtube_video_id, captured_at, window_start, window_end, views,"
                " watch_minutes, avg_view_duration_s, impressions, ctr, metrics_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    youtube_video_id, utcnow(), window_start, window_end, metrics.get("views"),
                    metrics.get("estimatedMinutesWatched"), metrics.get("averageViewDuration"),
                    metrics.get("impressions"), metrics.get("impressionsCtr"), json.dumps(metrics, default=str),
                ),
            )

    def latest_analytics(self) -> list[dict[str, Any]]:
        return self.query(
            """SELECT a.* FROM analytics_snapshots a
               JOIN (SELECT youtube_video_id, MAX(captured_at) m FROM analytics_snapshots GROUP BY youtube_video_id) l
                 ON a.youtube_video_id = l.youtube_video_id AND a.captured_at = l.m"""
        )

    # ------------------------------------------------------------------ media files
    def register_media(self, path: str | Path, kind: str, *, episode_id: int | None = None,
                       keep: bool = False) -> None:
        p = Path(path)
        size = p.stat().st_size if p.exists() else None
        with self.tx() as c:
            c.execute(
                """INSERT INTO media_files(episode_id, path, kind, size_bytes, keep, created_at) VALUES (?,?,?,?,?,?)
                   ON CONFLICT(path) DO UPDATE SET size_bytes=excluded.size_bytes, kind=excluded.kind,
                       deleted_at=NULL""",
                (episode_id, str(p), kind, size, int(keep), utcnow()),
            )

    def cleanup_candidates(self) -> list[dict[str, Any]]:
        """Local files for episodes whose long-form upload is fully processed ('live')."""
        return self.query(
            """SELECT m.* FROM media_files m
               JOIN uploads u ON u.episode_id = m.episode_id AND u.kind='long' AND u.status='live'
               WHERE m.deleted_at IS NULL AND m.keep=0"""
        )

    def mark_media_deleted(self, paths: Iterable[str]) -> None:
        now = utcnow()
        with self.tx() as c:
            c.executemany("UPDATE media_files SET deleted_at=? WHERE path=?", [(now, p) for p in paths])

    def disk_usage_bytes(self) -> int:
        row = self.conn.execute("SELECT COALESCE(SUM(size_bytes),0) FROM media_files WHERE deleted_at IS NULL")
        return int(row.fetchone()[0])

    # ------------------------------------------------------------------ kv
    def kv_set(self, key: str, value: Any) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO kv_state(key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, json.dumps(value, default=str), utcnow()),
            )

    def kv_get(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM kv_state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    # ------------------------------------------------------------------ dashboard
    def dashboard_summary(self) -> dict[str, Any]:
        q = self.quota_status()
        counts = {r["status"]: r["n"] for r in self.query("SELECT status, COUNT(*) n FROM episodes GROUP BY status")}
        uploads = {r["status"]: r["n"] for r in self.query("SELECT status, COUNT(*) n FROM uploads GROUP BY status")}
        return {
            "quota": {"day": q.day, "used": q.used, "limit": q.limit, "remaining": q.remaining,
                      "reserve": q.reserve},
            "episodes": counts,
            "uploads": uploads,
            "competitor_channels": self.conn.execute("SELECT COUNT(*) FROM competitor_channels").fetchone()[0],
            "competitor_videos": self.conn.execute("SELECT COUNT(*) FROM competitor_videos").fetchone()[0],
            "running_ab_tests": len(self.running_ab_tests()),
            "local_media_bytes": self.disk_usage_bytes(),
        }


_DB_SINGLETON: Database | None = None
_DB_LOCK = threading.Lock()


def get_db() -> Database:
    """Process-wide Database bound to the configured path and quota."""
    global _DB_SINGLETON
    with _DB_LOCK:
        if _DB_SINGLETON is None:
            from core.config import get_settings

            s = get_settings()
            _DB_SINGLETON = Database(s.db_path, daily_quota=s.daily_quota, quota_reserve=s.quota_reserve)
        return _DB_SINGLETON
