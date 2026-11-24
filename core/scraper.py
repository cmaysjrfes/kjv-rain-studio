"""Competitor data collection.

Two interchangeable sources produce the same normalised dicts:

* ``ApiCollector``   - YouTube Data API v3 (accurate stats, quota-metered).
* ``YtDlpCollector`` - yt-dlp public-page extraction (no quota; slower, best-effort stats).

``collect_competitors`` uses the API when a key is configured and falls back to yt-dlp
for anything the API could not provide (or for everything when no key is set).
Only public metadata is collected; no media is downloaded except thumbnails/banners
for colour/contrast analysis.
"""
from __future__ import annotations

import logging
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.youtube_client import YouTubeDataClient, parse_iso8601_duration
from db.database import Database, QuotaExceededError

log = logging.getLogger(__name__)

CHAPTER_RE = re.compile(r"^\s*\(?(\d{1,2}:)?\d{1,2}:\d{2}\)?\s+\S", re.MULTILINE)
ProgressFn = Callable[[str], None]


@dataclass
class CompetitorDataset:
    channels: list[dict[str, Any]] = field(default_factory=list)
    videos: list[dict[str, Any]] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


def has_chapters(description: str | None) -> bool:
    return len(CHAPTER_RE.findall(description or "")) >= 3


def _iso_from_yyyymmdd(v: str | None) -> str | None:
    if not v or len(v) != 8:
        return None
    return datetime.strptime(v, "%Y%m%d").replace(tzinfo=timezone.utc).isoformat()


def median_gap_days(published: list[str | None]) -> float | None:
    stamps = sorted(datetime.fromisoformat(p.replace("Z", "+00:00")) for p in published if p)
    if len(stamps) < 3:
        return None
    gaps = [(b - a).total_seconds() / 86400 for a, b in zip(stamps, stamps[1:])]
    return round(statistics.median(gaps), 2)


# ============================================================================ API
class ApiCollector:
    def __init__(self, client: YouTubeDataClient, settings: dict[str, Any]) -> None:
        self.client = client
        self.cfg = settings

    def discover(self, progress: ProgressFn) -> tuple[list[str], list[dict[str, Any]]]:
        """search.list each query (100 units each) -> keep only 8h+ videos via videos.list."""
        ids: list[str] = []
        for q in self.cfg["search_queries"]:
            progress(f"API search: {q!r}")
            for item in self.client.search_videos(q, self.cfg.get("max_results_per_query", 50)):
                vid = item.get("id", {}).get("videoId")
                if vid:
                    ids.append(vid)
        videos = [self._norm_video(v) for v in self.client.videos(ids)]
        min_d = int(self.cfg.get("min_video_duration_s", 28800))
        long_videos = [v for v in videos if v["duration_s"] >= min_d]
        channel_ids = list(dict.fromkeys(v["channel_id"] for v in long_videos))
        progress(f"API: {len(long_videos)} videos >= {min_d // 3600}h across {len(channel_ids)} channels")
        return channel_ids, long_videos

    def channel_details(self, channel_ids: list[str], progress: ProgressFn) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        per_ch = int(self.cfg.get("videos_per_channel", 30))
        for raw in self.client.channels(channel_ids):
            ch = self._norm_channel(raw)
            progress(f"API channel: {ch['title']}")
            uploads_pl = raw.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
            if uploads_pl:
                items = self.client.playlist_items(uploads_pl, per_ch)
                vids = self.client.videos([i["contentDetails"]["videoId"] for i in items])
                ch["uploads"] = [self._norm_video(v) for v in vids]
            ch["playlists"] = [
                {"title": p["snippet"]["title"], "item_count": p.get("contentDetails", {}).get("itemCount", 0)}
                for p in self.client.playlists(ch["channel_id"])
            ]
            out.append(ch)
        return out

    @staticmethod
    def _norm_video(v: dict[str, Any]) -> dict[str, Any]:
        sn, st = v.get("snippet", {}), v.get("statistics", {})
        thumbs = sn.get("thumbnails", {})
        thumb = (thumbs.get("maxres") or thumbs.get("high") or thumbs.get("medium") or {}).get("url")
        desc = sn.get("description", "")
        return {
            "video_id": v["id"], "channel_id": sn.get("channelId", ""), "channel_title": sn.get("channelTitle", ""),
            "title": sn.get("title", ""), "description": desc, "published_at": sn.get("publishedAt"),
            "duration_s": parse_iso8601_duration(v.get("contentDetails", {}).get("duration")),
            "view_count": int(st.get("viewCount", 0)), "like_count": int(st.get("likeCount", 0)),
            "comment_count": int(st.get("commentCount", 0)), "thumbnail_url": thumb,
            "tags": sn.get("tags", []), "has_chapters": has_chapters(desc), "source": "api",
        }

    @staticmethod
    def _norm_channel(c: dict[str, Any]) -> dict[str, Any]:
        sn, st, br = c.get("snippet", {}), c.get("statistics", {}), c.get("brandingSettings", {})
        thumbs = sn.get("thumbnails", {})
        return {
            "channel_id": c["id"], "title": sn.get("title", ""), "handle": sn.get("customUrl"),
            "description": sn.get("description", ""), "country": sn.get("country"),
            "keywords": br.get("channel", {}).get("keywords"),
            "banner_url": br.get("image", {}).get("bannerExternalUrl"),
            "avatar_url": (thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}).get("url"),
            "subscriber_count": int(st.get("subscriberCount", 0)) if not st.get("hiddenSubscriberCount") else None,
            "view_count": int(st.get("viewCount", 0)), "video_count": int(st.get("videoCount", 0)),
            "uploads": [], "playlists": [], "source": "api",
        }


# ============================================================================ yt-dlp
class YtDlpCollector:
    def __init__(self, settings: dict[str, Any]) -> None:
        import yt_dlp  # local import: heavy module

        self._yt_dlp = yt_dlp
        self.cfg = settings
        self.base_opts: dict[str, Any] = {
            "quiet": True, "no_warnings": True, "skip_download": True, "ignoreerrors": True,
            "socket_timeout": int(settings.get("request_timeout_s", 20)), "extractor_retries": 3,
        }

    def _extract(self, url: str, **opts: Any) -> dict[str, Any] | None:
        with self._yt_dlp.YoutubeDL({**self.base_opts, **opts}) as ydl:
            try:
                return ydl.extract_info(url, download=False)
            except self._yt_dlp.utils.DownloadError as exc:
                log.warning("yt-dlp failed for %s: %s", url, exc)
                return None

    def discover(self, progress: ProgressFn) -> tuple[list[str], list[dict[str, Any]]]:
        n = int(self.cfg.get("max_results_per_query", 50))
        min_d = int(self.cfg.get("min_video_duration_s", 28800))
        seen: dict[str, dict[str, Any]] = {}
        for q in self.cfg["search_queries"]:
            progress(f"yt-dlp search: {q!r}")
            info = self._extract(f"ytsearch{n}:{q}", extract_flat="in_playlist") or {}
            for e in info.get("entries") or []:
                if not e or not e.get("id"):
                    continue
                if (e.get("duration") or 0) >= min_d:
                    seen[e["id"]] = self._norm_flat(e)
        channel_ids = list(dict.fromkeys(v["channel_id"] for v in seen.values() if v["channel_id"]))
        progress(f"yt-dlp: {len(seen)} videos >= {min_d // 3600}h across {len(channel_ids)} channels")
        return channel_ids, list(seen.values())

    def enrich_video(self, video: dict[str, Any]) -> dict[str, Any]:
        """Full page extraction for description/tags/chapters/likes (one request per video)."""
        info = self._extract(f"https://www.youtube.com/watch?v={video['video_id']}")
        if not info:
            return video
        desc = info.get("description") or ""
        video.update({
            "description": desc, "tags": info.get("tags") or [],
            "like_count": info.get("like_count") or 0, "comment_count": info.get("comment_count") or 0,
            "view_count": info.get("view_count") or video.get("view_count") or 0,
            "published_at": _iso_from_yyyymmdd(info.get("upload_date")) or video.get("published_at"),
            "has_chapters": bool(info.get("chapters")) or has_chapters(desc),
            "thumbnail_url": info.get("thumbnail") or video.get("thumbnail_url"),
        })
        return video

    def channel_details(self, channel_ids: list[str], progress: ProgressFn) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        per_ch = int(self.cfg.get("videos_per_channel", 30))
        for cid in channel_ids:
            info = self._extract(f"https://www.youtube.com/channel/{cid}/videos",
                                 extract_flat="in_playlist", playlistend=per_ch)
            if not info:
                continue
            progress(f"yt-dlp channel: {info.get('channel') or cid}")
            thumbs = {t.get("id"): t.get("url") for t in info.get("thumbnails") or []}
            banner = thumbs.get("banner_uncropped") or next(
                (t.get("url") for t in info.get("thumbnails") or [] if (t.get("width") or 0) >= 2000), None)
            uploads = [self._norm_flat({**e, "channel_id": cid}) for e in info.get("entries") or [] if e]
            pl_info = self._extract(f"https://www.youtube.com/channel/{cid}/playlists",
                                    extract_flat="in_playlist", playlistend=30) or {}
            out.append({
                "channel_id": cid, "title": info.get("channel") or info.get("title") or cid,
                "handle": info.get("uploader_id"), "description": info.get("description") or "",
                "country": None, "keywords": " ".join(info.get("tags") or []) or None,
                "banner_url": banner, "avatar_url": thumbs.get("avatar_uncropped"),
                "subscriber_count": info.get("channel_follower_count"), "view_count": None,
                "video_count": info.get("playlist_count"), "uploads": uploads,
                "playlists": [{"title": e.get("title"), "item_count": e.get("playlist_count") or 0}
                              for e in pl_info.get("entries") or [] if e and e.get("title")],
                "source": "yt-dlp",
            })
        return out

    @staticmethod
    def _norm_flat(e: dict[str, Any]) -> dict[str, Any]:
        thumbs = e.get("thumbnails") or []
        thumb = thumbs[-1]["url"] if thumbs else f"https://i.ytimg.com/vi/{e['id']}/hqdefault.jpg"
        ts = e.get("timestamp") or e.get("release_timestamp")
        return {
            "video_id": e["id"], "channel_id": e.get("channel_id") or "", "channel_title": e.get("channel") or "",
            "title": e.get("title") or "", "description": e.get("description") or "",
            "published_at": datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts
            else _iso_from_yyyymmdd(e.get("upload_date")),
            "duration_s": int(e.get("duration") or 0), "view_count": int(e.get("view_count") or 0),
            "like_count": 0, "comment_count": 0, "thumbnail_url": thumb.split("?")[0],
            "tags": [], "has_chapters": False, "source": "yt-dlp",
        }


# ============================================================================ orchestration
def score_channel(ch: dict[str, Any], long_videos: list[dict[str, Any]]) -> float:
    """Rank by proven demand for *long-form* sleep content, not raw channel size."""
    mine = [v for v in long_videos if v["channel_id"] == ch["channel_id"]]
    long_views = sum(v.get("view_count") or 0 for v in mine)
    subs = ch.get("subscriber_count") or 0
    consistency = 1.0
    gap = ch.get("median_upload_gap_days")
    if gap:
        consistency = 1.25 if gap <= 3 else 1.1 if gap <= 7 else 1.0 if gap <= 21 else 0.8
    return round((long_views ** 0.5 * 0.7 + subs ** 0.5 * 0.3) * consistency + len(mine) * 10, 2)


def collect_competitors(db: Database, settings: dict[str, Any], *, api_key: str | None,
                        progress: ProgressFn, enrich_with_ytdlp: bool = True) -> CompetitorDataset:
    ds = CompetitorDataset()
    min_d = int(settings.get("min_video_duration_s", 28800))
    top_n = int(settings.get("top_channels", 15))
    channel_ids: list[str] = []
    long_videos: list[dict[str, Any]] = []
    api: ApiCollector | None = None

    if api_key:
        try:
            api = ApiCollector(YouTubeDataClient(db, api_key=api_key, agent="agent1"), settings)
            channel_ids, long_videos = api.discover(progress)
            ds.sources.add("api")
        except QuotaExceededError as exc:
            ds.warnings.append(f"API quota exhausted during discovery, falling back to yt-dlp: {exc}")
            api = None
        except Exception as exc:  # noqa: BLE001 - any API failure falls back to yt-dlp
            ds.warnings.append(f"YouTube API discovery failed ({type(exc).__name__}: {exc}); using yt-dlp")
            api = None

    ytdlp: YtDlpCollector | None = None
    if not long_videos:
        ytdlp = YtDlpCollector(settings)
        channel_ids, long_videos = ytdlp.discover(progress)
        ds.sources.add("yt-dlp")

    # Rank channels by long-form views before spending requests on details
    views_by_ch: dict[str, int] = {}
    for v in long_videos:
        views_by_ch[v["channel_id"]] = views_by_ch.get(v["channel_id"], 0) + (v.get("view_count") or 0)
    channel_ids = sorted(channel_ids, key=lambda c: views_by_ch.get(c, 0), reverse=True)[: top_n * 2]

    channels: list[dict[str, Any]] = []
    if api is not None:
        try:
            channels = api.channel_details(channel_ids, progress)
        except QuotaExceededError as exc:
            ds.warnings.append(f"Quota exhausted fetching channel details; yt-dlp fallback: {exc}")
    if not channels:
        ytdlp = ytdlp or YtDlpCollector(settings)
        channels = ytdlp.channel_details(channel_ids, progress)
        ds.sources.add("yt-dlp")

    for ch in channels:
        uploads = ch.get("uploads") or []
        ch["median_upload_gap_days"] = median_gap_days([u.get("published_at") for u in uploads])
        ch["long_video_count"] = sum(1 for u in uploads if (u.get("duration_s") or 0) >= min_d)
        ch["score"] = score_channel(ch, long_videos)
    channels.sort(key=lambda c: c["score"], reverse=True)
    channels = channels[:top_n]
    keep = {c["channel_id"] for c in channels}
    videos = sorted((v for v in long_videos if v["channel_id"] in keep),
                    key=lambda v: v.get("view_count") or 0, reverse=True)

    if enrich_with_ytdlp and videos and not videos[0].get("description"):
        ytdlp = ytdlp or YtDlpCollector(settings)
        for v in videos[: int(settings.get("thumbnails_to_analyze", 60))]:
            progress(f"yt-dlp enrich: {v['title'][:60]}")
            ytdlp.enrich_video(v)

    ds.channels, ds.videos = channels, videos
    return ds
