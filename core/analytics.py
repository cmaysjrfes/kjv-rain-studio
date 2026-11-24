"""YouTube Analytics API v2 + YouTube Reporting API (reach: thumbnail impressions & CTR).

Thumbnail impressions/CTR are not exposed by the Analytics API; they arrive in daily bulk
"reach" reports from the Reporting API (1-2 day lag, after the reporting job has existed
for ~48 h). ReachReports discovers the reach report type dynamically and degrades to
an empty result if your channel does not have it yet.
"""
from __future__ import annotations

import csv
import io
import logging
import random
import time
from collections.abc import Callable
from datetime import date
from typing import Any

from google.auth.transport.requests import AuthorizedSession
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from db.database import Database

log = logging.getLogger(__name__)

TOTAL_METRICS = "views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,likes,subscribersGained"


def _retry(fn: Callable[[], Any], what: str, attempts: int = 6) -> Any:
    for i in range(attempts):
        try:
            return fn()
        except HttpError as exc:
            status = int(getattr(exc.resp, "status", 0) or 0)
            if status in (429, 500, 502, 503, 504) and i < attempts - 1:
                delay = min(60.0, 2 ** i) + random.random()
                log.warning("%s HTTP %s; retry in %.1fs", what, status, delay)
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("unreachable")


def _rows(resp: dict[str, Any]) -> list[dict[str, Any]]:
    cols = [h["name"] for h in resp.get("columnHeaders", [])]
    return [dict(zip(cols, r)) for r in resp.get("rows", []) or []]


class AnalyticsClient:
    def __init__(self, credentials: Any, db: Database, agent: str = "agent5_analytics") -> None:
        self.svc = build("youtubeAnalytics", "v2", credentials=credentials, cache_discovery=False)
        self.db = db
        self.agent = agent

    def _query(self, **kw: Any) -> list[dict[str, Any]]:
        self.db.consume_quota("reports.query", 1, agent=self.agent, api="youtube_analytics", enforce=False)
        return _rows(_retry(lambda: self.svc.reports().query(ids="channel==MINE", **kw).execute(), "analytics"))

    def video_totals(self, video_ids: list[str], start: date, end: date) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for i in range(0, len(video_ids), 200):
            batch = video_ids[i:i + 200]
            for r in self._query(startDate=start.isoformat(), endDate=end.isoformat(), metrics=TOTAL_METRICS,
                                 dimensions="video", filters=f"video=={','.join(batch)}", maxResults=200):
                out[r["video"]] = r
        return out

    def daily(self, video_id: str, start: date, end: date) -> list[dict[str, Any]]:
        return self._query(startDate=start.isoformat(), endDate=end.isoformat(), dimensions="day",
                           metrics="views,estimatedMinutesWatched,averageViewDuration",
                           filters=f"video=={video_id}", sort="day")

    def retention(self, video_id: str, start: date, end: date) -> list[dict[str, Any]]:
        return self._query(startDate=start.isoformat(), endDate=end.isoformat(),
                           dimensions="elapsedVideoTimeRatio",
                           metrics="audienceWatchRatio,relativeRetentionPerformance",
                           filters=f"video=={video_id}")

    def traffic_sources(self, video_id: str, start: date, end: date) -> list[dict[str, Any]]:
        return self._query(startDate=start.isoformat(), endDate=end.isoformat(),
                           dimensions="insightTrafficSourceType", metrics="views,estimatedMinutesWatched",
                           filters=f"video=={video_id}", sort="-views")


class ReachReports:
    KV = "reach_daily"
    KV_SEEN = "reach_reports_seen"

    def __init__(self, credentials: Any, db: Database) -> None:
        self.svc = build("youtubereporting", "v1", credentials=credentials, cache_discovery=False)
        self.http = AuthorizedSession(credentials)
        self.db = db

    def _report_type(self) -> str | None:
        types = _retry(lambda: self.svc.reportTypes().list().execute(), "reportTypes").get("reportTypes", [])
        ids = [t["id"] for t in types]
        reach = [i for i in ids if "reach" in i and i.startswith("channel_")]
        return sorted(reach)[0] if reach else None

    def _job(self, report_type: str) -> str:
        jobs = _retry(lambda: self.svc.jobs().list().execute(), "jobs.list").get("jobs", [])
        for j in jobs:
            if j.get("reportTypeId") == report_type:
                return j["id"]
        created = _retry(lambda: self.svc.jobs().create(body={"reportTypeId": report_type,
                                                              "name": "kjv-studio-reach"}).execute(), "jobs.create")
        log.info("Created reporting job %s (%s); first reports arrive in ~48 h", created["id"], report_type)
        return created["id"]

    def sync(self) -> dict[str, dict[str, dict[str, float]]]:
        """Download new reach reports -> {video_id: {date: {impressions, ctr}}} persisted in kv_state."""
        data: dict[str, dict[str, dict[str, float]]] = self.db.kv_get(self.KV, {}) or {}
        rtype = self._report_type()
        if not rtype:
            log.info("No channel reach report type available; CTR A/B testing falls back to views/hour")
            return data
        job = self._job(rtype)
        seen = set(self.db.kv_get(self.KV_SEEN, []) or [])
        reports = _retry(lambda: self.svc.jobs().reports().list(jobId=job).execute(), "reports.list")
        for rep in reports.get("reports", []):
            if rep["id"] in seen:
                continue
            resp = self.http.get(rep["downloadUrl"], timeout=120)
            resp.raise_for_status()
            for row in csv.DictReader(io.StringIO(resp.content.decode("utf-8"))):
                vid, day = row.get("video_id"), row.get("date")
                if not vid or not day:
                    continue
                imp_key = next((k for k in row if "impressions" in k and "ctr" not in k), None)
                ctr_key = next((k for k in row if "ctr" in k), None)
                if not imp_key:
                    continue
                d = data.setdefault(vid, {}).setdefault(day, {"impressions": 0.0, "clicks": 0.0})
                imp = float(row[imp_key] or 0)
                d["impressions"] += imp
                if ctr_key and row[ctr_key]:
                    d["clicks"] += imp * float(row[ctr_key])
            seen.add(rep["id"])
        cutoff = date.fromordinal(date.today().toordinal() - 120)
        for vid in list(data):
            data[vid] = {k: v for k, v in data[vid].items() if parse_day(k) >= cutoff}
        self.db.kv_set(self.KV, data)
        self.db.kv_set(self.KV_SEEN, sorted(seen)[-1000:])
        return data


def parse_day(day: str) -> date:
    """Reporting API dates are YYYYMMDD; accept ISO too."""
    return date(int(day[:4]), int(day[4:6]), int(day[6:8])) if len(day) == 8 else date.fromisoformat(day)


def reach_window(reach: dict[str, dict[str, dict[str, float]]], video_id: str, start: date,
                 end: date) -> tuple[int, float | None]:
    """Sum impressions and derive CTR for a date window (dates stored as YYYYMMDD)."""
    imp = clicks = 0.0
    for day, v in (reach.get(video_id) or {}).items():
        if start <= parse_day(day) <= end:
            imp += v.get("impressions", 0.0)
            clicks += v.get("clicks", 0.0)
    return int(imp), (clicks / imp if imp else None)
