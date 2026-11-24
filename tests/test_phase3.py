"""Phase 3 tests: planner rotation, metadata, Shorts selection, resumable uploads, processing poll,
A/B rotation, cleanup safety and production planning - all offline with fakes."""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from agents.agent1_competitor_intel import BASELINE_PRIORS, build_content_blueprint
from agents.agent4_publisher import PublisherAgent
from agents.agent5_analytics import AnalyticsDirectorAgent
from core.config import get_settings
from core.metadata import build_description, render_title
from core.planner import EpisodePlanner
from core.shorts import select_windows
from core.uploader import ResumableUploader, UploadError
from core.youtube_client import YouTubeDataClient
from db.database import Database


def blueprint() -> dict[str, Any]:
    analysis = {**BASELINE_PRIORS, "data_source": "baseline_prior"}
    return build_content_blueprint(analysis, {"name": "T", "palette": {}}, get_settings().content, "", "")


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = Database(self.root / "s.sqlite3")
        self.settings = dataclasses.replace(get_settings(), output_dir=self.root / "out", db_path=self.root / "s.db",
                                            preview_seconds=0)
        self.settings.output_dir.mkdir()

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()


# ---------------------------------------------------------------------------- planner / metadata / shorts
class PlannerTests(Base):
    def test_rotation_gaps_and_unique_combos(self) -> None:
        bp = blueprint()
        planner = EpisodePlanner(self.db, bp)
        plans = [planner.plan_next(seed=i)[1] for i in range(12)]
        books = [p.book for p in plans]
        gap = bp["anti_repetition_rules"]["min_episodes_between"]["book"]
        for i in range(len(books)):
            self.assertNotIn(books[i], books[max(0, i - gap):i], f"book reused within {gap}: {books}")
        beds = [p.bed["id"] for p in plans]
        for i in range(1, len(beds)):
            self.assertNotEqual(beds[i], beds[i - 1])
        self.assertEqual(len({p.passage_set["id"] for p in plans}), 12)  # fresh sets first
        for p in plans:
            self.assertTrue(28800 <= p.duration_s <= 36000)

    def test_preview_plans_do_not_consume_rotation(self) -> None:
        planner = EpisodePlanner(self.db, blueprint())
        planner.plan_next(seed=1, preview=True)
        self.assertEqual(self.db.used_scripture_refs(), set())
        self.assertEqual(self.db.query("SELECT COUNT(*) n FROM uniqueness_history")[0]["n"], 0)

    def test_titles_unique_and_within_limit(self) -> None:
        bp = blueprint()
        planner = EpisodePlanner(self.db, bp)
        ep, plan = planner.plan_next(seed=3)
        t1, _ = render_title(plan, bp, self.db)
        self.assertLessEqual(len(t1), 100)
        self.assertNotIn("<", t1)
        self.db.update_episode(ep, title=t1)
        from core.metadata import norm_title
        from db.database import sha256_fingerprint
        self.db.register_fingerprint("title", sha256_fingerprint(norm_title(t1)), episode_id=ep)
        t2, _ = render_title(plan, bp, self.db)
        self.assertNotEqual(t1, t2)

    def test_description_rules(self) -> None:
        bp = blueprint()
        _, plan = EpisodePlanner(self.db, bp).plan_next(seed=5)
        chapters = [{"start_s": 0, "label": "Opening"}, {"start_s": 60, "label": "Psalm 1"},
                    {"start_s": 9000, "label": "Narration Ends"}]
        audio = {"narration_end_s": 9000, "passage_display": "Psalms 1–24", "chapters": chapters,
                 "chapters_read": ["Psalms 1"], "first_verse": "Psalms 1:1", "last_verse": "Psalms 24:10"}
        d = build_description(plan, audio, {"name": "X"}, bp, self.settings)
        self.assertIn("0:00 Opening", d)
        self.assertIn("2:30:00 Narration Ends", d)
        self.assertIn("2.5 hours", d)
        self.assertLessEqual(len(d), 5000)
        self.assertEqual(len([w for w in d.split() if w.startswith("#")]), 3)


class ShortsTests(unittest.TestCase):
    def test_windows_non_overlapping_and_bounded(self) -> None:
        tl, t = [], 0.0
        for ch in (1, 2, 3):
            for v in range(1, 15):
                text = "Peace I leave with you my peace I give unto you" if v % 5 == 0 else "And it came to pass"
                tl.append({"kind": "verse", "ref": f"John {ch}:{v}", "start": t, "end": t + 6, "display": text,
                           "chapter_key": f"John {ch}"})
                t += 8
        wins = select_windows(tl, 3)
        self.assertEqual(len(wins), 3)
        self.assertEqual(len({w["chapter"] for w in wins}), 3)
        for a, b in zip(wins, wins[1:]):
            self.assertLess(a["end"], b["start"])
        for w in wins:
            self.assertLessEqual(w["end"] - w["start"], 52)
            self.assertLessEqual(len(w["text"].split()), 45 + 12)


# ---------------------------------------------------------------------------- resumable upload
class FakeResp:
    def __init__(self, status: int, headers: dict[str, str] | None = None, body: Any = None) -> None:
        self.status_code = status
        self.headers = headers or {}
        self._body = body or {}
        self.text = json.dumps(self._body)

    def json(self) -> Any:
        return self._body


class FakeYouTubeUploadServer:
    """Implements the resumable protocol; can inject a 503 and a dropped connection."""

    def __init__(self, fail_plan: list[str] | None = None) -> None:
        self.received = bytearray()
        self.sessions = 0
        self.size = 0
        self.fail_plan = list(fail_plan or [])

    def post(self, url: str, params: Any = None, json: Any = None, headers: Any = None, timeout: Any = None):
        self.sessions += 1
        self.size = int(headers["X-Upload-Content-Length"])
        return FakeResp(200, {"Location": f"https://upload/session/{self.sessions}"})

    def put(self, uri: str, data: bytes | None = None, headers: Any = None, timeout: Any = None):
        cr = headers["Content-Range"]
        if cr.startswith("bytes */"):
            if len(self.received) >= self.size:
                return FakeResp(200, body={"id": "VID123", "status": {"uploadStatus": "uploaded"}})
            return FakeResp(308, {"Range": f"bytes=0-{len(self.received) - 1}"} if self.received else {})
        action = self.fail_plan.pop(0) if self.fail_plan else "ok"
        if action == "drop":
            raise requests.ConnectionError("connection reset")
        if action == "503":
            return FakeResp(503)
        start = int(cr.split()[1].split("-")[0])
        assert start == len(self.received), f"server expected offset {len(self.received)}, got {start}"
        self.received.extend(data or b"")
        if len(self.received) >= self.size:
            return FakeResp(201, body={"id": "VID123", "status": {"uploadStatus": "uploaded"}})
        return FakeResp(308, {"Range": f"bytes=0-{len(self.received) - 1}"})


class CrashAfter(Exception):
    pass


class UploaderTests(Base):
    def _file(self, n: int) -> Path:
        p = self.root / "video.mp4"
        p.write_bytes(bytes(range(256)) * (n // 256))
        return p

    def test_retries_and_byte_exactness(self) -> None:
        f = self._file(3 * 1024 * 1024)
        server = FakeYouTubeUploadServer(["ok", "503", "drop", "ok"])
        up = self.db.create_upload(f)
        u = ResumableUploader(None, self.db, chunk_mb=1, session=server)
        u._sleep = staticmethod(lambda *a: None)  # type: ignore[method-assign]
        res = u.upload(up, {"snippet": {}, "status": {}})
        self.assertEqual(res["id"], "VID123")
        self.assertEqual(bytes(server.received), f.read_bytes())
        self.assertEqual(server.sessions, 1)
        self.assertEqual(self.db.quota_status().used, 1600)

    def test_resume_after_process_crash(self) -> None:
        f = self._file(4 * 1024 * 1024)
        server = FakeYouTubeUploadServer()
        up = self.db.create_upload(f)
        calls = {"n": 0}
        real_put = server.put

        def crashing_put(*a: Any, **k: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 3:
                raise CrashAfter()
            return real_put(*a, **k)

        server.put = crashing_put  # type: ignore[method-assign]
        with self.assertRaises(CrashAfter):
            ResumableUploader(None, self.db, chunk_mb=1, session=server).upload(up, {"snippet": {}})
        row = self.db.get_upload(up)
        self.assertEqual(row["bytes_uploaded"], 2 * 1024 * 1024)
        self.assertTrue(row["resumable_uri"])
        server.put = real_put  # type: ignore[method-assign]
        res = ResumableUploader(None, self.db, chunk_mb=1, session=server).upload(up, {"snippet": {}})
        self.assertEqual(res["id"], "VID123")
        self.assertEqual(bytes(server.received), f.read_bytes())
        self.assertEqual(server.sessions, 1, "must resume the same session, not restart")
        self.assertEqual(self.db.quota_status().used, 1600)

    def test_non_retryable_error_refunds_quota(self) -> None:
        f = self._file(1024 * 1024)
        up = self.db.create_upload(f)

        class Bad(FakeYouTubeUploadServer):
            def post(self, *a: Any, **k: Any) -> FakeResp:
                return FakeResp(400, body={"error": {"errors": [{"reason": "invalidTitle"}]}})

        with self.assertRaises(UploadError):
            ResumableUploader(None, self.db, chunk_mb=1, session=Bad()).upload(up, {"snippet": {}})
        self.assertEqual(self.db.quota_status().used, 0)


# ---------------------------------------------------------------------------- processing poll / A-B
class _Req:
    def __init__(self, value: Any) -> None:
        self.value = value

    def execute(self) -> Any:
        return self.value


class FakeVideos:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items
        self.updated: list[dict[str, Any]] = []

    def list(self, **kw: Any) -> _Req:
        ids = kw["id"].split(",")
        return _Req({"items": [i for i in self.items if i["id"] in ids]})

    def update(self, part: str, body: dict[str, Any]) -> _Req:
        self.updated.append(body)
        return _Req(body)


class FakeThumbs:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def set(self, videoId: str, media_body: Any) -> _Req:
        self.calls.append(videoId)
        return _Req({})


class FakeService:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._videos = FakeVideos(items)
        self._thumbs = FakeThumbs()

    def videos(self) -> FakeVideos:
        return self._videos

    def thumbnails(self) -> FakeThumbs:
        return self._thumbs


def fake_client(db: Database, service: FakeService) -> YouTubeDataClient:
    c = YouTubeDataClient.__new__(YouTubeDataClient)
    c.db, c.agent, c.max_retries, c.service = db, "test", 0, service
    return c


class ProcessingAndABTests(Base):
    def test_poll_marks_live_only_when_hd(self) -> None:
        f = self.root / "v.mp4"
        f.write_bytes(b"x")
        ups = []
        for vid in ("A", "B", "C"):
            u = self.db.create_upload(f)
            self.db.set_upload_status(u, "uploading")
            self.db.set_upload_status(u, "uploaded", youtube_video_id=vid)
            ups.append(u)
        svc = FakeService([
            {"id": "A", "status": {"uploadStatus": "processed"}, "processingDetails": {"processingStatus": "succeeded"},
             "contentDetails": {"definition": "hd"}},
            {"id": "B", "status": {"uploadStatus": "processed"}, "processingDetails": {"processingStatus": "succeeded"},
             "contentDetails": {"definition": "sd"}},
            {"id": "C", "status": {"uploadStatus": "rejected", "rejectionReason": "duplicate"},
             "processingDetails": {"processingStatus": "failed"}, "contentDetails": {}},
        ])
        agent = PublisherAgent(self.settings, self.db)
        agent.client = fake_client(self.db, svc)
        self.assertEqual(agent._poll_processing(), 3)
        self.assertEqual([self.db.get_upload(u)["status"] for u in ups], ["live", "processing", "failed"])
        self.assertIn("duplicate", self.db.get_upload(ups[2])["last_error"])

    def test_ab_rotation_then_winner(self) -> None:
        thumbs = []
        for label in "AB":
            p = self.root / f"t{label}.jpg"
            p.write_bytes(b"jpg")
            thumbs.append(str(p))
        test = self.db.create_ab_test("VID", "thumbnail", {"A": thumbs[0], "B": thumbs[1]}, rotate_after_hours=72)
        past = (datetime.now(timezone.utc) - timedelta(hours=80)).isoformat(timespec="seconds")
        self.db.conn.execute("UPDATE ab_variants SET active_from=? WHERE test_id=? AND label='A'", (past, test))

        class FakeAnalytics:
            def __init__(self, views: int) -> None:
                self.views = views

            def daily(self, *a: Any) -> list[dict[str, Any]]:
                return [{"views": self.views, "averageViewDuration": 1800}]

        svc = FakeService([])
        agent = AnalyticsDirectorAgent(self.settings, self.db)
        agent.data = fake_client(self.db, svc)
        r = agent._rotate_tests(FakeAnalytics(100), {"VID": {"20260101": {"impressions": 1, "clicks": 0}}}, [])
        self.assertEqual(r, {"rotated": 1, "concluded": 0})
        self.assertEqual(svc._thumbs.calls, ["VID"])
        vs = self.db.ab_variants(test)
        self.assertEqual([v["is_active"] for v in vs], [0, 1])
        self.db.conn.execute("UPDATE ab_variants SET active_from=? WHERE test_id=? AND label='B'", (past, test))
        r = agent._rotate_tests(FakeAnalytics(400), {}, [])
        self.assertEqual(r["concluded"], 1)
        winner = self.db.query("SELECT winner_variant_id FROM ab_tests WHERE id=?", (test,))[0]["winner_variant_id"]
        self.assertEqual(winner, vs[1]["id"])  # B had more views/hour


class CleanupAndPlanTests(Base):
    def test_cleanup_never_leaves_output_dir(self) -> None:
        ep = self.db.create_episode("ep")
        inside = self.settings.output_dir / "episodes" / "ep" / "final.mp4"
        inside.parent.mkdir(parents=True)
        inside.write_bytes(b"x" * 100)
        outside = self.root / "precious.mp4"
        outside.write_bytes(b"y")
        for p in (inside, outside):
            self.db.register_media(p, "final_mp4", episode_id=ep)
        u = self.db.create_upload(inside, episode_id=ep)
        self.db.set_upload_status(u, "uploading")
        self.db.set_upload_status(u, "uploaded", youtube_video_id="V")
        self.db.set_upload_status(u, "live")
        agent = AnalyticsDirectorAgent(self.settings, self.db)
        res = agent._cleanup()
        self.assertEqual(res["files"], 1)
        self.assertFalse(inside.exists())
        self.assertTrue(outside.exists())

    def test_production_plan_respects_backlog_and_quota(self) -> None:
        from core.config import PROJECT_ROOT
        if not (PROJECT_ROOT / "config" / "channel_page_blueprint.json").exists():
            self.skipTest("run build_channel_page.py first")
        agent = AnalyticsDirectorAgent(self.settings, self.db, dry_run=True)
        plan = agent._plan_production([])
        self.assertEqual(plan["long_form_today"], 1)
        for i in range(2):
            self.db.create_episode(f"b{i}", status="video_ready")
        plan = agent._plan_production([])
        self.assertEqual(plan["long_form_today"], 0)
        self.assertTrue(any("backlog" in r for r in plan["reasons"]))


if __name__ == "__main__":
    unittest.main()


class ChannelGuardTests(Base):
    def _agent(self, expected: str) -> PublisherAgent:
        class Channels:
            def list(self, **kw: Any) -> _Req:
                return _Req({"items": [{"id": "UC_RIGHT", "snippet": {"title": "Mine"},
                                        "status": {"longUploadsStatus": "allowed"}}]})

        class Svc:
            def channels(self) -> Channels:
                return Channels()

        a = PublisherAgent(dataclasses.replace(self.settings, expected_channel_id=expected), self.db)
        a.client = fake_client(self.db, Svc())  # type: ignore[arg-type]
        return a

    def test_refuses_wrong_or_unset_channel(self) -> None:
        from agents.agent4_publisher import EligibilityError
        with self.assertRaises(EligibilityError):
            self._agent("UC_OTHER")._check_eligibility()
        with self.assertRaises(EligibilityError):
            self._agent("")._check_eligibility()
        a = self._agent("UC_RIGHT")
        a._check_eligibility()
        self.assertEqual(a.channel_id, "UC_RIGHT")
