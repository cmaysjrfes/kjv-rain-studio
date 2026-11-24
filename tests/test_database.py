"""Phase 2 tests: SQLite state, uniqueness and quota tracking.  Run: python -m unittest -v"""
from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

from db.database import (Database, DuplicateContentError, InvalidTransitionError, QuotaExceededError,
                         quota_day, sha256_fingerprint)


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.sqlite3", daily_quota=10_000, quota_reserve=2_000)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_quota_reserve_and_limit(self) -> None:
        self.db.consume_quota("search.list", agent="a1")            # 100
        self.assertEqual(self.db.quota_status().used, 100)
        for _ in range(4):
            self.db.consume_quota("videos.insert", use_reserve=True)  # 6400 -> 6500
        with self.assertRaises(QuotaExceededError):                 # 6500+1600 > 8000 unreserved
            self.db.consume_quota("videos.insert")
        self.db.consume_quota("videos.insert", use_reserve=True)    # 8100 ok with reserve
        self.db.consume_quota("videos.insert", use_reserve=True)    # 9700 ok with reserve
        with self.assertRaises(QuotaExceededError):                 # 11300 > 10000 hard cap
            self.db.consume_quota("videos.insert", use_reserve=True)
        self.assertEqual(self.db.quota_status().used, 9700)

    def test_quota_exact_accounting(self) -> None:
        self.db.consume_quota("videos.insert", use_reserve=True)
        self.db.refund_quota("videos.insert", 1600)
        self.assertEqual(self.db.quota_status().used, 0)
        self.assertTrue(self.db.can_spend(8000))
        self.assertFalse(self.db.can_spend(8001))
        self.assertTrue(self.db.can_spend(10_000, use_reserve=True))

    def test_quota_day_is_pacific(self) -> None:
        # 07:30 UTC on Jan 2 is still Jan 1 in Los Angeles
        self.assertEqual(quota_day(datetime(2026, 1, 2, 7, 30, tzinfo=timezone.utc)), "2026-01-01")
        self.assertEqual(quota_day(datetime(2026, 1, 2, 8, 30, tzinfo=timezone.utc)), "2026-01-02")

    def test_uniqueness_exact_and_perceptual(self) -> None:
        ep = self.db.create_episode("ep-1", scripture_book="Psalms", ambient_bed="steady_rainfall")
        fp = sha256_fingerprint("psa-001-024", "steady_rainfall", "rain_on_glass_night", "T1")
        self.db.register_fingerprint("combo", fp, episode_id=ep)
        self.assertFalse(self.db.is_unique("combo", fp))
        with self.assertRaises(DuplicateContentError):
            self.db.register_fingerprint("combo", fp)
        self.db.register_fingerprint("thumbnail", "ffff0000ffff0000", algo="dhash64", episode_id=ep)
        with self.assertRaises(DuplicateContentError):   # 2 bits away
            self.db.register_fingerprint("thumbnail", "ffff0000ffff0003", algo="dhash64")
        self.db.register_fingerprint("thumbnail", "0000ffff0000ffff", algo="dhash64")  # far away: OK
        self.assertEqual(self.db.release_fingerprints(ep), 2)
        self.assertTrue(self.db.is_unique("combo", fp))

    def test_rotation_history(self) -> None:
        for i, book in enumerate(["Psalms", "John", "Isaiah"]):
            self.db.create_episode(f"ep-{i}", scripture_book=book)
        self.assertEqual(self.db.recent_episode_values("scripture_book", 2), ["Isaiah", "John"])
        with self.assertRaises(ValueError):
            self.db.recent_episode_values("title; DROP TABLE episodes", 2)

    def test_upload_state_machine_and_resume(self) -> None:
        f = Path(self.tmp.name) / "v.mp4"
        f.write_bytes(b"x" * 1024)
        ep = self.db.create_episode("ep-up")
        up = self.db.create_upload(f, episode_id=ep)
        self.db.set_upload_status(up, "uploading", resumable_uri="https://upload/session")
        self.db.save_upload_progress(up, 512)
        row = self.db.get_upload(up)
        self.assertEqual((row["bytes_uploaded"], row["attempts"], row["file_size"]), (512, 1, 1024))
        self.assertEqual(len(self.db.resumable_uploads()), 1)
        with self.assertRaises(InvalidTransitionError):
            self.db.set_upload_status(up, "live")
        self.db.set_upload_status(up, "uploaded", youtube_video_id="abc123")
        self.db.set_upload_status(up, "live", processing_status="succeeded")
        self.db.register_media(f, "final_mp4", episode_id=ep)
        self.assertEqual([m["path"] for m in self.db.cleanup_candidates()], [str(f)])
        self.db.mark_media_deleted([str(f)])
        self.assertEqual(self.db.cleanup_candidates(), [])

    def test_ab_test_flow(self) -> None:
        t = self.db.create_ab_test("vid1", "thumbnail", {"A": "a.jpg", "B": "b.jpg"})
        a, b = self.db.ab_variants(t)
        self.assertEqual((a["is_active"], b["is_active"]), (1, 0))
        self.db.activate_variant(t, b["id"])
        self.assertEqual([v["is_active"] for v in self.db.ab_variants(t)], [0, 1])
        self.db.record_variant_metrics(a["id"], impressions=1000, views=52)
        self.assertAlmostEqual(self.db.ab_variants(t)[0]["ctr"], 0.052)
        self.db.conclude_ab_test(t, a["id"])
        self.assertEqual(self.db.running_ab_tests(), [])

    def test_render_jobs_and_competitors(self) -> None:
        ep = self.db.create_episode("ep-r")
        j = self.db.create_render_job(ep, "audio_master", {"bed": "x"})
        self.db.update_render_job(j, status="running", encoder="libx264")
        self.db.update_render_job(j, status="done", progress=1.0, output_path="/o.flac")
        row = self.db.render_jobs("done")[0]
        self.assertEqual((row["attempts"], row["progress"]), (1, 1.0))
        self.db.upsert_competitor_channel({"channel_id": "UC1", "title": "X", "subscriber_count": 10})
        self.db.upsert_competitor_channel({"channel_id": "UC1", "title": "X", "subscriber_count": 20})
        self.assertEqual(self.db.top_competitor_channels()[0]["subscriber_count"], 20)
        snaps = self.db.query("SELECT COUNT(*) n FROM competitor_channel_snapshots")[0]["n"]
        self.assertEqual(snaps, 2)

    def test_concurrent_quota_writes(self) -> None:
        def worker() -> None:
            for _ in range(50):
                self.db.consume_quota("videos.list")
            self.db.close()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.db.quota_status().used, 400)


if __name__ == "__main__":
    unittest.main()
