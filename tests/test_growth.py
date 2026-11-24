"""Publishing-settings tests: scheduled slots (DST-safe), Shorts spread, notifySubscribers, request bodies."""
from __future__ import annotations

import dataclasses
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from core.config import get_settings
from core.metadata import video_body
from core.scheduling import next_long_slot, reserve_slot, rfc3339, short_slots
from core.uploader import ResumableUploader
from db.database import Database
from tests.test_phase3 import FakeYouTubeUploadServer


class SchedulingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "s.db")
        self.s = dataclasses.replace(get_settings(), publish_time="19:30", publish_timezone="America/New_York")

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_slot_lead_time_and_reservation(self) -> None:
        now = datetime(2026, 3, 10, 22, 0, tzinfo=timezone.utc)  # 18:00 EDT -> 19:30 is only 1.5 h away
        slot = next_long_slot(self.db, self.s, now)
        local = slot.astimezone(ZoneInfo("America/New_York"))
        self.assertEqual((local.day, local.hour, local.minute), (11, 19, 30))
        reserve_slot(self.db, self.s, slot)
        self.assertEqual(next_long_slot(self.db, self.s, now).astimezone(ZoneInfo("America/New_York")).day, 12)

    def test_dst_keeps_local_time(self) -> None:
        winter = next_long_slot(self.db, self.s, datetime(2026, 1, 5, 12, tzinfo=timezone.utc))
        summer = next_long_slot(self.db, self.s, datetime(2026, 7, 5, 12, tzinfo=timezone.utc))
        self.assertEqual((winter.hour, summer.hour), (0, 23))  # 19:30 EST = 00:30Z, EDT = 23:30Z

    def test_unscheduled_and_shorts_spread(self) -> None:
        s = dataclasses.replace(self.s, publish_time="")
        self.assertIsNone(next_long_slot(self.db, s))
        base = datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)
        hours = [(x - base).total_seconds() / 3600 for x in short_slots(base, 3)]
        self.assertEqual(hours, [3.0, 10.0, 17.0])

    def test_body_scheduled_is_private_with_publish_at(self) -> None:
        b = video_body("t", "d", [], dataclasses.replace(self.s, default_privacy="public"),
                       publish_at=rfc3339(datetime(2026, 1, 1, tzinfo=timezone.utc)))
        self.assertEqual(b["status"]["privacyStatus"], "private")
        self.assertEqual(b["status"]["publishAt"], "2026-01-01T00:00:00Z")
        self.assertFalse(b["status"]["selfDeclaredMadeForKids"])
        self.assertTrue(b["status"]["embeddable"])

    def test_notify_subscribers_flag_sent(self) -> None:
        f = Path(self.tmp.name) / "v.mp4"
        f.write_bytes(b"z" * 300_000)
        seen = {}

        class Server(FakeYouTubeUploadServer):
            def post(self, url, params=None, **kw):  # type: ignore[override]
                seen.update(params)
                return super().post(url, params=params, **kw)

        up = self.db.create_upload(f)
        ResumableUploader(None, self.db, chunk_mb=1, session=Server()).upload(up, {"snippet": {}}, notify=False)
        self.assertEqual(seen["notifySubscribers"], "false")


if __name__ == "__main__":
    unittest.main()
