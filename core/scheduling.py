"""Consistent publish slots: one long-form release per day at PUBLISH_TIME (local), Shorts spread after it."""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from core.config import Settings
from db.database import Database

KV = "scheduled_slots"
MIN_LEAD = timedelta(hours=2)        # YouTube needs time to process HD before going public
SHORT_OFFSETS_H = (3.0, 10.0, 17.0)  # hours after the long-form slot


def next_long_slot(db: Database, settings: Settings, now: datetime | None = None) -> datetime | None:
    """Next free daily slot at PUBLISH_TIME at least MIN_LEAD from now (UTC), or None if unscheduled."""
    if not settings.publish_time:
        return None
    now = now or datetime.now(timezone.utc)
    tz = ZoneInfo(settings.publish_timezone)
    hh, mm = (int(x) for x in settings.publish_time.split(":"))
    taken = set(db.kv_get(KV, []) or [])
    day = now.astimezone(tz).date()
    for _ in range(366):
        local = datetime.combine(day, time(hh, mm), tzinfo=tz)   # zoneinfo handles DST per date
        utc = local.astimezone(timezone.utc)
        if utc - now >= MIN_LEAD and day.isoformat() not in taken:
            return utc
        day += timedelta(days=1)
    raise RuntimeError("No free publish slot in the next year")


def reserve_slot(db: Database, settings: Settings, slot_utc: datetime) -> None:
    tz = ZoneInfo(settings.publish_timezone)
    taken = set(db.kv_get(KV, []) or [])
    taken.add(slot_utc.astimezone(tz).date().isoformat())
    db.kv_set(KV, sorted(taken)[-400:])


def short_slots(long_slot: datetime | None, n: int) -> list[datetime | None]:
    if long_slot is None:
        return [None] * n
    return [long_slot + timedelta(hours=SHORT_OFFSETS_H[i % len(SHORT_OFFSETS_H)] + 24 * (i // 3))
            for i in range(n)]


def rfc3339(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None
