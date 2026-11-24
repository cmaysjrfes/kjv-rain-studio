"""Agent 5 - Analytics, A/B Testing & Scaling Director.

1. Pull per-video metrics (views, watch time, avg view duration/percentage, retention curve,
   traffic sources) + reach (impressions/CTR, when available) into ``analytics_snapshots``.
2. Rotate A/B thumbnail/title variants every ``rotate_after_hours``; once every variant has
   had a window, keep the winner (CTR when impressions are known, else views/hour x AVD).
3. Start title tests on long-form videos that underperform the channel median.
4. Re-weight ``content_blueprint.json`` toward high-retention scripture/rain/scene pairings
   (the planner samples with these weights; rotation gaps still apply).
5. Disk hygiene: delete local media once YouTube has finished HD processing.
6. Decide today's safe production volume from the cadence ramp, quota, disk and health.
"""
from __future__ import annotations

import json
import shutil
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from googleapiclient.http import MediaFileUpload

from agents.base import AgentResult, BaseAgent
from core.analytics import AnalyticsClient, ReachReports, reach_window
from core.metadata import norm_title, render_title
from core.planner import load_blueprint, plan_for_episode
from core.youtube_auth import load_credentials
from core.youtube_client import YouTubeDataClient
from db.database import QUOTA_COSTS, DuplicateContentError, sha256_fingerprint


def _dt(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


class AnalyticsDirectorAgent(BaseAgent):
    name = "agent5_analytics"

    def __init__(self, *args: Any, credentials: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._creds = credentials
        self.data: YouTubeDataClient | None = None

    def execute(self) -> AgentResult:
        warnings: list[str] = []
        summary: dict[str, Any] = {}
        if not self.dry_run:
            creds = self._creds or load_credentials(self.settings)
            self.data = YouTubeDataClient(self.db, credentials=creds, agent=self.name)
            analytics = AnalyticsClient(creds, self.db, self.name)
            reach: dict[str, Any] = {}
            try:
                reach = ReachReports(creds, self.db).sync()
            except Exception as exc:  # noqa: BLE001 - reach data is optional
                warnings.append(f"Reach reports unavailable: {exc}")
            summary["snapshots"] = self._collect(analytics, reach, warnings)
            summary["ab"] = self._rotate_tests(analytics, reach, warnings)
            summary["new_title_tests"] = self._start_title_tests(warnings)
            summary["blueprint_weights"] = self._update_weights()
        summary["cleanup"] = self._cleanup()
        summary["production_plan"] = self._plan_production(warnings)
        return AgentResult(self.name, True, summary=summary, warnings=warnings)

    # ------------------------------------------------------------------ 1. metrics
    def _collect(self, analytics: AnalyticsClient, reach: dict[str, Any], warnings: list[str]) -> int:
        live = [u for u in self.db.query("SELECT * FROM uploads WHERE youtube_video_id IS NOT NULL AND status IN "
                                         "('uploaded','processing','live')")]
        if not live:
            return 0
        today = date.today()
        start = min(_dt(u["created_at"]).date() for u in live)
        totals = analytics.video_totals([u["youtube_video_id"] for u in live], start, today)
        n = 0
        for u in live:
            vid = u["youtube_video_id"]
            m: dict[str, Any] = dict(totals.get(vid, {}))
            pub = _dt(u["created_at"]).date()
            imp, ctr = reach_window(reach, vid, pub, today)
            m.update({"impressions": imp or None, "impressionsCtr": ctr, "kind": u["kind"],
                      "days_live": max(1, (today - pub).days)})
            if u["kind"] == "long":
                try:
                    m["retention"] = analytics.retention(vid, pub, today)
                    m["traffic_sources"] = analytics.traffic_sources(vid, pub, today)
                except Exception as exc:  # noqa: BLE001 - new videos often have no retention data yet
                    warnings.append(f"retention/traffic unavailable for {vid}: {exc}")
            self.db.add_analytics_snapshot(vid, m, window_start=pub.isoformat(), window_end=today.isoformat())
            n += 1
        return n

    # ------------------------------------------------------------------ 2. A/B rotation
    def _apply_variant(self, test: dict[str, Any], variant: dict[str, Any]) -> None:
        assert self.data
        svc = self.data.service
        vid = test["youtube_video_id"]
        if test["element"] == "thumbnail":
            if not Path(variant["payload"]).exists():
                raise FileNotFoundError(f"thumbnail variant missing: {variant['payload']}")
            self.data.execute("thumbnails.set", lambda: svc.thumbnails().set(
                videoId=vid, media_body=MediaFileUpload(variant["payload"], mimetype="image/jpeg")))
        else:
            cur = self.data.execute("videos.list", lambda: svc.videos().list(part="snippet", id=vid))
            if not cur.get("items"):
                raise RuntimeError(f"video {vid} not found")
            sn = cur["items"][0]["snippet"]
            body = {"id": vid, "snippet": {k: sn[k] for k in ("title", "description", "categoryId", "tags",
                                                               "defaultLanguage") if k in sn}}
            body["snippet"]["title"] = variant["payload"]
            self.data.execute("videos.update", lambda: svc.videos().update(part="snippet", body=body))
        self.db.activate_variant(test["id"], variant["id"])

    def _rotate_tests(self, analytics: AnalyticsClient, reach: dict[str, Any], warnings: list[str]) -> dict:
        now = datetime.now(timezone.utc)
        rotated = concluded = 0
        for test in self.db.running_ab_tests():
            variants = self.db.ab_variants(test["id"])
            active = next((v for v in variants if v["is_active"]), None)
            if active is None or not active["active_from"]:
                continue
            since = _dt(active["active_from"])
            hours = (now - since).total_seconds() / 3600
            if hours < test["rotate_after_hours"]:
                continue
            start, end = since.date(), (now - timedelta(days=1)).date()
            rows = analytics.daily(test["youtube_video_id"], start, max(start, end))
            views = int(sum(r.get("views", 0) for r in rows))
            avds = [r["averageViewDuration"] for r in rows if r.get("averageViewDuration")]
            imp, ctr = reach_window(reach, test["youtube_video_id"], start, end)
            self.db.record_variant_metrics(active["id"], impressions=imp, views=views,
                                           avg_view_duration_s=statistics.mean(avds) if avds else None, ctr=ctr)
            variants = self.db.ab_variants(test["id"])
            untested = [v for v in variants if not v["active_from"]]
            need = QUOTA_COSTS["thumbnails.set"] if test["element"] == "thumbnail" else \
                QUOTA_COSTS["videos.update"] + QUOTA_COSTS["videos.list"]
            if not self.db.can_spend(need):
                warnings.append("Quota too low to rotate A/B variants today")
                break
            try:
                if untested:
                    self._apply_variant(test, untested[0])
                    rotated += 1
                    continue
                winner = self._winner(variants)
                if not winner["is_active"]:
                    self._apply_variant(test, winner)
                self.db.conclude_ab_test(test["id"], winner["id"])
                concluded += 1
                self.status(f"A/B {test['element']} on {test['youtube_video_id']}: winner {winner['label']}")
            except Exception as exc:  # noqa: BLE001 - one bad test must not stop the others
                warnings.append(f"A/B test {test['id']} rotation failed: {exc}")
        return {"rotated": rotated, "concluded": concluded}

    @staticmethod
    def _winner(variants: list[dict[str, Any]]) -> dict[str, Any]:
        if all((v["impressions"] or 0) >= 300 and v["ctr"] is not None for v in variants):
            return max(variants, key=lambda v: v["ctr"])

        def score(v: dict[str, Any]) -> float:
            start = _dt(v["active_from"]) if v["active_from"] else datetime.now(timezone.utc)
            end = _dt(v["active_to"]) if v["active_to"] else datetime.now(timezone.utc)
            hrs = max(1.0, (end - start).total_seconds() / 3600)
            return (v["views"] or 0) / hrs * (v["avg_view_duration_s"] or 1.0)
        return max(variants, key=score)

    # ------------------------------------------------------------------ 3. underperformers
    def _start_title_tests(self, warnings: list[str]) -> int:
        snaps = {s["youtube_video_id"]: s for s in self.db.latest_analytics()}
        longs = [u for u in self.db.uploads_by_kind("long", "live") if u["youtube_video_id"] in snaps]
        if len(longs) < 4:
            return 0
        per_day = {u["youtube_video_id"]: (snaps[u["youtube_video_id"]]["views"] or 0) /
                   max(1, json.loads(snaps[u["youtube_video_id"]]["metrics_json"]).get("days_live", 1)) for u in longs}
        median = statistics.median(per_day.values())
        tested = {t["youtube_video_id"] for t in self.db.query("SELECT youtube_video_id FROM ab_tests "
                                                               "WHERE element='title'")}
        bp = load_blueprint(self.settings.content_blueprint_path)
        for u in sorted(longs, key=lambda x: per_day[x["youtube_video_id"]]):
            vid = u["youtube_video_id"]
            age = (datetime.now(timezone.utc) - _dt(u["created_at"])).days
            if vid in tested or age < 7 or per_day[vid] >= 0.5 * median or not u["episode_id"]:
                continue
            ep = self.db.get_episode(u["episode_id"])
            plan = plan_for_episode(self.db, u["episode_id"])
            current = ep["title"] or json.loads(u["metadata_json"])["body"]["snippet"]["title"]
            others = [t["id"] for t in bp["rotation_pools"]["title_templates"] if t["id"] != ep["title_template"]]
            try:
                alt, _ = render_title(plan, bp, self.db, exclude={current}, template_id=others[0] if others else None)
                self.db.register_fingerprint("title", sha256_fingerprint(norm_title(alt)), episode_id=u["episode_id"])
            except (ValueError, DuplicateContentError) as exc:
                warnings.append(f"No alternative title for {vid}: {exc}")
                continue
            self.db.create_ab_test(vid, "title", {"A": current, "B": alt}, episode_id=u["episode_id"],
                                   rotate_after_hours=int(self.settings.publishing.get("ab_rotate_after_hours", 72)))
            self.status(f"Title test started on underperformer {vid}: {alt!r}")
            return 1  # at most one new title test per run
        return 0

    # ------------------------------------------------------------------ 4. blueprint weights
    def _update_weights(self) -> dict[str, Any]:
        snaps = {s["youtube_video_id"]: s for s in self.db.latest_analytics()}
        rows = []
        for u in self.db.uploads_by_kind("long"):
            s = snaps.get(u["youtube_video_id"] or "")
            if not s or not u["episode_id"]:
                continue
            m = json.loads(s["metrics_json"])
            plan = plan_for_episode(self.db, u["episode_id"])
            watch_h_day = (s["watch_minutes"] or 0) / 60 / max(1, m.get("days_live", 1))
            avp = float(m.get("averageViewPercentage") or 0)
            rows.append({"series": plan.passage_set["series"], "ambient_bed": plan.bed["id"],
                         "visual_scene": plan.scene["id"], "score": watch_h_day * (1 + avp / 100)})
        if len(rows) < 3:
            return {"updated": False, "reason": f"need >= 3 measured long-form videos, have {len(rows)}"}
        overall = statistics.mean(r["score"] for r in rows) or 1e-9
        weights: dict[str, dict[str, float]] = {}
        for dim in ("series", "ambient_bed", "visual_scene"):
            groups: dict[str, list[float]] = {}
            for r in rows:
                groups.setdefault(r[dim], []).append(r["score"])
            weights[dim] = {k: round(min(1.6, max(0.6, (statistics.mean(v) / overall) ** 0.5)), 3)
                            for k, v in groups.items() if len(v) >= 2}
        pairs: dict[str, list[float]] = {}
        for r in rows:
            pairs.setdefault(f"{r['series']} + {r['ambient_bed']}", []).append(r["score"])
        top = sorted(((k, statistics.mean(v)) for k, v in pairs.items()), key=lambda x: x[1], reverse=True)[:5]
        path = self.settings.content_blueprint_path
        bp = load_blueprint(path)
        bp["performance_weights"] = {**weights, "top_pairings": [{"pairing": k, "watch_score": round(v, 3)}
                                                                 for k, v in top],
                                     "sample_size": len(rows),
                                     "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(bp, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return {"updated": True, "sample_size": len(rows), "top_pairings": bp["performance_weights"]["top_pairings"]}

    # ------------------------------------------------------------------ 5. cleanup
    def _cleanup(self) -> dict[str, Any]:
        out_root = self.settings.output_dir.resolve()
        keep_days = self.settings.keep_final_video_days
        now = datetime.now(timezone.utc)
        cands = self.db.cleanup_candidates()
        stale_failed = self.db.query(
            "SELECT m.* FROM media_files m JOIN episodes e ON e.id = m.episode_id WHERE e.status='failed' "
            "AND m.deleted_at IS NULL AND m.keep=0 AND e.updated_at < ?",
            ((now - timedelta(days=7)).isoformat(timespec="seconds"),))
        deleted, freed, skipped = [], 0, 0
        for m in [*cands, *stale_failed]:
            p = Path(m["path"])
            if m["kind"] == "final_mp4" and keep_days > 0 and m in cands:
                up = self.db.upload_for_episode(m["episode_id"], "long")
                if up and (now - _dt(up["updated_at"])).days < keep_days:
                    skipped += 1
                    continue
            try:
                resolved = p.resolve()
            except OSError:
                continue
            if out_root not in resolved.parents:
                skipped += 1  # never delete outside OUTPUT_DIR
                continue
            size = p.stat().st_size if p.exists() else 0
            if not self.dry_run:
                p.unlink(missing_ok=True)
            deleted.append(str(p))
            freed += size
        if not self.dry_run and deleted:
            self.db.mark_media_deleted(deleted)
            for d in {Path(x).parent for x in deleted}:
                if d.exists() and not any(f.suffix in {".mp4", ".m4a", ".wav", ".flac"} for f in d.iterdir()):
                    for f in d.iterdir():
                        if f.suffix in {".txt", ".png"}:
                            f.unlink(missing_ok=True)
        return {"files": len(deleted), "freed_gb": round(freed / 1e9, 2), "skipped": skipped,
                "dry_run": self.dry_run}

    # ------------------------------------------------------------------ 6. scaling
    def _plan_production(self, warnings: list[str]) -> dict[str, Any]:
        reasons: list[str] = []
        now = datetime.now(timezone.utc)
        sc = self.settings.scaling
        channel_bp = load_blueprint(self.settings.channel_blueprint_path)
        ramp = channel_bp["channel_page_layout"]["upload_cadence"]["ramp"]
        longs = self.db.uploads_by_kind("long")
        published = [u for u in longs if u["status"] in {"uploaded", "processing", "live"}]
        first = min((_dt(u["created_at"]) for u in published), default=now)
        weeks = (now - first).days / 7
        tier = ramp[0] if weeks < 2 else ramp[1] if weeks < 6 else ramp[-1]
        last7 = sum(1 for u in published if now - _dt(u["created_at"]) < timedelta(days=7))
        target = int(tier["long_form_per_week"])
        long_today = 1 if last7 < target else 0
        if not long_today:
            reasons.append(f"weekly target reached ({last7}/{target}, ramp weeks {tier['weeks']})")
        shorts = self.settings.shorts_per_release

        backlog = len(self.db.episodes_by_status("video_ready", "audio_ready", "uploading"))
        if backlog >= 2:
            long_today = 0
            reasons.append(f"{backlog} episodes already in the pipeline - publish backlog first")
        recent_fail = [u for u in longs if u["status"] == "failed" and now - _dt(u["updated_at"]) < timedelta(days=3)]
        if len(recent_fail) >= 2:
            long_today = 0
            reasons.append(f"{len(recent_fail)} failed uploads in 3 days - holding production")
        processing = [u for u in longs if u["status"] in {"uploaded", "processing"}]
        if len(processing) > 2:
            long_today = 0
            reasons.append(f"{len(processing)} uploads still processing on YouTube")

        q = self.db.quota_status()
        long_units = QUOTA_COSTS["videos.insert"] + 4 * 50
        short_units = QUOTA_COSTS["videos.insert"] + 50
        if long_today and q.remaining < long_units:
            long_today = 0
            reasons.append(f"quota: {q.remaining} units left today, need {long_units}")
        elif long_today:
            affordable = max(0, (q.remaining - long_units) // short_units)
            if affordable < shorts:
                reasons.append(f"quota allows only {affordable} Shorts today (each costs {short_units} units)")
                shorts = int(affordable)

        free = shutil.disk_usage(self.settings.output_dir).free
        need = int(sc.get("est_bytes_per_episode", 14e9)) + int(sc.get("min_free_disk_bytes", 30e9))
        if long_today and free < need:
            long_today = 0
            reasons.append(f"disk: {free / 1e9:.0f} GB free, need {need / 1e9:.0f} GB")
        long_today = min(long_today, int(sc.get("max_long_form_per_day", 1)))
        releases_per_day_quota = self.settings.daily_quota // (long_units + shorts * short_units or 1)
        if releases_per_day_quota < 1:
            warnings.append("Default quota cannot fit one full release (long + Shorts) per day; "
                            "request a quota increase in Google Cloud Console.")
        plan = {"date": now.date().isoformat(), "long_form_today": long_today, "shorts_per_long": shorts,
                "ramp_tier": tier, "published_last_7d": last7, "quota_remaining": q.remaining,
                "disk_free_gb": round(free / 1e9, 1), "reasons": reasons or ["on schedule"]}
        self.db.kv_set("production_plan", plan)
        return plan
