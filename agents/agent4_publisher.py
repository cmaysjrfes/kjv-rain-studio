"""Agent 4 - Resumable YouTube Publisher & Funnel Agent.

Per ``video_ready`` episode (live mode):
  eligibility check (verified long uploads) -> unique title/description/tags/chapters ->
  resumable chunked upload (restart-safe) -> thumbnail A -> playlists -> funnel comment ->
  thumbnail A/B test registration -> 3 vertical Shorts (rendered + uploaded, linking back).
Every run also resumes interrupted uploads and polls processing until HD is ready ('live').

``--dry-run`` renders everything (including Shorts) and writes ``upload_manifest.json``
with the exact API bodies, but makes no YouTube API calls.

Note: the YouTube API cannot *pin* comments - pin the posted comment in Studio if desired.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from googleapiclient.http import MediaFileUpload

from agents.base import AgentResult, BaseAgent
from core.ffmpeg import detect_encoder
from core.growth import STUDIO_ONLY_CHECKLIST
from core.metadata import (build_comment, build_description, build_tags, norm_title, render_title, sanitize,
                           video_body)
from core.scheduling import next_long_slot, reserve_slot, rfc3339, short_slots
from core.planner import EpisodePlan, episode_meta, load_blueprint, plan_for_episode
from core.shorts import render_short, select_windows
from core.uploader import ResumableUploader, UploadError
from core.youtube_auth import load_credentials
from core.youtube_client import YouTubeDataClient
from db.database import QUOTA_COSTS, QuotaExceededError, sha256_fingerprint, sha256_file

THUNDER_BEDS = {"distant_soft_thunder", "heavy_night_storm_far"}


class EligibilityError(RuntimeError):
    pass


class PublisherAgent(BaseAgent):
    name = "agent4_publisher"

    def __init__(self, *args: Any, preview_seconds: int | None = None, credentials: Any = None,
                 apply_branding: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.apply_branding = apply_branding
        self.preview_seconds = preview_seconds if preview_seconds is not None else self.settings.preview_seconds
        self._creds = credentials
        self.client: YouTubeDataClient | None = None
        self.uploader: ResumableUploader | None = None
        self.channel_id: str = ""

    # ------------------------------------------------------------------ entry
    def execute(self) -> AgentResult:
        bp = load_blueprint(self.settings.content_blueprint_path)
        channel_bp = load_blueprint(self.settings.channel_blueprint_path)
        eps = [e for e in self.db.episodes_by_status("video_ready")
               if e["slug"].startswith("preview-") == bool(self.preview_seconds)]
        if self.dry_run:
            manifests = [self._dry_run(e["id"], plan_for_episode(self.db, e["id"]), bp, channel_bp) for e in eps]
            return AgentResult(self.name, True, summary={"mode": "dry-run", "episodes": len(manifests),
                                                         "manifests": manifests})
        if self.preview_seconds:
            raise RuntimeError("Refusing to upload preview-length episodes; unset PREVIEW_SECONDS for live runs")

        creds = self._creds or load_credentials(self.settings)
        self.client = YouTubeDataClient(self.db, credentials=creds, agent=self.name)
        self.uploader = ResumableUploader(creds, self.db, chunk_mb=self.settings.upload_chunk_mb, agent=self.name)
        self._check_eligibility()
        warnings: list[str] = []
        branding = self._apply_branding(channel_bp, warnings) if self.apply_branding else "skipped"
        playlists = self._ensure_playlists(channel_bp)

        resumed = self._resume_interrupted(bp, channel_bp, playlists, warnings)
        eps = [e for e in self.db.episodes_by_status("video_ready", "uploading")
               if not e["slug"].startswith("preview-")]
        published = []
        for ep in eps:
            plan = plan_for_episode(self.db, ep["id"])
            try:
                published.append(self._publish(ep["id"], plan, bp, channel_bp, playlists, warnings))
            except QuotaExceededError as exc:
                warnings.append(f"Quota: deferred {plan.slug}: {exc}")
                break
        processed = self._poll_processing()
        return AgentResult(self.name, True, warnings=warnings, summary={
            "published": published, "resumed": resumed, "processing_updates": processed, "branding": branding,
            "quota_used_today": self.db.quota_status().used})

    # ------------------------------------------------------------------ metadata
    def _metadata(self, ep_id: int, plan: EpisodePlan, bp: dict[str, Any],
                  channel_bp: dict[str, Any], publish_at: str | None = None) -> dict[str, Any]:
        meta = episode_meta(self.db, ep_id)
        audio = json.loads(Path(meta["audio_manifest"]).read_text(encoding="utf-8"))
        brand = self._brand(channel_bp)
        title, tpl = render_title(plan, bp, self.db)
        desc = build_description(plan, audio, brand, bp, self.settings)
        tags = build_tags(plan)
        return {"title": title, "title_template": tpl, "description": desc, "tags": tags, "audio": audio,
                "brand": brand, "video": meta["video"], "comment": build_comment(audio, self.settings),
                "body": video_body(title, desc, tags, self.settings, publish_at=publish_at)}

    def _brand(self, channel_bp: dict[str, Any]) -> dict[str, Any]:
        stored = self.db.kv_get("brand") or {}
        base = {k: channel_bp["brand"][k] for k in ("name", "handle", "tagline", "palette")}
        return {**base, **stored,
                "cadence_phrase": channel_bp["channel_page_layout"]["upload_cadence"]["public_phrase"]}

    def _playlist_keys(self, plan: EpisodePlan) -> list[str]:
        keys = [plan.passage_set["series"]]
        if plan.bed["id"] in THUNDER_BEDS:
            keys.append("storms")
        return keys

    def _shorts(self, ep_id: int, plan: EpisodePlan, md: dict[str, Any], long_video_id: str | None) -> list[dict]:
        n = self.settings.shorts_per_release
        if n <= 0:
            return []
        work = Path(md["video"]["final_video"]).parent
        windows = select_windows(md["audio"]["timeline"], n=n, max_s=52.0 if not self.preview_seconds else 30.0,
                                 min_s=18.0 if not self.preview_seconds else 4.0)
        encoder = detect_encoder(self.settings.video_encoder)
        shorts = []
        link = f"https://youtu.be/{long_video_id}" if long_video_id else "(link added after upload)"
        long_at = md["body"]["status"].get("publishAt")
        slots = short_slots(datetime.fromisoformat(long_at.replace("Z", "+00:00")) if long_at else None,
                            len(windows))
        for i, w in enumerate(windows, 1):
            out = work / f"short_{i}.mp4"
            if out.exists():
                info = {"path": str(out), "ref": w["ref"], "text": w["text"]}
            else:
                info = render_short(w, Path(md["audio"]["master_audio"]), Path(md["video"]["shorts_loop"]), out,
                                    md["brand"], encoder, self.settings.video_fps)
            self.db.register_media(out, "short_mp4", episode_id=ep_id)
            title = sanitize(f"{w['ref']} KJV - {plan.passage_set['theme']} | Scripture for Sleep #Shorts")[:100]
            desc = sanitize(f"“{w['text'][:300]}” - {w['ref']} (KJV)\n\nFull {plan.hours}-hour sleep "
                            f"reading with {plan.bed['label'].lower()}: {link}\n\n#Shorts #kjv #bibleverse")
            shorts.append({**info, "body": video_body(title, desc, build_tags(plan, ["shorts"])[:10],
                                                      self.settings, publish_at=rfc3339(slots[i - 1]))})
        return shorts

    # ------------------------------------------------------------------ dry run
    def _dry_run(self, ep_id: int, plan: EpisodePlan, bp: dict[str, Any], channel_bp: dict[str, Any]) -> str:
        md = self._metadata(ep_id, plan, bp, channel_bp, publish_at=rfc3339(next_long_slot(self.db, self.settings)))
        self.status(f"[dry-run] rendering {self.settings.shorts_per_release} Shorts for {plan.slug}")
        shorts = self._shorts(ep_id, plan, md, None)
        units = QUOTA_COSTS["videos.insert"] * (1 + len(shorts)) + QUOTA_COSTS["thumbnails.set"] + \
            QUOTA_COSTS["playlistItems.insert"] * (len(self._playlist_keys(plan)) + len(shorts)) + \
            QUOTA_COSTS["commentThreads.insert"]
        manifest = {
            "episode_id": ep_id, "slug": plan.slug, "video_file": md["video"]["final_video"],
            "video_bytes": md["video"]["bytes"], "duration_s": md["video"]["duration_s"],
            "request_body": md["body"], "thumbnails": md["video"]["thumbnails"],
            "playlists": self._playlist_keys(plan), "comment": md["comment"],
            "shorts": [{k: v for k, v in s.items()} for s in shorts],
            "notify_subscribers": {"long": self.settings.notify_subscribers_long,
                                   "shorts": self.settings.notify_subscribers_shorts},
            "studio_only_checklist": STUDIO_ONLY_CHECKLIST,
            "estimated_quota_units": units,
            "quota_remaining_today": self.db.quota_status().remaining,
        }
        out = Path(md["video"]["final_video"]).parent / "upload_manifest.json"
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        self.db.update_episode(ep_id, metadata={"dry_run_manifest": str(out)})
        return str(out)

    # ------------------------------------------------------------------ live
    def _check_eligibility(self) -> None:
        assert self.client
        resp = self.client.execute("channels.list", lambda: self.client.service.channels().list(
            part="status,snippet", mine=True))
        items = resp.get("items", [])
        if not items:
            raise EligibilityError("OAuth account has no YouTube channel")
        ch = items[0]
        self.channel_id = ch["id"]
        expected = self.settings.expected_channel_id
        if not expected:
            raise EligibilityError(
                f"Authorised channel is '{ch['snippet']['title']}' ({ch['id']}). Set YOUTUBE_CHANNEL_ID={ch['id']} "
                "in .env to confirm this is the channel to publish to.")
        if ch["id"] != expected:
            raise EligibilityError(
                f"Authorised channel '{ch['snippet']['title']}' ({ch['id']}) is not YOUTUBE_CHANNEL_ID={expected}. "
                "Delete config/token.json and re-run `python -m core.youtube_auth`, choosing the right channel.")
        long_status = ch.get("status", {}).get("longUploadsStatus", "longUploadsUnspecified")
        if long_status != "allowed":
            raise EligibilityError(
                f"Channel '{ch['snippet']['title']}' longUploadsStatus={long_status}. Videos > 15 min require a "
                "verified channel: verify at https://www.youtube.com/verify then re-run.")
        self.status(f"Channel '{ch['snippet']['title']}' eligible for long uploads")

    def _apply_branding(self, channel_bp: dict[str, Any], warnings: list[str]) -> str:
        """Banner + About text + keywords + language + watermark via API. Idempotent per asset hash.

        channels.update replaces the whole brandingSettings part, so current values are fetched and
        merged first. The watermark shows only for the first 2 minutes - a bright overlay for 8 h
        would be glare in a dark bedroom.
        """
        assert self.client
        a = channel_bp["assets"]
        banner, wm = Path(a["banner"]["path"]), Path(a["watermark"]["path"])
        about = Path(a["description_file"]).read_text(encoding="utf-8")
        keywords = Path(a["keywords_file"]).read_text(encoding="utf-8")
        digest = sha256_fingerprint(sha256_file(banner), sha256_file(wm), about, keywords)
        if (self.db.kv_get("branding_applied") or {}).get("digest") == digest:
            return "unchanged"
        need = sum(QUOTA_COSTS[k] for k in ("channelBanners.insert", "channels.update", "watermarks.set",
                                             "channels.list"))
        if not self.db.can_spend(need):
            warnings.append("Quota too low to apply branding today")
            return "deferred"
        svc = self.client.service
        if banner.stat().st_size > 6 * 1024 * 1024 and banner.with_suffix(".jpg").exists():
            banner = banner.with_suffix(".jpg")
        mime = "image/png" if banner.suffix == ".png" else "image/jpeg"
        up = self.client.execute("channelBanners.insert", lambda: svc.channelBanners().insert(
            media_body=MediaFileUpload(str(banner), mimetype=mime)))
        cur = self.client.execute("channels.list", lambda: svc.channels().list(part="brandingSettings",
                                                                               id=self.channel_id))
        branding = (cur.get("items") or [{}])[0].get("brandingSettings", {})
        ch = branding.setdefault("channel", {})
        ch.update({"description": about, "keywords": keywords,
                   "defaultLanguage": self.settings.publishing.get("default_language", "en")})
        branding.setdefault("image", {})["bannerExternalUrl"] = up["url"]
        self.client.execute("channels.update", lambda: svc.channels().update(
            part="brandingSettings", body={"id": self.channel_id, "brandingSettings": branding}))
        try:
            self.client.execute("watermarks.set", lambda: svc.watermarks().set(
                channelId=self.channel_id, media_body=MediaFileUpload(str(wm), mimetype="image/png"),
                body={"timing": {"type": "offsetFromStart", "offsetMs": 0, "durationMs": 120000},
                      "position": {"type": "corner", "cornerPosition": "topRight"}}))
        except Exception as exc:  # noqa: BLE001 - watermark is optional
            warnings.append(f"watermark not set: {exc}")
        self.db.kv_set("branding_applied", {"digest": digest, "at": datetime.now(timezone.utc).isoformat()})
        self.status("Channel branding applied (banner, About, keywords, watermark)")
        return "applied"

    def _ensure_playlists(self, channel_bp: dict[str, Any]) -> dict[str, str]:
        assert self.client
        known: dict[str, str] = self.db.kv_get("playlist_ids", {}) or {}
        plan = channel_bp["playlists"]
        if all(p["key"] in known for p in plan):
            return known
        existing: dict[str, str] = {}
        token = None
        while True:
            resp = self.client.execute("playlists.list", lambda t=token: self.client.service.playlists().list(
                part="snippet", mine=True, maxResults=50, pageToken=t))
            for it in resp.get("items", []):
                existing[it["snippet"]["title"]] = it["id"]
            token = resp.get("nextPageToken")
            if not token:
                break
        for p in plan:
            if p["key"] in known:
                continue
            if p["title"] in existing:
                known[p["key"]] = existing[p["title"]]
                continue
            body = {"snippet": {"title": p["title"], "description": p["description"],
                                "defaultLanguage": self.settings.publishing.get("default_language", "en")},
                    "status": {"privacyStatus": p.get("privacy", "public")}}
            created = self.client.execute("playlists.insert", lambda b=body: self.client.service.playlists().insert(
                part="snippet,status", body=b))
            known[p["key"]] = created["id"]
            self.status(f"Created playlist {p['title']}")
            self.db.kv_set("playlist_ids", known)
        self.db.kv_set("playlist_ids", known)
        return known

    def _release_units(self, plan: EpisodePlan) -> int:
        return (QUOTA_COSTS["videos.insert"] + QUOTA_COSTS["thumbnails.set"] + QUOTA_COSTS["commentThreads.insert"]
                + QUOTA_COSTS["playlistItems.insert"] * len(self._playlist_keys(plan)))

    def _publish(self, ep_id: int, plan: EpisodePlan, bp: dict[str, Any], channel_bp: dict[str, Any],
                 playlists: dict[str, str], warnings: list[str]) -> dict[str, Any]:
        assert self.client and self.uploader
        existing = self.db.upload_for_episode(ep_id, "long")
        if existing and existing["status"] in {"uploaded", "processing", "live"}:
            # crashed after the upload finished: never upload twice, just finish post steps
            body = json.loads(existing["metadata_json"] or "{}")["body"]
            md = self._metadata_for_resume(ep_id, plan, bp, channel_bp, body)
            vid = existing["youtube_video_id"]
            self.db.update_episode(ep_id, status="published", metadata={"youtube_video_id": vid})
            self._post_steps(existing["id"], ep_id, plan, md, playlists, warnings)
            return {"episode": plan.slug, "video_id": vid, "title": body["snippet"]["title"], "recovered": True}
        need = self._release_units(plan)
        if not self.db.can_spend(need, use_reserve=True):
            raise QuotaExceededError(f"need {need} units for long-form release")
        if existing and existing["status"] in {"pending", "uploading", "failed"}:
            upload_id = existing["id"]
            meta = json.loads(existing["metadata_json"] or "{}")
            body = meta["body"]
            at = body["status"].get("publishAt")
            if at and not existing["resumable_uri"] and \
                    datetime.fromisoformat(at.replace("Z", "+00:00")) - datetime.now(timezone.utc) < timedelta(hours=2):
                slot = next_long_slot(self.db, self.settings)  # stale slot: YouTube rejects past publishAt
                body["status"]["publishAt"] = rfc3339(slot)
                reserve_slot(self.db, self.settings, slot)
                self.db.set_upload_metadata(upload_id, meta)
            md = self._metadata_for_resume(ep_id, plan, bp, channel_bp, body)
        else:
            slot = next_long_slot(self.db, self.settings)
            md = self._metadata(ep_id, plan, bp, channel_bp, publish_at=rfc3339(slot))
            if slot:
                reserve_slot(self.db, self.settings, slot)
            self.db.register_fingerprint("title", sha256_fingerprint(norm_title(md["title"])), episode_id=ep_id)
            upload_id = self.db.create_upload(md["video"]["final_video"], episode_id=ep_id, kind="long",
                                              privacy=md["body"]["status"]["privacyStatus"],
                                              metadata={"body": md["body"], "steps": []})
            self.db.update_episode(ep_id, status="uploading", title=md["title"],
                                   metadata={"publish": {"title_template": md["title_template"]}})
        video_id = self._upload(upload_id, md["body"], plan.slug, notify=self.settings.notify_subscribers_long)
        self.db.update_episode(ep_id, status="published", metadata={
            "youtube_video_id": video_id, "publish_at": md["body"]["status"].get("publishAt")})
        self._post_steps(upload_id, ep_id, plan, md, playlists, warnings)
        return {"episode": plan.slug, "video_id": video_id, "title": md["title"]}

    def _upload(self, upload_id: int, body: dict[str, Any], label: str, *, notify: bool = True) -> str:
        assert self.uploader
        try:
            resource = self.uploader.upload(upload_id, body, notify=notify, progress=lambda done, total: self.progress(
                f"upload {label}", done / total))
        except (UploadError, QuotaExceededError) as exc:
            row = self.db.get_upload(upload_id)
            if row and row["status"] in {"pending", "uploading"}:
                self.db.set_upload_status(upload_id, "failed", last_error=str(exc)[:2000])
            raise
        vid = resource["id"]
        self.db.set_upload_status(upload_id, "uploaded", youtube_video_id=vid,
                                  processing_status=resource.get("status", {}).get("uploadStatus"))
        self.status(f"Uploaded {label} -> https://youtu.be/{vid}")
        return vid

    def _post_steps(self, upload_id: int, ep_id: int, plan: EpisodePlan, md: dict[str, Any],
                    playlists: dict[str, str], warnings: list[str]) -> None:
        """Idempotent post-upload steps; completed steps are recorded on the upload row."""
        assert self.client
        row = self.db.get_upload(upload_id)
        meta = json.loads(row["metadata_json"] or "{}")
        steps: list[str] = meta.setdefault("steps", [])
        vid = row["youtube_video_id"]
        svc = self.client.service

        def done(step: str) -> None:
            steps.append(step)
            self.db.set_upload_metadata(upload_id, meta)

        thumbs = {t["label"]: t["path"] for t in md["video"]["thumbnails"]}
        if "thumbnail" not in steps:
            try:
                self.client.execute("thumbnails.set", lambda: svc.thumbnails().set(
                    videoId=vid, media_body=MediaFileUpload(thumbs["A"], mimetype="image/jpeg")), use_reserve=True)
                done("thumbnail")
            except Exception as exc:  # noqa: BLE001 - custom thumbnails need a verified channel
                warnings.append(f"thumbnail set failed for {vid}: {exc}")
        for key in self._playlist_keys(plan):
            step = f"playlist:{key}"
            if step in steps or key not in playlists:
                continue
            body = {"snippet": {"playlistId": playlists[key], "resourceId": {"kind": "youtube#video", "videoId": vid}}}
            self.client.execute("playlistItems.insert", lambda b=body: svc.playlistItems().insert(
                part="snippet", body=b), use_reserve=True)
            done(step)
        if "comment" not in steps:
            body = {"snippet": {"channelId": self.channel_id, "videoId": vid,
                                "topLevelComment": {"snippet": {"textOriginal": md["comment"]}}}}
            try:
                self.client.execute("commentThreads.insert", lambda: svc.commentThreads().insert(
                    part="snippet", body=body), use_reserve=True)
                done("comment")
            except Exception as exc:  # noqa: BLE001 - comments may be disabled
                warnings.append(f"comment failed for {vid}: {exc}")
        if "ab_test" not in steps and len(thumbs) >= 2:
            self.db.create_ab_test(vid, "thumbnail", {k: thumbs[k] for k in sorted(thumbs)}, episode_id=ep_id,
                                   rotate_after_hours=int(self.settings.publishing.get("ab_rotate_after_hours", 72)))
            done("ab_test")
        if "shorts" not in steps:
            self._publish_shorts(ep_id, plan, md, vid, playlists, warnings)
            done("shorts")

    def _publish_shorts(self, ep_id: int, plan: EpisodePlan, md: dict[str, Any], long_vid: str,
                        playlists: dict[str, str], warnings: list[str]) -> None:
        assert self.client
        shorts = self._shorts(ep_id, plan, md, long_vid)
        already = {u["file_path"] for u in self.db.uploads_by_kind("short") if u["episode_id"] == ep_id
                   and u["status"] in {"uploaded", "processing", "live"}}
        for s in shorts:
            if s["path"] in already:
                continue
            per_short = QUOTA_COSTS["videos.insert"] + QUOTA_COSTS["playlistItems.insert"]
            if not self.db.can_spend(per_short, use_reserve=False):
                warnings.append(f"Quota: skipped remaining Shorts for {plan.slug}")
                return
            uid = self.db.create_upload(s["path"], episode_id=ep_id, kind="short",
                                        privacy=s["body"]["status"]["privacyStatus"], metadata={"body": s["body"]})
            vid = self._upload(uid, s["body"], f"short {s['ref']}", notify=self.settings.notify_subscribers_shorts)
            if "shorts" in playlists:
                body = {"snippet": {"playlistId": playlists["shorts"],
                                    "resourceId": {"kind": "youtube#video", "videoId": vid}}}
                self.client.execute("playlistItems.insert", lambda b=body: self.client.service.playlistItems()
                                    .insert(part="snippet", body=b), use_reserve=True)

    def _resume_interrupted(self, bp: dict[str, Any], channel_bp: dict[str, Any], playlists: dict[str, str],
                            warnings: list[str]) -> list[str]:
        resumed = []
        for row in self.db.resumable_uploads():
            if row["status"] == "pending" and not row["resumable_uri"]:
                continue
            body = json.loads(row["metadata_json"] or "{}").get("body")
            if not body or not Path(row["file_path"]).exists():
                continue
            self.status(f"Resuming interrupted upload {Path(row['file_path']).name}")
            try:
                vid = self._upload(row["id"], body, Path(row["file_path"]).stem,
                                   notify=self.settings.notify_subscribers_long if row["kind"] == "long"
                                   else self.settings.notify_subscribers_shorts)
            except (UploadError, QuotaExceededError) as exc:
                warnings.append(f"resume failed {row['file_path']}: {exc}")
                continue
            resumed.append(vid)
            if row["kind"] == "long" and row["episode_id"]:
                plan = plan_for_episode(self.db, row["episode_id"])
                self.db.update_episode(row["episode_id"], status="published", metadata={"youtube_video_id": vid})
                md = self._metadata_for_resume(row["episode_id"], plan, bp, channel_bp, body)
                self._post_steps(row["id"], row["episode_id"], plan, md, playlists, warnings)
        return resumed

    def _metadata_for_resume(self, ep_id: int, plan: EpisodePlan, bp: dict[str, Any], channel_bp: dict[str, Any],
                             body: dict[str, Any]) -> dict[str, Any]:
        meta = episode_meta(self.db, ep_id)
        audio = json.loads(Path(meta["audio_manifest"]).read_text(encoding="utf-8"))
        brand = self._brand(channel_bp)
        return {"title": body["snippet"]["title"], "audio": audio, "brand": brand, "video": meta["video"],
                "comment": build_comment(audio, self.settings), "body": body}

    def _poll_processing(self) -> int:
        """videos.list (1 unit / 50 ids) -> 'live' once processed and HD (long-form) is available."""
        assert self.client
        rows = self.db.uploads_awaiting_processing()
        rows = rows[: int(self.settings.publishing.get("processing_poll_max_checks", 40)) * 50]
        by_id = {r["youtube_video_id"]: r for r in rows if r["youtube_video_id"]}
        changed = 0
        ids = list(by_id)
        for i in range(0, len(ids), 50):
            batch = ids[i:i + 50]
            resp = self.client.execute("videos.list", lambda b=batch: self.client.service.videos().list(
                part="status,processingDetails,contentDetails", id=",".join(b)))
            for item in resp.get("items", []):
                row = by_id[item["id"]]
                st = item.get("status", {})
                proc = item.get("processingDetails", {}).get("processingStatus", "processing")
                hd = item.get("contentDetails", {}).get("definition") == "hd"
                if st.get("uploadStatus") in {"rejected", "failed"} or proc in {"failed", "terminated"}:
                    reason = st.get("rejectionReason") or st.get("failureReason") or proc
                    self.db.set_upload_status(row["id"], "failed", last_error=f"YouTube processing: {reason}")
                    if row["kind"] == "long" and row["episode_id"]:
                        self.db.update_episode(row["episode_id"], status="failed",
                                               metadata={"error": f"youtube: {reason}"})
                elif st.get("uploadStatus") == "processed" and proc == "succeeded" and (hd or row["kind"] == "short"):
                    self.db.set_upload_status(row["id"], "live", processing_status="succeeded_hd" if hd else
                                              "succeeded")
                else:
                    self.db.set_upload_status(row["id"], "processing", processing_status=f"{proc}/hd={hd}")
                changed += 1
            missing = set(batch) - {it["id"] for it in resp.get("items", [])}
            for vid in missing:  # deleted on YouTube
                self.db.set_upload_status(by_id[vid]["id"], "failed", last_error="video not found on YouTube")
                changed += 1
        return changed
