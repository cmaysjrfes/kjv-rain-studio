"""Agent 3 - Dark Visual & Thumbnail Generator.

For each ``audio_ready`` episode:
1. Render a seamless ultra-dark scene loop (procedural, or your own footage from
   ``assets/visuals/<scene_id>/`` made seamless with a crossfade wrap and darkened to target luma).
2. Mux the 8-10 h master by *stream-copying* the loop N times (concat demuxer) - an 8 h
   video is produced in minutes with zero re-encoding of the repeated segment.
3. Render a vertical 9:16 loop for Agent 4's Shorts.
4. Render two A/B thumbnails (1280x720, < 2 MB) that are perceptually unique vs. history
   and vs. competitor thumbnails.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from agents.base import AgentResult, BaseAgent
from core.catalog import THUMBNAIL_TEMPLATES
from core.ffmpeg import FFmpegWriter, detect_encoder, encoder_args, media_duration, probe, run_ffmpeg
from core.planner import EpisodePlan, episode_meta, load_blueprint, plan_for_episode
from core.scenes import SceneRenderer
from core.thumbnails import ThumbContext, render_thumbnail
from db.database import DuplicateContentError, hamming_hex, sha256_head_tail

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


class VisualsAgent(BaseAgent):
    name = "agent3_visuals"

    def __init__(self, *args: Any, preview_seconds: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.preview_seconds = preview_seconds if preview_seconds is not None else self.settings.preview_seconds

    def execute(self) -> AgentResult:
        bp = load_blueprint(self.settings.content_blueprint_path)
        eps = [e for e in self.db.episodes_by_status("audio_ready")
               if e["slug"].startswith("preview-") == bool(self.preview_seconds)]
        if not eps:
            return AgentResult(self.name, True, summary={"rendered": 0, "note": "no audio_ready episodes"})
        encoder = detect_encoder(self.settings.video_encoder)
        done = []
        for ep in eps:
            plan = plan_for_episode(self.db, ep["id"])
            try:
                done.append(self.produce(ep["id"], plan, bp, encoder))
            except Exception as exc:
                self.db.update_episode(ep["id"], status="failed", metadata={"error": f"agent3: {exc}"})
                self.db.release_fingerprints(ep["id"])
                raise
        return AgentResult(self.name, True, summary={"rendered": len(done), "encoder": encoder,
                                                     "videos": [d["final_video"] for d in done]})

    # ------------------------------------------------------------------ pipeline
    def produce(self, ep_id: int, plan: EpisodePlan, bp: dict[str, Any], encoder: str) -> dict[str, Any]:
        meta = episode_meta(self.db, ep_id)
        audio_manifest = json.loads(Path(meta["audio_manifest"]).read_text(encoding="utf-8"))
        master = Path(audio_manifest["master_audio"])
        total = media_duration(master)
        work = master.parent
        vspec = bp["episode_spec"]["video"]
        fps = self.settings.video_fps
        loop_s = float(self.settings.render.get("loop_seconds", 120))
        loop_s = min(loop_s, max(10.0, total))
        size = tuple(self.settings.render.get("internal_resolution", [960, 540]))
        out_w, out_h = self.settings.render.get("output_resolution") or vspec["resolution"]
        max_luma = float(vspec.get("max_mean_luma", 32))

        job = self.db.create_render_job(ep_id, "visual_loop", {"scene": plan.scene["id"], "loop_s": loop_s},
                                        encoder)
        self.db.update_render_job(job, status="running")
        loop = work / "loop.mp4"
        try:
            footage = self._footage(plan.scene["id"])
            if footage:
                luma = self._footage_loop(footage, loop, loop_s, fps, (out_w, out_h), encoder, max_luma, plan)
            else:
                luma = self._procedural_loop(plan, loop, loop_s, fps, size, (out_w, out_h), encoder, max_luma)
            if luma > max_luma + 2:
                raise RuntimeError(f"Visual loop too bright for sleep content: mean luma {luma:.1f} > {max_luma}")
            self.db.update_render_job(job, status="done", progress=1.0, output_path=str(loop))
        except Exception as exc:
            self.db.update_render_job(job, status="failed", error=str(exc)[:2000])
            raise
        if not plan.preview:
            try:
                self.db.register_fingerprint("visual_loop_file", sha256_head_tail(loop), episode_id=ep_id)
            except DuplicateContentError:
                raise RuntimeError("Visual loop file identical to a previous episode") from None

        # final mux: concat-copy the loop, copy AAC audio
        job = self.db.create_render_job(ep_id, "final_mux", {"loop": str(loop), "audio": str(master)}, "copy")
        self.db.update_render_job(job, status="running")
        final = work / f"{plan.slug}.mp4"
        try:
            self._mux(loop, master, final, total, loop_s)
            self._verify_final(final, total, bp)
            self.db.update_render_job(job, status="done", progress=1.0, output_path=str(final))
        except Exception as exc:
            self.db.update_render_job(job, status="failed", error=str(exc)[:2000])
            raise

        # vertical loop for Shorts
        shorts_loop = work / "shorts_loop.mp4"
        self._procedural_loop(plan, shorts_loop, float(self.settings.render.get("shorts_loop_seconds", 20)), fps,
                              (540, 960), (1080, 1920), encoder, max_luma, luma_boost=1.35)

        thumbs = self._thumbnails(ep_id, plan, bp, audio_manifest, work)
        for p, kind in ((loop, "intermediate"), (shorts_loop, "intermediate"), (final, "final_mp4")):
            self.db.register_media(p, kind, episode_id=ep_id)
        for t in thumbs:
            self.db.register_media(t["path"], "thumbnail", episode_id=ep_id, keep=True)
        video = {"final_video": str(final), "bytes": final.stat().st_size, "duration_s": round(total, 2),
                 "loop": str(loop), "shorts_loop": str(shorts_loop), "encoder": encoder,
                 "mean_luma": round(luma, 1), "thumbnails": thumbs}
        self.db.update_episode(ep_id, status="video_ready", metadata={"video": video})
        self.status(f"Video ready: {final.name} ({final.stat().st_size / 1e9:.2f} GB)")
        return video

    # ------------------------------------------------------------------ loops
    def _procedural_loop(self, plan: EpisodePlan, out: Path, loop_s: float, fps: int, size: tuple[int, int],
                         out_size: tuple[int, int], encoder: str, max_luma: float, luma_boost: float = 1.0) -> float:
        r = SceneRenderer(plan.scene, plan.visual_params, size=size, loop_s=loop_s, fps=fps,
                          max_mean_luma=max_luma)
        if luma_boost != 1.0:
            r.gain *= luma_boost
        w, h = size
        args = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                "-vf", f"scale={out_size[0]}:{out_size[1]}:flags=lanczos,noise=alls=2:allf=u",
                *encoder_args(encoder, fps, quality="loop"), "-an", "-movflags", "+faststart", str(out)]
        lumas = []
        with FFmpegWriter(args) as wr:
            for i in range(r.n_frames):
                f = r.frame(i)
                if i % max(1, r.n_frames // 12) == 0:
                    lumas.append(r.mean_luma(f))
                wr.write(f.tobytes())
                if i % fps == 0:
                    self.progress(f"render {out.stem}", i / r.n_frames)
        return statistics.mean(lumas)

    @staticmethod
    def _footage(scene_id: str) -> list[Path]:
        from core.config import PROJECT_ROOT

        d = PROJECT_ROOT / "assets" / "visuals" / scene_id
        return sorted(p for p in d.glob("*") if p.suffix.lower() in VIDEO_EXTS) if d.exists() else []

    def _footage_loop(self, files: list[Path], out: Path, loop_s: float, fps: int, out_size: tuple[int, int],
                      encoder: str, max_luma: float, plan: EpisodePlan) -> float:
        """User footage -> seamless loop: tail crossfades into head; gain set to the target luma."""
        src = files[int(plan.visual_params.get("seed", 0)) % len(files)]
        dur = media_duration(src)
        xf = 2.0
        loop_s = min(loop_s, dur - xf)
        if loop_s < 8:
            raise RuntimeError(f"Footage {src.name} too short for a loop ({dur:.1f}s)")
        lines = run_ffmpeg(["-t", "10", "-i", str(src), "-vf",
                            "signalstats,metadata=print:key=lavfi.signalstats.YAVG", "-f", "null", "-"])
        yavgs = [float(m) for line in lines for m in re.findall(r"YAVG=([\d.]+)", line)]
        yavg = max(1.0, statistics.mean(yavgs) if yavgs else 60.0)
        target = min(max_luma, float(plan.visual_params.get("brightness", 0.08)) * 255)
        g = max(0.05, min(1.0, target / yavg))
        w, h = out_size
        graph = (f"[0:v]fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
                 f"colorchannelmixer=rr={g:.3f}:gg={g:.3f}:bb={g:.3f},split=2[a][b];"
                 f"[a]trim={xf}:{loop_s + xf},setpts=PTS-STARTPTS[main];"
                 f"[b]trim=0:{xf},setpts=PTS-STARTPTS[head];"
                 f"[main][head]xfade=transition=fade:duration={xf}:offset={loop_s - xf},format=yuv420p[v]")
        run_ffmpeg(["-i", str(src), "-filter_complex", graph, "-map", "[v]",
                    *encoder_args(encoder, fps, quality="loop"), "-an", "-t", f"{loop_s:.3f}", str(out)],
                   duration_s=loop_s, progress=lambda f: self.progress("footage loop", f))
        return min(max_luma, target)

    def _mux(self, loop: Path, master: Path, final: Path, total: float, loop_s: float) -> None:
        reps = math.ceil(total / loop_s) + 1
        lst = final.with_suffix(".concat.txt")
        safe = str(loop.resolve()).replace("'", "'\\''")
        lst.write_text("".join(f"file '{safe}'\n" for _ in range(reps)), encoding="utf-8")
        self.status(f"Muxing {total / 3600:.2f} h video ({reps} loop repeats, stream copy)")
        try:
            run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(master), "-map", "0:v:0",
                        "-map", "1:a:0", "-c", "copy", "-t", f"{total:.3f}", "-movflags", "+faststart",
                        str(final)], duration_s=total, progress=lambda f: self.progress("final mux", f))
        finally:
            lst.unlink(missing_ok=True)

    def _verify_final(self, final: Path, total: float, bp: dict[str, Any]) -> None:
        info = probe(final)
        kinds = {s["codec_type"] for s in info["streams"]}
        if kinds != {"video", "audio"}:
            raise RuntimeError(f"Final video streams wrong: {kinds}")
        dur = float(info["format"]["duration"])
        if abs(dur - total) > 2.0:
            raise RuntimeError(f"Final duration {dur:.1f}s differs from audio {total:.1f}s")
        lim = bp["episode_spec"]["youtube_limits"]
        if dur > lim["max_duration_s"] or final.stat().st_size > lim["max_bytes"]:
            raise RuntimeError("Final video exceeds YouTube 12 h / 256 GB limits")

    # ------------------------------------------------------------------ thumbnails
    def _thumbnails(self, ep_id: int, plan: EpisodePlan, bp: dict[str, Any], audio: dict[str, Any],
                    work: Path) -> list[dict[str, Any]]:
        brand = self.db.kv_get("brand") or {}
        palette = brand.get("palette") or bp.get("brand_ref", {}).get("palette")
        if not palette:
            raise RuntimeError("Brand palette missing - run Agent 1 first")
        rules = bp["anti_repetition_rules"]
        own_min = int(rules.get("thumbnail_dhash_min_distance", 10))
        comp_min = int(rules.get("competitor_thumb_dhash_min_distance", 14))
        comp_hashes = rules.get("competitor_thumb_dhashes", [])
        quote, qref = _pick_quote(audio)
        lo, hi = bp["thumbnail_rules"]["target_mean_luma"]
        valid_ids = {t["id"] for t in THUMBNAIL_TEMPLATES}
        out: list[dict[str, Any]] = []
        for label, tid in zip("AB", plan.thumb_templates):
            if tid not in valid_ids:
                raise RuntimeError(f"Unknown thumbnail template {tid}")
            for attempt in range(6):
                params = {**plan.visual_params, "seed": plan.visual_params["seed"] + 101 * attempt + ord(label),
                          "glow_x": (plan.visual_params.get("glow_x", 0.5) + 0.17 * attempt) % 1.0}
                still = SceneRenderer(plan.scene, params, size=(1280, 720), loop_s=10, fps=1,
                                      target_luma=(lo + hi) / 2 + 10).still(t=attempt * 1.7)
                ctx = ThumbContext(headline=plan.passage_set["theme"], book_ref=plan.book_ref, hours=plan.hours,
                                   bed_label=plan.bed["label"], verse_quote=quote, verse_ref=qref,
                                   palette=palette, seed=plan.seed + attempt)
                info = render_thumbnail(still, tid, ctx, work / f"thumb_{label}.jpg")
                if any(hamming_hex(info["dhash"], h) < comp_min for h in comp_hashes):
                    continue
                if self.db.find_similar_phash("thumbnail", info["dhash"], own_min - 1):
                    continue
                if any(hamming_hex(info["dhash"], o["dhash"]) < own_min for o in out):
                    continue
                if not plan.preview:
                    try:
                        self.db.register_fingerprint("thumbnail", info["dhash"], algo="dhash64", episode_id=ep_id,
                                                     phash_max_distance=own_min - 1, detail={"label": label})
                    except DuplicateContentError:
                        continue
                out.append({**info, "label": label})
                break
            else:
                raise RuntimeError(f"Could not render a unique thumbnail for variant {label}")
        return out


def _pick_quote(audio: dict[str, Any]) -> tuple[str, str]:
    """Shortest complete verse (6-16 words) actually read in this episode, for the verse-card template."""
    verses = audio.get("timeline", [])
    cands = [t for t in verses if t["kind"] == "verse" and 6 <= len(t["display"].split()) <= 16]
    if not cands:
        cands = [t for t in verses if t["kind"] == "verse"] or [{"display": "Be still, and know that I am God.",
                                                                  "ref": "Psalms 46:10"}]
    best = min(cands, key=lambda t: abs(len(t["display"].split()) - 10))
    return best["display"].rstrip(":;,"), best["ref"]

