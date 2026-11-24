"""Agent 2 - 8-Hour KJV Scripture & Ambient Audio Engineer.

Produces ``output/episodes/<slug>/master.m4a``:

* 0 -> ~2.5 h: slow KJV narration (TTS, cached per verse) with natural verse/chapter pauses,
  warm EQ, de-essing, gentle compression and two-pass EBU R128 loudness normalisation,
  laid over the ambient bed which is side-chain ducked under the voice.
* narration fades out over the blueprint crossfade, the ducking releases, and the bed
  continues uninterrupted (no loop seams) to the 8-10 h end.

Also writes a verse-level timeline (used for chapters and Shorts) and registers the audio
fingerprint so no two uploads share an identical master.
"""
from __future__ import annotations

import json
import random
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agents.base import AgentResult, BaseAgent
from core.ambient import build_ambient_graph, write_thunder_track
from core.ffmpeg import (FFmpegError, loudnorm_second_pass, measure_loudness, media_duration, run_ffmpeg)
from core.kjv import KJVBible, chapter_heading, speech_text, spoken_book, display_ref
from core.planner import EpisodePlan, EpisodePlanner, load_blueprint, plan_for_episode
from core.tts import CachedTTS, TTSEngine, make_engine
from db.database import DuplicateContentError, sha256_head_tail

INTRO_TEXT = ("Rest now, and be still. Tonight's reading is from {ref}, in the King James Version. "
              "Let the rain carry your thoughts away.")
OUTRO_TEXT = ("The Lord bless thee, and keep thee: the Lord make his face shine upon thee, and be gracious "
              "unto thee: the Lord lift up his countenance upon thee, and give thee peace. Sleep now, in His peace.")


@dataclass
class ScriptItem:
    kind: str           # intro | heading | verse | outro
    text: str           # what is spoken
    display: str        # what is shown (Shorts captions)
    ref: str
    pause_after: float
    chapter_key: str = ""


@dataclass
class TimelineEntry:
    kind: str
    ref: str
    start: float
    end: float
    display: str = ""
    chapter_key: str = ""


def build_script(bible: KJVBible, plan: EpisodePlan, voice: dict[str, Any]) -> list[ScriptItem]:
    rng = random.Random(plan.seed ^ 0x5EED)
    lo, hi = voice.get("pause_after_verse_s", [1.2, 2.0])
    after_ch = float(voice.get("pause_after_chapter_s", 6.0))
    spoken_ref = " and ".join(spoken_book(b) for b in dict.fromkeys(s[0] for s in plan.passage_set["segments"]))
    items = [ScriptItem("intro", INTRO_TEXT.format(ref=spoken_ref), "", "intro", 3.5)]
    for book, first, last in plan.passage_set["segments"]:
        for chapter in bible.passage(book, first, last):
            c = chapter[0].chapter
            key = f"{book} {c}"
            items.append(ScriptItem("heading", chapter_heading(book, c), key, key, 2.5, key))
            for i, v in enumerate(chapter):
                pause = after_ch if i == len(chapter) - 1 else round(rng.uniform(lo, hi), 2)
                items.append(ScriptItem("verse", speech_text(v.text), v.text, v.ref, pause, key))
    items.append(ScriptItem("outro", OUTRO_TEXT, "", "Numbers 6:24-26", 0.0))
    return items


class AudioEngineerAgent(BaseAgent):
    name = "agent2_audio"

    def __init__(self, *args: Any, engine: TTSEngine | None = None, bible: KJVBible | None = None,
                 preview_seconds: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._engine = engine
        self._bible = bible
        self.preview_seconds = preview_seconds if preview_seconds is not None else self.settings.preview_seconds

    # ------------------------------------------------------------------ entry
    def execute(self) -> AgentResult:
        bp = load_blueprint(self.settings.content_blueprint_path)
        planned = [e for e in self.db.episodes_by_status("planned")
                   if e["slug"].startswith("preview-") == bool(self.preview_seconds)]
        if planned:
            ep_id, plan = planned[0]["id"], plan_for_episode(self.db, planned[0]["id"])
            self.status(f"Resuming planned episode {plan.slug}")
        else:
            ep_id, plan = EpisodePlanner(self.db, bp).plan_next(preview=bool(self.preview_seconds))
            self.status(f"Planned {plan.slug} ({plan.book_ref}, {plan.bed['label']}, {plan.hours}h)")
        try:
            manifest = self.produce(ep_id, plan, bp)
        except Exception as exc:
            self.db.update_episode(ep_id, status="failed", metadata={"error": f"agent2: {exc}"})
            self.db.release_fingerprints(ep_id)
            raise
        return AgentResult(self.name, True, summary={
            "episode_id": ep_id, "slug": plan.slug, "passage": plan.book_ref, "bed": plan.bed["id"],
            "duration_s": manifest["duration_s"], "narration_end_s": manifest["narration_end_s"],
            "verses": manifest["verses_read"], "master": manifest["master_audio"],
        })

    # ------------------------------------------------------------------ pipeline
    def produce(self, ep_id: int, plan: EpisodePlan, bp: dict[str, Any]) -> dict[str, Any]:
        spec = bp["episode_spec"]
        narr = spec["narration"]
        total = float(self.preview_seconds or plan.duration_s)
        scale = total / plan.duration_s
        window_end = narr["window_s"]["end"] * scale
        crossfade = max(3.0, narr["crossfade_out_s"] * scale)
        intro_silence = min(float(narr["voice"].get("intro_silence_s", 8.0)), window_end * 0.05)
        work = self.settings.output_dir / "episodes" / plan.slug
        work.mkdir(parents=True, exist_ok=True)

        bible = self._bible or KJVBible.load()
        engine = CachedTTS(self._engine or make_engine(self.settings), self.settings.output_dir / "tts_cache")
        script = build_script(bible, plan, narr["voice"])
        target_tempo = float(narr["voice"].get("tempo", 0.88))
        post_tempo = round(target_tempo / engine.native_tempo, 4) if engine.native_tempo != target_tempo else 1.0

        # 1. narration (raw)
        raw = work / "narration_raw.wav"
        budget = window_end - crossfade - intro_silence
        timeline = self._synthesize(engine, script, raw, budget, post_tempo)
        verses = [t for t in timeline if t.kind == "verse"]
        if not verses:
            raise RuntimeError("Narration budget too small to read a single verse")
        self.status(f"Narrated {len(verses)} verses ({timeline[-1].end / 60:.1f} min)")

        # 2. voice processing: warmth EQ, de-ess, compress, two-pass loudnorm, 48 kHz
        proc = work / "narration.wav"
        self._process_voice(raw, proc, narr["processing"], post_tempo)
        actual = media_duration(proc)
        expected = timeline[-1].end
        if expected > 0 and abs(actual - expected) / expected > 0.01:
            k = actual / expected
            timeline = [TimelineEntry(t.kind, t.ref, t.start * k, t.end * k, t.display, t.chapter_key)
                        for t in timeline]
        raw.unlink(missing_ok=True)
        narration_end = intro_silence + timeline[-1].end
        fade_start = narration_end  # fade the tail (outro + room tone) into rain
        timeline = [TimelineEntry(t.kind, t.ref, t.start + intro_silence, t.end + intro_silence, t.display,
                                  t.chapter_key) for t in timeline]

        # 3. thunder
        thunder = None
        rate = float(plan.bed.get("thunder_per_hour", 0)) * float(plan.ambient_params.get("thunder_rate_scale", 1))
        if rate > 0:
            thunder = work / "thunder.flac"
            n = write_thunder_track(thunder, total, rate, int(plan.ambient_params["seed"]),
                                    quiet_until=narration_end)
            self.status(f"Thunder track: {n} distant rolls")

        # 4. ambient calibration + master mix
        master = work / "master.m4a"
        amb = spec["ambient"]
        gain = self._ambient_gain(plan, min(90.0, total), float(amb["loudnorm_lufs"]))
        self._mix(plan, proc, thunder, master, total=total, intro=intro_silence, fade_start=fade_start,
                  crossfade=crossfade, ambient_gain_db=gain, ducking=narr["ducking"],
                  true_peak=float(narr["processing"]["true_peak_db"]))

        dur = media_duration(master)
        if abs(dur - total) > 1.5:
            raise RuntimeError(f"Master duration {dur:.1f}s != planned {total:.1f}s")
        if dur > spec["youtube_limits"]["max_duration_s"]:
            raise RuntimeError("Master exceeds YouTube's 12 h limit")
        fp = sha256_head_tail(master)
        if not plan.preview:
            try:
                self.db.register_fingerprint("audio_master", fp, episode_id=ep_id)
            except DuplicateContentError:
                raise RuntimeError("Identical audio master already published - aborting episode") from None

        chapters = build_chapters(timeline, narration_end, total, plan)
        manifest = {
            "episode_id": ep_id, "slug": plan.slug, "duration_s": round(dur, 2), "planned_duration_s": total,
            "narration_start_s": intro_silence, "narration_end_s": round(narration_end, 2),
            "crossfade_s": crossfade, "ambient_gain_db": gain, "post_tempo": post_tempo,
            "tts_engine": engine.engine.cache_key, "verses_read": len(verses),
            "chapters_read": sorted({t.chapter_key for t in verses}, key=lambda k: next(
                i for i, t in enumerate(timeline) if t.chapter_key == k)),
            "passage_display": display_ref(plan.passage_set["segments"]),
            "first_verse": verses[0].ref, "last_verse": verses[-1].ref,
            "master_audio": str(master), "master_sha256_head_tail": fp,
            "narration_audio": str(proc), "thunder_audio": str(thunder) if thunder else None,
            "timeline": [asdict(t) for t in timeline], "chapters": chapters,
        }
        (work / "audio_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.db.register_media(master, "audio_master", episode_id=ep_id)
        self.db.register_media(proc, "intermediate", episode_id=ep_id)
        if thunder:
            self.db.register_media(thunder, "intermediate", episode_id=ep_id)
        self.db.update_episode(ep_id, status="audio_ready", metadata={"audio": {
            k: v for k, v in manifest.items() if k != "timeline"}, "audio_manifest": str(work / "audio_manifest.json")})
        return manifest

    # ------------------------------------------------------------------ steps
    def _synthesize(self, engine: CachedTTS, script: list[ScriptItem], out: Path, budget_s: float,
                    post_tempo: float) -> list[TimelineEntry]:
        """Synthesize verse by verse into a streaming WAV until the narration budget is reached.

        Times are in *final* (post-atempo) seconds; pauses are pre-stretched so they land exactly.
        """
        sr = engine.sample_rate
        outro = script[-1]
        outro_audio = engine.synthesize(outro.text)
        reserve = len(outro_audio) / sr / post_tempo + 4.0
        timeline: list[TimelineEntry] = []
        t = 0.0
        n_items = len(script) - 1
        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)

            def emit(audio: np.ndarray, pause: float, item: ScriptItem) -> None:
                nonlocal t
                speech = len(audio) / sr / post_tempo
                timeline.append(TimelineEntry(item.kind, item.ref, round(t, 3), round(t + speech, 3),
                                              item.display, item.chapter_key))
                silence = np.zeros(int(pause * post_tempo * sr), dtype=np.float32)
                pcm = (np.clip(np.concatenate([audio, silence]), -1, 1) * 32767).astype("<i2")
                wf.writeframes(pcm.tobytes())
                t += speech + pause

            for i, item in enumerate(script[:-1]):
                if item.kind == "heading" and t > 0.92 * (budget_s - reserve):
                    break  # prefer to stop on a chapter boundary
                audio = engine.synthesize(item.text)
                if t + len(audio) / sr / post_tempo + item.pause_after > budget_s - reserve:
                    break
                emit(audio, item.pause_after, item)
                if i % 50 == 0:
                    self.status(f"TTS {i}/{n_items} items, {t / 60:.1f} min narrated")
            # never leave a dangling chapter heading without verses
            while timeline and timeline[-1].kind == "heading":
                timeline.pop()
            emit(outro_audio, 3.0, outro)
        return timeline

    def _process_voice(self, raw: Path, out: Path, p: dict[str, Any], post_tempo: float) -> None:
        chain = (f"highpass=f={p['highpass_hz']},lowpass=f={p['lowpass_hz']},"
                 f"equalizer=f=180:t=q:w=0.9:g={p['warmth_shelf_db']},"
                 f"equalizer=f=3200:t=q:w=1.2:g=-1.5,"
                 + ("deesser=i=0.4:m=0.5:f=0.5," if p.get("deesser") else "")
                 + f"acompressor=threshold=-24dB:ratio={p['compressor_ratio']}:attack=20:release=300:makeup=2")
        if post_tempo != 1.0:
            chain = f"atempo={post_tempo}," + chain
        target, tp = float(p["loudnorm_lufs"]), float(p["true_peak_db"])
        self.status("Voice processing: loudness analysis pass")
        measured = measure_loudness(["-i", str(raw)], chain, target_i=target, target_tp=tp)
        dur = media_duration(raw) / post_tempo
        self.status("Voice processing: normalise pass")
        run_ffmpeg(["-i", str(raw), "-af", f"{chain},{loudnorm_second_pass(measured, target_i=target, target_tp=tp)},"
                    "aresample=48000", "-ac", "1", "-c:a", "pcm_s16le", str(out)],
                   duration_s=dur, progress=lambda f: self.progress("voice processing", f))

    def _ambient_gain(self, plan: EpisodePlan, seconds: float, target_lufs: float) -> float:
        g = build_ambient_graph(plan.bed, plan.ambient_params, seconds)
        lines = run_ffmpeg([*g.input_args, "-filter_complex",
                            f"{g.filter};[{g.out_label}]loudnorm=I={target_lufs}:TP=-2:print_format=json[m]",
                            "-map", "[m]", "-f", "null", "-"])
        text = "\n".join(lines)
        start = text.rfind('"input_i"')
        if start < 0:
            raise FFmpegError(["ambient-calibration"], 0, text[-1500:])
        measured_i = float(text[start:].split(":", 1)[1].split(",", 1)[0].strip().strip('"'))
        gain = round(max(-30.0, min(30.0, target_lufs - measured_i)), 2)
        self.status(f"Ambient bed measured {measured_i:.1f} LUFS -> gain {gain:+.1f} dB")
        return gain

    def _mix(self, plan: EpisodePlan, narration: Path, thunder: Path | None, out: Path, *, total: float,
             intro: float, fade_start: float, crossfade: float, ambient_gain_db: float,
             ducking: dict[str, Any], true_peak: float) -> None:
        g = build_ambient_graph(plan.bed, plan.ambient_params, total, thunder_path=thunder, input_offset=0)
        n_idx = g.n_inputs
        limit = round(10 ** (true_peak / 20), 4)
        fade_out = min(30.0, total * 0.05)
        # threshold chosen so a -20 LUFS voice drives ~ambient_reduction_db of gain reduction
        ratio = max(2.0, float(ducking.get("ambient_reduction_db", 9)) / 2.5)
        graph = ";".join([
            g.filter,
            f"[{g.out_label}]volume={ambient_gain_db}dB,afade=t=in:d={min(5.0, total * 0.02):.2f}[bed]",
            f"[{n_idx}:a]aresample=48000,pan=stereo|c0=c0|c1=c0,adelay={int(intro * 1000)}:all=1,"
            f"apad=whole_dur={total:.3f},afade=t=out:st={max(0.0, fade_start - crossfade):.3f}:d={crossfade:.3f},"
            "asplit=2[nmix][nsc]",
            f"[bed][nsc]sidechaincompress=threshold=0.015:ratio={ratio:.1f}:attack={ducking.get('attack_ms', 120)}:"
            f"release={ducking.get('release_ms', 1600)}:makeup=1[duck]",
            f"[duck][nmix]amix=inputs=2:normalize=0:duration=first,atrim=0:{total:.3f},"
            f"afade=t=out:st={total - fade_out:.3f}:d={fade_out:.3f},alimiter=limit={limit}:level=0[out]",
        ])
        script = out.with_suffix(".filtergraph.txt")
        script.write_text(graph, encoding="utf-8")
        self.status(f"Mixing master ({total / 3600:.2f} h)...")
        run_ffmpeg([*g.input_args, "-i", str(narration), "-filter_complex_script", str(script),
                    "-map", "[out]", "-c:a", "aac", "-b:a", str(self.settings.render.get("audio_bitrate", "192k")),
                    "-ar", "48000", "-ac", "2", "-t", f"{total:.3f}", "-movflags", "+faststart", str(out)],
                   duration_s=total, progress=lambda f: self.progress("master mix", f))
        script.unlink(missing_ok=True)


def build_chapters(timeline: list[TimelineEntry], narration_end: float, total: float,
                   plan: EpisodePlan) -> list[dict[str, Any]]:
    """YouTube chapters: start 0:00, ascending, >= 10 s apart, >= 3 entries."""
    raw: list[tuple[float, str]] = [(0.0, "Opening - Settle In")]
    headings = [t for t in timeline if t.kind == "heading"]
    if len(headings) > 40:  # group long Psalm runs into pairs
        headings = headings[::2]
    for h in headings:
        raw.append((h.start, h.chapter_key if not h.chapter_key.startswith("Psalms ") else
                    h.chapter_key.replace("Psalms", "Psalm")))
    raw.append((narration_end, "Narration Ends - Rain Only for Deep Sleep"))
    hour = 3600.0
    n = int(narration_end // hour) + 1
    while n * hour < total - 60:
        raw.append((n * hour, f"Hour {n + 1} - {plan.bed['label']}"))
        n += 1
    chapters: list[dict[str, Any]] = []
    for at, label in sorted(raw):
        if chapters and at - chapters[-1]["start_s"] < 10:
            continue
        chapters.append({"start_s": round(at, 2), "label": label})
    return chapters
