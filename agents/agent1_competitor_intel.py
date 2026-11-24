"""Agent 1 - Competitor Intelligence & Channel Page Architect.

Pipeline
--------
1. Collect 8h+ "KJV + rain + sleep" competitors (YouTube Data API v3, yt-dlp fallback).
2. Persist channels/videos + metric snapshots to SQLite.
3. Analyse *structural* patterns only - title anatomy, description layout, thumbnail
   darkness/contrast/text placement, banner/avatar hierarchy, playlists, cadence.
4. Design a brand that is measurably distinct from every competitor (name similarity,
   palette colour distance) while following the winning structure.
5. Emit ``channel_page_blueprint.json``, ``content_blueprint.json`` and the brand kit
   (banner, avatar, watermark, playlist plan, channel description).

No competitor titles, descriptions, artwork or audio are reused - only aggregate statistics.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from agents.base import AgentResult, BaseAgent
from core import catalog
from core.growth import STUDIO_ONLY_CHECKLIST
from core.imaging import (analyze_url, contrast_ratio, delta_e, render_avatar, render_banner,
                          render_watermark)
from core.scraper import CompetitorDataset, collect_competitors
from db.database import sha256_fingerprint

BLUEPRINT_VERSION = "1.0"

TITLE_FEATURES: dict[str, str] = {
    "duration_cue": r"\b(?:8|9|10|11|12)\s*(?:\+\s*)?(?:hours?|hrs?|h)\b",
    "black_or_dark_screen": r"\b(?:black|dark)\s*screen\b",
    "rain": r"\brain",
    "thunder": r"\bthunder",
    "kjv": r"\bkjv\b|king\s+james",
    "sleep": r"\bsleep",
    "bible_or_scripture": r"\bbible\b|\bscriptures?\b|\bverses?\b|god'?s\s+word",
    "psalms": r"\bpsalms?\b",
    "emotional_benefit": r"\b(?:peace|anxiety|calm|rest|comfort|protection|healing|fear|worry|stress)\b",
    "no_ads_claim": r"\bno\s+(?:mid-?roll\s+)?ads\b",
    "pipe_separator": r"\|",
    "dash_separator": r"\s[-–—]\s",
    "all_caps_word": r"\b[A-Z]{4,}\b",
    "emoji": "[\U0001F300-\U0001FAFF☀-➿]",
    "book_name": r"\b(?:" + "|".join(re.escape(b) for b in catalog.BIBLE_BOOKS) + r")\b",
}
STOPWORDS = set("""a an the of to and in for with on at by from is are be your you my me i it this that as or
into over all his her our their god's""".split())
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
URL_RE = re.compile(r"https?://\S+")


# ============================================================================ analysis
class PatternAnalyzer:
    """Turns raw competitor data into aggregate, non-copyable structural patterns."""

    def __init__(self, ds: CompetitorDataset, timeout: int = 20, max_images: int = 60) -> None:
        self.ds = ds
        self.timeout = timeout
        self.max_images = max_images

    # ---- titles
    def title_patterns(self) -> dict[str, Any]:
        vids = [v for v in self.ds.videos if v.get("title")]
        if not vids:
            return {}
        feats: dict[str, dict[str, float]] = {}
        for name, pattern in TITLE_FEATURES.items():
            rx = re.compile(pattern, re.IGNORECASE if name not in {"all_caps_word"} else 0)
            with_f = [v["view_count"] or 0 for v in vids if rx.search(v["title"])]
            without = [v["view_count"] or 0 for v in vids if not rx.search(v["title"])]
            prev = len(with_f) / len(vids)
            lift = (statistics.mean(with_f) / statistics.mean(without)) if with_f and without and \
                statistics.mean(without) > 0 else 1.0
            feats[name] = {"prevalence": round(prev, 3), "view_lift": round(lift, 2)}

        dur_rx = re.compile(TITLE_FEATURES["duration_cue"], re.IGNORECASE)
        positions = [m.start() / max(1, len(v["title"])) for v in vids if (m := dur_rx.search(v["title"]))]
        lengths = [len(v["title"]) for v in vids]
        return {
            "sample_size": len(vids),
            "length_chars": {"median": statistics.median(lengths), "p25": _pct(lengths, 25),
                             "p75": _pct(lengths, 75)},
            "features": feats,
            "duration_cue_position": ("front" if positions and statistics.median(positions) < 0.33 else
                                      "end" if positions and statistics.median(positions) > 0.66 else "middle"),
            "top_view_weighted_phrases": self._weighted_ngrams(vids),
            "segments_per_title_median": statistics.median(
                [len(re.split(r"\s[|•\-–—]\s|\|", v["title"])) for v in vids]),
        }

    @staticmethod
    def _weighted_ngrams(vids: list[dict[str, Any]], top: int = 25) -> list[dict[str, Any]]:
        """Generic 2-3 word phrases weighted by sqrt(views) - signals vocabulary, not titles to copy."""
        score: Counter[str] = Counter()
        for v in vids:
            words = [w for w in re.findall(r"[a-z0-9']+", v["title"].lower())]
            w = (v.get("view_count") or 0) ** 0.5
            grams = set()
            for n in (2, 3):
                for i in range(len(words) - n + 1):
                    g = words[i:i + n]
                    if g[0] in STOPWORDS or g[-1] in STOPWORDS:
                        continue
                    grams.add(" ".join(g))
            for g in grams:
                score[g] += w
        channels_using: dict[str, set[str]] = {}
        for v in vids:
            t = v["title"].lower()
            for g, _ in score.most_common(200):
                if g in t:
                    channels_using.setdefault(g, set()).add(v["channel_id"])
        # Only phrases used by >= 2 channels are genre vocabulary (not one creator's signature)
        return [{"phrase": g, "score": round(s, 1), "channels": len(channels_using.get(g, ()))}
                for g, s in score.most_common(200) if len(channels_using.get(g, ())) >= 2][:top]

    # ---- descriptions
    def description_patterns(self) -> dict[str, Any]:
        descs = [v for v in self.ds.videos if v.get("description")]
        if not descs:
            return {}
        tags: Counter[str] = Counter()
        for v in descs:
            tags.update(h.lower() for h in HASHTAG_RE.findall(v["description"]))
        lens = [len(v["description"]) for v in descs]
        first_lines = [len(v["description"].strip().splitlines()[0]) for v in descs if v["description"].strip()]
        return {
            "sample_size": len(descs),
            "length_chars_median": statistics.median(lens),
            "first_line_chars_median": statistics.median(first_lines) if first_lines else 0,
            "share_with_chapters": round(sum(bool(v.get("has_chapters")) for v in descs) / len(descs), 3),
            "share_with_hashtags": round(sum(bool(HASHTAG_RE.search(v["description"])) for v in descs)
                                         / len(descs), 3),
            "share_with_links": round(sum(bool(URL_RE.search(v["description"])) for v in descs) / len(descs), 3),
            "share_mentions_subscribe": round(sum("subscribe" in v["description"].lower() for v in descs)
                                              / len(descs), 3),
            "share_mentions_kjv_public_domain": round(sum("public domain" in v["description"].lower()
                                                          for v in descs) / len(descs), 3),
            "common_hashtags": [t for t, n in tags.most_common(15) if n >= 2],
        }

    # ---- thumbnails / banners / avatars
    def visual_patterns(self) -> dict[str, Any]:
        thumbs = [v for v in self.ds.videos if v.get("thumbnail_url")][: self.max_images]
        with ThreadPoolExecutor(max_workers=8) as pool:
            thumb_metrics = list(pool.map(lambda v: analyze_url(v["thumbnail_url"], self.timeout), thumbs))
            banner_metrics = list(pool.map(lambda c: analyze_url(c.get("banner_url"), self.timeout),
                                           self.ds.channels))
            avatar_metrics = list(pool.map(lambda c: analyze_url(c.get("avatar_url"), self.timeout),
                                           self.ds.channels))
        for v, m in zip(thumbs, thumb_metrics):
            if m:
                v["thumb_metrics"] = m
        return {
            "thumbnails": _aggregate_metrics([m for m in thumb_metrics if m]),
            "banners": _aggregate_metrics([m for m in banner_metrics if m]),
            "avatars": _aggregate_metrics([m for m in avatar_metrics if m]),
            "competitor_colors": sorted({c for m in [*thumb_metrics, *banner_metrics] if m
                                         for c in m["palette"][:3]}),
            "competitor_thumb_dhashes": [m["dhash"] for m in thumb_metrics if m],
        }

    # ---- channel level
    def channel_patterns(self) -> dict[str, Any]:
        chs = self.ds.channels
        if not chs:
            return {}
        gaps = [c["median_upload_gap_days"] for c in chs if c.get("median_upload_gap_days")]
        about_lens = [len(c.get("description") or "") for c in chs]
        pl_tokens: Counter[str] = Counter()
        pl_counts: list[int] = []
        for c in chs:
            pls = c.get("playlists") or []
            pl_counts.append(len(pls))
            for p in pls:
                pl_tokens.update(w for w in re.findall(r"[a-z]+", (p.get("title") or "").lower())
                                 if w not in STOPWORDS and len(w) > 2)
        long_share = [c["long_video_count"] / max(1, len(c.get("uploads") or [])) for c in chs if c.get("uploads")]
        about = [c.get("description") or "" for c in chs]
        return {
            "sample_size": len(chs),
            "channels": [{"title": c["title"], "subs": c.get("subscriber_count"), "score": c["score"],
                          "median_upload_gap_days": c.get("median_upload_gap_days"),
                          "long_video_count": c.get("long_video_count")} for c in chs],
            "upload_gap_days_median": statistics.median(gaps) if gaps else None,
            "upload_gap_days_best_quartile": _pct(gaps, 25) if gaps else None,
            "about_length_median": statistics.median(about_lens),
            "about_mentions": {
                "kjv": _share(about, r"kjv|king james"), "sleep": _share(about, r"sleep"),
                "schedule": _share(about, r"\b(daily|every|new videos?|each week|weekly)\b"),
                "no_ads": _share(about, r"no (mid-?roll )?ads"), "prayer": _share(about, r"pray"),
            },
            "playlists_per_channel_median": statistics.median(pl_counts) if pl_counts else 0,
            "playlist_vocabulary": [w for w, _ in pl_tokens.most_common(20)],
            "long_form_share_median": round(statistics.median(long_share), 3) if long_share else None,
        }


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


def _share(texts: list[str], pattern: str) -> float:
    rx = re.compile(pattern, re.IGNORECASE)
    return round(sum(bool(rx.search(t)) for t in texts) / max(1, len(texts)), 3)


def _aggregate_metrics(ms: list[dict[str, Any]]) -> dict[str, Any]:
    if not ms:
        return {"sample_size": 0}
    keys = ["mean_luma", "luma_std", "p95_luma", "dark_ratio", "bright_ratio", "saturation", "warmth",
            "edge_density"]
    return {
        "sample_size": len(ms),
        **{k: round(statistics.median(m[k] for m in ms), 3) for k in keys},
        "text_column_mode": Counter(m["text_column"] for m in ms).most_common(1)[0][0],
        "text_row_mode": Counter(m["text_row"] for m in ms).most_common(1)[0][0],
    }


# Industry-informed priors used ONLY when no live competitor data could be collected
# (offline/no network). The blueprint flags this explicitly via `data_source`.
BASELINE_PRIORS: dict[str, Any] = {
    "titles": {
        "sample_size": 0, "length_chars": {"median": 78, "p25": 64, "p75": 92},
        "duration_cue_position": "end",
        "features": {k: {"prevalence": v, "view_lift": 1.0} for k, v in {
            "duration_cue": 0.9, "black_or_dark_screen": 0.55, "rain": 0.85, "thunder": 0.25, "kjv": 0.6,
            "sleep": 0.9, "bible_or_scripture": 0.95, "psalms": 0.35, "emotional_benefit": 0.45,
            "no_ads_claim": 0.1, "pipe_separator": 0.6, "dash_separator": 0.3, "all_caps_word": 0.3,
            "emoji": 0.1, "book_name": 0.5}.items()},
        "top_view_weighted_phrases": [], "segments_per_title_median": 3,
    },
    "descriptions": {"sample_size": 0, "length_chars_median": 1400, "first_line_chars_median": 110,
                     "share_with_chapters": 0.5, "share_with_hashtags": 0.7, "share_with_links": 0.6,
                     "share_mentions_subscribe": 0.7, "share_mentions_kjv_public_domain": 0.15,
                     "common_hashtags": ["#kjv", "#biblesleep", "#rainsounds", "#sleepwithgodsword"]},
    "visuals": {
        "thumbnails": {"sample_size": 0, "mean_luma": 38.0, "luma_std": 48.0, "dark_ratio": 0.62,
                       "saturation": 0.32, "warmth": 14.0, "text_column_mode": "center", "text_row_mode": "middle"},
        "banners": {"sample_size": 0, "mean_luma": 30.0, "dark_ratio": 0.7},
        "avatars": {"sample_size": 0, "mean_luma": 45.0},
        "competitor_colors": ["#0A0A0A", "#1A2333", "#C8A15A", "#3B4A63", "#FFFFFF"],
        "competitor_thumb_dhashes": [],
    },
    "channels": {"sample_size": 0, "channels": [], "upload_gap_days_median": 2.0,
                 "upload_gap_days_best_quartile": 1.0, "about_length_median": 600,
                 "about_mentions": {"kjv": 0.7, "sleep": 0.9, "schedule": 0.4, "no_ads": 0.1, "prayer": 0.5},
                 "playlists_per_channel_median": 6, "playlist_vocabulary": ["psalms", "sleep", "rain", "kjv",
                                                                            "gospel", "promises"],
                 "long_form_share_median": 0.8},
}


# ============================================================================ brand design
class BrandDesigner:
    def __init__(self, brand_cfg: dict[str, Any], *, competitor_titles: list[str], competitor_colors: list[str],
                 env_name: str = "", env_handle: str = "", env_tagline: str = "") -> None:
        self.cfg = brand_cfg
        self.comp_titles = [t.lower() for t in competitor_titles]
        self.comp_colors = competitor_colors
        self.env_name, self.env_handle, self.env_tagline = env_name, env_handle, env_tagline

    def name_similarity(self, name: str) -> tuple[float, str | None]:
        n = name.lower()
        best, who = 0.0, None
        tok_n = set(re.findall(r"[a-z]+", n)) - STOPWORDS - {"sleep", "rain", "kjv", "bible"}
        for t in self.comp_titles:
            ratio = SequenceMatcher(None, n, t).ratio()
            tok_t = set(re.findall(r"[a-z]+", t)) - STOPWORDS - {"sleep", "rain", "kjv", "bible"}
            jacc = len(tok_n & tok_t) / max(1, len(tok_n | tok_t))
            s = max(ratio, jacc)
            if s > best:
                best, who = s, t
        return round(best, 3), who

    def choose_name(self) -> dict[str, Any]:
        candidates = [self.env_name] if self.env_name else list(self.cfg.get("name_candidates", []))
        scored = [(c, *self.name_similarity(c)) for c in candidates]
        scored.sort(key=lambda x: x[1])
        name, sim, nearest = scored[0]
        warning = None
        if sim > 0.6:
            warning = f"Brand name '{name}' is {sim:.0%} similar to competitor '{nearest}'. Consider renaming."
        return {"name": name, "max_similarity_to_competitor": sim, "nearest_competitor": nearest,
                "alternatives": [{"name": c, "max_similarity": s} for c, s, _ in scored[1:]], "warning": warning}

    def choose_palette(self) -> dict[str, Any]:
        palettes: dict[str, list[str]] = self.cfg.get("palettes", {})
        results = []
        for pid, (bg, mid, accent, text) in palettes.items():
            if contrast_ratio(text, bg) < 7.0:
                continue  # must meet WCAG AAA for legibility at thumbnail size
            dists = sorted(delta_e(accent, c) for c in self.comp_colors) or [100.0]
            distinct = statistics.mean(dists[: max(1, len(dists) // 5)])  # mean of nearest 20%
            results.append({"id": pid, "bg": bg, "mid": mid, "accent": accent, "text": text,
                            "distinctiveness_delta_e": round(distinct, 1),
                            "text_contrast": round(contrast_ratio(text, bg), 1)})
        if not results:
            raise ValueError("No configured palette meets the 7:1 text contrast requirement")
        results.sort(key=lambda r: r["distinctiveness_delta_e"], reverse=True)
        return {"selected": results[0], "ranked": results}

    def handle_for(self, name: str) -> str:
        if self.env_handle:
            return self.env_handle if self.env_handle.startswith("@") else f"@{self.env_handle}"
        core_words = [w for w in re.findall(r"[A-Za-z]+", name) if w.lower() not in STOPWORDS]
        return "@" + "".join(w.capitalize() for w in core_words)[:30]

    def tagline(self) -> str:
        return self.env_tagline or "King James Scripture for Sleep  ·  Gentle Rain  ·  Dark Screen"


# ============================================================================ blueprint builders
def build_title_templates(tp: dict[str, Any]) -> list[dict[str, Any]]:
    """Structural templates ordered by which competitor features correlate with views."""
    feats = tp.get("features", {})
    front = tp.get("duration_cue_position") == "front"
    dark = feats.get("black_or_dark_screen", {}).get("prevalence", 0) >= 0.3
    benefit = feats.get("emotional_benefit", {}).get("view_lift", 1.0) >= 1.0
    dark_s = " | Dark Screen" if dark else ""
    hrs = "{hours} Hours"
    t = [
        {"id": "TT1", "pattern": (f"{hrs} " if front else "") + "{theme} | {book_ref} KJV for Sleep with "
                                 "{bed_label}" + ("" if front else f" | {hrs}") + dark_s},
        {"id": "TT2", "pattern": "Fall Asleep in God's Word: {book_ref} (KJV) + {bed_label}" + f" | {hrs}"},
        {"id": "TT3", "pattern": "{theme} — King James Bible Reading Over Rain" + f" ({hrs}){dark_s}"},
        {"id": "TT4", "pattern": "Sleep With Scripture: {book_ref} KJV | {bed_label}" + f" | {hrs} Black Screen"},
        {"id": "TT5", "pattern": "{benefit_hook}: {book_ref} KJV Bible + {bed_label}" + f" ({hrs})"},
        {"id": "TT6", "pattern": f"{hrs} of " + "{book_ref} Read Softly | KJV Scripture & {bed_label}"},
    ]
    if not benefit:
        t = [x for x in t if x["id"] != "TT5"] + [x for x in t if x["id"] == "TT5"]
    return [{**x, "max_chars": 100} for x in t]


def build_description_template(dp: dict[str, Any], funnel_url: str, funnel_label: str) -> dict[str, Any]:
    first_line = int(min(140, max(90, dp.get("first_line_chars_median", 110))))
    sections = [
        {"key": "hook", "rule": f"<= {first_line} chars, restate theme + benefit; visible above 'more'"},
        {"key": "whats_inside", "rule": "2-3 lines: narration hours, ambient bed, dark-screen promise"},
    ]
    if dp.get("share_with_chapters", 0.5) >= 0.25:
        sections.append({"key": "chapters", "rule": "00:00 first; >= 3 entries; >= 10 s apart; one per passage"})
    sections += [
        {"key": "scripture_list", "rule": "Full references read, in order"},
        {"key": "about_channel", "rule": "1-2 lines brand promise + upload cadence"},
    ]
    if funnel_url:
        sections.append({"key": "resource", "rule": f"Single calm line: '{funnel_label}: {funnel_url}'"})
    sections += [
        {"key": "attribution", "rule": "Scripture: King James Version (public domain). Narration & audio "
                                        "design original to this channel."},
        {"key": "hashtags", "rule": "Exactly 3 hashtags on the last line (first 3 show above the title)"},
    ]
    return {"sections": sections, "max_chars": 5000,
            "hashtag_pool": (dp.get("common_hashtags") or [])[:6] + ["#kjvbible", "#sleepscripture",
                                                                     "#rainforsleep"]}


def build_playlists(brand: str) -> list[dict[str, Any]]:
    series_names = {
        "psalms": ("Psalms for Sleep", "The Psalms read slowly over rain, a different section every night."),
        "gospels": ("The Gospels at Night", "Matthew, Mark, Luke and John for peaceful, unhurried sleep."),
        "wisdom": ("Wisdom by Lamplight", "Proverbs, Ecclesiastes and Job, read gently for rest."),
        "promises": ("Promises of Peace", "Isaiah and the promises of God for anxious, restless nights."),
        "letters": ("Letters of Comfort", "The epistles - Romans to 1 John - softly read over rain."),
        "stories": ("Bible Stories for Sleep", "Genesis, Ruth, Jonah, Esther and more, told calmly."),
    }
    pls = [{"key": k, "title": f"{t} | {brand}"[:150], "description": d, "privacy": "public",
            "passage_set_ids": [p["id"] for p in catalog.PASSAGE_SETS if p["series"] == k], "order": i}
           for i, (k, (t, d)) in enumerate(series_names.items())]
    pls.append({"key": "storms", "title": f"Thunderstorm Scripture Nights | {brand}"[:150],
                "description": "Episodes with distant thunder and heavier rain.", "privacy": "public",
                "rule": "ambient_bed in {distant_soft_thunder, heavy_night_storm_far}", "order": len(pls)})
    pls.append({"key": "shorts", "title": f"60-Second Verses for Rest | {brand}"[:150],
                "description": "Short scripture moments from our full-length sleep readings.",
                "privacy": "public", "rule": "all Shorts", "order": len(pls)})
    return pls


def build_channel_description(name: str, cadence_phrase: str, funnel_url: str, funnel_label: str) -> str:
    lines = [
        f"Welcome to {name} - a quiet place to fall asleep in the Word of God.",
        "",
        "Every episode is 8 to 10 hours long: the first hours are slow, gentle King James (KJV) scripture "
        "narration over soft rain, then the voice fades and uninterrupted rain carries you through the night. "
        "Dark, low-glare screens make them safe to leave playing in a bedroom.",
        "",
        f"New full-length sleep readings {cadence_phrase} - Psalms, the Gospels, Proverbs, Isaiah, the "
        "letters of Paul and more, each paired with its own rain, thunder or cabin-wind soundscape.",
        "",
        "Use these for sleep, insomnia, anxiety, prayer, quiet time, or a peaceful night with the Lord.",
    ]
    if funnel_url:
        lines += ["", f"{funnel_label}: {funnel_url}"]
    lines += ["", "Scripture: King James Version (public domain). Narration and soundscapes are original."]
    text = "\n".join(lines)
    return text[:1000]


def cadence_from(cp: dict[str, Any]) -> dict[str, Any]:
    best = cp.get("upload_gap_days_best_quartile") or 1.0
    median = cp.get("upload_gap_days_median") or 2.0
    # Start slower than competitors; ramp to the top-quartile pace as uniqueness pools allow.
    start_gap = max(1.0, round(median, 1))
    target_gap = max(1.0, round(best, 1))
    phrase = "every day" if target_gap <= 1.0 else f"every {int(round(target_gap))} days"
    return {
        "competitor_median_gap_days": median, "competitor_top_quartile_gap_days": best,
        "ramp": [
            {"weeks": "1-2", "long_form_per_week": max(2, round(7 / max(start_gap, 2.0))), "shorts_per_long": 3},
            {"weeks": "3-6", "long_form_per_week": max(3, round(7 / max(start_gap, 1.5))), "shorts_per_long": 3},
            {"weeks": "7+", "long_form_per_week": min(7, max(3, round(7 / target_gap))), "shorts_per_long": 3},
        ],
        "max_long_form_per_day": 1,
        "public_phrase": phrase,
    }


def build_content_blueprint(analysis: dict[str, Any], brand: dict[str, Any], content_cfg: dict[str, Any],
                            funnel_url: str, funnel_label: str) -> dict[str, Any]:
    tp, dp, vp = analysis["titles"], analysis["descriptions"], analysis["visuals"]
    thumbs = vp.get("thumbnails", {})
    dur = content_cfg.get("duration_s", {"min": 28800, "max": 36000})
    narr = content_cfg.get("narration_window_s", {"start": 0, "end": 9000})
    loud = content_cfg.get("loudness", {})
    return {
        "version": BLUEPRINT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": analysis["data_source"],
        "brand_ref": {"name": brand["name"], "palette": brand["palette"]},
        "episode_spec": {
            "duration_s": {"min": int(dur["min"]), "max": int(dur["max"])},
            "youtube_limits": {"max_duration_s": 43200, "max_bytes": 274877906944},
            "narration": {
                "window_s": {"start": int(narr["start"]), "end": int(narr["end"])},
                "crossfade_out_s": int(content_cfg.get("narration_crossfade_s", 90)),
                "voice": {"tempo": 0.88, "register": "deep, warm, unhurried",
                          "pause_after_verse_s": [1.2, 2.0], "pause_after_chapter_s": 6.0,
                          "intro_silence_s": 8.0},
                "processing": {"highpass_hz": 70, "lowpass_hz": 7500, "warmth_shelf_db": 2.0,
                               "deesser": True, "compressor_ratio": 2.5,
                               "loudnorm_lufs": float(loud.get("narration_lufs", -20)),
                               "true_peak_db": float(loud.get("true_peak_db", -2.0))},
                "ducking": {"method": "sidechaincompress", "ambient_reduction_db": 9, "attack_ms": 120,
                            "release_ms": 1600},
            },
            "ambient": {"loudnorm_lufs": float(loud.get("ambient_lufs", -26)), "loop_seam": "equal-power "
                        "crossfade >= 8 s between non-identical segments; no audible period < 20 min",
                        "warm_lowpass_hz": [3800, 5200]},
            "video": {"resolution": [1920, 1080], "fps": [15, 24], "max_mean_luma": 32,
                      "encoders_priority": ["h264_videotoolbox", "h264_nvenc", "libx264"],
                      "gop_seconds": 4, "pix_fmt": "yuv420p", "tune": "stillimage/grain-friendly"},
        },
        "rotation_pools": {
            "passage_sets": catalog.PASSAGE_SETS,
            "ambient_beds": catalog.AMBIENT_BEDS, "ambient_jitter": catalog.AMBIENT_JITTER,
            "visual_scenes": catalog.VISUAL_SCENES, "visual_jitter": catalog.VISUAL_JITTER,
            "thumbnail_templates": catalog.THUMBNAIL_TEMPLATES,
            "title_templates": build_title_templates(tp),
            "benefit_hooks": ["Peaceful Sleep Tonight", "When Anxiety Keeps You Awake", "Rest for the Weary",
                              "Comfort in the Night", "Fear Not", "Quiet Your Mind", "Sleep in His Peace"],
        },
        "anti_repetition_rules": {
            "min_episodes_between": {
                "book": int(content_cfg.get("min_episodes_between_book_reuse", 6)),
                "visual_scene": int(content_cfg.get("min_episodes_between_scene_reuse", 3)),
                "ambient_bed": int(content_cfg.get("min_episodes_between_bed_reuse", 2)),
                "title_template": int(content_cfg.get("min_episodes_between_title_template_reuse", 4)),
                "thumbnail_template": int(content_cfg.get("min_episodes_between_thumb_template_reuse", 3)),
            },
            "passage_set_reuse": "never until every set has been used once; afterwards only with a different "
                                 "bed AND scene AND narrator take",
            "unique_forever": ["combo(passage_set, ambient_bed, visual_scene, thumbnail_template)",
                               "title_exact", "ambient_mix_params", "visual_loop_params", "audio_master_sha256",
                               "thumbnail_dhash"],
            "fingerprint_kinds": {
                "combo": "sha256(passage_set_id, bed_id, scene_id, thumb_template_id)",
                "ambient_mix": "sha256(bed_id, jitter params rounded to 2 dp, seed)",
                "visual_loop": "sha256(scene_id, jitter params rounded to 2 dp, seed)",
                "title": "sha256(normalised lowercase title)",
                "thumbnail": "dhash64 of final thumbnail; reject if Hamming <= 10 vs any prior",
                "audio_master": "sha256 of first+last 64 MB and file size",
            },
            "title_similarity_max": 0.85,
            "thumbnail_dhash_min_distance": 10,
            "competitor_thumb_dhash_min_distance": 14,
            "competitor_thumb_dhashes": vp.get("competitor_thumb_dhashes", [])[:200],
        },
        "title_rules": {
            "length_chars_target": [int(tp.get("length_chars", {}).get("p25", 64)),
                                    min(100, int(tp.get("length_chars", {}).get("p75", 92)))],
            "duration_cue_position": tp.get("duration_cue_position", "end"),
            "must_include_one_of": ["KJV", "King James"], "must_include": ["hours", "Sleep|sleep"],
            "competitor_feature_stats": tp.get("features", {}),
            "genre_vocabulary": [p["phrase"] for p in tp.get("top_view_weighted_phrases", [])][:15],
        },
        "description": build_description_template(dp, funnel_url, funnel_label),
        "chapters_template": [
            {"at": "start", "label": "Opening & Prayer of Rest"},
            {"at": "each_passage", "label": "{book} {chapter_range}"},
            {"at": "narration_end", "label": "Narration Ends - Rain Only for Deep Sleep"},
            {"at": "hourly_after_narration", "label": "Hour {n} - {bed_label}"},
        ],
        "thumbnail_rules": {
            "size": [1280, 720], "max_bytes": 2_000_000, "format": "jpg",
            "target_mean_luma": [max(18, thumbs.get("mean_luma", 38) - 8), thumbs.get("mean_luma", 38) + 6],
            "min_text_contrast": 7.0, "max_words": 6, "text_column_competitors": thumbs.get("text_column_mode"),
            "variants_per_episode": 2,
        },
        "shorts": {"per_long_form": 3, "duration_s": [45, 59], "aspect": "9:16", "resolution": [1080, 1920],
                   "content": "one complete verse/short passage from that episode, burned-in serif captions",
                   "cta": "Full 8-hour reading linked in related video"},
        "pinned_comment": "One gentle line inviting viewers to share a prayer request + full scripture list.",
    }


# ============================================================================ agent
class CompetitorIntelAgent(BaseAgent):
    name = "agent1_competitor_intel"

    def __init__(self, *args: Any, offline: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.offline = offline

    def execute(self) -> AgentResult:
        s = self.settings
        warnings: list[str] = []
        ds = CompetitorDataset()
        if not self.offline:
            self.status("Collecting competitors (8h+ KJV rain sleep)...")
            try:
                ds = collect_competitors(self.db, s.research, api_key=s.youtube_api_key or None,
                                         progress=self.status)
            except Exception as exc:  # noqa: BLE001 - network may be unavailable; continue on priors
                warnings.append(f"Competitor collection failed ({type(exc).__name__}: {exc}); using priors")
            warnings += ds.warnings
            if not ds.videos:
                warnings.append("No competitor data collected (network blocked, no API key results, or "
                                "YouTube changed its pages). Blueprint uses BASELINE PRIORS - rerun with "
                                "network access before relying on it.")

        if ds.channels or ds.videos:
            self.status(f"Persisting {len(ds.channels)} channels / {len(ds.videos)} videos")

        self.status("Analysing structural patterns...")
        analysis = self._analyse(ds)
        # Persist after visual analysis so thumbnail metrics are stored too
        for ch in ds.channels:
            self.db.upsert_competitor_channel(ch)
        for v in ds.videos:
            self.db.upsert_competitor_video(v)

        designer = BrandDesigner(
            s.brand, competitor_titles=[c["title"] for c in ds.channels],
            competitor_colors=analysis["visuals"].get("competitor_colors", []),
            env_name=s.brand_name, env_handle=s.brand_handle, env_tagline=s.brand_tagline,
        )
        name_info = designer.choose_name()
        if name_info["warning"]:
            warnings.append(name_info["warning"])
        palette_info = designer.choose_palette()
        pal = palette_info["selected"]
        palette = {k: pal[k] for k in ("bg", "mid", "accent", "text")}
        name = name_info["name"]
        cadence = cadence_from(analysis["channels"])
        tagline = designer.tagline()
        brand = {"name": name, "handle": designer.handle_for(name), "tagline": tagline, "palette": palette}

        kit = s.brand_kit_dir
        kit.mkdir(parents=True, exist_ok=True)
        seed = int(sha256_fingerprint(name, pal["id"])[:8], 16)
        self.status("Rendering banner / avatar / watermark...")
        banner = render_banner(kit / "banner_2560x1440.png", name=name, tagline=tagline,
                               cadence_line=f"New 8-10 hour sleep readings {cadence['public_phrase']}",
                               palette=palette, seed=seed)
        monogram = "".join(w[0] for w in re.findall(r"[A-Za-z]+", name) if w.lower() not in STOPWORDS)[:3].upper()
        avatar = render_avatar(kit / "avatar_800x800.png", monogram=monogram, palette=palette, seed=seed + 1)
        watermark = render_watermark(kit / "watermark_150x150.png", palette=palette)

        about = build_channel_description(name, cadence["public_phrase"], s.funnel_url, s.funnel_label)
        (kit / "channel_description.txt").write_text(about, encoding="utf-8")
        playlists = build_playlists(name)
        _write_json(kit / "playlist_plan.json", playlists)
        keywords = _channel_keywords(analysis)
        (kit / "channel_keywords.txt").write_text(keywords, encoding="utf-8")

        channel_bp = {
            "version": BLUEPRINT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_source": analysis["data_source"],
            "competitor_sample": {"channels": len(ds.channels), "videos": len(ds.videos),
                                  "sources": sorted(ds.sources)},
            "brand": {**brand, "name_check": name_info, "palette_check": palette_info,
                      "monogram": monogram},
            "assets": {"banner": banner, "avatar": avatar, "watermark": watermark,
                       "description_file": str(kit / "channel_description.txt"),
                       "playlist_plan_file": str(kit / "playlist_plan.json"),
                       "keywords_file": str(kit / "channel_keywords.txt")},
            "channel_page_layout": {
                "visual_hierarchy": {
                    "banner": "brand name (serif, tracked caps) > flame rule > tagline > cadence line, all in "
                              "1546x423 safe area; dark rain-on-glass field "
                              f"(competitor banner median luma {analysis['visuals']['banners'].get('mean_luma')})",
                    "avatar": "single flame emblem + monogram; reads at 98 px",
                },
                "home_tab_sections": [
                    {"type": "channel_trailer_for_non_subscribers",
                     "spec": "60-90 s: what the channel is, 3 rain beds previewed, voice sample"},
                    {"type": "featured_video_for_returning_subscribers", "spec": "latest long-form episode"},
                    {"type": "playlist", "playlist_key": "psalms"},
                    {"type": "playlist", "playlist_key": "gospels"},
                    {"type": "popular_uploads"},
                    {"type": "playlist", "playlist_key": "promises"},
                    {"type": "shorts", "playlist_key": "shorts"},
                    {"type": "playlist", "playlist_key": "storms"},
                ],
                "about": {"description": about, "chars": len(about), "keywords": keywords,
                          "links": [{"title": s.funnel_label, "url": s.funnel_url}] if s.funnel_url else []},
                "series_naming": "{Series} · Night {n}: {theme}  (e.g. 'Psalms for Sleep · Night 3')",
                "upload_cadence": cadence,
                "defaults": {"category_id": "22", "default_language": "en", "made_for_kids": False,
                             "license": "youtube", "embeddable": True, "comments": "on, held for review"},
            },
            "playlists": playlists,
            "competitor_patterns": {
                "channels": analysis["channels"], "titles": analysis["titles"],
                "descriptions": analysis["descriptions"],
                "visuals": {k: v for k, v in analysis["visuals"].items() if k != "competitor_thumb_dhashes"},
            },
            "studio_only_growth_settings": STUDIO_ONLY_CHECKLIST,
            "manual_setup_steps": [
                "YouTube Studio > Customization > Branding: upload banner, avatar and watermark from this kit.",
                "Customization > Basic info: paste channel_description.txt; set handle "
                f"{brand['handle']} if available.",
                "Settings > Channel > Basic info: paste channel_keywords.txt.",
                "Verify the channel (phone) at youtube.com/verify to unlock >15 min uploads & custom thumbnails.",
                "Playlists are created automatically by Agent 4 from playlist_plan.json on first live run.",
                "Banner, About text, keywords and watermark can be applied via API: python main.py --apply-branding",
            ],
            "warnings": warnings,
        }
        content_bp = build_content_blueprint(analysis, brand, s.content, s.funnel_url, s.funnel_label)

        prev = s.content_blueprint_path
        if prev.exists():  # keep Agent 5's learned weights across research refreshes
            weights = json.loads(prev.read_text(encoding="utf-8")).get("performance_weights")
            if weights:
                content_bp["performance_weights"] = weights
        _write_json(s.channel_blueprint_path, channel_bp)
        _write_json(s.content_blueprint_path, content_bp)
        _write_json(kit / "channel_page_blueprint.json", channel_bp)
        self.db.kv_set("brand", brand)
        self.db.kv_set("blueprint_version", BLUEPRINT_VERSION)

        return AgentResult(self.name, True, summary={
            "data_source": analysis["data_source"], "channels": len(ds.channels), "videos": len(ds.videos),
            "brand": name, "handle": brand["handle"], "palette": pal["id"],
            "palette_distinctiveness": pal["distinctiveness_delta_e"],
            "name_similarity": name_info["max_similarity_to_competitor"], "brand_kit": str(kit),
        }, warnings=warnings)

    def _analyse(self, ds: CompetitorDataset) -> dict[str, Any]:
        pa = PatternAnalyzer(ds, timeout=int(self.settings.research.get("request_timeout_s", 20)),
                             max_images=int(self.settings.research.get("thumbnails_to_analyze", 60)))
        live = bool(ds.videos)
        titles = pa.title_patterns() if live else {}
        descs = pa.description_patterns() if live else {}
        visuals = pa.visual_patterns() if live and not self.offline else {}
        chans = pa.channel_patterns() if ds.channels else {}
        merged = {
            "titles": titles or BASELINE_PRIORS["titles"],
            "descriptions": descs or BASELINE_PRIORS["descriptions"],
            "visuals": visuals if visuals.get("thumbnails", {}).get("sample_size") else BASELINE_PRIORS["visuals"],
            "channels": chans or BASELINE_PRIORS["channels"],
        }
        used_priors = [k for k, v in (("titles", titles), ("descriptions", descs), ("channels", chans)) if not v]
        if not visuals.get("thumbnails", {}).get("sample_size"):
            used_priors.append("visuals")
        merged["data_source"] = ("live" if not used_priors else
                                 "baseline_prior" if len(used_priors) == 4 else f"live+prior({','.join(used_priors)})")
        return merged


def _channel_keywords(analysis: dict[str, Any]) -> str:
    base = ["KJV bible sleep", "bible verses for sleep", "rain sounds for sleep", "scripture for sleep",
            "king james bible reading", "8 hours", "dark screen", "psalms for sleep", "sleep with gods word",
            "christian sleep", "thunderstorm"]
    vocab = [p["phrase"] for p in analysis["titles"].get("top_view_weighted_phrases", [])][:10]
    seen: list[str] = []
    for k in [*base, *vocab]:
        if k not in seen:
            seen.append(k)
    out = " ".join(f'"{k}"' if " " in k else k for k in seen)
    return out[:500]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _json_default(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def iter_unused(values: Iterable[str], recent: list[str], gap: int) -> list[str]:
    """Helper for later agents: candidates not used within the last `gap` episodes."""
    blocked = set(recent[:gap])
    return [v for v in values if v not in blocked]
