"""Titles, descriptions, tags and comments built from the blueprint templates, with uniqueness checks."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from core.config import Settings
from core.ffmpeg import fmt_ts
from core.planner import EpisodePlan
from db.database import Database, sha256_fingerprint

MAX_TITLE = 100
MAX_DESC = 5000


def sanitize(text: str) -> str:
    """YouTube rejects '<' and '>' in titles/descriptions."""
    return re.sub(r"[ \t]+", " ", text.replace("<", "‹").replace(">", "›")).strip()


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def _shorten(title: str) -> str:
    """Shrink an over-long title by dropping its least important ' | ' segments."""
    t = title
    for pat in (r" \| Dark Screen", r" Black Screen", r" Read Softly"):
        if len(t) <= MAX_TITLE:
            break
        t = re.sub(pat, "", t, count=1)
    while len(t) > MAX_TITLE and " | " in t:
        t = t.rsplit(" | ", 1)[0]
    return t[:MAX_TITLE].rstrip(" |-—")


def render_title(plan: EpisodePlan, bp: dict[str, Any], db: Database, *, exclude: set[str] | None = None,
                 template_id: str | None = None) -> tuple[str, str]:
    """-> (title, template_id). Unique (exact) and not too similar to the last 50 titles."""
    templates = bp["rotation_pools"]["title_templates"]
    hooks = [plan.benefit_hook] + [h for h in bp["rotation_pools"].get("benefit_hooks", []) if h != plan.benefit_hook]
    first = template_id or plan.title_template["id"]
    ordered = sorted(templates, key=lambda t: t["id"] != first)
    max_sim = float(bp["anti_repetition_rules"].get("title_similarity_max", 0.85))
    recent = [norm_title(t) for t in db.recent_titles(50)] + [norm_title(t) for t in (exclude or set())]
    for tpl in ordered:
        for hook in hooks:
            fields = {"hours": plan.hours, "theme": plan.passage_set["theme"], "book_ref": plan.book_ref,
                      "bed_label": plan.bed["label"], "benefit_hook": hook}
            title = _shorten(sanitize(tpl["pattern"].format(**fields)))
            n = norm_title(title)
            if not db.is_unique("title", sha256_fingerprint(n)):
                continue
            if any(SequenceMatcher(None, n, r).ratio() > max_sim for r in recent):
                continue
            return title, tpl["id"]
    raise ValueError("No unique title available - add title templates or benefit hooks to the blueprint")


def build_description(plan: EpisodePlan, audio: dict[str, Any], brand: dict[str, Any], bp: dict[str, Any],
                      settings: Settings) -> str:
    cadence = brand.get("cadence_phrase", "regularly")
    chapters = audio.get("chapters", [])
    hashtags = _hashtags(bp)
    hook = sanitize(f"{plan.passage_set['theme']} - {plan.hours} hours of King James scripture and "
                    f"{plan.bed['label'].lower()} for deep, peaceful sleep.")
    narr_s = float(audio["narration_end_s"])
    narr_txt = f"{narr_s / 3600:.1f} hours" if narr_s >= 3600 else f"{max(1, round(narr_s / 60))} minutes"
    blocks: dict[str, str] = {
        "hook": hook,
        "whats_inside": (f"The first {narr_txt} are a slow, gentle reading of {audio['passage_display']} "
                         f"(KJV) over soft {plan.bed['label'].lower()}. Then the voice fades and the rain continues, "
                         f"uninterrupted, for the rest of the night. Dark, low-glare screen - safe to leave playing."),
        "chapters": ("Chapters\n" + "\n".join(f"{fmt_ts(c['start_s'])} {c['label']}" for c in chapters))
        if len(chapters) >= 3 else "",
        "scripture_list": "Scripture read tonight: " + ", ".join(audio.get("chapters_read", [])),
        "about_channel": f"{brand.get('name', '')} - new 8-10 hour sleep readings {cadence}. Subscribe to fall "
                         "asleep in God's Word every night.",
        "resource": f"{settings.funnel_label}: {settings.funnel_url}" if settings.funnel_url else "",
        "attribution": "Scripture: King James Version (public domain). Narration and soundscapes are original to "
                       "this channel." + (" Narration voice is AI-generated." if settings.disclose_synthetic_media
                                          else ""),
        "hashtags": " ".join(hashtags),
    }
    order = [s["key"] for s in bp["description"]["sections"]]
    text = "\n\n".join(blocks[k] for k in order if blocks.get(k))
    if len(text) > MAX_DESC and len(chapters) >= 3:  # keep YouTube's 5,000 char limit: thin out chapter list
        keep = chapters[::2] if len(chapters) > 6 else chapters
        if keep[0]["start_s"] != 0:
            keep = [chapters[0], *keep]
        blocks["chapters"] = "Chapters\n" + "\n".join(f"{fmt_ts(c['start_s'])} {c['label']}" for c in keep)
        text = "\n\n".join(blocks[k] for k in order if blocks.get(k))
    return sanitize_multiline(text)[:MAX_DESC]


def sanitize_multiline(text: str) -> str:
    return "\n".join(sanitize(line) for line in text.splitlines())


def _hashtags(bp: dict[str, Any]) -> list[str]:
    pool = [h if h.startswith("#") else f"#{h}" for h in bp["description"].get("hashtag_pool", [])]
    seen: list[str] = []
    for h in pool:
        h = re.sub(r"[^#\w]", "", h)
        if h.lower() not in {s.lower() for s in seen}:
            seen.append(h)
    return (seen or ["#kjv", "#biblesleep", "#rainsounds"])[:3]


def build_tags(plan: EpisodePlan, extra: list[str] | None = None) -> list[str]:
    books = list(dict.fromkeys(s[0] for s in plan.passage_set["segments"]))
    tags = ["kjv bible sleep", "bible for sleep", "scripture for sleep", "rain sounds for sleep",
            f"{plan.hours} hours", "king james bible", "dark screen", plan.bed["label"].lower(),
            plan.passage_set["theme"].lower(), *[f"{b.lower()} kjv" for b in books], "sleep with gods word",
            "christian sleep", *(extra or [])]
    out: list[str] = []
    total = 0
    for t in dict.fromkeys(tags):
        cost = len(t) + (2 if " " in t else 0) + 1
        if total + cost > 480:
            break
        out.append(t)
        total += cost
    return out


def build_comment(audio: dict[str, Any], settings: Settings) -> str:
    lines = [f"Scripture tonight: {audio['passage_display']} (KJV), verses {audio['first_verse']} to "
             f"{audio['last_verse']}.",
             "If you have a prayer request, share it below. Sleep well in His peace."]
    if settings.funnel_url:
        lines.append(f"{settings.funnel_label}: {settings.funnel_url}")
    return sanitize_multiline("\n".join(lines))


def video_body(title: str, description: str, tags: list[str], settings: Settings, *, privacy: str | None = None,
               publish_at: str | None = None) -> dict[str, Any]:
    pub = settings.publishing
    status: dict[str, Any] = {
        "privacyStatus": "private" if publish_at else (privacy or settings.default_privacy),
        "selfDeclaredMadeForKids": False, "embeddable": True, "license": "youtube",
        "publicStatsViewable": True, "containsSyntheticMedia": settings.disclose_synthetic_media,
    }
    if publish_at:
        status["publishAt"] = publish_at
    return {
        "snippet": {"title": title, "description": description, "tags": tags,
                    "categoryId": str(pub.get("category_id", "22")),
                    "defaultLanguage": pub.get("default_language", "en"),
                    "defaultAudioLanguage": pub.get("default_language", "en")},
        "status": status,
    }
