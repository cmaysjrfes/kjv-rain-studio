"""Vertical 9:16 Shorts cut from each long-form release: one complete verse group + burned-in captions."""
from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from core.ffmpeg import encoder_args, run_ffmpeg
from core.imaging import hex_to_rgb, load_font

W, H = 1080, 1920
QUOTABLE = re.compile(r"\b(peace|rest|sleep|fear not|light|shepherd|comfort|strength|love|still|refuge|"
                      r"trust|heart|troubled|lord is|wings|mercy|joy|hope|faith)\b", re.IGNORECASE)


def select_windows(timeline: list[dict[str, Any]], n: int = 3, *, min_s: float = 18.0,
                   max_s: float = 52.0, max_words: int = 45) -> list[dict[str, Any]]:
    """Pick up to `n` non-overlapping runs of consecutive verses (same chapter) lasting min_s..max_s
    and at most `max_words` words, so the burned-in caption stays readable on a phone."""
    verses = [t for t in timeline if t["kind"] == "verse"]
    cands: list[dict[str, Any]] = []
    for i in range(len(verses)):
        group = []
        for v in verses[i:]:
            if group and v["chapter_key"] != group[0]["chapter_key"]:
                break
            group.append(v)
            span = group[-1]["end"] - group[0]["start"]
            words = sum(len(g["display"].split()) for g in group)
            if span > max_s or (words > max_words and len(group) > 1):
                group.pop()
                break
        if not group:
            continue
        span = group[-1]["end"] - group[0]["start"]
        if span < min_s:
            continue
        text = " ".join(g["display"] for g in group)
        score = len(QUOTABLE.findall(text)) * 3 - abs(span - 38) / 6 - len(group) * 0.3
        cands.append({"start": group[0]["start"], "end": group[-1]["end"], "verses": group, "score": score,
                      "chapter": group[0]["chapter_key"]})
    cands.sort(key=lambda c: c["score"], reverse=True)
    chosen: list[dict[str, Any]] = []
    for c in cands:
        if any(not (c["end"] < o["start"] or c["start"] > o["end"]) for o in chosen):
            continue
        if any(o["chapter"] == c["chapter"] for o in chosen) and len({x["chapter"] for x in cands}) > len(chosen):
            continue
        chosen.append(c)
        if len(chosen) == n:
            break
    for c in chosen:
        first, last = c["verses"][0]["ref"], c["verses"][-1]["ref"]
        c["ref"] = first if first == last else f"{first}-{last.split(':')[-1]}"
        c["text"] = " ".join(v["display"] for v in c["verses"])
    return sorted(chosen, key=lambda c: c["start"])


def caption_png(window: dict[str, Any], brand: dict[str, Any], out: Path) -> Path:
    pal = brand["palette"]
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    plate = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(plate).rounded_rectangle([70, 520, W - 70, 1420], 48, fill=(*hex_to_rgb(pal["bg"]), 175))
    img.alpha_composite(plate.filter(ImageFilter.GaussianBlur(26)))
    d = ImageDraw.Draw(img)
    text = window["text"]
    for size in range(74, 40, -4):
        font = load_font("serif", size)
        lines = textwrap.wrap(text, width=max(12, int(900 / (size * 0.5))))
        if len(lines) * size * 1.3 <= 760 and all(d.textlength(line, font=font) <= 900 for line in lines):
            break
    y = 960 - len(lines) * size * 1.3 / 2
    for line in lines:
        d.text(((W - d.textlength(line, font=font)) / 2, y), line, font=font, fill=pal["text"])
        y += size * 1.3
    rf = load_font("sans", 44)
    ref = f"{window['ref']}  (KJV)"
    d.text(((W - d.textlength(ref, font=rf)) / 2, y + 36), ref, font=rf, fill=pal["accent"])
    bf = load_font("sans", 34)
    name = brand.get("name", "")
    d.text(((W - d.textlength(name, font=bf)) / 2, 1640), name, font=bf, fill=(*hex_to_rgb(pal["text"]), 170))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def render_short(window: dict[str, Any], master: Path, shorts_loop: Path, out: Path, brand: dict[str, Any],
                 encoder: str, fps: int) -> dict[str, Any]:
    start = max(0.0, window["start"] - 1.0)
    dur = min(59.0, window["end"] - window["start"] + 3.0)
    cap = caption_png(window, brand, out.with_suffix(".caption.png"))
    graph = (f"[0:v]trim=0:{dur:.3f},setpts=PTS-STARTPTS[bg];[bg][2:v]overlay=0:0:format=auto,"
             f"fade=t=in:st=0:d=0.6,fade=t=out:st={dur - 0.8:.3f}:d=0.8,format=yuv420p[v];"
             f"[1:a]afade=t=in:d=0.5,afade=t=out:st={dur - 1.2:.3f}:d=1.2,volume=4dB,alimiter=limit=0.89:level=0[a]")
    run_ffmpeg(["-stream_loop", "-1", "-i", str(shorts_loop), "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                "-i", str(master), "-loop", "1", "-i", str(cap), "-filter_complex", graph, "-map", "[v]",
                "-map", "[a]", *encoder_args(encoder, fps, quality="short"), "-r", str(fps), "-c:a", "aac",
                "-b:a", "160k", "-t", f"{dur:.3f}", "-movflags", "+faststart", str(out)], duration_s=dur)
    cap.unlink(missing_ok=True)
    return {"path": str(out), "ref": window["ref"], "duration_s": round(dur, 2), "text": window["text"],
            "source_start_s": round(start, 2)}
