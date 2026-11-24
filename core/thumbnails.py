"""A/B thumbnail rendering (1280x720 JPEG < 2 MB) on top of the episode's own scene still."""
from __future__ import annotations

import io
import random
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from core.imaging import contrast_ratio, dhash64, draw_tracked, fit_font, hex_to_rgb, load_font

W, H = 1280, 720
MAX_BYTES = 2_000_000


@dataclass
class ThumbContext:
    headline: str          # <= 6 words, e.g. the passage theme
    book_ref: str          # "Psalms 42–58"
    hours: int
    bed_label: str
    verse_quote: str       # short KJV verse for the verse-card template
    verse_ref: str
    palette: dict[str, str]
    seed: int


def _plate(img: Image.Image, box: tuple[float, float, float, float], color: str, alpha: int,
           radius: int = 28, blur: int = 18) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle(box, radius, fill=(*hex_to_rgb(color), alpha))
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)) if blur else layer)


def _badge(img: Image.Image, ctx: ThumbContext, xy: tuple[int, int], anchor_right: bool = False) -> None:
    d = ImageDraw.Draw(img)
    text = f"{ctx.hours} HOURS"
    font = load_font("sans", 46)
    w = d.textlength(text, font=font) + 44
    x, y = xy
    if anchor_right:
        x -= int(w)
    d.rounded_rectangle([x, y, x + w, y + 68], 14, fill=(*hex_to_rgb(ctx.palette["accent"]), 235))
    d.text((x + 22, y + 9), text, font=font, fill=ctx.palette["bg"])


def _wrap_fit(d: ImageDraw.ImageDraw, text: str, kind: str, max_w: int, max_h: int, start: int,
              max_lines: int = 3) -> tuple[list[str], Any]:
    for size in range(start, 24, -4):
        font = load_font(kind, size)
        for width in range(8, 40):
            lines = textwrap.wrap(text, width=width)
            if len(lines) > max_lines:
                continue
            widest = max(d.textlength(line, font=font) for line in lines)
            if widest <= max_w and len(lines) * size * 1.15 <= max_h:
                return lines, font
    return textwrap.wrap(text, 18)[:max_lines], load_font(kind, 24)


def _lines(d: ImageDraw.ImageDraw, lines: list[str], font: Any, x: int, y: int, fill: str,
           align: str = "left", width: int = 0, spacing: float = 1.15) -> int:
    size = getattr(font, "size", 32)
    for line in lines:
        lw = d.textlength(line, font=font)
        lx = x if align == "left" else int(x + (width - lw) / 2) if align == "center" else int(x + width - lw)
        d.text((lx, y), line, font=font, fill=fill)
        y += int(size * spacing)
    return y


# ------------------------------------------------------------------ templates
def t1_split(img: Image.Image, ctx: ThumbContext) -> None:
    _plate(img, (W * 0.50, 60, W - 40, H - 60), ctx.palette["bg"], 205)
    d = ImageDraw.Draw(img)
    lines, font = _wrap_fit(d, ctx.headline.upper(), "serif", int(W * 0.42), 300, 92)
    y = _lines(d, lines, font, int(W * 0.54), 190, ctx.palette["text"])
    d.line([(W * 0.54, y + 14), (W * 0.54 + 160, y + 14)], fill=ctx.palette["accent"], width=5)
    sub = f"{ctx.book_ref} · KJV"
    sub_lines, sub_font = _wrap_fit(d, sub, "sans", int(W * 0.42), 110, 40, max_lines=2)
    _lines(d, sub_lines, sub_font, int(W * 0.54), y + 36, ctx.palette["accent"])
    _badge(img, ctx, (W - 60, 90), anchor_right=True)


def t2_center_band(img: Image.Image, ctx: ThumbContext) -> None:
    _plate(img, (0, H * 0.30, W, H * 0.72), ctx.palette["bg"], 200, radius=0, blur=24)
    d = ImageDraw.Draw(img)
    lines, font = _wrap_fit(d, ctx.headline.upper(), "serif", int(W * 0.86), 210, 96, max_lines=2)
    y = _lines(d, lines, font, 0, int(H * 0.34), ctx.palette["text"], "center", W)
    sub = f"{ctx.book_ref.upper()} • KING JAMES"
    draw_tracked(d, (W // 2, y + 34), sub, fit_font(d, sub, "sans", int(W * 0.86), 40, start=34, tracking=4),
                 ctx.palette["accent"], tracking=4)
    _badge(img, ctx, (48, 48))


def t3_big_numeral(img: Image.Image, ctx: ThumbContext) -> None:
    d = ImageDraw.Draw(img)
    _plate(img, (30, 40, W * 0.40, H - 40), ctx.palette["bg"], 170)
    num = str(ctx.hours)
    num_font = fit_font(d, num, "serif", int(W * 0.33), 400, start=440)
    l, t, r, b = d.textbbox((0, 0), num, font=num_font)
    x0 = int(30 + (W * 0.40 - 30 - (r - l)) / 2)
    d.text((x0 - l, 90 - t), num, font=num_font, fill=ctx.palette["accent"])
    hours_font = load_font("sans", 64)
    hw = d.textlength("HOURS", font=hours_font)
    d.text((int(30 + (W * 0.40 - 30 - hw) / 2), 90 + (b - t) + 30), "HOURS", font=hours_font,
           fill=ctx.palette["text"])
    _plate(img, (W * 0.44, H * 0.22, W - 40, H * 0.78), ctx.palette["bg"], 190)
    lines, font = _wrap_fit(d, ctx.book_ref, "serif", int(W * 0.48), 220, 96, max_lines=2)
    y = _lines(d, lines, font, int(W * 0.48), int(H * 0.28), ctx.palette["text"])
    d.text((W * 0.48, y + 20), "KJV • RAIN • SLEEP", font=load_font("sans", 40), fill=ctx.palette["accent"])


def t4_verse_card(img: Image.Image, ctx: ThumbContext) -> None:
    _plate(img, (W * 0.10, H * 0.12, W * 0.90, H * 0.88), ctx.palette["bg"], 215, radius=36)
    d = ImageDraw.Draw(img)
    quote = f"“{ctx.verse_quote}”"
    lines, font = _wrap_fit(d, quote, "serif", int(W * 0.70), 330, 72, max_lines=4)
    y = _lines(d, lines, font, int(W * 0.15), int(H * 0.20), ctx.palette["text"], "center", int(W * 0.70))
    draw_tracked(d, (W // 2, y + 40), ctx.verse_ref.upper(), load_font("sans", 34), ctx.palette["accent"],
                 tracking=5)
    draw_tracked(d, (W // 2, int(H * 0.80)), f"{ctx.hours} HOURS • KJV • {ctx.bed_label.upper()}",
                 load_font("sans", 28), ctx.palette["text"], tracking=3)


def t5_minimal_book(img: Image.Image, ctx: ThumbContext) -> None:
    _plate(img, (0, H * 0.25, W, H * 0.80), ctx.palette["bg"], 150, radius=0, blur=40)
    d = ImageDraw.Draw(img)
    book = ctx.book_ref.split(" ")[0] if "&" not in ctx.book_ref else ctx.book_ref.split(",")[0]
    font = fit_font(d, book.upper(), "serif", int(W * 0.84), 210, start=210, tracking=16)
    draw_tracked(d, (W // 2, int(H * 0.44)), book.upper(), font, ctx.palette["text"], tracking=16)
    d.line([(W / 2 - 200, H * 0.60), (W / 2 + 200, H * 0.60)], fill=ctx.palette["accent"], width=4)
    draw_tracked(d, (W // 2, int(H * 0.69)), f"KJV • RAIN • SLEEP • {ctx.hours} HOURS", load_font("sans", 38),
                 ctx.palette["accent"], tracking=6)


def t6_lower_third(img: Image.Image, ctx: ThumbContext) -> None:
    _plate(img, (0, H * 0.58, W, H), ctx.palette["bg"], 225, radius=0, blur=30)
    d = ImageDraw.Draw(img)
    lines, font = _wrap_fit(d, ctx.headline, "serif", int(W * 0.80), 150, 86, max_lines=2)
    y = _lines(d, lines, font, 64, int(H * 0.62), ctx.palette["text"])
    d.line([(64, y + 10), (300, y + 10)], fill=ctx.palette["accent"], width=5)
    sub = f"{ctx.book_ref} · King James Bible · {ctx.bed_label}"
    d.text((64, y + 26), sub, font=fit_font(d, sub, "sans", W - 128, 40, start=34), fill=ctx.palette["accent"])
    _badge(img, ctx, (W - 48, 48), anchor_right=True)


TEMPLATES: dict[str, Callable[[Image.Image, ThumbContext], None]] = {
    "T1_split_scene_text": t1_split, "T2_center_title_band": t2_center_band,
    "T3_big_hours_numeral": t3_big_numeral, "T4_verse_card": t4_verse_card,
    "T5_minimal_book": t5_minimal_book, "T6_lower_third": t6_lower_third,
}


def render_thumbnail(background: Image.Image, template_id: str, ctx: ThumbContext, out: Path) -> dict[str, Any]:
    if contrast_ratio(ctx.palette["text"], ctx.palette["bg"]) < 7.0:
        raise ValueError("Palette text/bg contrast below 7:1 - thumbnails would be illegible")
    rng = random.Random(ctx.seed)
    bg = background.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    bg = ImageEnhance.Brightness(bg).enhance(rng.uniform(1.05, 1.25))
    img = bg.convert("RGBA")
    TEMPLATES[template_id](img, ctx)
    rgb = img.convert("RGB")
    out.parent.mkdir(parents=True, exist_ok=True)
    for q in (93, 90, 86, 82, 78, 72):
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
        if buf.tell() < MAX_BYTES:
            out.write_bytes(buf.getvalue())
            break
    else:
        raise ValueError("Thumbnail could not be compressed under 2 MB")
    return {"path": str(out), "template": template_id, "bytes": out.stat().st_size, "dhash": dhash64(rgb)}
