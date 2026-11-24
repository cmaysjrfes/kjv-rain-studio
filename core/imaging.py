"""Image analysis (competitor thumbnails/banners) and procedural brand-art rendering."""
from __future__ import annotations

import io
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

log = logging.getLogger(__name__)

FONT_DIRS = [
    Path(__file__).resolve().parent.parent / "assets" / "fonts",
    Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
    Path("/Library/Fonts"), Path("/System/Library/Fonts"), Path.home() / "Library/Fonts",
    Path("C:/Windows/Fonts"),
]
SERIF_PREFS = ["CormorantGaramond-SemiBold", "EBGaramond-SemiBold", "PlayfairDisplay-Bold", "DejaVuSerif-Bold",
               "LiberationSerif-Bold", "Georgia Bold", "georgiab", "Times New Roman Bold", "FreeSerifBold"]
SANS_PREFS = ["Inter-Medium", "Inter-Regular", "DejaVuSans", "LiberationSans-Regular", "Helvetica", "arial",
              "FreeSans"]


# ============================================================================ colour helpers
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb: tuple[int, int, int] | np.ndarray) -> str:
    r, g, b = (int(max(0, min(255, c))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    def lin(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a: str, b: str) -> float:
    la, lb = rgb_to_lab(hex_to_rgb(a)), rgb_to_lab(hex_to_rgb(b))
    return math.dist(la, lb)


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG 2.1 contrast ratio - used to guarantee legible thumbnail/banner text."""
    def rel(h: str) -> float:
        def ch(c: int) -> float:
            v = c / 255
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        r, g, b = (ch(c) for c in hex_to_rgb(h))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    l1, l2 = sorted((rel(fg), rel(bg)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


# ============================================================================ analysis
def download_image(url: str, timeout: int = 20) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (research-bot)"})
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - network/decoding errors are non-fatal
        log.debug("image download failed %s: %s", url, exc)
        return None


def dhash64(img: Image.Image) -> str:
    """64-bit difference hash; Hamming distance <= ~6 means visually near-identical."""
    g = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def dominant_palette(img: Image.Image, k: int = 5) -> list[str]:
    small = img.copy()
    small.thumbnail((160, 160))
    q = small.quantize(colors=k, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette() or []
    counts = sorted(q.getcolors() or [], reverse=True)
    return [rgb_to_hex(tuple(pal[i * 3:i * 3 + 3])) for _, i in counts[:k]]


def image_metrics(img: Image.Image) -> dict[str, Any]:
    """Darkness, contrast, saturation, warmth, text-placement hint and palette of an image."""
    small = img.copy()
    small.thumbnail((480, 480))
    arr = np.asarray(small, dtype=np.float32)
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    bright = luma > 200
    thirds = np.array_split(bright, 3, axis=1)
    third_share = [float(t.mean()) for t in thirds]
    vthirds = np.array_split(bright, 3, axis=0)
    edges = np.asarray(small.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return {
        "mean_luma": round(float(luma.mean()), 1),
        "luma_std": round(float(luma.std()), 1),
        "p95_luma": round(float(np.percentile(luma, 95)), 1),
        "dark_ratio": round(float((luma < 40).mean()), 3),
        "bright_ratio": round(float(bright.mean()), 4),
        "saturation": round(float(hsv[..., 1].mean() / 255), 3),
        "warmth": round(float((arr[..., 0] - arr[..., 2]).mean()), 1),
        "edge_density": round(float((edges > 40).mean()), 4),
        "text_column": ["left", "center", "right"][int(np.argmax(third_share))],
        "text_row": ["top", "middle", "bottom"][int(np.argmax([float(t.mean()) for t in vthirds]))],
        "palette": dominant_palette(small),
        "dhash": dhash64(img),
    }


def analyze_url(url: str | None, timeout: int = 20) -> dict[str, Any] | None:
    if not url:
        return None
    img = download_image(url, timeout)
    return image_metrics(img) if img else None


# ============================================================================ fonts
def find_font(prefs: list[str]) -> Path | None:
    candidates: dict[str, Path] = {}
    for d in FONT_DIRS:
        if d.exists():
            for p in d.rglob("*"):
                if p.suffix.lower() in {".ttf", ".otf", ".ttc"}:
                    candidates.setdefault(p.stem.lower(), p)
    for name in prefs:
        if (hit := candidates.get(name.lower())) is not None:
            return hit
    return None


_FONT_CACHE: dict[str, Path | None] = {}


def load_font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if kind not in _FONT_CACHE:
        _FONT_CACHE[kind] = find_font(SERIF_PREFS if kind == "serif" else SANS_PREFS)
    path = _FONT_CACHE[kind]
    if path is not None:
        return ImageFont.truetype(str(path), size)
    log.warning("No %s TTF found (add one to assets/fonts); using Pillow default font", kind)
    return ImageFont.load_default(size=size)


def fit_font(draw: ImageDraw.ImageDraw, text: str, kind: str, max_w: int, max_h: int,
             start: int = 200, tracking: int = 0) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = start
    while size > 10:
        font = load_font(kind, size)
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        if (r - l) + tracking * len(text) <= max_w and (b - t) <= max_h:
            return font
        size -= 4
    return load_font(kind, 10)


def draw_tracked(draw: ImageDraw.ImageDraw, center: tuple[int, int], text: str, font: Any, fill: Any,
                 tracking: int = 0) -> None:
    """Draw text centred on `center` with letter-spacing (tracking)."""
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    x = center[0] - total / 2
    y = center[1] - (b + t) / 2
    for ch, w in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=fill)
        x += w + tracking


# ============================================================================ procedural art
def _radial_glow(size: tuple[int, int], center: tuple[float, float], radius: float, color: str,
                 strength: float = 0.6) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / radius
    a = np.clip(1 - d, 0, 1) ** 2.2 * strength
    r, g, b = hex_to_rgb(color)
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[..., 0], layer[..., 1], layer[..., 2] = r, g, b
    layer[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(layer, "RGBA")


def _vertical_gradient(size: tuple[int, int], top: str, bottom: str) -> Image.Image:
    w, h = size
    t, b = np.array(hex_to_rgb(top), np.float32), np.array(hex_to_rgb(bottom), np.float32)
    ramp = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    col = (t * (1 - ramp) + b * ramp).astype(np.uint8)
    return Image.fromarray(np.repeat(col[:, None, :], w, axis=1), "RGB").convert("RGBA")


def _rain_layer(size: tuple[int, int], rng: random.Random, color: str, density: int,
                length: tuple[int, int], alpha: tuple[int, int], width: int, angle_deg: float) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    r, g, b = hex_to_rgb(color)
    dx = math.tan(math.radians(angle_deg))
    for _ in range(density):
        x, y = rng.uniform(-100, size[0]), rng.uniform(-100, size[1])
        ln = rng.uniform(*length)
        d.line([(x, y), (x + dx * ln, y + ln)], fill=(r, g, b, rng.randint(*alpha)), width=width)
    return layer


def _grain(img: Image.Image, rng: random.Random, amount: float = 4.0) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, amount, arr.shape[:2])[..., None]
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), "RGB")


def _flame(draw: ImageDraw.ImageDraw, cx: float, base_y: float, h: float, accent: str, text: str) -> None:
    """Stylised lamp flame (Psalm 119:105 motif): teardrop with a bright inner core; tip points up."""
    def teardrop(scale_w: float, scale_h: float) -> list[tuple[float, float]]:
        pts = []
        for i in range(73):
            t = i / 72 * 2 * math.pi
            x = math.sin(t) * math.sin(t / 2) ** 1.5 * h * 0.36 * scale_w
            y = (1 + math.cos(t)) / 2 * h * scale_h
            pts.append((cx + x, base_y - y))
        return pts

    draw.polygon(teardrop(1.0, 1.0), fill=accent)
    draw.polygon(teardrop(0.5, 0.55), fill=text)


def render_banner(path: Path, *, name: str, tagline: str, cadence_line: str, palette: dict[str, str],
                  seed: int) -> dict[str, Any]:
    """2560x1440 banner; all text inside the 1546x423 safe area visible on every device."""
    W, H = 2560, 1440
    safe = (int((W - 1546) / 2), int((H - 423) / 2), int((W + 1546) / 2), int((H + 423) / 2))
    rng = random.Random(seed)
    img = _vertical_gradient((W, H), palette["mid"], palette["bg"])
    img.alpha_composite(_radial_glow((W, H), (W * 0.18, H * 0.62), 900, palette["accent"], 0.22))
    img.alpha_composite(_radial_glow((W, H), (W * 0.82, H * 0.35), 700, palette["mid"], 0.5))
    far = _rain_layer((W, H), rng, palette["text"], 1400, (40, 110), (10, 28), 1, 8)
    img.alpha_composite(far.filter(ImageFilter.GaussianBlur(1.2)))
    img.alpha_composite(_rain_layer((W, H), rng, palette["text"], 260, (120, 260), (18, 45), 2, 8))
    # Window mullions framing the safe area (rain-on-glass cue, very low contrast)
    frame = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(frame)
    fr = (*hex_to_rgb(palette["bg"]), 150)
    fd.rectangle([0, 0, W, 70], fill=fr)
    fd.rectangle([0, H - 70, W, H], fill=fr)
    fd.rectangle([W // 2 - 10, 0, W // 2 + 10, safe[1] - 40], fill=fr)
    fd.rectangle([W // 2 - 10, safe[3] + 40, W // 2 + 10, H], fill=fr)
    img.alpha_composite(frame.filter(ImageFilter.GaussianBlur(3)))
    # Dim plate behind text for legibility
    plate = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(plate).rounded_rectangle([safe[0] + 60, safe[1] + 10, safe[2] - 60, safe[3] - 10], 40,
                                            fill=(*hex_to_rgb(palette["bg"]), 150))
    img.alpha_composite(plate.filter(ImageFilter.GaussianBlur(30)))

    d = ImageDraw.Draw(img)
    sx0, sy0, sx1, sy1 = safe
    sw = sx1 - sx0
    title_font = fit_font(d, name.upper(), "serif", int(sw * 0.86), 150, start=150, tracking=10)
    draw_tracked(d, (W // 2, sy0 + 150), name.upper(), title_font, palette["text"], tracking=10)
    rule_y = sy0 + 245
    d.line([(W // 2 - 220, rule_y), (W // 2 + 220, rule_y)], fill=palette["accent"], width=3)
    _flame(d, W / 2, rule_y + 6, 34, palette["accent"], palette["text"])
    sub_font = fit_font(d, tagline, "sans", int(sw * 0.8), 56, start=56)
    draw_tracked(d, (W // 2, sy0 + 305), tagline, sub_font, palette["text"])
    cad_font = fit_font(d, cadence_line.upper(), "sans", int(sw * 0.7), 34, start=34, tracking=6)
    draw_tracked(d, (W // 2, sy0 + 370), cadence_line.upper(), cad_font, palette["accent"], tracking=6)

    out = _grain(img, rng, 3.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, "PNG", optimize=True)
    _shrink_to(path, out, 6 * 1024 * 1024)
    return {"path": str(path), "size": [W, H], "safe_area": list(safe), "bytes": path.stat().st_size,
            "metrics": image_metrics(out)}


def render_avatar(path: Path, *, monogram: str, palette: dict[str, str], seed: int) -> dict[str, Any]:
    """800x800 avatar; emblem kept inside the inner 75% circle so the round crop never clips it."""
    S = 800
    rng = random.Random(seed)
    img = _vertical_gradient((S, S), palette["mid"], palette["bg"])
    img.alpha_composite(_radial_glow((S, S), (S / 2, S * 0.48), S * 0.55, palette["accent"], 0.35))
    img.alpha_composite(_rain_layer((S, S), rng, palette["text"], 90, (30, 70), (10, 26), 2, 8))
    d = ImageDraw.Draw(img)
    r = S * 0.36
    d.ellipse([S / 2 - r, S / 2 - r, S / 2 + r, S / 2 + r], outline=palette["accent"], width=6)
    _flame(d, S / 2, S * 0.53, S * 0.30, palette["accent"], palette["text"])
    mono_font = fit_font(d, monogram, "serif", int(S * 0.34), int(S * 0.12), start=110, tracking=12)
    draw_tracked(d, (S // 2, int(S * 0.66)), monogram, mono_font, palette["text"], tracking=12)
    out = _grain(img, rng, 2.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, "PNG", optimize=True)
    return {"path": str(path), "size": [S, S], "bytes": path.stat().st_size, "metrics": image_metrics(out)}


def render_watermark(path: Path, *, palette: dict[str, str]) -> dict[str, Any]:
    """150x150 transparent PNG for YouTube's 'branding watermark' (subscribe button)."""
    S = 600  # supersample then downscale for smooth edges
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = S * 0.44
    d.ellipse([S / 2 - r, S / 2 - r, S / 2 + r, S / 2 + r], fill=(*hex_to_rgb(palette["bg"]), 170),
              outline=(*hex_to_rgb(palette["text"]), 230), width=18)
    _flame(d, S / 2, S * 0.70, S * 0.46, palette["accent"], palette["text"])
    out = img.resize((150, 150), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, "PNG", optimize=True)
    return {"path": str(path), "size": [150, 150], "bytes": path.stat().st_size}


def _shrink_to(path: Path, img: Image.Image, max_bytes: int) -> None:
    """YouTube banner limit is 6 MB; fall back to high-quality JPEG-in-PNG-name if exceeded."""
    if path.stat().st_size <= max_bytes:
        return
    jpg = path.with_suffix(".jpg")
    for q in (92, 88, 84, 80):
        img.convert("RGB").save(jpg, "JPEG", quality=q, optimize=True, progressive=True)
        if jpg.stat().st_size <= max_bytes:
            break
    log.info("Banner PNG exceeded %d bytes; JPEG version saved to %s", max_bytes, jpg)
