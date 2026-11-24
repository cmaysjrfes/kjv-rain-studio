"""Procedural ultra-dark sleep scenes rendered as mathematically seamless loops.

Every time-varying term (rain, glints, flicker, fog drift, embers) completes an integer number of
cycles per loop, so frame N wraps perfectly to frame 0 and the loop can be stream-copied for
8-10 hours without a visible seam or re-encoding.

Photosensitivity: flicker amplitude is capped at 6 % luminance and there are no flashes.
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from core.imaging import hex_to_rgb


@dataclass(frozen=True)
class SceneStyle:
    rain: str            # streaks | glass | window | none
    flicker: float       # relative glow flicker amplitude (<= 0.06)
    fog: float           # fog layer strength
    embers: bool = False
    window_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # rain region (relative)


SCENE_STYLES: dict[str, SceneStyle] = {
    "rain_on_glass_night": SceneStyle("glass", 0.0, 0.25),
    "dim_cabin_lantern": SceneStyle("window", 0.05, 0.10, window_box=(0.58, 0.12, 0.92, 0.62)),
    "candle_by_window": SceneStyle("window", 0.06, 0.08, window_box=(0.08, 0.10, 0.48, 0.66)),
    "misty_forest_night": SceneStyle("streaks", 0.0, 0.55),
    "porch_lamp_rain": SceneStyle("streaks", 0.025, 0.20),
    "chapel_window_moon": SceneStyle("glass", 0.0, 0.18, window_box=(0.30, 0.08, 0.70, 0.80)),
    "open_bible_lamplight": SceneStyle("window", 0.04, 0.08, window_box=(0.20, 0.06, 0.80, 0.42)),
    "fireplace_embers_low": SceneStyle("none", 0.06, 0.05, embers=True),
}


def _shift_hue(hex_color: str, degrees: float) -> np.ndarray:
    r, g, b = (c / 255 for c in hex_to_rgb(hex_color))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb((h + degrees / 360) % 1.0, l, s)
    return np.array([r2, g2, b2], dtype=np.float32)


def _tileable_noise(rng: np.random.Generator, w: int, h: int, cell: int) -> np.ndarray:
    gw, gh = max(2, w // cell), max(2, h // cell)
    g = rng.random((gh, gw)).astype(np.float32)
    tiled = np.tile(g, (3, 3))
    img = Image.fromarray((tiled * 255).astype(np.uint8), "L").resize((w * 3, h * 3), Image.Resampling.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(cell / 3))
    arr = np.asarray(img, dtype=np.float32)[h:2 * h, w:2 * w] / 255.0
    return (arr - arr.min()) / max(1e-6, arr.max() - arr.min())


class SceneRenderer:
    def __init__(self, scene: dict[str, Any], params: dict[str, float], *, size: tuple[int, int] = (960, 540),
                 loop_s: float = 120.0, fps: int = 15, max_mean_luma: float = 32.0,
                 target_luma: float | None = None) -> None:
        self.scene = scene
        self.style = SCENE_STYLES.get(scene["id"], SceneStyle("streaks", 0.02, 0.2))
        self.W, self.H = size
        self.fps = fps
        self.loop_s = loop_s
        self.n_frames = int(round(loop_s * fps))
        self.rng = np.random.default_rng(int(params.get("seed", 1)))
        self.p = params
        hue = float(params.get("hue_shift_deg", 0.0))
        self.base = _shift_hue(scene["base"], hue)
        self.glow_rgb = _shift_hue(scene["glow"], hue)
        self.gx = float(params.get("glow_x", 0.5)) * self.W
        self.gy = float(params.get("glow_y", 0.6)) * self.H
        self.target = target_luma if target_luma is not None else min(
            max_mean_luma, max(8.0, float(params.get("brightness", 0.08)) * 255))
        self._build_static()
        self._build_dynamic()
        self.gain = 1.0
        self.gain = self._calibrate()

    # ---------------------------------------------------------------- static layers
    def _build_static(self) -> None:
        W, H = self.W, self.H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        grad = (0.6 + 0.4 * (1 - yy / H))[..., None]
        self.bg = grad * self.base[None, None, :]
        r = math.hypot(W, H) * 0.45
        d = np.sqrt((xx - self.gx) ** 2 + (yy - self.gy) ** 2) / r
        self.glow = (np.clip(1 - d, 0, 1) ** 2.4).astype(np.float32)
        cx, cy = W / 2, H / 2
        vd = np.sqrt(((xx - cx) / (W / 2)) ** 2 + ((yy - cy) / (H / 2)) ** 2) / math.sqrt(2)
        v = float(self.p.get("vignette", 0.7))
        self.vignette = (1 - v * np.clip(vd, 0, 1) ** 1.6).astype(np.float32)
        self.structure = self._silhouettes()            # 0 = occluded (dark object), 1 = open
        wb = self.style.window_box
        self.rain_mask = np.zeros((H, W), np.float32)
        x0, y0, x1, y1 = int(wb[0] * W), int(wb[1] * H), int(wb[2] * W), int(wb[3] * H)
        self.rain_mask[y0:y1, x0:x1] = 1.0
        self.rain_mask = np.asarray(Image.fromarray((self.rain_mask * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(4)), dtype=np.float32) / 255
        self.fog = _tileable_noise(self.rng, W, H, max(24, W // 12))

    def _silhouettes(self) -> np.ndarray:
        W, H = self.W, self.H
        m = Image.new("L", (W, H), 255)
        d = ImageDraw.Draw(m)
        sid = self.scene["id"]
        wb = self.style.window_box
        box = [wb[0] * W, wb[1] * H, wb[2] * W, wb[3] * H]
        t = max(3, W // 160)
        if sid in {"dim_cabin_lantern", "candle_by_window", "open_bible_lamplight"}:
            d.rectangle(box, outline=40, width=t * 2)                       # window frame
            d.line([((box[0] + box[2]) / 2, box[1]), ((box[0] + box[2]) / 2, box[3])], fill=40, width=t)
            d.line([(box[0], (box[1] + box[3]) / 2), (box[2], (box[1] + box[3]) / 2)], fill=40, width=t)
        if sid == "dim_cabin_lantern":
            for y in range(0, H, max(18, H // 14)):                         # plank seams
                d.line([(0, y), (W, y)], fill=200, width=1)
            lx, ly = self.gx, self.gy
            d.rectangle([lx - W * 0.025, ly - H * 0.03, lx + W * 0.025, ly + H * 0.09], fill=90)
            d.arc([lx - W * 0.02, ly - H * 0.10, lx + W * 0.02, ly - H * 0.02], 180, 360, fill=60, width=t)
        if sid == "candle_by_window":
            cx, cy = self.gx, self.gy
            d.rectangle([cx - W * 0.012, cy + H * 0.02, cx + W * 0.012, cy + H * 0.16], fill=150)
            d.rectangle([0, H * 0.82, W, H], fill=70)                       # sill / table
        if sid == "misty_forest_night":
            for _ in range(9):
                x = float(self.rng.uniform(0, W))
                w = float(self.rng.uniform(W * 0.01, W * 0.035))
                d.polygon([(x - w, H), (x + w, H), (x + w * 0.4, 0), (x - w * 0.4, 0)],
                          fill=int(self.rng.uniform(30, 110)))
        if sid == "porch_lamp_rain":
            d.rectangle([W * 0.08, 0, W * 0.12, H], fill=40)                 # post
            d.rectangle([0, H * 0.70, W, H * 0.73], fill=60)                 # rail
            for x in np.linspace(0, W, 14):
                d.rectangle([x, H * 0.73, x + W * 0.008, H], fill=60)
        if sid == "chapel_window_moon":
            m.paste(40, (0, 0, W, H))
            x0, y0, x1, y1 = box
            d.rectangle([x0, y0 + (x1 - x0) / 2, x1, y1], fill=255)
            d.pieslice([x0, y0, x1, y0 + (x1 - x0)], 180, 360, fill=255)
            d.line([((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1)], fill=40, width=t)
            for k in (1, 2, 3):
                yk = y0 + (y1 - y0) * k / 4 + (x1 - x0) / 4
                d.line([(x0, yk), (x1, yk)], fill=40, width=t)
        if sid == "open_bible_lamplight":
            bx, by, bw = W / 2, H * 0.80, W * 0.26
            d.polygon([(bx - bw, by), (bx, by - H * 0.05), (bx, H * 0.97), (bx - bw * 1.05, H * 0.97)], fill=170)
            d.polygon([(bx + bw, by), (bx, by - H * 0.05), (bx, H * 0.97), (bx + bw * 1.05, H * 0.97)], fill=170)
        if sid == "fireplace_embers_low":
            fx0, fx1, fy = W * 0.25, W * 0.75, H * 0.45
            m.paste(70, (0, 0, W, H))
            d.rectangle([fx0, fy + (fx1 - fx0) / 4, fx1, H], fill=255)
            d.pieslice([fx0, fy, fx1, fy + (fx1 - fx0) / 2], 180, 360, fill=255)
            d.rectangle([fx0 - W * 0.05, H * 0.9, fx1 + W * 0.05, H], fill=50)
        m = m.filter(ImageFilter.GaussianBlur(max(1, W // 480)))
        return np.asarray(m, dtype=np.float32) / 255

    # ---------------------------------------------------------------- dynamic layers
    def _build_dynamic(self) -> None:
        W, H, L = self.W, self.H, self.loop_s
        dens = float(self.p.get("rain_density", 1.0))
        rain = self.style.rain
        n = 0 if rain == "none" else int((260 if rain in {"streaks", "window"} else 90) * dens * W / 960)
        self.drop_len = self.rng.uniform(H * 0.04, H * 0.11, n)
        speed_target = self.rng.uniform(0.9, 1.6, n) * H                     # px / s
        self.drop_k = np.maximum(1, np.round(speed_target * L / (H + self.drop_len))).astype(np.int64)
        self.drop_phase = self.rng.random(n)
        self.drop_x = self.rng.uniform(-0.1 * W, 1.1 * W, n)
        self.drop_a = self.rng.uniform(0.25, 0.7, n)
        self.slant = math.tan(math.radians(float(self.rng.uniform(4, 10))))
        # glass beads: static droplets + slow sliders (1-3 cycles per loop)
        nb = int(140 * dens * W / 960) if rain == "glass" else 0
        self.beads = np.column_stack([self.rng.uniform(0, W, nb), self.rng.uniform(0, H, nb),
                                      self.rng.uniform(1.0, 3.2, nb) * W / 960])
        ns = int(14 * dens) if rain == "glass" else 0
        self.sliders = np.column_stack([self.rng.uniform(0, W, ns), self.rng.random(ns),
                                        self.rng.integers(1, 4, ns), self.rng.uniform(2.0, 4.0, ns) * W / 960])
        f = [0.7, 1.9, 4.3]
        self.flick = [(max(1, round(L * fr)), float(self.rng.uniform(0, 2 * math.pi)), a)
                      for fr, a in zip(f, (0.55, 0.3, 0.15))]
        self.breath_k = max(1, round(L * 0.05))
        self.fog_k = 1 if float(self.p.get("drift_speed", 0.4)) < 0.4 else 2
        ne = 60 if self.style.embers else 0
        self.embers = np.column_stack([self.rng.uniform(0.3 * W, 0.7 * W, ne), self.rng.random(ne),
                                       self.rng.integers(1, 4, ne), self.rng.uniform(0, 2 * math.pi, ne)])

    def _flicker(self, t: float) -> float:
        L = self.loop_s
        amp = min(0.06, self.style.flicker)
        v = sum(a * math.sin(2 * math.pi * k * t / L + ph) for k, ph, a in self.flick)
        breath = 0.02 * math.sin(2 * math.pi * self.breath_k * t / L)
        return 1.0 + amp * v + breath

    def _rain_layer(self, t: float) -> np.ndarray:
        W, H, L = self.W, self.H, self.loop_s
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        if self.style.rain in {"streaks", "window"} and len(self.drop_k):
            span = H + self.drop_len
            y = ((self.drop_phase + self.drop_k * t / L) % 1.0) * span - self.drop_len
            x = self.drop_x + self.slant * y
            for xi, yi, li, ai in zip(x, y, self.drop_len, self.drop_a):
                d.line([(xi, yi), (xi + self.slant * li, yi + li)], fill=int(255 * ai), width=1)
        if self.style.rain == "glass":
            for bx, by, br in self.beads:
                d.ellipse([bx - br, by - br, bx + br, by + br], outline=90, width=1)
            for sx, ph, k, sr in self.sliders:
                y = ((ph + k * t / L) % 1.0) * (H + 40) - 20
                d.ellipse([sx - sr, y - sr * 1.3, sx + sr, y + sr * 1.3], fill=150)
                d.line([(sx, y - sr * 8), (sx, y - sr)], fill=50, width=max(1, int(sr / 2)))
            if len(self.drop_k):
                span = H + self.drop_len
                y = ((self.drop_phase + self.drop_k * t / L) % 1.0) * span - self.drop_len
                x = self.drop_x + self.slant * y
                for xi, yi, li, ai in zip(x[::2], y[::2], self.drop_len[::2], self.drop_a[::2]):
                    d.line([(xi, yi), (xi + self.slant * li, yi + li)], fill=int(120 * ai), width=1)
        if self.style.embers and len(self.embers):
            for ex, ph, k, wob in self.embers:
                frac = (ph + k * t / L) % 1.0
                ey = H * (0.95 - 0.55 * frac)
                exx = ex + 6 * math.sin(2 * math.pi * k * 3 * t / L + wob)
                a = int(220 * (1 - frac))
                d.ellipse([exx - 1.5, ey - 1.5, exx + 1.5, ey + 1.5], fill=a)
        return np.asarray(img, dtype=np.float32) / 255

    # ---------------------------------------------------------------- compose
    def _compose(self, t: float) -> np.ndarray:
        fl = self._flicker(t)
        glow = self.glow * fl
        shift = int(round(self.fog_k * self.W * t / self.loop_s)) % self.W
        fog = np.roll(self.fog, shift, axis=1) * self.style.fog * (0.4 + glow)
        rain = self._rain_layer(t)
        if self.style.rain == "window":
            rain *= self.rain_mask
        rain_light = rain * (0.25 + 1.6 * self.glow)
        if self.style.embers:
            rain_light = rain * 2.2
        img = self.bg * self.structure[..., None]
        img = img + glow[..., None] * self.glow_rgb * 1.4 * (0.35 + 0.65 * self.structure[..., None])
        img = img + fog[..., None] * self.glow_rgb * 0.35
        rain_rgb = (self.glow_rgb * 0.6 + np.array([0.55, 0.6, 0.7], np.float32) * 0.4)
        if self.style.embers:
            rain_rgb = np.array([1.0, 0.45, 0.15], np.float32)
        img = img + rain_light[..., None] * rain_rgb * 0.55
        return img * self.vignette[..., None] * self.gain

    def _calibrate(self) -> float:
        lumas = []
        for t in (0.0, self.loop_s / 3, 2 * self.loop_s / 3):
            f = np.clip(self._compose(t), 0, 1) * 255
            lumas.append(float((0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]).mean()))
        mean = max(0.5, float(np.mean(lumas)))
        return float(np.clip(self.target / mean, 0.2, 40.0))

    def frame(self, index: int) -> np.ndarray:
        t = (index % self.n_frames) / self.fps
        return (np.clip(self._compose(t), 0, 1) * 255).astype(np.uint8)

    def still(self, t: float = 0.0) -> Image.Image:
        return Image.fromarray((np.clip(self._compose(t), 0, 1) * 255).astype(np.uint8), "RGB")

    def mean_luma(self, frame: np.ndarray) -> float:
        f = frame.astype(np.float32)
        return float((0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]).mean())
