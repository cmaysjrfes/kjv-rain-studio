"""Ambient bed construction: procedural FFmpeg noise layers, optional field recordings, numpy thunder.

Procedural beds are generated over the *full* duration from seeded noise sources, so there is no
loop period at all (nothing to hear repeating). If you drop licensed/CC0 rain recordings into
``assets/ambient/<bed_id>/`` they replace the synthetic rain layers; random, non-repeating segments
are joined with 8 s equal-power crossfades.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config import PROJECT_ROOT
from core.ffmpeg import media_duration

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif"}
SR = 48000

# color, highpass Hz, lowpass Hz, gain dB, velvet density (None = continuous)
LAYER_RECIPES: dict[str, dict[str, Any]] = {
    "pink_rain_fine": {"color": "pink", "hp": 300, "lp": 5200, "gain": -4.0},
    "pink_rain_dense": {"color": "pink", "hp": 150, "lp": 4200, "gain": 0.0},
    "brown_rain_heavy": {"color": "brown", "hp": 60, "lp": 2400, "gain": 0.0},
    "roof_patter": {"color": "velvet", "hp": 900, "lp": 4500, "gain": -6.0, "density": 0.02},
    "glass_drips": {"color": "velvet", "hp": 1800, "lp": 5000, "gain": -12.0, "density": 0.002},
    "leaf_drips": {"color": "velvet", "hp": 600, "lp": 3000, "gain": -10.0, "density": 0.004},
    "canvas_patter": {"color": "velvet", "hp": 300, "lp": 2200, "gain": -6.0, "density": 0.03},
    "low_rumble": {"color": "brown", "hp": 25, "lp": 140, "gain": -10.0},
    "brown_wind": {"color": "brown", "hp": 40, "lp": 520, "gain": -6.0, "wind": True},
}
RAIN_LAYERS = {"pink_rain_fine", "pink_rain_dense", "brown_rain_heavy"}


@dataclass
class AmbientGraph:
    input_args: list[str]
    filter: str
    out_label: str
    n_inputs: int
    uses_recordings: bool


def find_recordings(bed_id: str) -> list[Path]:
    d = PROJECT_ROOT / "assets" / "ambient" / bed_id
    return sorted(p for p in d.glob("*") if p.suffix.lower() in AUDIO_EXTS) if d.exists() else []


def build_ambient_graph(bed: dict[str, Any], params: dict[str, float], duration: float, *,
                        thunder_path: Path | None = None, input_offset: int = 0) -> AmbientGraph:
    """Return FFmpeg input args + filtergraph producing stereo 48 kHz ``[bedraw]`` of `duration` s."""
    rng = random.Random(int(params.get("seed", 0)))
    density = float(params.get("density", 1.0))
    inputs: list[str] = []
    chains: list[str] = []
    labels: list[str] = []
    idx = input_offset
    recordings = find_recordings(bed["id"])

    def add_input(args: list[str]) -> int:
        nonlocal idx
        inputs.extend(args)
        idx += 1
        return idx - 1

    if recordings:
        chains_rec, label, used = _recording_chain(recordings, duration, rng, add_input)
        chains.extend(chains_rec)
        labels.append(label)
        layer_names = [n for n in bed["layers"] if n not in RAIN_LAYERS]
    else:
        used = 0
        layer_names = list(bed["layers"])

    for n, name in enumerate(layer_names):
        r = LAYER_RECIPES[name]
        src = []
        for _side in (0, 1):  # decorrelated L/R
            seed = rng.randrange(1, 2**31)
            extra = ""
            if r["color"] == "velvet":
                extra = f":density={min(1.0, r['density'] * density):.5f}"
            src.append(add_input(["-f", "lavfi", "-i",
                                  f"anoisesrc=c={r['color']}:r={SR}:a=0.5:seed={seed}:d={duration:.3f}{extra}"]))
        hp = r["hp"] * (params.get("rain_highpass_hz", 200) / 200 if name in RAIN_LAYERS else 1)
        lp = min(r["lp"], params.get("rain_lowpass_hz", 9999)) if name in RAIN_LAYERS else r["lp"]
        mods = ""
        if r.get("wind"):
            f = float(params.get("wind_lfo_hz", 0.06))
            depth = min(0.9, 0.3 + float(bed.get("wind", 0.1)))
            mods = f",volume=eval=frame:volume='{1 - depth / 2:.3f}+{depth / 2:.3f}*sin(2*PI*{f}*t)'"
        elif name in RAIN_LAYERS:
            # two incommensurate slow swells -> natural intensity drift with no audible period
            a1, a2 = rng.uniform(0.0011, 0.0023), rng.uniform(0.0003, 0.0007) * math.sqrt(2)
            mods = f",volume=eval=frame:volume='1+0.10*sin(2*PI*{a1:.5f}*t)+0.06*sin(2*PI*{a2:.6f}*t)'"
        lab = f"l{n}"
        chains.append(f"[{src[0]}:a][{src[1]}:a]amerge=inputs=2,highpass=f={hp:.0f},lowpass=f={lp:.0f}"
                      f"{mods},volume={r['gain']:.1f}dB[{lab}]")
        labels.append(lab)

    if thunder_path is not None:
        t_in = add_input(["-i", str(thunder_path)])
        chains.append(f"[{t_in}:a]aresample={SR},pan=stereo|c0=c0|c1=c0,lowpass=f=320,volume=2.0[thd]")
        labels.append("thd")

    mix = "".join(f"[{lab}]" for lab in labels)
    width = float(params.get("stereo_width", 0.8))
    dens_db = 20 * math.log10(max(0.3, density))
    chains.append(f"{mix}amix=inputs={len(labels)}:normalize=0:duration=longest,"
                  f"extrastereo=m={width:.2f},volume={dens_db:.2f}dB,atrim=0:{duration:.3f},"
                  f"aformat=sample_rates={SR}:channel_layouts=stereo[bedraw]")
    return AmbientGraph(inputs, ";".join(chains), "bedraw", idx - input_offset, bool(used))


def _recording_chain(files: list[Path], duration: float, rng: random.Random,
                     add_input: Any) -> tuple[list[str], str, int]:
    durs = {f: media_duration(f) for f in files}
    usable = [f for f, d in durs.items() if d >= 30]
    if not usable:
        raise ValueError("Ambient recordings must be at least 30 s long")
    xf = 8.0
    segs: list[int] = []
    covered = 0.0
    while covered < duration:
        f = rng.choice(usable)
        seg = min(durs[f], rng.uniform(720, 1200), duration + xf)
        off = rng.uniform(0, max(0.0, durs[f] - seg))
        segs.append(add_input(["-ss", f"{off:.2f}", "-t", f"{seg:.2f}", "-i", str(f)]))
        covered += seg - (xf if segs[1:] else 0)
    chains = [f"[{segs[0]}:a]aresample={SR},aformat=channel_layouts=stereo[r0]"]
    for i, s in enumerate(segs[1:], 1):
        chains.append(f"[{s}:a]aresample={SR},aformat=channel_layouts=stereo[s{i}]")
        chains.append(f"[r{i - 1}][s{i}]acrossfade=d={xf}:c1=qsin:c2=qsin[r{i}]")
    last = f"r{len(segs) - 1}"
    chains.append(f"[{last}]atrim=0:{duration:.3f}[rec]")
    return chains, "rec", len(segs)


# ------------------------------------------------------------------ thunder
def _thunder_event(rng: np.random.Generator, sr: int, dur: float) -> np.ndarray:
    n = int(dur * sr)
    brown = np.cumsum(rng.normal(0, 1, n))
    k = max(3, sr // 4)
    brown -= np.convolve(brown, np.ones(k) / k, mode="same")       # remove DC drift
    for _ in range(2):
        brown = np.convolve(brown, np.ones(24) / 24, mode="same")   # gentle low-pass (~300 Hz @ 8 kHz)
    t = np.arange(n) / sr
    env = np.zeros(n)
    for _ in range(int(rng.integers(1, 4))):                       # 1-3 rolling peaks
        start = rng.uniform(0, dur * 0.4)
        a, d = rng.uniform(0.3, 1.4), rng.uniform(2.5, 7.0)
        tt = np.clip(t - start, 0, None)
        env += (1 - np.exp(-tt / a)) * np.exp(-tt / d) * (t >= start) * rng.uniform(0.5, 1.0)
    fade = np.minimum(1, np.minimum(t, dur - t) / 0.5)
    out = brown * env * fade
    peak = np.abs(out).max() or 1.0
    return (out / peak).astype(np.float32)


def write_thunder_track(path: Path, duration: float, events_per_hour: float, seed: int, *,
                        quiet_until: float = 0.0, sr: int = 8000) -> int:
    """Sparse low-rate thunder track written in chunks as FLAC (silence compresses to ~nothing)."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    n_events = int(rng.poisson(max(0.0, events_per_hour) * duration / 3600))
    events = []
    for _ in range(n_events):
        at = float(rng.uniform(5, max(6, duration - 20)))
        dur = float(rng.uniform(6, 14))
        amp = float(rng.uniform(0.18, 0.5)) * (0.45 if at < quiet_until else 1.0)
        events.append((at, dur, amp, _thunder_event(rng, sr, dur)))
    events.sort(key=lambda e: e[0])
    chunk = 60 * sr
    total = int(duration * sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=1, format="FLAC", subtype="PCM_16") as f:
        for c0 in range(0, total, chunk):
            c1 = min(total, c0 + chunk)
            buf = np.zeros(c1 - c0, dtype=np.float32)
            for at, _, amp, wav in events:
                s0 = int(at * sr)
                s1 = s0 + len(wav)
                if s1 <= c0 or s0 >= c1:
                    continue
                a, b = max(s0, c0), min(s1, c1)
                buf[a - c0:b - c0] += wav[a - s0:b - s0] * amp
            f.write(np.clip(buf, -1, 1))
    return n_events
