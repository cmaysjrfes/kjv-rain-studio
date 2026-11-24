"""FFmpeg/ffprobe wrappers: progress parsing, stderr capture, encoder detection, loudness measurement."""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import IO, Any

log = logging.getLogger(__name__)
ProgressCb = Callable[[float], None]

HW_ENCODER_ORDER = ["h264_videotoolbox", "h264_nvenc", "libx264"]


class FFmpegError(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int, stderr_tail: str) -> None:
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        short = " ".join(self.cmd[:12]) + (" ..." if len(self.cmd) > 12 else "")
        super().__init__(f"ffmpeg exited {returncode}: {short}\n--- stderr (tail) ---\n{stderr_tail}")


class FFmpegMissingError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegMissingError("ffmpeg not found on PATH (brew install ffmpeg / apt install ffmpeg)")
    return path


@lru_cache(maxsize=1)
def ffprobe_bin() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise FFmpegMissingError("ffprobe not found on PATH (ships with ffmpeg)")
    return path


def _drain(stream: IO[bytes], sink: deque[str]) -> None:
    for raw in iter(stream.readline, b""):
        sink.append(raw.decode("utf-8", "replace").rstrip())
    stream.close()


def run_ffmpeg(args: Sequence[str], *, duration_s: float | None = None, progress: ProgressCb | None = None,
               timeout_s: float | None = None, stderr_lines: int = 400) -> list[str]:
    """Run ffmpeg; returns captured stderr lines. Raises FFmpegError with the stderr tail on failure."""
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-y", "-progress", "pipe:1", "-nostats", *map(str, args)]
    log.debug("ffmpeg %s", " ".join(cmd[1:]))
    errs: deque[str] = deque(maxlen=stderr_lines)
    # own session: a terminal Ctrl-C reaches the orchestrator (graceful stop), not ffmpeg mid-render
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    t = threading.Thread(target=_drain, args=(proc.stderr, errs), daemon=True)
    t.start()
    last = -1.0
    assert proc.stdout is not None
    try:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if progress and duration_s and line.startswith("out_time_us="):
                try:
                    frac = min(1.0, int(line.split("=", 1)[1]) / 1e6 / duration_s)
                except ValueError:
                    continue
                if frac - last >= 0.005:
                    last = frac
                    progress(frac)
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        t.join(5)
        raise FFmpegError(cmd, -9, "timeout\n" + "\n".join(list(errs)[-40:])) from None
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    t.join(5)
    if proc.returncode != 0:
        raise FFmpegError(cmd, proc.returncode, "\n".join(list(errs)[-40:]))
    if progress:
        progress(1.0)
    return list(errs)


class FFmpegWriter:
    """Pipe raw RGB frames into an ffmpeg encode (used for procedural visual loops)."""

    def __init__(self, args: Sequence[str]) -> None:
        self.cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)]
        self._errs: deque[str] = deque(maxlen=200)
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, start_new_session=True)
        self._t = threading.Thread(target=_drain, args=(self.proc.stderr, self._errs), daemon=True)
        self._t.start()

    def write(self, frame_bytes: bytes) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(frame_bytes)
        except BrokenPipeError:
            self.proc.wait()
            self._t.join(5)
            raise FFmpegError(self.cmd, self.proc.returncode or -1, "\n".join(self._errs)) from None

    def close(self) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.wait()
        self._t.join(5)
        if self.proc.returncode != 0:
            raise FFmpegError(self.cmd, self.proc.returncode, "\n".join(self._errs))

    def abort(self) -> None:
        self.proc.kill()
        self.proc.wait()

    def __enter__(self) -> FFmpegWriter:
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def probe(path: str | Path) -> dict[str, Any]:
    out = subprocess.run(
        [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise FFmpegError(["ffprobe", str(path)], out.returncode, out.stderr[-2000:])
    return json.loads(out.stdout)


def media_duration(path: str | Path) -> float:
    return float(probe(path)["format"]["duration"])


@lru_cache(maxsize=4)
def detect_encoder(preference: str = "auto") -> str:
    """Pick the first H.264 encoder that actually works on this machine (not just compiled in)."""
    candidates = HW_ENCODER_ORDER if preference in ("", "auto") else [preference, "libx264"]
    for enc in candidates:
        try:
            run_ffmpeg(["-f", "lavfi", "-i", "color=c=black:s=256x144:r=15:d=0.4", "-frames:v", "4",
                        "-c:v", enc, "-pix_fmt", "yuv420p", "-f", "null", "-"], timeout_s=30)
            log.info("Video encoder selected: %s", enc)
            return enc
        except (FFmpegError, FFmpegMissingError) as exc:
            log.debug("encoder %s unavailable: %s", enc, str(exc).splitlines()[0])
    raise FFmpegMissingError("No working H.264 encoder (need libx264, h264_nvenc or h264_videotoolbox)")


def encoder_args(encoder: str, fps: int, *, quality: str = "loop") -> list[str]:
    """Encoder settings for dark, low-motion sleep visuals with closed 2-second GOPs.

    The loop is stream-copied for the whole 8-10 h, so its bitrate sets the final file size:
    ~2 Mbps cap -> roughly 9 GB for 10 h at 1080p.
    """
    gop = str(int(fps * 2))
    hq = quality == "loop"
    if encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "medium" if hq else "veryfast", "-crf", "23" if hq else "24",
                "-profile:v", "high", "-maxrate", "2500k" if hq else "6M", "-bufsize", "5000k" if hq else "12M",
                "-g", gop,
                "-x264-params", f"keyint={gop}:min-keyint={gop}:scenecut=0:open-gop=0", "-pix_fmt", "yuv420p"]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p6" if hq else "p4", "-rc", "vbr", "-cq", "25" if hq else "24",
                "-b:v", "0", "-maxrate", "2500k" if hq else "6M", "-profile:v", "high", "-g", gop, "-forced-idr", "1", "-pix_fmt", "yuv420p"]
    if encoder == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-b:v", "1800k" if hq else "4M", "-maxrate", "2500k" if hq else "6M",
                "-profile:v", "high", "-g", gop, "-allow_sw", "1", "-pix_fmt", "yuv420p"]
    return ["-c:v", encoder, "-g", gop, "-pix_fmt", "yuv420p"]


_LOUDNORM_JSON = re.compile(r"\{\s*\"input_i\"[\s\S]*?\}")


def measure_loudness(input_args: Sequence[str], pre_filter: str = "", *, target_i: float = -23.0,
                     target_tp: float = -2.0, target_lra: float = 11.0) -> dict[str, float]:
    """First loudnorm pass. Returns input_i/input_tp/input_lra/input_thresh/target_offset."""
    chain = f"{pre_filter + ',' if pre_filter else ''}loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}" \
            ":print_format=json"
    lines = run_ffmpeg([*input_args, "-af", chain, "-f", "null", "-"])
    m = _LOUDNORM_JSON.search("\n".join(lines))
    if not m:
        raise FFmpegError(["loudnorm-measure"], 0, "loudnorm JSON not found in ffmpeg output")
    data = json.loads(m.group(0))
    return {k: float(v) for k, v in data.items() if k != "normalization_type"}


def loudnorm_second_pass(measured: dict[str, float], *, target_i: float, target_tp: float,
                         target_lra: float = 11.0) -> str:
    """Filter string for an accurate (linear) second loudnorm pass."""
    return (f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}:measured_I={measured['input_i']}"
            f":measured_TP={measured['input_tp']}:measured_LRA={measured['input_lra']}"
            f":measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:linear=true")


def fmt_ts(seconds: float) -> str:
    """YouTube chapter timestamp: M:SS under an hour, H:MM:SS above."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
