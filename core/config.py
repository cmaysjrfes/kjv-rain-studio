"""Central configuration: merges .env secrets with config/settings.yaml tunables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _abs(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    return default if not raw else raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    # Secrets / env
    youtube_api_key: str
    client_secrets_file: Path
    token_file: Path
    daily_quota: int
    quota_reserve: int
    brand_name: str
    brand_handle: str
    brand_tagline: str
    funnel_url: str
    funnel_label: str
    tts_engine: str
    video_encoder: str
    video_fps: int
    db_path: Path
    output_dir: Path
    brand_kit_dir: Path
    log_level: str
    # Agents 2-5
    kokoro_voice: str = "am_michael"
    piper_model_path: Path = PROJECT_ROOT / "assets/voices/en_US-ryan-low.onnx"
    chatterbox_voice_prompt: str = ""
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    preview_seconds: int = 0
    default_privacy: str = "private"
    disclose_synthetic_media: bool = True
    upload_chunk_mb: int = 64
    shorts_per_release: int = 3
    keep_final_video_days: int = 0
    expected_channel_id: str = ""         # safety: refuse to publish to any other channel
    publish_time: str = ""                 # "19:30" -> scheduled publishing; "" -> DEFAULT_PRIVACY now
    publish_timezone: str = "America/New_York"
    notify_subscribers_long: bool = True
    notify_subscribers_shorts: bool = False
    # YAML sections
    research: dict[str, Any] = field(default_factory=dict)
    brand: dict[str, Any] = field(default_factory=dict)
    content: dict[str, Any] = field(default_factory=dict)
    render: dict[str, Any] = field(default_factory=dict)
    publishing: dict[str, Any] = field(default_factory=dict)
    scaling: dict[str, Any] = field(default_factory=dict)
    raw_env: dict[str, str] = field(default_factory=dict)

    @property
    def content_blueprint_path(self) -> Path:
        return PROJECT_ROOT / "config" / "content_blueprint.json"

    @property
    def channel_blueprint_path(self) -> Path:
        return PROJECT_ROOT / "config" / "channel_page_blueprint.json"

    @property
    def has_api_key(self) -> bool:
        return bool(self.youtube_api_key)

    def ensure_dirs(self) -> None:
        for d in (self.output_dir, self.brand_kit_dir, self.db_path.parent, PROJECT_ROOT / "logs"):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    yaml_path = PROJECT_ROOT / "config" / "settings.yaml"
    data: dict[str, Any] = {}
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    s = Settings(
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", "").strip(),
        client_secrets_file=_abs(os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", "config/client_secret.json")),
        token_file=_abs(os.getenv("YOUTUBE_TOKEN_FILE", "config/token.json")),
        daily_quota=_env_int("YOUTUBE_DAILY_QUOTA", 10_000),
        quota_reserve=_env_int("YOUTUBE_QUOTA_RESERVE", 2_000),
        brand_name=os.getenv("BRAND_NAME", "").strip(),
        brand_handle=os.getenv("BRAND_HANDLE", "").strip(),
        brand_tagline=os.getenv("BRAND_TAGLINE", "").strip(),
        funnel_url=os.getenv("FUNNEL_URL", "").strip(),
        funnel_label=os.getenv("FUNNEL_LABEL", "Free printable sleep scripture guide").strip(),
        tts_engine=os.getenv("TTS_ENGINE", "kokoro").strip().lower(),
        video_encoder=os.getenv("VIDEO_ENCODER", "auto").strip(),
        video_fps=_env_int("VIDEO_FPS", int(data.get("render", {}).get("fps", 15))),
        db_path=_abs(os.getenv("DB_PATH", "db/state.sqlite3")),
        output_dir=_abs(os.getenv("OUTPUT_DIR", "output")),
        brand_kit_dir=_abs(os.getenv("BRAND_KIT_DIR", "channel_brand_kit")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        kokoro_voice=os.getenv("KOKORO_VOICE", "am_michael").strip(),
        piper_model_path=_abs(os.getenv("PIPER_MODEL_PATH", "assets/voices/en_US-ryan-low.onnx")),
        chatterbox_voice_prompt=os.getenv("CHATTERBOX_VOICE_PROMPT", "").strip(),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", "").strip(),
        elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "").strip(),
        preview_seconds=_env_int("PREVIEW_SECONDS", 0),
        default_privacy=os.getenv("DEFAULT_PRIVACY", "private").strip().lower(),
        disclose_synthetic_media=_env_bool("DISCLOSE_SYNTHETIC_MEDIA", True),
        upload_chunk_mb=_env_int("UPLOAD_CHUNK_MB", 64),
        shorts_per_release=_env_int("SHORTS_PER_RELEASE", 3),
        keep_final_video_days=_env_int("KEEP_FINAL_VIDEO_DAYS", 0),
        expected_channel_id=os.getenv("YOUTUBE_CHANNEL_ID", "").strip(),
        publish_time=os.getenv("PUBLISH_TIME", "").strip(),
        publish_timezone=os.getenv("PUBLISH_TIMEZONE", "America/New_York").strip(),
        notify_subscribers_long=_env_bool("NOTIFY_SUBSCRIBERS_LONG", True),
        notify_subscribers_shorts=_env_bool("NOTIFY_SUBSCRIBERS_SHORTS", False),
        research=data.get("competitor_research", {}),
        brand=data.get("brand", {}),
        content=data.get("content", {}),
        render=data.get("render", {}),
        publishing=data.get("publishing", {}),
        scaling=data.get("scaling", {}),
        raw_env={k: v for k, v in os.environ.items() if k.startswith(("YOUTUBE_", "BRAND_", "TTS_"))},
    )
    if s.default_privacy not in {"private", "unlisted", "public"}:
        raise ValueError(f"DEFAULT_PRIVACY must be private|unlisted|public, got {s.default_privacy!r}")
    if s.publish_time:
        import re
        from zoneinfo import ZoneInfo

        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", s.publish_time):
            raise ValueError(f"PUBLISH_TIME must be HH:MM (24h), got {s.publish_time!r}")
        ZoneInfo(s.publish_timezone)  # raises on an unknown zone
    if s.upload_chunk_mb < 1:
        raise ValueError("UPLOAD_CHUNK_MB must be >= 1")
    if not 15 <= s.video_fps <= 24:
        raise ValueError(f"VIDEO_FPS must be 15-24 for sleep content, got {s.video_fps}")
    s.ensure_dirs()
    return s
