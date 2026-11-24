#!/usr/bin/env python3
"""Run Agent 1 standalone so the channel page blueprint and brand kit can be reviewed first.

Usage:
    python build_channel_page.py            # live research (API key if set, else yt-dlp)
    python build_channel_page.py --offline  # no network: baseline priors + brand kit only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.agent1_competitor_intel import CompetitorIntelAgent
from core.config import PROJECT_ROOT, get_settings
from core.logging_setup import setup_logging
from db.database import get_db

console = Console()


def render_preview(kit: Path, bp: dict) -> Path:
    """Mock desktop channel header: safe-area crop of the banner + round avatar + name/handle."""
    banner = Image.open(bp["assets"]["banner"]["path"]).convert("RGB")
    x0, y0, x1, y1 = bp["assets"]["banner"]["safe_area"]
    desktop = banner.crop((0, y0, 2560, y1)).resize((1280, int(423 * 1280 / 2560)))
    pal = bp["brand"]["palette"]
    canvas = Image.new("RGB", (1280, desktop.height + 220), pal["bg"])
    canvas.paste(desktop, (0, 0))
    av = Image.open(bp["assets"]["avatar"]["path"]).convert("RGB").resize((140, 140))
    mask = Image.new("L", (140, 140), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, 139, 139], fill=255)
    canvas.paste(av, (40, desktop.height + 40), mask.filter(ImageFilter.GaussianBlur(0.6)))
    d = ImageDraw.Draw(canvas)
    from core.imaging import load_font
    d.text((210, desktop.height + 55), bp["brand"]["name"], font=load_font("serif", 44), fill=pal["text"])
    d.text((210, desktop.height + 115), f'{bp["brand"]["handle"]}  ·  {bp["brand"]["tagline"]}',
           font=load_font("sans", 22), fill=pal["accent"])
    out = kit / "channel_page_preview.png"
    canvas.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="skip all network calls; use baseline priors")
    args = ap.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)
    db = get_db()

    mode = "offline (baseline priors)" if args.offline else (
        "YouTube Data API v3 + yt-dlp" if settings.has_api_key else "yt-dlp only (no YOUTUBE_API_KEY)")
    console.print(Panel.fit(f"[bold]Agent 1 - Competitor Intelligence & Channel Page Architect[/]\nSource: {mode}"))

    result = CompetitorIntelAgent(settings, db, offline=args.offline).run()
    if not result.ok:
        console.print(f"[red]Agent 1 failed:[/] {result.error}")
        return 1

    bp_path = PROJECT_ROOT / "config" / "channel_page_blueprint.json"
    bp = json.loads(bp_path.read_text(encoding="utf-8"))
    preview = render_preview(settings.brand_kit_dir, bp)

    t = Table(title="Channel page blueprint", show_header=False)
    for k, v in result.summary.items():
        t.add_row(k, str(v))
    t.add_row("cadence", json.dumps(bp["channel_page_layout"]["upload_cadence"]["ramp"]))
    q = db.quota_status()
    t.add_row("API quota today (PT)", f"{q.used}/{q.limit} units")
    t.add_row("elapsed", f"{result.seconds:.1f}s")
    console.print(t)

    files = Table(title="Review these files", show_header=False)
    for label, p in [
        ("channel blueprint", bp_path), ("content blueprint", PROJECT_ROOT / "config" / "content_blueprint.json"),
        ("banner", bp["assets"]["banner"]["path"]), ("avatar", bp["assets"]["avatar"]["path"]),
        ("watermark", bp["assets"]["watermark"]["path"]), ("about text", bp["assets"]["description_file"]),
        ("playlist plan", bp["assets"]["playlist_plan_file"]), ("page preview", preview),
    ]:
        files.add_row(label, str(Path(p).relative_to(PROJECT_ROOT)))
    console.print(files)
    for w in result.warnings:
        console.print(f"[yellow]warning:[/] {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
