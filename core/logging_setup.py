"""Rich-backed logging with a rotating file sink."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from rich.console import Console
from rich.logging import RichHandler

from core.config import PROJECT_ROOT

_CONFIGURED = False


def setup_logging(level: str = "INFO", *, console: Console | None = None, rich_console: bool = True,
                  extra_handlers: list[logging.Handler] | None = None) -> None:
    """File log always; Rich console log unless a live dashboard owns the terminal."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(log_dir / "studio.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    handlers: list[logging.Handler] = [file_handler, *(extra_handlers or [])]
    if rich_console:
        handlers.append(RichHandler(console=console, rich_tracebacks=True, show_path=False))
    logging.basicConfig(level=level, format="%(message)s", datefmt="[%X]", handlers=handlers, force=True)
    for noisy in ("googleapiclient.discovery_cache", "urllib3", "PIL", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True
