#!/usr/bin/env python3
"""KJV Rain Studio - orchestrator for the 5-agent pipeline.

Modes
-----
  python main.py --channel-page-only [--offline]   Agent 1 only: research + blueprints + brand kit
  python main.py --dry-run [--preview 180]         full pipeline locally, writes upload manifests, no YouTube writes
  python main.py                                   live: one production cycle (analytics -> produce -> publish)
  python main.py --daemon [--interval 60]          live, repeating; publishes when the production plan says so
  python main.py --status                          print the dashboard once and exit

Cycle order: Agent 1 (only if blueprints are missing/stale) -> Agent 5 (analytics, cleanup, production plan)
-> Agent 2 + Agent 3 per planned episode -> Agent 4 (resume interrupted uploads, publish, poll processing).

Ctrl-C once: finish the current agent step, then stop.  Ctrl-C twice: abort now (uploads resume next run).
Exit codes: 0 ok, 1 an agent failed, 2 configuration/preflight error, 3 another instance running, 130 aborted.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import logging
import shutil
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from agents.agent1_competitor_intel import CompetitorIntelAgent
from agents.agent2_audio import AudioEngineerAgent
from agents.agent3_visuals import VisualsAgent
from agents.agent4_publisher import PublisherAgent
from agents.agent5_analytics import AnalyticsDirectorAgent
from agents.base import AgentResult, BaseAgent
from core.config import PROJECT_ROOT, Settings, get_settings
from core.ffmpeg import FFmpegMissingError, ffmpeg_bin, ffprobe_bin
from core.growth import STUDIO_ONLY_CHECKLIST
from core.lockfile import InstanceLock, LockHeldError
from core.logging_setup import setup_logging
from db.database import Database, get_db

log = logging.getLogger("orchestrator")
console = Console()

AGENT_ORDER = [
    ("agent1_competitor_intel", "1 Research & Channel Page"),
    ("agent5_analytics", "5 Analytics & Scaling"),
    ("agent2_audio", "2 Scripture & Ambient Audio"),
    ("agent3_visuals", "3 Visuals & Thumbnails"),
    ("agent4_publisher", "4 Publisher & Shorts"),
]
STATE_STYLE = {"idle": ("·", "dim"), "running": ("▶", "bold cyan"), "done": ("✔", "green"),
               "failed": ("✖", "bold red"), "skipped": ("–", "dim yellow")}
TTS_PACKAGES = {"kokoro": "kokoro", "piper": "piper", "chatterbox": "chatterbox"}


class PreflightError(RuntimeError):
    pass


# =============================================================================== dashboard
class Dashboard:
    def __init__(self, settings: Settings, db: Database, mode: str) -> None:
        self.settings, self.db, self.mode = settings, db, mode
        self.lock = threading.Lock()
        self.agents: dict[str, dict[str, Any]] = {k: {"state": "idle", "msg": "", "t0": None, "t1": None}
                                                  for k, _ in AGENT_ORDER}
        self.logs: deque[str] = deque(maxlen=14)
        self.cycle = 0
        self.next_cycle_at: datetime | None = None
        self.stopping = False
        self._db_cache: tuple[float, dict[str, Any]] = (0.0, {})

    def status_cb(self, agent: str, state: str, msg: str) -> None:
        with self.lock:
            a = self.agents.setdefault(agent, {"state": "idle", "msg": "", "t0": None, "t1": None})
            if state == "running" and a["state"] != "running":
                a["t0"], a["t1"] = time.monotonic(), None
            if state in {"done", "failed", "skipped"}:
                a["t1"] = time.monotonic()
            a["state"], a["msg"] = state, msg

    def add_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(line)

    def _db_summary(self) -> dict[str, Any]:
        t, cached = self._db_cache
        if time.monotonic() - t < 2.0 and cached:
            return cached
        try:
            data = self.db.dashboard_summary()
            data["plan"] = self.db.kv_get("production_plan") or {}
            data["brand"] = (self.db.kv_get("brand") or {}).get("name", "(run Agent 1)")
        except Exception as exc:  # noqa: BLE001 - never let the dashboard crash the run
            data = {"error": str(exc)}
        self._db_cache = (time.monotonic(), data)
        return data

    def render(self) -> Layout:
        d = self._db_summary()
        layout = Layout()
        layout.split_column(Layout(name="head", size=3), Layout(name="mid", ratio=3), Layout(name="logs", ratio=2))
        layout["mid"].split_row(Layout(name="agents", ratio=3), Layout(name="side", ratio=2))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tail = " · [yellow]stopping after current step[/]" if self.stopping else ""
        nxt = f" · next cycle {self.next_cycle_at.astimezone().strftime('%H:%M')}" if self.next_cycle_at else ""
        layout["head"].update(Panel(Text.from_markup(
            f"[bold]KJV Rain Studio[/] · {d.get('brand', '')} · mode [bold magenta]{self.mode}[/] · "
            f"cycle {self.cycle}{nxt} · {now}{tail}"), style="on grey7"))
        layout["agents"].update(self._agents_panel())
        layout["side"].update(self._side_panel(d))
        with self.lock:
            logs = "\n".join(self.logs)
        layout["logs"].update(Panel(Text(logs, overflow="ellipsis", no_wrap=True), title="log", border_style="dim"))
        return layout

    def _agents_panel(self) -> Panel:
        t = Table(expand=True, box=None, pad_edge=False)
        t.add_column("", width=2)
        t.add_column("Agent", style="bold", no_wrap=True)
        t.add_column("Time", justify="right", width=8)
        t.add_column("Status", overflow="ellipsis", no_wrap=True, ratio=1)
        with self.lock:
            for key, label in AGENT_ORDER:
                a = self.agents[key]
                icon, style = STATE_STYLE.get(a["state"], ("?", ""))
                el = ""
                if a["t0"]:
                    secs = (a["t1"] or time.monotonic()) - a["t0"]
                    el = f"{int(secs // 60)}:{int(secs % 60):02d}"
                t.add_row(Text(icon, style=style), label, el, Text(a["msg"], style=style if a["state"] in
                                                                     {"failed"} else ""))
        return Panel(t, title="agents", border_style="cyan")

    def _side_panel(self, d: dict[str, Any]) -> Panel:
        if "error" in d:
            return Panel(Text(d["error"], style="red"), title="state")
        q = d["quota"]
        bar = ProgressBar(total=q["limit"], completed=max(0, q["used"]), width=28,
                          complete_style="red" if q["remaining"] < q["reserve"] else "green")
        quota = Table.grid(padding=(0, 1))
        quota.add_row(Text("API quota", style="bold"), bar, Text(f"{q['used']}/{q['limit']}"))
        quota.add_row("", Text(f"day {q['day']} PT · {q['reserve']} reserved for uploads", style="dim"), "")
        pipe = Table.grid(padding=(0, 1))
        eps = d.get("episodes", {})
        ups = d.get("uploads", {})
        pipe.add_row("Episodes", ", ".join(f"{k} {v}" for k, v in sorted(eps.items())) or "none")
        pipe.add_row("Uploads", ", ".join(f"{k} {v}" for k, v in sorted(ups.items())) or "none")
        pipe.add_row("A/B tests", str(d.get("running_ab_tests", 0)))
        free = shutil.disk_usage(self.settings.output_dir).free / 1e9
        pipe.add_row("Local media", f"{d.get('local_media_bytes', 0) / 1e9:.1f} GB · disk free {free:.0f} GB")
        pipe.add_row("Competitors", f"{d.get('competitor_channels', 0)} channels / {d.get('competitor_videos', 0)} videos")
        plan = d.get("plan") or {}
        plan_t = Table.grid(padding=(0, 1))
        if plan:
            plan_t.add_row("Today", f"{plan.get('long_form_today', 0)} long-form + "
                                    f"{plan.get('shorts_per_long', 0)} Shorts each")
            plan_t.add_row("Ramp", f"weeks {plan.get('ramp_tier', {}).get('weeks', '?')} · "
                                   f"{plan.get('published_last_7d', 0)} published in 7 d")
            plan_t.add_row("Why", "; ".join(plan.get("reasons", []))[:160])
        else:
            plan_t.add_row("Plan", "computed by Agent 5")
        sched = self.settings.publish_time and f"{self.settings.publish_time} {self.settings.publish_timezone}" \
            or f"immediate ({self.settings.default_privacy})"
        plan_t.add_row("Publish", sched)
        return Panel(Group(quota, Text(""), pipe, Text(""), plan_t), title="pipeline", border_style="magenta")


class DashboardLogHandler(logging.Handler):
    def __init__(self, dash: Dashboard) -> None:
        super().__init__(logging.INFO)
        self.dash = dash
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.dash.add_log(self.format(record).replace("\n", " ⏎ ")[:300])
        except Exception:  # noqa: BLE001
            self.handleError(record)


# =============================================================================== orchestrator
class Orchestrator:
    def __init__(self, settings: Settings, db: Database, args: argparse.Namespace, dash: Dashboard) -> None:
        self.s, self.db, self.args, self.dash = settings, db, args, dash
        self.stop = threading.Event()
        self.results: list[AgentResult] = []
        self.live = not (args.dry_run or args.channel_page_only)

    # ---------------------------------------------------------------- helpers
    def _kw(self) -> dict[str, Any]:
        return {"dry_run": not self.live, "status_cb": self.dash.status_cb}

    def _run(self, agent: BaseAgent) -> AgentResult:
        if self.stop.is_set():
            self.dash.status_cb(agent.name, "skipped", "stop requested")
            return AgentResult(agent.name, False, error="stopped")
        self.dash.status_cb(agent.name, "running", "starting")
        result = agent.run()
        self.results.append(result)
        for w in result.warnings:
            log.warning("%s: %s", agent.name, w)
        if not result.ok:
            log.error("%s failed: %s", agent.name, result.error)
        return result

    def skip(self, name: str, why: str) -> None:
        self.dash.status_cb(name, "skipped", why)

    # ---------------------------------------------------------------- preflight
    def preflight(self) -> None:
        try:
            ffmpeg_bin()
            ffprobe_bin()
        except FFmpegMissingError as exc:
            raise PreflightError(str(exc)) from exc
        if self.args.channel_page_only:
            return
        engine = self.s.tts_engine
        if engine == "kokoro" and sys.version_info >= (3, 13):
            raise PreflightError("Kokoro needs Python 3.10-3.12 (PyTorch wheels); recreate the venv with python3.11")
        pkg = TTS_PACKAGES.get(engine)
        if pkg and importlib.util.find_spec(pkg) is None:
            raise PreflightError(f"TTS_ENGINE={engine} but package '{pkg}' is not installed (see requirements.txt)")
        if engine == "elevenlabs" and not (self.s.elevenlabs_api_key and self.s.elevenlabs_voice_id):
            raise PreflightError("TTS_ENGINE=elevenlabs needs ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID")
        if self.live:
            if self.s.preview_seconds:
                raise PreflightError("PREVIEW_SECONDS is set - live mode would refuse to upload previews. "
                                     "Unset it or use --dry-run.")
            from core.youtube_auth import AuthError, load_credentials
            try:
                load_credentials(self.s)
            except AuthError as exc:
                raise PreflightError(str(exc)) from exc
            if self.s.default_privacy == "public" and not self.s.publish_time:
                log.warning("DEFAULT_PRIVACY=public: videos go live immediately with no review window")

    def blueprints_stale(self) -> str | None:
        if self.args.refresh_research:
            return "--refresh-research"
        for p in (self.s.channel_blueprint_path, self.s.content_blueprint_path):
            if not p.exists():
                return f"{p.name} missing"
        age_days = (time.time() - self.s.channel_blueprint_path.stat().st_mtime) / 86400
        if self.live and age_days > self.args.research_max_age_days:
            return f"blueprint {age_days:.0f} days old"
        return None

    # ---------------------------------------------------------------- cycle
    def cycle(self) -> bool:
        self.dash.cycle += 1
        ok = True
        why = self.blueprints_stale()
        if self.args.channel_page_only or why:
            if why:
                log.info("Running Agent 1: %s", why)
            r = self._run(CompetitorIntelAgent(self.s, self.db, offline=self.args.offline, **self._kw()))
            if not r.ok or self.args.channel_page_only:
                return r.ok
        else:
            self.skip("agent1_competitor_intel", "blueprints fresh")

        r5 = self._run(AnalyticsDirectorAgent(self.s, self.db, **self._kw()))
        ok &= r5.ok
        plan = self.db.kv_get("production_plan") or {}
        if self.args.episodes is not None:
            n = self.args.episodes
        elif self.s.preview_seconds:
            n = 1
        else:
            n = int(plan.get("long_form_today", 0))
        preview = self.s.preview_seconds or None

        # finish work-in-progress first (audio rendered but no video yet)
        backlog_audio = [e for e in self.db.episodes_by_status("audio_ready")
                         if e["slug"].startswith("preview-") == bool(preview)]
        if backlog_audio:
            ok &= self._run(VisualsAgent(self.s, self.db, preview_seconds=preview, **self._kw())).ok

        if n <= 0:
            reason = "; ".join(plan.get("reasons", [])) or "production plan: 0 today"
            self.skip("agent2_audio", reason)
            if not backlog_audio:
                self.skip("agent3_visuals", "nothing to render")
        for i in range(n):
            if self.stop.is_set():
                break
            log.info("Producing episode %d/%d", i + 1, n)
            r2 = self._run(AudioEngineerAgent(self.s, self.db, preview_seconds=preview, **self._kw()))
            if not r2.ok:
                ok = False
                break
            r3 = self._run(VisualsAgent(self.s, self.db, preview_seconds=preview, **self._kw()))
            if not r3.ok:
                ok = False
                break

        r4 = self._run(PublisherAgent(self.s, self.db, preview_seconds=preview, apply_branding=self.args.apply_branding,
                                      **self._kw()))
        return ok and r4.ok

    def daemon(self) -> bool:
        ok = True
        while not self.stop.is_set():
            ok = self.cycle() and ok
            if self.stop.is_set():
                break
            wake = time.time() + self.args.interval * 60
            self.dash.next_cycle_at = datetime.fromtimestamp(wake, timezone.utc)
            while time.time() < wake and not self.stop.is_set():
                time.sleep(1)
            self.dash.next_cycle_at = None
            for k, _ in AGENT_ORDER:
                self.dash.status_cb(k, "idle", "")
        return ok


# =============================================================================== CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--channel-page-only", action="store_true", help="run Agent 1 only")
    mode.add_argument("--dry-run", action="store_true", help="render locally, write manifests, no YouTube writes")
    mode.add_argument("--daemon", action="store_true", help="live mode, repeat every --interval minutes")
    mode.add_argument("--status", action="store_true", help="print the dashboard once and exit")
    ap.add_argument("--preview", type=int, metavar="SECONDS",
                    help="render short preview episodes (implies --dry-run)")
    ap.add_argument("--episodes", type=int, metavar="N", help="override the production plan's episode count")
    ap.add_argument("--offline", action="store_true", help="Agent 1 without network (baseline priors)")
    ap.add_argument("--refresh-research", action="store_true", help="force Agent 1 to re-run")
    ap.add_argument("--research-max-age-days", type=int, default=14, help="auto re-run Agent 1 after N days")
    ap.add_argument("--apply-branding", action="store_true",
                    help="live: push banner/About/keywords/watermark to the channel via API")
    ap.add_argument("--interval", type=int, default=60, help="daemon cycle interval in minutes (default 60)")
    ap.add_argument("--no-dashboard", action="store_true", help="plain log output (cron/CI)")
    return ap


def print_summary(orch: Orchestrator, show_checklist: bool) -> None:
    t = Table(title="Run summary")
    t.add_column("Agent")
    t.add_column("Result")
    t.add_column("Time", justify="right")
    t.add_column("Details", overflow="fold")
    for r in orch.results:
        detail = r.error or ", ".join(f"{k}={str(v)[:48]}" for k, v in r.summary.items()
                                      if not isinstance(v, (list, dict)) and "/" not in str(v))[:300]
        t.add_row(r.agent, "[green]ok[/]" if r.ok else "[red]failed[/]", f"{r.seconds:.0f}s", detail)
    console.print(t)
    for r in orch.results:
        for m in r.summary.get("manifests", []) if isinstance(r.summary.get("manifests"), list) else []:
            console.print(f"[cyan]upload manifest:[/] {m}")
        for w in r.warnings:
            console.print(f"[yellow]warning ({r.agent}):[/] {w}")
    if show_checklist:
        console.print(Panel("\n".join(f"• {c}" for c in STUDIO_ONLY_CHECKLIST),
                            title="YouTube Studio settings the API cannot set", border_style="yellow"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preview:
        args.dry_run = True
    settings = get_settings()
    if args.preview:
        settings = dataclasses.replace(settings, preview_seconds=args.preview)
    elif args.dry_run is False and settings.preview_seconds and not args.channel_page_only and not args.status:
        console.print("[red]PREVIEW_SECONDS is set in .env; live runs never upload previews. Use --dry-run.[/]")
        return 2
    mode = ("CHANNEL PAGE" if args.channel_page_only else
            f"PREVIEW {args.preview}s (dry-run)" if args.preview else
            "DRY-RUN" if args.dry_run else "STATUS" if args.status else
            "LIVE DAEMON" if args.daemon else "LIVE")
    db = get_db()
    dash = Dashboard(settings, db, mode)

    if args.status:
        setup_logging(settings.log_level)
        console.print(dash.render(), height=34)
        return 0

    use_dash = not args.no_dashboard and console.is_terminal
    setup_logging(settings.log_level, console=console, rich_console=not use_dash,
                  extra_handlers=[DashboardLogHandler(dash)] if use_dash else None)
    orch = Orchestrator(settings, db, args, dash)

    def on_signal(signum: int, _frame: Any) -> None:
        if orch.stop.is_set():
            raise KeyboardInterrupt
        orch.stop.set()
        dash.stopping = True
        log.warning("Stop requested - finishing the current step (press Ctrl-C again to abort now)")

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)

    try:
        with InstanceLock(PROJECT_ROOT / "db" / "orchestrator.lock"):
            orch.preflight()
            if use_dash:
                with Live(get_renderable=dash.render, console=console, refresh_per_second=4, screen=False,
                          transient=False):
                    ok = orch.daemon() if args.daemon else orch.cycle()
            else:
                ok = orch.daemon() if args.daemon else orch.cycle()
    except LockHeldError as exc:
        console.print(f"[red]{exc}[/]")
        return 3
    except PreflightError as exc:
        console.print(f"[red]Preflight failed:[/] {exc}")
        return 2
    except KeyboardInterrupt:
        console.print("[red]Aborted.[/] Interrupted uploads resume automatically on the next run.")
        print_summary(orch, False)
        return 130

    if orch.stop.is_set():
        console.print("[yellow]Stopped by request after the current step.[/] Re-run to continue; "
                      "work in progress resumes automatically.")
        print_summary(orch, False)
        return 130
    today = datetime.now().date().isoformat()
    show = args.channel_page_only or args.dry_run or db.kv_get("checklist_shown") != today
    if show and not args.dry_run:
        db.kv_set("checklist_shown", today)
    print_summary(orch, show)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
