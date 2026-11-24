# KJV Rain Studio: 5-agent 8-10 hour sleep-scripture pipeline

| Phase | Status |
|---|---|
| 1. Scaffold, config, `build_channel_page.py` (Agent 1) | done |
| 2. SQLite state, uniqueness and quota DB (`db/database.py`) | done |
| 3. Agents 2-5 (audio, visuals and thumbnails, publisher, analytics) | done |
| 4. `main.py` orchestrator with Rich dashboard | done |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add YOUTUBE_API_KEY (optional: yt-dlp is the fallback)
```
FFmpeg 6+ must be on PATH.

## Run
```bash
python main.py --channel-page-only          # Agent 1: research, blueprints, brand kit
python main.py --dry-run --preview 180      # whole pipeline on a 3-min episode, no YouTube writes
python main.py --dry-run                    # full 8-10 h render + upload_manifest.json, no YouTube writes
python -m core.youtube_auth                 # one-time OAuth
python main.py --apply-branding             # live cycle + push banner/About/keywords/watermark
python main.py --daemon --interval 60       # live, hourly cycles; production follows Agent 5's plan
python main.py --status                     # dashboard snapshot
```
Each cycle runs Agent 1 only if the blueprints are missing or older than 14 days, then Agent 5 (analytics, cleanup, today's plan), then Agent 2 and Agent 3 for each planned episode, then Agent 4 (resume interrupted uploads, publish, poll HD processing).

Press Ctrl-C once to finish the current step and stop; press it twice to abort. Only one orchestrator can run at a time. Use `--no-dashboard` for cron.

Exit codes: 0 ok · 1 agent failed · 2 preflight · 3 already running · 130 stopped.

**Episodes from a non-preview `--dry-run` are real.** They get a full-length render and count toward rotation, and the next live run uploads them. Use `--preview` for throwaway checks.

## Publishing settings for growth
- **`PUBLISH_TIME` + `PUBLISH_TIMEZONE`:** each episode uploads as private and YouTube makes it public at the same local time every day. Shorts go public 3, 10 and 17 hours later. Choose the time from Studio > Audience > "When your viewers are on YouTube".
- **Subscriber notifications:** on for long-form, off for Shorts (`NOTIFY_SUBSCRIBERS_*`).
- **`render.output_resolution: [2560, 1440]`:** YouTube then serves its VP9/AV1 encode, which means less banding in dark scenes.
- **`--apply-branding`:** sets the banner, About text, keywords and language, and adds a watermark shown only for the first 2 minutes.
- **Studio-only settings:** `main.py` prints the checklist (mid-roll ads, avatar, trailer and layout, Shorts related video, end screens, comment pinning).

## Review the channel page first
```bash
python build_channel_page.py            # live research
python build_channel_page.py --offline  # brand kit from baseline priors, no network
```
Outputs:
- `config/channel_page_blueprint.json`: brand, layout, playlists, competitor patterns, setup steps
- `config/content_blueprint.json`: episode spec, rotation pools, anti-repetition rules
- `channel_brand_kit/`: banner (2560x1440), avatar (800x800), watermark (150x150), About text, keywords, playlist plan, page preview

`data_source` in each blueprint shows whether it came from `live` data or `baseline_prior` data. Don't act on prior-only blueprints.

## Agents 2-5
| Agent | Module | Output |
|---|---|---|
| 2 Audio | `agents/agent2_audio.py` | `output/episodes/<slug>/master.m4a`, verse timeline, chapters |
| 3 Visuals | `agents/agent3_visuals.py` | seamless loop, final 8-10 h MP4 (stream-copied), Shorts loop, A/B thumbnails |
| 4 Publisher | `agents/agent4_publisher.py` | resumable upload, thumbnail, playlists, comment, 3 Shorts; `upload_manifest.json` in dry-run |
| 5 Analytics | `agents/agent5_analytics.py` | snapshots, A/B rotation, blueprint weights, disk cleanup, daily production plan |

**One-time YouTube auth** (Desktop OAuth client JSON saved at `config/client_secret.json`):
```bash
python -m core.youtube_auth
```

**Optional real recordings and footage.** Procedural rain and scenes are used when these folders are empty.
- `assets/ambient/<bed_id>/*.wav|flac`: rain recordings, at least 30 s each. Segments are picked at random and joined with 8 s crossfades.
- `assets/visuals/<scene_id>/*.mp4`: footage. It's made into a seamless loop and darkened to the target brightness.

Only use recordings and footage that you own or that are CC0/licensed for commercial use.

**Preview without spending hours.** Set `PREVIEW_SECONDS=180` to render 3-minute episodes through the full pipeline. Preview episodes never consume rotation history, and Agent 4 refuses to upload them.
```bash
python -m tests.smoke_preview 180   # Agents 2-4 dry-run on a scratch DB with a test voice
```

**Per-episode cost on a 4-core CPU with libx264.** The master mix runs at about 22x realtime, so a 10 h episode takes about 27 min. The 2-minute visual loop takes about 1 min, and the final mux is a stream copy that finishes in minutes. A 10 h MP4 is about 5 GB.

## Quota
Research costs about 600 units of `search.list` plus about 100 more units of other reads, out of the default 10,000 per day. The DB pre-charges each call and keeps `YOUTUBE_QUOTA_RESERVE` units back that only uploads (1,600 each) can use. The quota day resets at midnight Pacific.

A full release (1 long-form video + 3 Shorts) costs about 6,700 units, so the default 10,000 units/day allows **one release per day**. To publish more Shorts or more episodes per day, request a quota increase. Agent 5's production plan cuts Shorts automatically when quota is short.

## Tests
```bash
python -m unittest discover -s tests -v
```
