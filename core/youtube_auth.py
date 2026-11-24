"""OAuth2 for uploads/management/analytics. Run once interactively:  python -m core.youtube_auth"""
from __future__ import annotations

import logging
import sys

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from core.config import Settings, get_settings

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


class AuthError(RuntimeError):
    pass


def load_credentials(settings: Settings, *, interactive: bool = False) -> Credentials:
    token = settings.token_file
    creds: Credentials | None = None
    if token.exists():
        creds = Credentials.from_authorized_user_file(str(token), SCOPES)
        missing = set(SCOPES) - set(creds.scopes or SCOPES)
        if missing:
            log.warning("Stored token lacks scopes %s; re-authorisation required", sorted(missing))
            creds = None
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save(creds, settings)
            return creds
        except RefreshError as exc:
            log.warning("Token refresh failed (%s); re-authorisation required", exc)
    if not interactive:
        raise AuthError("No valid YouTube OAuth token. Run once:  python -m core.youtube_auth")
    if not settings.client_secrets_file.exists():
        raise AuthError(f"OAuth client file missing: {settings.client_secrets_file} "
                        "(Google Cloud Console > Credentials > OAuth client ID > Desktop app > Download JSON)")
    flow = InstalledAppFlow.from_client_secrets_file(str(settings.client_secrets_file), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True, prompt="consent", access_type="offline")
    _save(creds, settings)
    return creds


def _save(creds: Credentials, settings: Settings) -> None:
    settings.token_file.parent.mkdir(parents=True, exist_ok=True)
    settings.token_file.write_text(creds.to_json(), encoding="utf-8")
    try:
        settings.token_file.chmod(0o600)
    except OSError:
        pass


if __name__ == "__main__":
    from googleapiclient.discovery import build

    s = get_settings()
    c = load_credentials(s, interactive=True)
    items = build("youtube", "v3", credentials=c, cache_discovery=False).channels().list(
        part="snippet", mine=True).execute().get("items", [])
    print("Token saved to", s.token_file)
    if not items:
        print("WARNING: this Google account has no YouTube channel. Create one at youtube.com first.")
        sys.exit(1)
    ch = items[0]
    print(f"Authorised channel: {ch['snippet']['title']}  (id {ch['id']})")
    if s.expected_channel_id and s.expected_channel_id != ch["id"]:
        print(f"MISMATCH: YOUTUBE_CHANNEL_ID={s.expected_channel_id}. Delete {s.token_file} and re-run, "
              "choosing the correct channel on the consent screen.")
        sys.exit(1)
    if not s.expected_channel_id:
        print(f"Add to .env to lock publishing to this channel:  YOUTUBE_CHANNEL_ID={ch['id']}")
    sys.exit(0)
