"""Growth settings that the YouTube Data API cannot set - printed by main.py and kept in the blueprint."""
from __future__ import annotations

STUDIO_ONLY_CHECKLIST: list[str] = [
    "Create the channel yourself (the API cannot) and verify it at youtube.com/verify for >15 min uploads; "
    "then set YOUTUBE_CHANNEL_ID in .env.",
    "Monetization > Ad settings: turn OFF automatic mid-roll ads. If you want any, place 1-2 manually in the "
    "first ~30 min while viewers are still awake - mid-rolls later wake sleepers and cut watch time.",
    "Customization > Branding: upload channel_brand_kit/avatar_800x800.png (profile pictures can't be set via API).",
    "Customization > Layout: set a channel trailer for new visitors and 'featured video' for returning subscribers; "
    "add playlist shelves in the order given in channel_page_blueprint.json.",
    "Each Short: Studio > Content > Shorts > 'Related video' -> the full-length episode (not settable via API).",
    "Long-form end screen: add 'Best for viewer' + subscribe element on the final 20 s (Studio only).",
    "Settings > Upload defaults: comments 'Hold potentially inappropriate comments for review'.",
    "Settings > Channel > Advanced: country of residence set (affects RPM reporting / eligibility).",
    "Pin the funnel comment Agent 4 posts on each video (pinning isn't available via API).",
]
