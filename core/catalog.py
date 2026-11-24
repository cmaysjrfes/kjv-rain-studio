"""Static rotation pools shared by Agents 1-3: KJV passage sets, ambient beds, scenes, thumbnail layouts.

The King James Version is in the public domain in the United States (and most of the
world). In the UK it remains under Crown letters patent - review before targeting UK
commercial use.
"""
from __future__ import annotations

from typing import Any

BIBLE_BOOKS: list[str] = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", "1 Samuel",
    "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
    "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations", "Ezekiel",
    "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah",
    "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians",
    "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James", "1 Peter",
    "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation",
]

# Each set is one episode's narration (~2-2.5 h at slow sleep pacing). `segments` are
# (book, first_chapter, last_chapter). `theme` feeds title hooks; `series` maps to playlists.
PASSAGE_SETS: list[dict[str, Any]] = [
    {"id": "psa-001-024", "segments": [("Psalms", 1, 24)], "theme": "The Lord Is My Shepherd", "series": "psalms"},
    {"id": "psa-025-041", "segments": [("Psalms", 25, 41)], "theme": "Wait on the Lord", "series": "psalms"},
    {"id": "psa-042-058", "segments": [("Psalms", 42, 58)], "theme": "Be Still and Know", "series": "psalms"},
    {"id": "psa-059-077", "segments": [("Psalms", 59, 77)], "theme": "My Refuge in the Night", "series": "psalms"},
    {"id": "psa-078-089", "segments": [("Psalms", 78, 89)], "theme": "Mercies Forever", "series": "psalms"},
    {"id": "psa-090-106", "segments": [("Psalms", 90, 106)], "theme": "Under His Wings", "series": "psalms"},
    {"id": "psa-107-119", "segments": [("Psalms", 107, 119)], "theme": "A Lamp Unto My Feet", "series": "psalms"},
    {"id": "psa-120-150", "segments": [("Psalms", 120, 150)], "theme": "Songs of Ascent", "series": "psalms"},
    {"id": "pro-001-015", "segments": [("Proverbs", 1, 15)], "theme": "The Beginning of Wisdom", "series": "wisdom"},
    {"id": "pro-016-031", "segments": [("Proverbs", 16, 31)], "theme": "Words of the Wise", "series": "wisdom"},
    {"id": "ecc-job", "segments": [("Ecclesiastes", 1, 12), ("Job", 38, 42)], "theme": "A Time for Everything",
     "series": "wisdom"},
    {"id": "isa-040-052", "segments": [("Isaiah", 40, 52)], "theme": "Comfort Ye My People", "series": "promises"},
    {"id": "isa-053-066", "segments": [("Isaiah", 53, 66)], "theme": "Peace Like a River", "series": "promises"},
    {"id": "mat-005-014", "segments": [("Matthew", 5, 14)], "theme": "Come Unto Me and Rest", "series": "gospels"},
    {"id": "mat-015-028", "segments": [("Matthew", 15, 28)], "theme": "Lo, I Am With You Always", "series": "gospels"},
    {"id": "mrk-001-016", "segments": [("Mark", 1, 16)], "theme": "Peace, Be Still", "series": "gospels"},
    {"id": "luk-001-012", "segments": [("Luke", 1, 12)], "theme": "Good Tidings of Great Joy", "series": "gospels"},
    {"id": "luk-013-024", "segments": [("Luke", 13, 24)], "theme": "The Lost Is Found", "series": "gospels"},
    {"id": "jhn-001-010", "segments": [("John", 1, 10)], "theme": "In the Beginning Was the Word", "series": "gospels"},
    {"id": "jhn-011-021", "segments": [("John", 11, 21)], "theme": "Let Not Your Heart Be Troubled",
     "series": "gospels"},
    {"id": "rom-001-016", "segments": [("Romans", 1, 16)], "theme": "Nothing Can Separate Us", "series": "letters"},
    {"id": "eph-col", "segments": [("Ephesians", 1, 6), ("Philippians", 1, 4), ("Colossians", 1, 4)],
     "theme": "The Peace That Passeth Understanding", "series": "letters"},
    {"id": "heb-001-013", "segments": [("Hebrews", 1, 13)], "theme": "Faith, the Substance of Hope",
     "series": "letters"},
    {"id": "jas-1pe-1jn", "segments": [("James", 1, 5), ("1 Peter", 1, 5), ("1 John", 1, 5)],
     "theme": "God Is Love", "series": "letters"},
    {"id": "gen-001-011", "segments": [("Genesis", 1, 11)], "theme": "In the Beginning", "series": "stories"},
    {"id": "gen-037-050", "segments": [("Genesis", 37, 50)], "theme": "The Story of Joseph", "series": "stories"},
    {"id": "rut-jon-est", "segments": [("Ruth", 1, 4), ("Jonah", 1, 4), ("Esther", 1, 10)],
     "theme": "Quiet Stories of Faithfulness", "series": "stories"},
    {"id": "rev-019-022", "segments": [("Revelation", 19, 22), ("Isaiah", 35, 35), ("Psalms", 23, 23)],
     "theme": "All Things New", "series": "promises"},
]

# Ambient beds: Agent 2 synthesises/blends these with per-episode jitter inside `jitter`.
AMBIENT_BEDS: list[dict[str, Any]] = [
    {"id": "light_rain_window", "label": "Light Rain on the Window", "layers": ["pink_rain_fine", "glass_drips"],
     "thunder_per_hour": 0, "wind": 0.0},
    {"id": "steady_rainfall", "label": "Steady Rainfall", "layers": ["pink_rain_dense", "roof_patter"],
     "thunder_per_hour": 0, "wind": 0.05},
    {"id": "distant_soft_thunder", "label": "Rain & Distant Soft Thunder",
     "layers": ["pink_rain_dense", "low_rumble"], "thunder_per_hour": 5, "wind": 0.1},
    {"id": "cabin_wind_rain", "label": "Cabin Rain & Gentle Wind", "layers": ["brown_wind", "pink_rain_fine"],
     "thunder_per_hour": 1, "wind": 0.35},
    {"id": "tin_roof_rain", "label": "Rain on a Tin Roof", "layers": ["roof_patter", "pink_rain_fine"],
     "thunder_per_hour": 0, "wind": 0.05},
    {"id": "forest_canopy_rain", "label": "Forest Rain Under the Trees",
     "layers": ["pink_rain_dense", "leaf_drips"], "thunder_per_hour": 2, "wind": 0.15},
    {"id": "heavy_night_storm_far", "label": "Far-Off Night Storm", "layers": ["brown_rain_heavy", "low_rumble"],
     "thunder_per_hour": 8, "wind": 0.2},
    {"id": "rain_on_tent", "label": "Rain on a Canvas Tent", "layers": ["canvas_patter", "pink_rain_fine"],
     "thunder_per_hour": 1, "wind": 0.1},
]
AMBIENT_JITTER: dict[str, tuple[float, float]] = {
    "rain_lowpass_hz": (2800.0, 5200.0), "rain_highpass_hz": (120.0, 320.0), "density": (0.75, 1.25),
    "stereo_width": (0.6, 1.0), "thunder_rate_scale": (0.6, 1.4), "wind_lfo_hz": (0.03, 0.11),
}

VISUAL_SCENES: list[dict[str, Any]] = [
    {"id": "rain_on_glass_night", "label": "Rain on Night Glass", "base": "#05070B", "glow": "#2B3A55"},
    {"id": "dim_cabin_lantern", "label": "Dim Cabin Lantern", "base": "#080605", "glow": "#7A4A1E"},
    {"id": "candle_by_window", "label": "Candle by the Window", "base": "#070506", "glow": "#8A5A2A"},
    {"id": "misty_forest_night", "label": "Misty Forest at Night", "base": "#050806", "glow": "#1F3326"},
    {"id": "porch_lamp_rain", "label": "Porch Lamp in the Rain", "base": "#060606", "glow": "#6B5530"},
    {"id": "chapel_window_moon", "label": "Moonlit Chapel Window", "base": "#06060A", "glow": "#3A3F5C"},
    {"id": "open_bible_lamplight", "label": "Open Bible by Lamplight", "base": "#070504", "glow": "#6E4B22"},
    {"id": "fireplace_embers_low", "label": "Low Fireplace Embers", "base": "#070403", "glow": "#5E2A12"},
]
VISUAL_JITTER: dict[str, tuple[float, float]] = {
    "brightness": (0.05, 0.12), "drift_speed": (0.2, 0.6), "rain_density": (0.5, 1.5), "vignette": (0.55, 0.85),
    "glow_x": (0.2, 0.8), "glow_y": (0.4, 0.75), "hue_shift_deg": (-10.0, 10.0),
}

THUMBNAIL_TEMPLATES: list[dict[str, Any]] = [
    {"id": "T1_split_scene_text", "layout": "scene left 55%, stacked title right, hours badge top-right"},
    {"id": "T2_center_title_band", "layout": "full-bleed scene, centred 2-line title on dark band, hours badge"},
    {"id": "T3_big_hours_numeral", "layout": "giant '8' / '10' numeral left, book + KJV right, rain overlay"},
    {"id": "T4_verse_card", "layout": "short verse quote in serif on lantern-lit card, book ref small caps"},
    {"id": "T5_minimal_book", "layout": "book name huge, thin rule, 'KJV • RAIN • SLEEP' tracking line"},
    {"id": "T6_lower_third", "layout": "scene with title in lower third, accent underline, flame emblem"},
]
