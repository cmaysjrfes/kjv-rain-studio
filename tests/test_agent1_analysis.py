"""Agent 1 analysis on a synthetic competitor dataset (no network).  Run: python -m unittest -v"""
from __future__ import annotations

import unittest

from agents.agent1_competitor_intel import (BrandDesigner, PatternAnalyzer, build_content_blueprint,
                                            build_title_templates, cadence_from)
from core.scraper import CompetitorDataset, median_gap_days


def _ds() -> CompetitorDataset:
    titles = [
        ("Psalm 91 KJV | Bible Verses for Sleep with Rain | 8 Hours Dark Screen", 900_000, "UC1"),
        ("10 Hours KJV Scripture & Thunder for Peaceful Sleep | Black Screen", 450_000, "UC2"),
        ("Fall Asleep to the Gospel of John KJV | Rain Sounds 8 Hours", 300_000, "UC1"),
        ("Bible for Sleep with Rain | Psalms KJV | 8 Hours", 120_000, "UC3"),
        ("Gentle Rain Bible Verses for Sleep | KJV Proverbs | 9 Hours", 80_000, "UC2"),
    ]
    vids = [{"video_id": f"v{i}", "channel_id": c, "title": t, "view_count": v, "duration_s": 28800 + i,
             "description": "Sleep with God's Word.\n00:00 Intro\n10:00 Psalm 1\n20:00 Psalm 2\n#kjv #sleep",
             "has_chapters": True, "thumbnail_url": None, "published_at": f"2026-0{i+1}-01T00:00:00Z"}
            for i, (t, v, c) in enumerate(titles)]
    chans = [{"channel_id": c, "title": n, "description": "KJV bible for sleep. New videos daily.",
              "playlists": [{"title": "Psalms for Sleep"}], "uploads": vids, "long_video_count": 5,
              "median_upload_gap_days": g, "score": 1.0, "subscriber_count": 1000}
             for c, n, g in [("UC1", "Sleepy Scripture Rain", 1.5), ("UC2", "Bible Night Rain", 3.0),
                             ("UC3", "KJV Storm Sleep", 2.0)]]
    return CompetitorDataset(channels=chans, videos=vids)


class Agent1AnalysisTests(unittest.TestCase):
    def test_title_patterns(self) -> None:
        tp = PatternAnalyzer(_ds()).title_patterns()
        self.assertEqual(tp["sample_size"], 5)
        self.assertEqual(tp["features"]["kjv"]["prevalence"], 1.0)
        self.assertEqual(tp["features"]["duration_cue"]["prevalence"], 1.0)
        self.assertIn(tp["duration_cue_position"], {"front", "middle", "end"})
        phrases = {p["phrase"] for p in tp["top_view_weighted_phrases"]}
        self.assertIn("bible verses", phrases)  # used by 2 channels -> genre vocabulary
        for t in build_title_templates(tp):
            self.assertIn("{hours}", t["pattern"])

    def test_description_and_channel_patterns(self) -> None:
        pa = PatternAnalyzer(_ds())
        dp = pa.description_patterns()
        self.assertEqual(dp["share_with_chapters"], 1.0)
        self.assertIn("#kjv", dp["common_hashtags"])
        cp = pa.channel_patterns()
        self.assertEqual(cp["upload_gap_days_median"], 2.0)
        self.assertEqual(cadence_from(cp)["max_long_form_per_day"], 1)

    def test_brand_distinctiveness(self) -> None:
        bd = BrandDesigner({"name_candidates": ["Bible Night Rain TV", "Cedar Cabin Scripture"],
                            "palettes": {"a": ["#0B0A0F", "#1C1820", "#E0A458", "#F3EBDD"],
                                         "b": ["#07090D", "#161C26", "#8FB3D9", "#E8EEF5"]}},
                           competitor_titles=["Bible Night Rain"], competitor_colors=["#E0A050", "#D9A55A"])
        self.assertEqual(bd.choose_name()["name"], "Cedar Cabin Scripture")
        self.assertEqual(bd.choose_palette()["selected"]["id"], "b")  # amber accent too close to competitors

    def test_median_gap(self) -> None:
        self.assertEqual(median_gap_days(["2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z",
                                          "2026-01-04T00:00:00Z"]), 1.5)

    def test_content_blueprint_contract(self) -> None:
        pa = PatternAnalyzer(_ds())
        analysis = {"titles": pa.title_patterns(), "descriptions": pa.description_patterns(),
                    "visuals": {"thumbnails": {"mean_luma": 35}}, "channels": pa.channel_patterns(),
                    "data_source": "live"}
        bp = build_content_blueprint(analysis, {"name": "X", "palette": {}}, {}, "", "")
        self.assertEqual(bp["episode_spec"]["duration_s"], {"min": 28800, "max": 36000})
        self.assertEqual(bp["episode_spec"]["narration"]["window_s"]["end"], 9000)
        self.assertGreaterEqual(len(bp["rotation_pools"]["passage_sets"]), 20)


if __name__ == "__main__":
    unittest.main()
