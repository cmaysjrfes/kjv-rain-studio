"""Episode planning: picks the next passage/bed/scene/templates under the anti-repetition rules."""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.kjv import display_ref
from db.database import Database, DuplicateContentError, sha256_fingerprint

log = logging.getLogger(__name__)


class PlanningError(RuntimeError):
    pass


@dataclass
class EpisodePlan:
    slug: str
    passage_set: dict[str, Any]
    bed: dict[str, Any]
    scene: dict[str, Any]
    title_template: dict[str, Any]
    thumb_templates: list[str]
    hours: int
    duration_s: int
    seed: int
    ambient_params: dict[str, float] = field(default_factory=dict)
    visual_params: dict[str, float] = field(default_factory=dict)
    benefit_hook: str = ""
    preview: bool = False

    @property
    def book(self) -> str:
        return self.passage_set["segments"][0][0]

    @property
    def book_ref(self) -> str:
        return display_ref(self.passage_set["segments"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EpisodePlan:
        return cls(**d)


def load_blueprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PlanningError(f"{path} missing - run build_channel_page.py (Agent 1) first")
    return json.loads(path.read_text(encoding="utf-8"))


def _jitter(ranges: dict[str, list[float] | tuple[float, float]], rng: random.Random) -> dict[str, float]:
    return {k: round(rng.uniform(float(lo), float(hi)), 2) for k, (lo, hi) in ranges.items()}


def _weighted_choice(items: list[dict[str, Any]], weights: dict[str, float], key: str,
                     rng: random.Random) -> dict[str, Any]:
    w = [max(0.05, float(weights.get(it[key], 1.0))) for it in items]
    return rng.choices(items, weights=w, k=1)[0]


class EpisodePlanner:
    def __init__(self, db: Database, blueprint: dict[str, Any]) -> None:
        self.db = db
        self.bp = blueprint
        self.pools = blueprint["rotation_pools"]
        self.gaps = blueprint["anti_repetition_rules"]["min_episodes_between"]
        self.perf = blueprint.get("performance_weights", {})

    def _allowed(self, column: str, gap_key: str, values: list[str], relax: int) -> set[str]:
        gap = max(0, int(self.gaps.get(gap_key, 0)) - relax)
        blocked = set(self.db.recent_episode_values(column, gap)) if gap else set()
        allowed = {v for v in values if v not in blocked}
        return allowed or set(values)

    def plan_next(self, *, seed: int | None = None, preview: bool = False) -> tuple[int, EpisodePlan]:
        seed = seed if seed is not None else random.SystemRandom().randrange(2**31)
        rng = random.Random(seed)
        sets = self.pools["passage_sets"]
        used = self.db.used_scripture_refs()
        fresh = [p for p in sets if p["id"] not in used]

        for attempt in range(300):
            relax = attempt // 60  # progressively relax gaps if pools are exhausted
            books_ok = self._allowed("scripture_book", "book", [p["segments"][0][0] for p in sets], relax)
            pool = [p for p in (fresh or sets) if p["segments"][0][0] in books_ok] or (fresh or sets)
            pset = _weighted_choice(pool, self.perf.get("series", {}), "series", rng)

            beds_ok = self._allowed("ambient_bed", "ambient_bed", [b["id"] for b in self.pools["ambient_beds"]],
                                    relax)
            bed = _weighted_choice([b for b in self.pools["ambient_beds"] if b["id"] in beds_ok],
                                   self.perf.get("ambient_bed", {}), "id", rng)
            scenes_ok = self._allowed("visual_scene", "visual_scene",
                                      [s["id"] for s in self.pools["visual_scenes"]], relax)
            scene = _weighted_choice([s for s in self.pools["visual_scenes"] if s["id"] in scenes_ok],
                                     self.perf.get("visual_scene", {}), "id", rng)
            tts_ok = self._allowed("title_template", "title_template",
                                   [t["id"] for t in self.pools["title_templates"]], relax)
            title_t = rng.choice([t for t in self.pools["title_templates"] if t["id"] in tts_ok])
            thumbs_ok = sorted(self._allowed("thumb_template", "thumbnail_template",
                                             [t["id"] for t in self.pools["thumbnail_templates"]], relax))
            rng.shuffle(thumbs_ok)
            others = [t["id"] for t in self.pools["thumbnail_templates"] if t["id"] != thumbs_ok[0]]
            thumb_ids = [thumbs_ok[0], rng.choice(others)]

            combo = sha256_fingerprint(pset["id"], bed["id"], scene["id"], thumb_ids[0])
            if not self.db.is_unique("combo", combo):
                continue

            hours = rng.choice([8, 9, 10])
            dmin, dmax = self.bp["episode_spec"]["duration_s"]["min"], self.bp["episode_spec"]["duration_s"]["max"]
            base = min(max(hours * 3600, dmin), dmax)
            duration = base + rng.randint(0, max(0, min(240, dmax - base)))
            ambient = {**_jitter(self.pools["ambient_jitter"], rng), "seed": rng.randrange(2**31)}
            visual = {**_jitter(self.pools["visual_jitter"], rng), "seed": rng.randrange(2**31)}
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            slug = f"{'preview-' if preview else ''}{stamp}-{pset['id']}-{bed['id'][:12]}"
            plan = EpisodePlan(
                slug=slug, passage_set=pset, bed=bed, scene=scene, title_template=title_t,
                thumb_templates=thumb_ids, hours=hours, duration_s=duration, seed=seed,
                ambient_params=ambient, visual_params=visual,
                benefit_hook=rng.choice(self.pools.get("benefit_hooks") or ["Peaceful Sleep"]), preview=preview,
            )
            ep_id = self.db.create_episode(
                slug, status="planned", scripture_book=plan.book, scripture_ref=pset["id"], ambient_bed=bed["id"],
                visual_scene=scene["id"], title_template=title_t["id"], thumb_template=thumb_ids[0],
                duration_s=duration, metadata={"plan": plan.to_dict()},
            )
            if not preview:
                try:
                    self.db.register_fingerprint("combo", combo, episode_id=ep_id)
                    self.db.register_fingerprint("ambient_mix", sha256_fingerprint(bed["id"], ambient),
                                                 episode_id=ep_id)
                    self.db.register_fingerprint("visual_loop", sha256_fingerprint(scene["id"], visual),
                                                 episode_id=ep_id)
                except DuplicateContentError as exc:
                    log.info("plan collision, retrying: %s", exc)
                    self.db.release_fingerprints(ep_id)
                    self.db.update_episode(ep_id, status="archived")
                    continue
            if relax:
                log.warning("Rotation gaps relaxed by %d to find a unique plan - add pool variety", relax)
            log.info("Planned episode %s: %s / %s / %s / %dh", slug, pset["id"], bed["id"], scene["id"], hours)
            return ep_id, plan
        raise PlanningError("Could not find a unique episode combination - extend rotation pools")


def plan_for_episode(db: Database, episode_id: int) -> EpisodePlan:
    ep = db.get_episode(episode_id)
    if not ep:
        raise PlanningError(f"episode {episode_id} not found")
    return EpisodePlan.from_dict(json.loads(ep["metadata_json"])["plan"])


def episode_meta(db: Database, episode_id: int) -> dict[str, Any]:
    ep = db.get_episode(episode_id)
    return json.loads(ep["metadata_json"]) if ep and ep.get("metadata_json") else {}
