"""Public-domain King James Version text: download once, validate, cache, and prepare for speech."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import requests

from core.catalog import BIBLE_BOOKS
from core.config import PROJECT_ROOT

log = logging.getLogger(__name__)

KJV_SOURCES = [
    # 66 books in canonical order: [{"abbrev": "gn", "chapters": [[verse, ...], ...]}, ...]
    "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json",
]
CACHE_PATH = PROJECT_ROOT / "assets" / "kjv" / "kjv_normalized.json"
EXPECTED_CHAPTERS = {"Genesis": 50, "Psalms": 150, "Isaiah": 66, "Matthew": 28, "John": 21, "Revelation": 22}

ORDINALS = {"1": "First", "2": "Second", "3": "Third"}
SPEECH_FIXES = [
    (re.compile(r"\bLORD\b"), "Lord"), (re.compile(r"\bGOD\b"), "God"), (re.compile(r"\bJAH\b"), "Jah"),
    (re.compile(r"\bJEHOVAH\b"), "Jehovah"), (re.compile(r"\bI AM\b"), "I am"),
    (re.compile(r"[¶{}\[\]]"), ""), (re.compile(r"\s+"), " "),
]


@dataclass(frozen=True)
class Verse:
    book: str
    chapter: int
    verse: int
    text: str

    @property
    def ref(self) -> str:
        return f"{self.book} {self.chapter}:{self.verse}"


class KJVBible:
    def __init__(self, data: dict[str, list[list[str]]]) -> None:
        self.books = data
        self._validate()

    @classmethod
    def load(cls, cache_path: Path = CACHE_PATH, *, timeout: int = 60) -> KJVBible:
        if cache_path.exists():
            return cls(json.loads(cache_path.read_text(encoding="utf-8"))["books"])
        last_exc: Exception | None = None
        for url in KJV_SOURCES:
            try:
                log.info("Downloading KJV text from %s", url)
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                raw = json.loads(resp.content.decode("utf-8-sig"))
                books = {name: [[_clean(v) for v in ch] for ch in b["chapters"]]
                         for name, b in zip(BIBLE_BOOKS, raw, strict=True)}
                bible = cls(books)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {"source": url, "sha256": hashlib.sha256(resp.content).hexdigest(), "books": books}
                cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return bible
            except Exception as exc:  # noqa: BLE001 - try next source
                last_exc = exc
                log.warning("KJV source failed %s: %s", url, exc)
        raise RuntimeError(f"Could not obtain KJV text (place it at {cache_path}): {last_exc}")

    def _validate(self) -> None:
        if len(self.books) != 66:
            raise ValueError(f"KJV must have 66 books, got {len(self.books)}")
        for book, n in EXPECTED_CHAPTERS.items():
            if len(self.books.get(book, [])) != n:
                raise ValueError(f"KJV validation failed: {book} should have {n} chapters")

    def chapter(self, book: str, chapter: int) -> list[Verse]:
        chs = self.books[book]
        if not 1 <= chapter <= len(chs):
            raise ValueError(f"{book} has {len(chs)} chapters, asked for {chapter}")
        return [Verse(book, chapter, i + 1, t) for i, t in enumerate(chs[chapter - 1]) if t]

    def passage(self, book: str, first: int, last: int) -> list[list[Verse]]:
        return [self.chapter(book, c) for c in range(first, last + 1)]


def _clean(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", t).strip()


def speech_text(text: str) -> str:
    """Display text -> TTS-friendly text (all-caps divine names read as words, no markup)."""
    for rx, rep in SPEECH_FIXES:
        text = rx.sub(rep, text)
    return text.strip()


def spoken_book(book: str) -> str:
    num, _, rest = book.partition(" ")
    return f"{ORDINALS[num]} {rest}" if num in ORDINALS and rest else book


def chapter_heading(book: str, chapter: int) -> str:
    if book == "Psalms":
        return f"Psalm {chapter}."
    return f"{spoken_book(book)}, chapter {chapter}."


def display_ref(segments: list[tuple[str, int, int]] | list[list]) -> str:
    """('Psalms', 42, 58) -> 'Psalms 42–58'; multi-book -> 'Ephesians, Philippians & Colossians'."""
    segs = [tuple(s) for s in segments]
    if len(segs) == 1:
        b, a, z = segs[0]
        return f"{b} {a}" if a == z else f"{b} {a}–{z}"
    books = list(dict.fromkeys(s[0] for s in segs))
    return books[0] if len(books) == 1 else ", ".join(books[:-1]) + " & " + books[-1]
