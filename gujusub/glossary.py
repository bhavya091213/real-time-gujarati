"""BAPS glossary: rewrite known names/terms to their house spelling.

Longest-match, case-insensitive, word-boundary phrase replacement over
whitespace tokens. Punctuation attached to the first/last token of a match is
preserved. Tokens containing Gujarati script are never touched.

Entries marked `safe: false` (ordinary English words such as "beta") are only
rewritten when the matched text differs from the canonical spelling by more
than letter case.

Override file: env GUJUSUB_GLOSSARY (operator-editable, hot-reloaded).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_VAR = "GUJUSUB_GLOSSARY"
RELOAD_INTERVAL_S = 2.0
PACKAGED = "glossary_baps.json"

_GUJARATI = re.compile("[઀-૿]")
_SPLIT = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)

# first token -> [(variant tokens, canonical, safe)], longest variants first
_Index = dict[str, list[tuple[tuple[str, ...], str, bool]]]


def _tokens(phrase: str) -> tuple[str, ...]:
    return tuple(phrase.lower().split())


def _build_index(entries: list[dict]) -> _Index:
    index: _Index = {}
    for entry in entries:
        canonical = entry["canonical"]
        safe = entry.get("safe", True) is not False
        for variant in entry.get("variants", ()):
            toks = _tokens(variant)
            if toks:
                index.setdefault(toks[0], []).append((toks, canonical, safe))
    for rules in index.values():
        rules.sort(key=lambda r: len(r[0]), reverse=True)
    return index


class Glossary:
    def __init__(self, entries: list[dict], override: Path | None = None) -> None:
        self._index = _build_index(entries)
        self._max_len = max((len(r[0]) for v in self._index.values() for r in v), default=1)
        self._override = override
        self._mtime = self._stat()
        self._checked = time.monotonic()

    def __len__(self) -> int:
        return sum(len(v) for v in self._index.values())

    def _stat(self) -> float | None:
        try:
            return self._override.stat().st_mtime if self._override else None
        except OSError:
            return None

    def maybe_reload(self) -> None:
        """Re-read the override file if its mtime changed (checked every 2 s)."""
        if self._override is None:
            return
        now = time.monotonic()
        if now - self._checked < RELOAD_INTERVAL_S:
            return
        self._checked = now
        mtime = self._stat()
        if mtime is None or mtime == self._mtime:
            return
        try:
            entries = _read_entries(self._override)
        except (OSError, ValueError, KeyError, TypeError):
            logger.exception("glossary reload failed; keeping previous entries")
            self._mtime = mtime  # don't retry the same broken file every 2 s
            return
        self._index = _build_index(entries)
        self._max_len = max((len(r[0]) for v in self._index.values() for r in v), default=1)
        self._mtime = mtime
        logger.info("glossary reloaded from %s", self._override)

    def apply(self, text: str) -> str:
        if not text:
            return text
        words = text.split()
        parts = [_SPLIT.match(w).groups() for w in words]  # (lead, core, trail)
        cores = [c.lower() if not _GUJARATI.search(c) else "" for _, c, _ in parts]
        out: list[str] = []
        i, n = 0, len(words)
        while i < n:
            hit = self._match(parts, cores, i, n)
            if hit is None:
                out.append(words[i])
                i += 1
                continue
            length, canonical = hit
            out.append(parts[i][0] + canonical + parts[i + length - 1][2])
            i += length
        return " ".join(out) if out != words else text

    def _match(self, parts, cores, i: int, n: int) -> tuple[int, str] | None:
        rules = self._index.get(cores[i])
        if not rules:
            return None
        for toks, canonical, safe in rules:  # longest first
            length = len(toks)
            if i + length > n:
                continue
            if any(cores[i + k] != toks[k] for k in range(length)):
                continue
            # interior tokens must be bare words: punctuation breaks a phrase
            if any(parts[i + k][2] or parts[i + k + 1][0] for k in range(length - 1)):
                continue
            matched = " ".join(parts[i + k][1] for k in range(length))
            if not safe and matched.lower() == canonical.lower():
                return length, matched  # ordinary word: leave exactly as spoken
            return length, canonical
        return None


def _read_entries(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["entries"]


def _packaged_entries() -> list[dict]:
    text = resources.files("gujusub.data").joinpath(PACKAGED).read_text(encoding="utf-8")
    return json.loads(text)["entries"]


def load_glossary(path: str | os.PathLike | None = None) -> Glossary:
    """Load `path`, else $GUJUSUB_GLOSSARY, else the packaged glossary.

    A missing/broken override is logged once and the packaged glossary is used
    (the override path is still watched, so creating it later hot-loads it).
    """
    chosen = path or os.environ.get(ENV_VAR)
    if not chosen:
        return Glossary(_packaged_entries())
    override = Path(chosen)
    try:
        return Glossary(_read_entries(override), override)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("glossary override %s unusable (%s); using packaged glossary", override, exc)
        return Glossary(_packaged_entries(), override)
