"""Server-side settings: display look + which models run.

One immutable `Settings` snapshot, persisted as JSON by `SettingsStore`
(default ~/.cache/gujusub/settings.json, override with GUJUSUB_SETTINGS).
The 34 display keys mirror DEFAULTS in static/display.html exactly; the rest
(asr_mode, asr_backend, translate, conf_*) are server-only and never sent to /ws/view.
"""

import dataclasses
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gujusub.cloud.registry import BACKENDS

logger = logging.getLogger("settings")

SCHEMA = 4
# Schema-3 defaults that schema 4 changed. A saved schema-3 file stores every
# key, so values still equal to these are dropped on load to pick up the new
# look; values the operator actually changed are kept.
_V3_DEFAULTS = {
    "clear": 6, "font": 'system-ui, "Noto Sans Gujarati", sans-serif', "size": 64,
    "weight": 700, "lh": 1.3, "align": "left", "bar": True,
}


@dataclass(frozen=True)
class Settings:
    # --- display keys (same names/defaults as display.html DEFAULTS) ---
    aspect: str = "16:9"
    mode: str = "translation"  # primary | translation | both
    stable: bool = False
    clear: float = 20  # seconds without new text before fade, 0 = never
    font: str = '"Noto Sans Gujarati", system-ui, sans-serif'  # Gujarati stack
    size: float = 56  # px at a 1080-px-high stage (scales with the stage)
    minsize: float = 36  # legacy: accepted, ignored by the row-based display
    weight: float = 500
    lh: float = 1.28  # English line height; Gujarati uses lh + 0.07
    ls: float = 0
    tt: str = "none"
    color: str = "#ffffff"
    ow: float = 0  # text stroke width, px at 1080p
    oc: str = "#000000"  # outline / shadow colour
    shadow: bool = False  # soft character shadow
    bg: str = "#000000"
    align: str = "center"
    pad: float = 5  # inner horizontal padding, % of stage width
    bottom: float = 8  # legacy: superseded by y; ignored by the display
    bar: bool = False  # legacy caption bar (bar/barw/barc): accepted, ignored
    barw: float = 8
    barc: str = "#ffffff"
    rows_primary: int = 2  # visible rows, primary (ASR) block
    rows_translation: int = 3  # visible rows, translation block
    x: float = 50  # block centre, % of stage width
    y: float = 88  # bottom edge of the lowest block, % of stage height
    gap: float = 16  # primary block's bottom edge sits this % above translation's
    scale: float = 1  # font scale multiplier
    maxw: float = 100  # block width, % of stage width
    panel: str = "bottom"  # backing: bottom | lines | none
    panel_h: float = 25  # min bottom-panel height, % (grows to cover the lowest block)
    panel_opacity: float = 0.75
    wordfade: bool = True  # fade new words in
    font_en: str = 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
    # --- server-only keys ---
    asr_mode: Literal["gu", "en", "auto"] = "gu"
    # "local" (bundled models) or a cloud provider id (gujusub/cloud/registry.py)
    asr_backend: str = "local"
    translate: bool = True
    # Confidence trimming (gujusub/confidence.py), provisional until real clips
    # in samples/ refine it (unit 4.3). Calibrated in
    # .orchestrate/english-auto-lang-modes/02-units/calibration-3.4.md: 0.5 is the
    # highest word_min that leaves the clean Gujarati sample unchanged (its lowest
    # word is 0.64); utt 0.7 drops wrong-language decodes (English audio on the gu
    # engine, Gujarati audio on the en engine) while clean gu/en speech passes.
    conf_word_min: float = 0.5
    conf_utt_min: float = 0.7
    schema: int = SCHEMA

    def with_updates(self, partial: dict) -> "Settings":
        return dataclasses.replace(self, **validate(partial))


DISPLAY_KEYS = (
    "aspect", "mode", "stable", "clear", "font", "size", "minsize", "weight",
    "lh", "ls", "tt", "color", "ow", "oc", "shadow", "bg", "align", "pad",
    "bottom", "bar", "barw", "barc", "rows_primary", "rows_translation", "x", "y",
    "gap", "scale", "maxw", "panel", "panel_h", "panel_opacity", "wordfade", "font_en",
)

_COLOUR = re.compile(
    r"#[0-9a-fA-F]{3,8}|rgba?\(\s*[0-9.%,\s]+\)|hsla?\(\s*[0-9.%,\sdeg]+\)|[a-zA-Z]{1,30}"
)
_ASPECT = re.compile(r"\d{1,3}:\d{1,3}")
_FONT_BAD = set(";{}<>\\")
_MAX_STR = 200
# Old content-mode values -> new names; the stored value is always new-style.
_LEGACY_MODE = {"gu": "primary", "en": "translation"}


def _enum(*allowed: str):
    def check(v: Any) -> bool:
        return isinstance(v, str) and v in allowed
    return check


def _num(lo: float | None = None, hi: float | None = None, *, positive=False):
    def check(v: Any) -> bool:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return False
        if positive and v <= 0:
            return False
        return (lo is None or v >= lo) and (hi is None or v <= hi)
    return check


def _int(lo: int, hi: int):
    def check(v: Any) -> bool:
        return (not isinstance(v, bool) and isinstance(v, (int, float))
                and math.isfinite(v) and v == int(v) and lo <= v <= hi)
    return check


def _bool(v: Any) -> bool:
    return isinstance(v, bool)


def _colour(v: Any) -> bool:
    return isinstance(v, str) and len(v) <= 64 and _COLOUR.fullmatch(v.strip()) is not None


def _font(v: Any) -> bool:
    return isinstance(v, str) and 0 < len(v) <= _MAX_STR and not (_FONT_BAD & set(v))


def _aspect(v: Any) -> bool:
    return isinstance(v, str) and _ASPECT.fullmatch(v) is not None


_CHECKS = {
    "aspect": _aspect,
    "mode": _enum("primary", "translation", "both", "gu", "en"),  # gu/en = legacy
    "stable": _bool,
    "clear": _num(0, 600),
    "font": _font,
    "size": _num(24, 160),
    "minsize": _num(16, 160),
    "weight": _num(1, 1000),
    "lh": _num(1.0, 2.2),
    "ls": _num(-2, 20),
    "tt": _enum("none", "uppercase", "lowercase", "capitalize"),
    "color": _colour,
    "ow": _num(0, 12),
    "oc": _colour,
    "shadow": _bool,
    "bg": _colour,
    "align": _enum("left", "center", "right"),
    "pad": _num(0, 25),
    "bottom": _num(0, 50),
    "bar": _bool,
    "barw": _num(0, 24),
    "barc": _colour,
    "rows_primary": _int(1, 6),
    "rows_translation": _int(1, 6),
    "x": _num(0, 100),
    "y": _num(0, 100),
    "gap": _num(0, 60),
    "scale": _num(0.5, 2),
    "maxw": _num(20, 100),
    "panel": _enum("bottom", "lines", "none"),
    "panel_h": _num(0, 100),
    "panel_opacity": _num(0, 1),
    "wordfade": _bool,
    "font_en": _font,
    "asr_mode": _enum("gu", "en", "auto"),
    "asr_backend": _enum(*BACKENDS),
    "translate": _bool,
    "conf_word_min": _num(0, 1),
    "conf_utt_min": _num(0, 1),
}
assert set(_CHECKS) == {f.name for f in dataclasses.fields(Settings)} - {"schema"}


def validate(partial: dict) -> dict:
    """Return a cleaned copy of `partial`; ValueError on any bad key/value."""
    if not isinstance(partial, dict):
        raise ValueError("settings must be an object")
    for key, value in partial.items():
        check = _CHECKS.get(key)
        if check is None:
            raise ValueError(f"unknown or read-only setting: {key!r}")
        if not check(value):
            raise ValueError(f"invalid value for {key!r}: {value!r}")
    cleaned = dict(partial)
    for key in ("rows_primary", "rows_translation"):
        if key in cleaned:
            cleaned[key] = int(cleaned[key])
    if "mode" in cleaned:
        cleaned["mode"] = _LEGACY_MODE.get(cleaned["mode"], cleaned["mode"])
    return cleaned


def display_payload(settings: Settings) -> dict:
    """What /display needs: the display keys + schema."""
    return {**{k: getattr(settings, k) for k in DISPLAY_KEYS}, "schema": settings.schema}


def control_payload(settings: Settings) -> dict:
    return dataclasses.asdict(settings)


def snapshot_payload(settings: Settings) -> dict:
    """/ws/control snapshot: all keys plus the factory defaults (for Reset)."""
    return {**control_payload(settings), "defaults": control_payload(Settings())}


def default_path() -> Path:
    env = os.environ.get("GUJUSUB_SETTINGS")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "gujusub" / "settings.json"


class SettingsStore:
    """Holds the current Settings; persists every update atomically."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else default_path()
        self._current = Settings()

    def load(self) -> Settings:
        self._current = self._read()
        return self._current

    def _read(self) -> Settings:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return Settings()
        except (OSError, ValueError) as e:
            logger.warning("settings file %s unreadable (%s); using defaults", self.path, e)
            return Settings()
        if not isinstance(raw, dict):
            logger.warning("settings file %s is not an object; using defaults", self.path)
            return Settings()
        good = {}
        old = not isinstance(raw.get("schema"), int) or raw["schema"] < 4
        for key, value in raw.items():
            if key == "schema":
                continue
            if old and key in _V3_DEFAULTS and value == _V3_DEFAULTS[key]:
                continue  # untouched schema-3 default -> take the new one
            try:
                good.update(validate({key: value}))
            except ValueError as e:
                logger.warning("ignoring saved setting: %s", e)
        return Settings().with_updates(good)

    def save(self, settings: Settings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(control_payload(settings), indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def get(self) -> Settings:
        return self._current

    def update(self, partial: dict) -> Settings:
        new = self._current.with_updates(partial)
        if new == self._current:
            return self._current
        self.save(new)
        self._current = new
        return new
