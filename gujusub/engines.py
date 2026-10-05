"""Loaded ASR engines keyed by caption language ("gu", "en").

Built once in server.main() from ``--engines gu,en``. Gujarati is mandatory: it
is the fallback whenever the selected mode needs an engine that was not loaded.
"""

import logging
from collections.abc import Callable, Iterable

from gujusub.engine import ASREngine

logger = logging.getLogger("engines")

KNOWN_LANGS = ("gu", "en")
FALLBACK_LANG = "gu"


class EngineNotLoaded(LookupError):
    """A mode asked for an engine that this server process did not load."""


class EngineRegistry:
    """Immutable mapping lang -> loaded engine."""

    def __init__(self, engines: dict[str, ASREngine]):
        self._engines = dict(engines)

    @property
    def langs(self) -> tuple[str, ...]:
        return tuple(self._engines)

    def __contains__(self, lang: object) -> bool:
        return lang in self._engines

    def get(self, lang: str) -> ASREngine:
        try:
            return self._engines[lang]
        except KeyError:
            raise EngineNotLoaded(
                f"ASR engine {lang!r} is not loaded (loaded: {', '.join(self.langs)}); "
                f"start the server with --engines including {lang!r}"
            ) from None

    def warmup(self) -> None:
        for lang, engine in self._engines.items():
            logger.info("warming up %s engine ...", lang)
            engine.warmup()


def parse_engine_list(text: str) -> tuple[str, ...]:
    """'gu,en' -> ('gu', 'en'); de-duplicated, order kept; gu required."""
    langs = tuple(dict.fromkeys(p.strip() for p in text.split(",") if p.strip()))
    unknown = [lang for lang in langs if lang not in KNOWN_LANGS]
    if unknown:
        raise ValueError(f"unknown engine(s) {unknown}; choose from {list(KNOWN_LANGS)}")
    if FALLBACK_LANG not in langs:
        raise ValueError(f"--engines must include {FALLBACK_LANG!r} (the fallback engine)")
    return langs


def _load_gu(device: str) -> ASREngine:
    from gujusub.asr_engine import ASREngine as IndicConformer

    return IndicConformer(lang="gu", device=device)


def _load_en(device: str) -> ASREngine:
    from gujusub.engine_parakeet import ParakeetEngine

    return ParakeetEngine()  # ONNX on CPU; device does not apply


LOADERS: dict[str, Callable[[str], ASREngine]] = {"gu": _load_gu, "en": _load_en}


def load_engines(langs: Iterable[str], device: str = "cpu") -> EngineRegistry:
    """Load (not warm up) every requested engine."""
    return EngineRegistry({lang: LOADERS[lang](device) for lang in langs})
