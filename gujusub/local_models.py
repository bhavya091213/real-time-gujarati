"""One-time loader for the local ASR models (engines, translator, language ID).

The server loads them synchronously at startup when ``asr_backend`` is
"local". With a cloud backend they are skipped; if the operator later switches
back to "local", ``start_background`` loads them once in a daemon thread and
the server assigns its globals when ``on_done`` runs on the event loop.
Pipelines opened meanwhile ``await wait()``.

A separate one-time "translator only" job (``start_translator``) loads just
the Translator, for cloud providers that do not translate themselves.

The loader callables are injected (the server passes late-binding wrappers so
tests can monkeypatch ``server.load_engines`` etc.).
"""

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gujusub.engines import DEFAULT_GU_MODEL

logger = logging.getLogger("local_models")

IDLE, LOADING, READY, ERROR = "idle", "loading", "ready", "error"


@dataclass(frozen=True)
class LocalSpec:
    """What to load: mirrors the server's --engines/--device/--gu-model/--no-translate."""

    langs: tuple[str, ...] = ("gu", "en")
    device: str = "cpu"
    gu_model: str = DEFAULT_GU_MODEL
    translate: bool = True


@dataclass(frozen=True)
class Loaded:
    engines: Any
    translator: Any
    language_id: Any


class LocalModels:
    """Loads the local models at most once (sync or in the background)."""

    def __init__(self, spec: LocalSpec, *, load_engines: Callable[..., Any],
                 make_translator: Callable[[], Any], load_language_id: Callable[[], Any]):
        self.spec = spec
        self._load_engines = load_engines
        self._make_translator = make_translator
        self._load_language_id = load_language_id
        self.state = IDLE
        self.error: str | None = None
        self.engines: Any = None
        self.translator: Any = None
        self.language_id: Any = None
        self._done: asyncio.Event | None = None
        self.translator_state = IDLE  # the translator-only job
        self.translator_error: str | None = None

    def _load(self, need_translator: bool) -> Loaded:
        spec = self.spec
        engines = self._load_engines(spec.langs, spec.device, gu_model=spec.gu_model)
        translator = None
        if spec.translate and need_translator:
            translator = self._make_translator()
            translator.warmup()
        engines.warmup()
        language_id = None
        if "en" in engines:  # Auto mode needs gu + en + LID
            language_id = self._load_language_id()
            language_id.warmup()
        return Loaded(engines, translator, language_id)

    def _apply(self, loaded: Loaded) -> None:
        self.engines = loaded.engines
        self.translator = loaded.translator
        self.language_id = loaded.language_id
        self.state = READY

    def load(self, need_translator: bool = True) -> "LocalModels":
        """Synchronous load (startup); exceptions propagate."""
        self.state = LOADING
        try:
            self._apply(self._load(need_translator))
        except Exception as e:
            self.state, self.error = ERROR, str(e) or type(e).__name__
            raise
        return self

    def start_background(self, loop: asyncio.AbstractEventLoop, need_translator: bool,
                         on_done: Callable[["LocalModels"], None]) -> bool:
        """Start the one-time background load; False if already started/loaded."""
        if self.state != IDLE:
            return False
        self.state = LOADING
        self._done = asyncio.Event()

        def target() -> None:
            loaded, error = None, None
            try:
                loaded = self._load(need_translator)
            except Exception as e:
                logger.exception("loading local models failed")
                error = str(e) or type(e).__name__
            loop.call_soon_threadsafe(self._finish, loaded, error, on_done)

        logger.info("loading local models in the background (%s)", ",".join(self.spec.langs))
        threading.Thread(target=target, name="local-models", daemon=True).start()
        return True

    def _finish(self, loaded: Loaded | None, error: str | None,
                on_done: Callable[["LocalModels"], None]) -> None:
        if loaded is not None:
            self._apply(loaded)
        else:
            self.state, self.error = ERROR, error
        try:
            on_done(self)
        except Exception:
            logger.exception("local models on_done callback failed")
        finally:
            if self._done is not None:
                self._done.set()

    def start_translator(self, loop: asyncio.AbstractEventLoop,
                         on_done: Callable[["LocalModels"], None]) -> bool:
        """One-time background load of only the Translator; False if already started."""
        if self.translator_state != IDLE:
            return False
        self.translator_state = LOADING

        def target() -> None:
            loaded, error = None, None
            try:
                loaded = self._make_translator()
                loaded.warmup()
            except Exception as e:
                logger.exception("loading the translator failed")
                error = str(e) or type(e).__name__
            loop.call_soon_threadsafe(self._finish_translator, loaded, error, on_done)

        logger.info("loading the translator in the background")
        threading.Thread(target=target, name="translator", daemon=True).start()
        return True

    def _finish_translator(self, loaded: Any, error: str | None,
                           on_done: Callable[["LocalModels"], None]) -> None:
        if loaded is not None:
            self.translator = loaded
            self.translator_state = READY
        else:
            self.translator_state, self.translator_error = ERROR, error
        try:
            on_done(self)
        except Exception:
            logger.exception("translator on_done callback failed")

    async def wait(self) -> bool:
        """Wait for a background load in progress; True when the models are ready."""
        if self.state == LOADING and self._done is not None:
            await self._done.wait()
        return self.state == READY
