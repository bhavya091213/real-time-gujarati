"""Shared types for cloud streaming ASR providers.

A provider module (``gujusub/cloud/<id>.py``) exposes ``INFO: ProviderInfo``
and ``Session`` (a ``CloudSession``). The server opens one session per mic
connection, pushes 16 kHz mono PCM16 into it and reads ``TranscriptEvent``s
back, exactly the shape the local ``StreamingTranscriber`` produces, so the
rest of the pipeline (fillers, glossary, translation, display) is unchanged.

Most providers speak WebSocket; ``WebSocketSession`` does the connection,
reader task, reconnects and the event queue, and leaves four hooks to the
provider: ``connect``, ``encode_audio``, ``handle_message``, ``finish``.
``SegmentTracker`` turns a provider's "final text so far + pending text" into
numbered utterances with partial/final events.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from gujusub.streaming import TranscriptEvent

logger = logging.getLogger("cloud")

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2  # PCM16 mono
MAX_RECONNECTS = 3
RECONNECT_BACKOFF_S = (0.5, 1.0, 2.0)
BUFFER_WHILE_RECONNECTING_S = 2.0  # audio kept while the socket is re-opened
CLOSE_DRAIN_TIMEOUT_S = 4.0  # how long close() waits for the provider's last finals


class CloudError(Exception):
    """Provider refused or dropped the stream; message is safe to show the operator."""


class CloudAuthError(CloudError):
    """The API key (or region) was rejected."""


class CloudConfigError(CloudError):
    """The requested language/feature is not supported by this provider."""


@dataclass(frozen=True)
class ProviderInfo:
    id: str  # settings value, module name: gujusub/cloud/<id>.py
    label: str
    key_env: str  # environment variable consulted when no key is saved
    key_url: str  # where the operator gets a key
    gujarati: str  # "yes" | "partial" | "no" (see note)
    translates: bool  # returns English for Gujarati speech itself
    language_id: bool  # can decide gu/en per utterance itself (asr_mode auto)
    needs_region: bool = False  # Azure: key + region
    note: str = ""  # one line shown under the provider in the mic page


@dataclass(frozen=True)
class Credentials:
    key: str
    region: str | None = None


@dataclass(frozen=True)
class SessionConfig:
    language: str = "gu"  # "gu" | "en" | "auto"
    translate: bool = True  # ask for English translation when the provider can
    sample_rate: int = SAMPLE_RATE
    terms: tuple[str, ...] = ()  # vocabulary hints (glossary) when supported


@runtime_checkable
class CloudSession(Protocol):
    """One live stream. ``state`` is one of connecting/connected/reconnecting/closed/error."""

    state: str
    error: str | None

    async def start(self) -> None: ...

    async def send_audio(self, pcm16: bytes) -> None: ...

    def events(self) -> AsyncIterator[TranscriptEvent]: ...

    async def close(self) -> None: ...


Emit = Callable[[TranscriptEvent], None]


class SegmentTracker:
    """Numbered utterances from a provider's (final text, pending text) view.

    ``update(committed, pending)`` emits a partial for the open utterance (a new
    one opens on the first non-empty update after a final). ``finalize(text)``
    emits the final and closes it; an empty final with no partial sent is
    dropped silently (no id consumed). Repeated identical partials are not
    re-emitted. Events are immutable; ``translation`` is attached when the
    provider supplies it (``None`` = let the server translate locally).
    """

    def __init__(self, emit: Emit, lang: str = "gu"):
        self._emit = emit
        self.default_lang = lang
        self.utterance_id = 0
        self.open = False
        self._last: tuple[str, str, str | None, str] | None = None

    def _lang(self, lang: str | None) -> str:
        return lang if lang in ("gu", "en") else self.default_lang

    def update(self, committed: str, pending: str = "", *, translation: str | None = None,
               lang: str | None = None) -> None:
        committed, pending = committed.strip(), pending.strip()
        if not committed and not pending:
            return
        if not self.open:
            self.utterance_id += 1
            self.open = True
            self._last = None
        key = (committed, pending, translation, self._lang(lang))
        if key == self._last:
            return
        self._last = key
        self._emit(TranscriptEvent("partial", self.utterance_id, committed, pending,
                                   lang=self._lang(lang), translation=translation))

    def finalize(self, text: str, *, translation: str | None = None,
                 lang: str | None = None) -> None:
        text = text.strip()
        if not self.open and not text:
            return
        if not self.open:
            self.utterance_id += 1
        self.open = False
        self._last = None
        self._emit(TranscriptEvent("final", self.utterance_id, text, "",
                                   lang=self._lang(lang), translation=translation))

    def flush(self) -> None:
        """Close the open utterance with whatever was last shown (reconnect/close)."""
        if not self.open:
            return
        committed, pending, translation, lang = self._last or ("", "", None, self.default_lang)
        text = f"{committed} {pending}".strip()
        self.finalize(text, translation=translation, lang=lang)


class WebSocketSession:
    """Base for WebSocket providers; subclasses implement the four hooks.

    Lifecycle: ``start()`` connects and starts the reader; ``send_audio()``
    forwards PCM16 (buffered for a short while during a reconnect, then
    dropped); ``events()`` yields TranscriptEvents until the session ends;
    ``close()`` sends the provider's end-of-stream, waits briefly for the last
    finals, and closes. A dropped connection is re-opened up to MAX_RECONNECTS
    times (the pending segment is finalised first); after that the session
    ends with ``state == "error"`` and ``error`` set.
    """

    keepalive_s: float | None = None  # subclass: send keepalive() this often while idle

    def __init__(self, creds: Credentials, config: SessionConfig):
        self.creds = creds
        self.config = config
        self.state = "connecting"
        self.error: str | None = None
        self.ws: Any = None
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._keeper: asyncio.Task | None = None
        self._closing = False
        self._reconnects = 0
        self._pending_audio: list[bytes] = []
        self._pending_bytes = 0
        self._last_audio_at = time.monotonic()
        self.tracker = SegmentTracker(self._queue.put_nowait,
                                      "gu" if config.language == "auto" else config.language)

    # ---- hooks -------------------------------------------------------------
    async def connect(self) -> Any:
        """Open the socket and complete any handshake; return the connection."""
        raise NotImplementedError

    def encode_audio(self, pcm16: bytes) -> bytes | str:
        return pcm16  # most providers take raw binary frames

    async def handle_message(self, raw: str | bytes) -> None:
        """Parse one provider message and drive ``self.tracker``."""
        raise NotImplementedError

    async def finish(self) -> None:
        """Tell the provider the audio is over (default: nothing)."""

    async def keepalive(self) -> None:
        """Sent every ``keepalive_s`` while no audio flows (default: nothing)."""

    # ---- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        self.ws = await self.connect()
        self.state = "connected"
        self._reader = asyncio.create_task(self._read_loop())
        if self.keepalive_s:
            self._keeper = asyncio.create_task(self._keepalive_loop())

    async def send_audio(self, pcm16: bytes) -> None:
        self._last_audio_at = time.monotonic()
        if self.state == "reconnecting":
            self._buffer(pcm16)
            return
        if self.ws is None or self.state != "connected":
            return
        try:
            await self.ws.send(self.encode_audio(pcm16))
        except Exception as e:  # the reader notices the drop and reconnects
            logger.debug("send failed (%r); buffering", e)
            self._buffer(pcm16)

    def _buffer(self, pcm16: bytes) -> None:
        self._pending_audio.append(pcm16)
        self._pending_bytes += len(pcm16)
        while self._pending_bytes > BUFFER_WHILE_RECONNECTING_S * BYTES_PER_SECOND:
            self._pending_bytes -= len(self._pending_audio.pop(0))

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._keeper:
            self._keeper.cancel()
        if self.ws is not None and self.state == "connected":
            try:
                await self.finish()
            except Exception as e:
                logger.debug("finish() failed: %r", e)
        if self._reader is not None:
            try:
                await asyncio.wait_for(self._reader, CLOSE_DRAIN_TIMEOUT_S)
            except TimeoutError:
                self._reader.cancel()
            except Exception:
                pass
        await self._close_ws()
        self.tracker.flush()  # nothing pending: no event; else the last partial becomes final
        if self.state not in ("error",):
            self.state = "closed"
        self._queue.put_nowait(None)

    async def _close_ws(self) -> None:
        ws, self.ws = self.ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    # ---- internals ---------------------------------------------------------
    async def _read_loop(self) -> None:
        try:
            while True:
                try:
                    raw = await self.ws.recv()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if self._closing:
                        return
                    if not await self._reconnect(e):
                        return
                    continue
                try:
                    await self.handle_message(raw)
                except CloudError as e:
                    self._fail(str(e))
                    return
                except Exception:
                    logger.exception("bad provider message ignored")
        except asyncio.CancelledError:
            pass
        except CloudError as e:
            self._fail(str(e))
        finally:
            if not self._closing:
                self._queue.put_nowait(None)

    def _fail(self, message: str) -> None:
        self.state = "error"
        self.error = message
        logger.error("cloud session failed: %s", message)

    async def _reconnect(self, cause: Exception) -> bool:
        self.tracker.flush()  # close whatever was pending on the old socket
        while self._reconnects < MAX_RECONNECTS:
            delay = RECONNECT_BACKOFF_S[min(self._reconnects, len(RECONNECT_BACKOFF_S) - 1)]
            self._reconnects += 1
            self.state = "reconnecting"
            logger.warning("provider connection dropped (%r); reconnecting in %.1f s (%d/%d)",
                           cause, delay, self._reconnects, MAX_RECONNECTS)
            await self._close_ws()
            await asyncio.sleep(delay)
            try:
                self.ws = await self.connect()
            except CloudAuthError as e:
                self._fail(str(e))
                return False
            except Exception as e:
                cause = e
                continue
            self.state = "connected"
            pending, self._pending_audio, self._pending_bytes = self._pending_audio, [], 0
            for chunk in pending:
                await self.send_audio(chunk)
            return True
        self._fail(f"connection lost ({cause!r}); gave up after {MAX_RECONNECTS} reconnects")
        return False

    async def _keepalive_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.keepalive_s)
                if self.state == "connected" and (
                        time.monotonic() - self._last_audio_at >= self.keepalive_s):
                    try:
                        await self.keepalive()
                    except Exception as e:
                        logger.debug("keepalive failed: %r", e)
        except asyncio.CancelledError:
            pass
