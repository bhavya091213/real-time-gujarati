"""Azure AI Speech continuous recognition (gu-IN / en-IN) via the Speech SDK.

Researched 2026-10-16 from learn.microsoft.com; untested against the live
service here. Azure's raw WebSocket protocol is not documented for custom
clients, so this provider drives the official ``azure-cognitiveservices-speech``
SDK (an optional dependency, imported lazily by ``_sdk()``) instead of
subclassing ``WebSocketSession``.

- Credentials: key + region (``Credentials.region``, e.g. ``eastus``).
- ``gu``/``en``: ``SpeechConfig(subscription, region)`` with
  ``speech_recognition_language`` ``gu-IN``/``en-IN``.
- ``auto``: the ``.../speech/universal/v2`` endpoint form plus continuous
  language ID over ``["gu-IN", "en-IN"]``; the detected language is read from
  ``AutoDetectSourceLanguageResult`` per phrase.
- Audio: a 16 kHz PCM16 mono ``PushAudioInputStream``; glossary terms go into
  a ``PhraseListGrammar`` (capped at ``MAX_PHRASES``).
- ``recognizing`` = full current phrase (replaces the partial), ``recognized``
  = final, ``canceled`` with an error ends the session (auth-looking details
  become ``CloudAuthError``), ``session_stopped`` ends the event stream.

SDK callbacks run on SDK threads; everything that touches the tracker or the
queue is marshalled onto the asyncio loop with ``call_soon_threadsafe``.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from gujusub.cloud.base import (
    CloudAuthError,
    CloudConfigError,
    CloudError,
    Credentials,
    SegmentTracker,
    SessionConfig,
)
from gujusub.streaming import TranscriptEvent

logger = logging.getLogger("cloud.azure")

LOCALES = {"gu": "gu-IN", "en": "en-IN"}
LID_LOCALES = ["gu-IN", "en-IN"]
LANG_BY_LOCALE = {"gu-IN": "gu", "en-IN": "en"}
MAX_PHRASES = 500
AUTH_MARKERS = ("401", "403", "authentication", "websocket upgrade failed", "unauthorized")
SDK_HINT = "Azure Speech SDK not installed: pip install azure-cognitiveservices-speech"


def _sdk() -> Any:
    """Import the optional Azure Speech SDK (absolute import; never this module)."""
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        raise CloudConfigError(SDK_HINT) from None
    return speechsdk


def _universal_endpoint(region: str) -> str:
    return f"wss://{region}.stt.speech.microsoft.com/speech/universal/v2"


def _looks_like_auth(details: str) -> bool:
    low = details.lower()
    return any(marker in low for marker in AUTH_MARKERS)


class Session:
    """One Azure continuous-recognition stream implementing ``CloudSession``."""

    def __init__(self, creds: Credentials, config: SessionConfig):
        self.creds = creds
        self.config = config
        self.state = "connecting"
        self.error: str | None = None
        self._start_failure: tuple[str, bool] | None = None  # set on the SDK thread
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self.tracker = SegmentTracker(self._queue.put_nowait,
                                      "gu" if config.language == "auto" else config.language)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sdk_mod: Any = None
        self._stream: Any = None
        self._reco: Any = None
        self._ended = False
        self._closing = False

    # ---- setup -------------------------------------------------------------
    def _build(self, sdk: Any) -> Any:
        region = (self.creds.region or "").strip()
        if not region:
            raise CloudConfigError("Azure Speech needs a region (e.g. eastus) with the key")
        auto = self.config.language == "auto"
        if auto:
            cfg = sdk.SpeechConfig(subscription=self.creds.key,
                                   endpoint=_universal_endpoint(region))
            cfg.set_property(sdk.PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous")
        else:
            locale = LOCALES.get(self.config.language)
            if locale is None:
                raise CloudConfigError(f"Azure: unsupported language {self.config.language!r}")
            cfg = sdk.SpeechConfig(subscription=self.creds.key, region=region)
            cfg.speech_recognition_language = locale
        fmt = sdk.audio.AudioStreamFormat(samples_per_second=self.config.sample_rate,
                                          bits_per_sample=16, channels=1)
        self._stream = sdk.audio.PushAudioInputStream(stream_format=fmt)
        audio_cfg = sdk.audio.AudioConfig(stream=self._stream)
        kwargs: dict[str, Any] = {"speech_config": cfg, "audio_config": audio_cfg}
        if auto:
            kwargs["auto_detect_source_language_config"] = (
                sdk.languageconfig.AutoDetectSourceLanguageConfig(languages=LID_LOCALES))
        reco = sdk.SpeechRecognizer(**kwargs)
        terms = [t.strip() for t in self.config.terms if t and t.strip()][:MAX_PHRASES]
        if terms:
            phrases = sdk.PhraseListGrammar.from_recognizer(reco)
            for term in terms:
                phrases.addPhrase(term)
        reco.recognizing.connect(self._threadsafe(self._on_recognizing))
        reco.recognized.connect(self._threadsafe(self._on_recognized))
        marshal_cancel = self._threadsafe(self._on_canceled)

        def on_canceled(evt: Any) -> None:  # SDK thread: also record it for start()
            failure = self._cancel_failure(evt)
            if failure is not None and self.state == "connecting":
                self._start_failure = failure
            marshal_cancel(evt)

        reco.canceled.connect(on_canceled)
        reco.session_stopped.connect(self._threadsafe(self._on_stopped))
        return reco

    def _threadsafe(self, handler: Callable[[Any], None]) -> Callable[[Any], None]:
        def callback(evt: Any) -> None:  # runs on an SDK thread
            loop = self._loop
            if loop is None:
                return
            try:
                loop.call_soon_threadsafe(handler, evt)
            except RuntimeError:  # loop already closed
                pass
        return callback

    # ---- SDK event handlers (loop thread) ----------------------------------
    def _lang_of(self, result: Any) -> str | None:
        if self.config.language != "auto":
            return None
        try:
            locale = self._sdk_mod.AutoDetectSourceLanguageResult(result).language
        except Exception:
            return None
        return LANG_BY_LOCALE.get(locale)

    def _on_recognizing(self, evt: Any) -> None:
        if self._ended:
            return
        self.tracker.update("", evt.result.text or "", lang=self._lang_of(evt.result))

    def _on_recognized(self, evt: Any) -> None:
        if self._ended or evt.result.reason != self._sdk_mod.ResultReason.RecognizedSpeech:
            return  # NoMatch and friends: nothing heard
        self.tracker.finalize(evt.result.text or "", lang=self._lang_of(evt.result))

    def _cancel_failure(self, evt: Any) -> tuple[str, bool] | None:
        """(operator message, looks like auth) for an error cancel, else None."""
        details = evt.cancellation_details
        if details.reason != self._sdk_mod.CancellationReason.Error:
            return None  # EndOfStream: session_stopped follows
        message = str(details.error_details or "recognition canceled")
        return f"Azure Speech: {message}", _looks_like_auth(message)

    def _on_canceled(self, evt: Any) -> None:
        failure = self._cancel_failure(evt)
        if failure is None:
            return
        self.state, self.error = "error", failure[0]
        logger.error("cloud session failed: %s", self.error)
        self._end()

    def _on_stopped(self, _evt: Any) -> None:
        self._end()

    def _end(self) -> None:
        if self._ended:
            return
        self.tracker.finalize("")
        self._ended = True
        self._queue.put_nowait(None)

    # ---- CloudSession ------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            self._sdk_mod = _sdk()
            self._reco = self._build(self._sdk_mod)
            await asyncio.to_thread(self._reco.start_continuous_recognition_async().get)
        except CloudError as e:
            self.state, self.error = "error", str(e)
            raise
        except Exception as e:
            self.state = "error"
            self.error = f"Azure Speech: {e}"
            raise (CloudAuthError if _looks_like_auth(str(e)) else CloudError)(self.error) from e
        if self._start_failure is not None:  # canceled fired before/at start
            message, auth = self._start_failure
            self.state, self.error = "error", message
            raise (CloudAuthError if auth else CloudError)(message)
        self.state = "connected"

    async def send_audio(self, pcm16: bytes) -> None:
        if self._stream is None or self.state != "connected" or self._closing:
            return
        self._stream.write(pcm16)

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
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception as e:
                logger.debug("stream close failed: %r", e)
        if self._reco is not None:
            try:
                await asyncio.to_thread(self._reco.stop_continuous_recognition_async().get)
            except Exception as e:
                logger.debug("stop recognition failed: %r", e)
        if self.state != "error":
            self.state = "closed"
        self._end()  # finalizes anything pending, then the sentinel (once)
