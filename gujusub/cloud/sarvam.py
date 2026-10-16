"""Sarvam AI Saaras v3 realtime streaming client.

Protocol (researched 2026-10-16 from docs.sarvam.ai; untested against the live
service here):

- URL ``wss://api.sarvam.ai/speech-to-text-realtime/ws`` with query
  ``language_code=gu-IN|en-IN|auto&model=saaras:v3-realtime&stream_type=fast
  &mode=transcribe&encoding=linear16&sample_rate=16000&endpointing=vad
  &silence_duration_ms=600`` plus ``keyterms`` (one ", "-joined string, kept
  under the service's 1000-char limit). ``mode=translate`` would replace the
  transcript with English, so translation stays local.
- Auth: ``API-SUBSCRIPTION-KEY`` header. HTTP 401/403 means a rejected key.
- Client sends JSON: ``{"event": "audio_input", "audio": <base64 PCM16>}``,
  ``{"event": "ping"}`` (the server closes idle streams with 1008) and
  ``{"event": "end"}`` (graceful end; the server finalises then closes).
- Server sends ``session.begin``, ``vad.speech_start``/``vad.speech_end``,
  ``config.updated``, ``pong`` (all ignored), ``transcript.partial`` (full
  current text of the utterance, replaces the previous one),
  ``transcript.final`` (with ``language`` such as ``gu-IN`` when
  language_code=auto), ``session.end`` and ``error`` (``is_fatal``,
  ``status_code``, ``code``, ``message``).
- Close codes: 1003 rate limit / quota / bad key, 4000 bad model, language or
  parameter (or account not enabled for realtime); others are reconnected.
"""

import base64
import json
import logging
import urllib.parse
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from gujusub.cloud.base import (
    SAMPLE_RATE,
    CloudAuthError,
    CloudConfigError,
    CloudError,
    WebSocketSession,
)
from gujusub.cloud.registry import provider_info

logger = logging.getLogger("cloud.sarvam")

INFO = provider_info("sarvam")

ENDPOINT = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
MODEL = "saaras:v3-realtime"
SILENCE_MS = 600
MAX_KEYTERMS_CHARS = 900
KEEPALIVE_S = 5.0

LANGUAGE_CODES = {"gu": "gu-IN", "en": "en-IN", "auto": "auto"}  # config -> request
EVENT_LANGS = {"gu-in": "gu", "gu": "gu", "en-in": "en", "en": "en"}  # reply -> event lang
IGNORED_EVENTS = frozenset({
    "session.begin", "vad.speech_start", "vad.speech_end", "config.updated", "pong",
})
AUTH_STATUSES = frozenset({401, 403})
CLOSE_AUTH = 1003
CLOSE_CONFIG = 4000


async def _connect(url: str, headers: dict[str, str]) -> Any:
    """Open the socket (module-level so tests can replace it)."""
    return await websockets.connect(url, additional_headers=headers)


def keyterms(terms: tuple[str, ...]) -> str:
    """", "-joined hints, whole terms only, at most MAX_KEYTERMS_CHARS long."""
    out = ""
    for term in dict.fromkeys(t.strip() for t in terms):
        if not term:
            continue
        candidate = f"{out}, {term}" if out else term
        if len(candidate) > MAX_KEYTERMS_CHARS:
            break
        out = candidate
    return out


def build_url(language: str, terms: tuple[str, ...] = ()) -> str:
    code = LANGUAGE_CODES.get(language)
    if code is None:
        raise CloudConfigError(f"Sarvam does not support language {language!r}")
    params: list[tuple[str, str]] = [
        ("language_code", code),
        ("model", MODEL),
        ("stream_type", "fast"),
        ("mode", "transcribe"),
        ("encoding", "linear16"),
        ("sample_rate", str(SAMPLE_RATE)),
        ("endpointing", "vad"),
        ("silence_duration_ms", str(SILENCE_MS)),
    ]
    hints = keyterms(terms)
    if hints:
        params.append(("keyterms", hints))
    return f"{ENDPOINT}?{urllib.parse.urlencode(params)}"


def audio_message(pcm16: bytes) -> str:
    return json.dumps({"event": "audio_input",
                       "audio": base64.b64encode(pcm16).decode("ascii")})


def _close_code(e: BaseException) -> int | None:
    rcvd = getattr(e, "rcvd", None)
    return getattr(rcvd, "code", None)


def _classify_close(code: int | None) -> CloudError | None:
    """Close codes that retrying cannot fix; None means "reconnect"."""
    if code == CLOSE_AUTH:
        return CloudAuthError("Sarvam rejected the key or quota")
    if code == CLOSE_CONFIG:
        return CloudConfigError("Sarvam rejected the model, language or parameters "
                                "(or the account is not enabled for realtime)")
    return None


class Session(WebSocketSession):
    keepalive_s = KEEPALIVE_S

    def __init__(self, creds, config):
        super().__init__(creds, config)
        self._ended = False  # server sent session.end

    async def connect(self) -> Any:
        url = build_url(self.config.language, self.config.terms)
        try:
            return await _connect(url, {"API-SUBSCRIPTION-KEY": self.creds.key})
        except InvalidStatus as e:
            status = getattr(e.response, "status_code", None)
            if status in AUTH_STATUSES:
                raise CloudAuthError(f"Sarvam rejected the API key (HTTP {status})") from e
            if status == 400:
                raise CloudConfigError(f"Sarvam rejected the request (HTTP {status})") from e
            raise CloudError(f"Sarvam refused the connection (HTTP {status})") from e
        except ConnectionClosed as e:
            raise (_classify_close(_close_code(e))
                   or CloudError(f"Sarvam closed the connection: {e}")) from e
        except CloudError:
            raise
        except Exception as e:
            raise CloudError(f"could not reach Sarvam: {e}") from e

    async def _reconnect(self, cause: Exception) -> bool:
        # A close with 1003/4000 or after session.end is final; others retry.
        if self._ended:
            self.state = "closed"
            return False
        err = _classify_close(_close_code(cause)) if isinstance(cause, ConnectionClosed) else None
        if err is not None:
            self.tracker.finalize("")
            self._fail(str(err))
            return False
        return await super()._reconnect(cause)

    def encode_audio(self, pcm16: bytes) -> str:
        return audio_message(pcm16)

    async def keepalive(self) -> None:
        if self.ws is not None:
            await self.ws.send(json.dumps({"event": "ping"}))

    async def finish(self) -> None:
        await self.ws.send(json.dumps({"event": "end"}))

    async def handle_message(self, raw: str | bytes) -> None:
        msg = json.loads(raw)
        if not isinstance(msg, dict):
            return
        kind = msg.get("event", "")
        if kind == "transcript.partial":
            self.tracker.update("", str(msg.get("text") or ""))
        elif kind == "transcript.final":
            lang = EVENT_LANGS.get(str(msg.get("language") or "").lower())
            self.tracker.finalize(str(msg.get("text") or ""), lang=lang)
        elif kind == "session.end":
            self.tracker.finalize("")
            self._ended = True
        elif kind == "error":
            self._handle_error(msg)
        elif kind not in IGNORED_EVENTS:
            logger.debug("ignored Sarvam message %r", kind)

    def _handle_error(self, msg: dict) -> None:
        status = msg.get("status_code")
        text = f"Sarvam {msg.get('code') or 'error'}: {msg.get('message') or 'no details'}"
        if status in AUTH_STATUSES:
            raise CloudAuthError(text)
        if msg.get("is_fatal") is True:
            raise CloudError(text)
        logger.warning("%s (non-fatal)", text)
