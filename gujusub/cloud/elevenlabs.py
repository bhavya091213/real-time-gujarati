"""ElevenLabs Scribe v2 Realtime streaming client.

Protocol (researched 2026-10-16 from elevenlabs.io/docs; untested against the
live service here):

- URL ``wss://api.elevenlabs.io/v1/speech-to-text/realtime`` with query
  ``model_id=scribe_v2_realtime&audio_format=pcm_16000&commit_strategy=vad
  &vad_silence_threshold_secs=0.8&include_timestamps=true
  &include_language_detection=true`` plus ``language_code=guj|eng`` (omitted
  for auto) and ``keyterms`` repeated once per term (max 50, each <= 20 chars).
- Auth: ``xi-api-key`` header. A 401/403 handshake means a rejected key.
- Client sends one JSON message type: ``{"message_type": "input_audio_chunk",
  "audio_base_64": ..., "commit": false, "sample_rate": 16000}``. A forced
  commit is the same message with empty audio and ``commit: true``. There are
  no ping/close messages; the server closes after 15 s without client
  messages, so a 100 ms silent chunk is sent every 5 s while idle.
- Server sends ``session_started`` (ignored), ``partial_transcript`` (full
  uncommitted segment text, replaces the previous one),
  ``committed_transcript`` and ``committed_transcript_with_timestamps`` (both
  per segment when timestamps are on; only the latter is used as the final
  because it carries ``language_code``), ``warning`` (logged) and error
  messages ``{"message_type": <type>, "error": ...}`` followed by a close.
"""

import base64
import json
import logging
import urllib.parse
from typing import Any

import websockets
from websockets.exceptions import InvalidStatus

from gujusub.cloud.base import (
    SAMPLE_RATE,
    CloudAuthError,
    CloudError,
    WebSocketSession,
)
from gujusub.cloud.registry import provider_info

logger = logging.getLogger("cloud.elevenlabs")

INFO = provider_info("elevenlabs")

ENDPOINT = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
MODEL_ID = "scribe_v2_realtime"
VAD_SILENCE_S = 0.8
MAX_KEYTERMS = 50
MAX_KEYTERM_CHARS = 20
KEEPALIVE_S = 5.0
KEEPALIVE_CHUNK_S = 0.1

LANGUAGE_CODES = {"gu": "guj", "en": "eng"}  # config.language -> request param
EVENT_LANGS = {"guj": "gu", "gu": "gu", "eng": "en", "en": "en"}  # reply -> event lang
AUTH_ERRORS = frozenset({"auth_error", "unaccepted_terms"})
ERROR_TYPES = frozenset({
    "error", "auth_error", "quota_exceeded", "commit_throttled", "unaccepted_terms",
    "rate_limited", "queue_overflow", "resource_exhausted", "session_time_limit_exceeded",
    "input_error", "invalid_request", "chunk_size_exceeded", "insufficient_audio_activity",
    "transcriber_error",
})
IGNORED_TYPES = frozenset({"session_started", "committed_transcript"})


async def _connect(url: str, headers: dict[str, str]) -> Any:
    """Open the socket (module-level so tests can replace it)."""
    return await websockets.connect(url, additional_headers=headers)


def keyterms(terms: tuple[str, ...]) -> list[str]:
    """Vocabulary hints within the service limits (longer terms are skipped)."""
    out: list[str] = []
    for term in terms:
        term = term.strip()
        if term and len(term) <= MAX_KEYTERM_CHARS and term not in out:
            out.append(term)
        if len(out) >= MAX_KEYTERMS:
            break
    return out


def build_url(language: str, terms: tuple[str, ...] = ()) -> str:
    params: list[tuple[str, str]] = [
        ("model_id", MODEL_ID),
        ("audio_format", f"pcm_{SAMPLE_RATE}"),
        ("commit_strategy", "vad"),
        ("vad_silence_threshold_secs", str(VAD_SILENCE_S)),
        ("include_timestamps", "true"),
        ("include_language_detection", "true"),
    ]
    code = LANGUAGE_CODES.get(language)  # "auto" (or anything else) -> service detects
    if code:
        params.append(("language_code", code))
    params.extend(("keyterms", t) for t in keyterms(terms))
    return f"{ENDPOINT}?{urllib.parse.urlencode(params)}"


def audio_message(pcm16: bytes, *, commit: bool = False) -> str:
    return json.dumps({
        "message_type": "input_audio_chunk",
        "audio_base_64": base64.b64encode(pcm16).decode("ascii"),
        "commit": commit,
        "sample_rate": SAMPLE_RATE,
    })


SILENT_CHUNK = bytes(int(SAMPLE_RATE * KEEPALIVE_CHUNK_S) * 2)


class Session(WebSocketSession):
    keepalive_s = KEEPALIVE_S

    async def connect(self) -> Any:
        url = build_url(self.config.language, self.config.terms)
        try:
            return await _connect(url, {"xi-api-key": self.creds.key})
        except InvalidStatus as e:
            status = getattr(e.response, "status_code", None)
            if status in (401, 403):
                raise CloudAuthError(f"ElevenLabs rejected the API key (HTTP {status})") from e
            raise CloudError(f"ElevenLabs refused the connection (HTTP {status})") from e
        except CloudError:
            raise
        except Exception as e:
            raise CloudError(f"could not reach ElevenLabs: {e}") from e

    def encode_audio(self, pcm16: bytes) -> str:
        return audio_message(pcm16)

    async def keepalive(self) -> None:
        if self.ws is not None:
            await self.ws.send(audio_message(SILENT_CHUNK))

    async def finish(self) -> None:
        await self.ws.send(audio_message(b"", commit=True))

    async def handle_message(self, raw: str | bytes) -> None:
        msg = json.loads(raw)
        if not isinstance(msg, dict):
            return
        kind = msg.get("message_type", "")
        if kind == "partial_transcript":
            self.tracker.update("", str(msg.get("text") or ""))
        elif kind == "committed_transcript_with_timestamps":
            lang = EVENT_LANGS.get(str(msg.get("language_code") or "").lower())
            self.tracker.finalize(str(msg.get("text") or ""), lang=lang)
        elif kind == "warning":
            logger.warning("ElevenLabs warning: %s", msg.get("warning"))
        elif kind in IGNORED_TYPES:
            # committed_transcript duplicates the _with_timestamps final we always request
            return
        elif kind in ERROR_TYPES or "error" in msg:
            text = f"ElevenLabs {kind or 'error'}: {msg.get('error') or 'no details'}"
            if kind in AUTH_ERRORS:
                raise CloudAuthError(text)
            raise CloudError(text)
        else:
            logger.debug("ignored ElevenLabs message %r", kind)
