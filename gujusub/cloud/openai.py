"""OpenAI Realtime API, transcription-only session (GA shape).

Protocol (researched 2026-10-16 from platform.openai.com docs; untested against
the live service here). Gujarati support is NOT confirmed by OpenAI's docs:
Whisper-class models do cover ``gu``, hence ``INFO.gujarati == "partial"``.

- URL ``wss://api.openai.com/v1/realtime?intent=transcription``, header
  ``Authorization: Bearer <key>`` (no ``OpenAI-Beta`` header in the GA API).
  A 401/403 handshake means a rejected key.
- First client event: ``session.update`` with ``session.type ==
  "transcription"`` and ``audio.input`` = 24 kHz ``audio/pcm`` format, the
  transcription model/language/prompt, ``server_vad`` turn detection and
  ``near_field`` noise reduction. ``language`` is omitted for auto.
- Audio: ``{"type": "input_audio_buffer.append", "audio": <base64 PCM16 24k>}``.
  The pipeline's 16 kHz audio is resampled here (linear, phase kept across
  chunks). No keepalive exists; with server VAD no commit is needed (a commit
  on an empty buffer is an error), so ``finish()`` sends nothing.
- Server events: ``input_audio_buffer.speech_started {item_id}`` opens a turn;
  ``conversation.item.input_audio_transcription.delta {item_id, delta}``
  appends text; ``...completed {item_id, transcript}`` is the final;
  ``...failed`` ends a turn without text; ``error {"error": {type, code,
  message, param}}``. Completions can arrive out of order, so text is kept
  per item and turns are finalised in speech order. Session max 60 min; a
  reconnect is a new session (``session.update`` is re-sent).
"""

import base64
import json
import logging
from math import gcd
from typing import Any

import numpy as np
import websockets
from websockets.exceptions import InvalidStatus

from gujusub.cloud.base import (
    SAMPLE_RATE,
    CloudAuthError,
    CloudConfigError,
    CloudError,
    Credentials,
    SessionConfig,
    WebSocketSession,
)
from gujusub.cloud.registry import provider_info

logger = logging.getLogger("cloud.openai")

INFO = provider_info("openai")

ENDPOINT = "wss://api.openai.com/v1/realtime?intent=transcription"
MODEL = "gpt-4o-transcribe"
OUTPUT_RATE = 24000
MAX_PROMPT_CHARS = 800
MAX_OPEN_ITEMS = 4  # turns waiting on an earlier completion before we force them out
VAD = {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
       "silence_duration_ms": 500}
PROMPT_PREFIX = {"gu": "Gujarati devotional discourse.", "en": "English devotional discourse.",
                 "auto": "Gujarati and English devotional discourse."}
AUTH_CODES = frozenset({"invalid_api_key", "authentication_error"})
DELTA = "conversation.item.input_audio_transcription.delta"
COMPLETED = "conversation.item.input_audio_transcription.completed"
FAILED = "conversation.item.input_audio_transcription.failed"
IGNORED_TYPES = frozenset({"session.created", "session.updated", "input_audio_buffer.committed",
                           "input_audio_buffer.speech_stopped"})


async def _connect(url: str, headers: dict[str, str]) -> Any:
    """Open the socket (module-level so tests can replace it)."""
    return await websockets.connect(url, additional_headers=headers)


def build_prompt(language: str, terms: tuple[str, ...] = ()) -> str:
    """Context prompt with glossary terms, kept within MAX_PROMPT_CHARS."""
    prompt = PROMPT_PREFIX.get(language, PROMPT_PREFIX["gu"])
    used: list[str] = []
    for term in terms:
        term = " ".join(term.split())
        if not term or term in used:
            continue
        candidate = f"{prompt} Terms: {', '.join([*used, term])}"
        if len(candidate) > MAX_PROMPT_CHARS:
            break
        used.append(term)
    return f"{prompt} Terms: {', '.join(used)}" if used else prompt


def session_update(language: str, terms: tuple[str, ...] = ()) -> dict:
    transcription: dict[str, str] = {"model": MODEL}
    if language in ("gu", "en"):  # "auto": the model detects the language itself
        transcription["language"] = language
    transcription["prompt"] = build_prompt(language, terms)
    return {"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
        "format": {"type": "audio/pcm", "rate": OUTPUT_RATE},
        "transcription": transcription,
        "turn_detection": dict(VAD),
        "noise_reduction": {"type": "near_field"},
    }}}}


class Resampler:
    """Stateful linear-interpolation resampler for PCM16 mono.

    Positions are exact integers in units of 1/den input samples; index 0 of
    each chunk's working array is the previous chunk's last sample, so output
    is continuous across chunk boundaries (one input sample of latency).
    """

    def __init__(self, src: int = SAMPLE_RATE, dst: int = OUTPUT_RATE):
        g = gcd(src, dst)
        self.step, self.den = src // g, dst // g
        self.pos = self.step
        self.tail: int | None = None

    def __call__(self, pcm16: bytes) -> bytes:
        x = np.frombuffer(pcm16[: len(pcm16) // 2 * 2], dtype="<i2")
        if x.size == 0:
            return b""
        tail = int(x[0]) if self.tail is None else self.tail
        xs = np.concatenate(([tail], x)).astype(np.float64)
        last = x.size * self.den
        n = (last - self.pos) // self.step + 1 if self.pos <= last else 0
        p = self.pos + self.step * np.arange(n, dtype=np.int64)
        idx, frac = p // self.den, (p % self.den) / self.den
        nxt = np.minimum(idx + 1, x.size)
        y = xs[idx] + (xs[nxt] - xs[idx]) * frac
        self.pos += self.step * n - last
        self.tail = int(x[-1])
        return np.clip(np.rint(y), -32768, 32767).astype("<i2").tobytes()


def audio_message(pcm24: bytes) -> str:
    return json.dumps({"type": "input_audio_buffer.append",
                       "audio": base64.b64encode(pcm24).decode("ascii")})


class Session(WebSocketSession):
    def __init__(self, creds: Credentials, config: SessionConfig) -> None:
        super().__init__(creds, config)
        self.lang = config.language if config.language in ("gu", "en") else "gu"
        self._reset()

    def _reset(self) -> None:
        self.resampler = Resampler(SAMPLE_RATE, OUTPUT_RATE)
        self.items: dict[str, str] = {}  # item_id -> text so far, in speech order
        self.done: dict[str, str] = {}  # item_id -> completed transcript awaiting its turn

    async def connect(self) -> Any:
        self._reset()
        headers = {"Authorization": f"Bearer {self.creds.key}"}
        try:
            ws = await _connect(ENDPOINT, headers)
        except InvalidStatus as e:
            status = getattr(e.response, "status_code", None)
            if status in (401, 403):
                raise CloudAuthError(f"OpenAI rejected the API key (HTTP {status})") from e
            raise CloudError(f"OpenAI refused the connection (HTTP {status})") from e
        except CloudError:
            raise
        except Exception as e:
            raise CloudError(f"could not reach OpenAI: {e}") from e
        try:
            await ws.send(json.dumps(session_update(self.config.language, self.config.terms)))
        except Exception as e:
            raise CloudError(f"OpenAI session setup failed: {e}") from e
        return ws

    def encode_audio(self, pcm16: bytes) -> str:
        return audio_message(self.resampler(pcm16))

    async def finish(self) -> None:
        return  # server VAD commits turns itself; commit on an empty buffer is an error

    async def handle_message(self, raw: str | bytes) -> None:
        msg = json.loads(raw)
        if not isinstance(msg, dict):
            return
        kind = str(msg.get("type") or "")
        item = str(msg.get("item_id") or "")
        if kind == "input_audio_buffer.speech_started":
            self.items.setdefault(item, "")
        elif kind == DELTA:
            self.items[item] = self.items.get(item, "") + str(msg.get("delta") or "")
            self._flush()
        elif kind in (COMPLETED, FAILED):
            if kind == FAILED:
                logger.warning("OpenAI transcription failed for %s: %s", item, msg.get("error"))
            self.items.setdefault(item, "")
            self.done[item] = str(msg.get("transcript") or "") if kind == COMPLETED else ""
            self._flush()
        elif kind == "error":
            raise _error(msg.get("error"))
        elif kind not in IGNORED_TYPES:
            logger.debug("ignored OpenAI event %r", kind)

    def _flush(self) -> None:
        """Finalise completed turns in speech order; show the oldest open one as pending."""
        while self.items:
            oldest = next(iter(self.items))
            if oldest in self.done:
                self.items.pop(oldest)
                self.tracker.finalize(self.done.pop(oldest), lang=self.lang)
            elif len(self.items) > MAX_OPEN_ITEMS:  # stuck turn: emit what we have
                self.tracker.finalize(self.items.pop(oldest), lang=self.lang)
            else:
                self.tracker.update("", self.items[oldest], lang=self.lang)
                return


def _error(err: Any) -> CloudError:
    err = err if isinstance(err, dict) else {"message": str(err)}
    code = str(err.get("code") or "")
    message = str(err.get("message") or "no details")
    if code in AUTH_CODES or err.get("type") == "authentication_error" or "API key" in message:
        # the provider message may quote part of the key: do not echo it
        return CloudAuthError(f"OpenAI rejected the API key ({code or 'auth error'})")
    if "language" in message.lower() or "language" in str(err.get("param") or ""):
        return CloudConfigError(f"OpenAI: {message}")
    return CloudError(f"OpenAI {code or err.get('type') or 'error'}: {message}")
