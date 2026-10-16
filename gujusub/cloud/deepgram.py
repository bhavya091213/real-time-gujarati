"""Deepgram Listen v1 streaming (Nova-3) provider.

Protocol (researched 2026-10-16, developers.deepgram.com):

- Connect ``wss://api.deepgram.com/v1/listen?model=nova-3&language=gu&...`` with
  header ``Authorization: Token <key>``. Gujarati (``gu``) is Nova-3 only;
  ``language=multi`` does not include it and there is no per-utterance language
  ID, so asr_mode "auto" requests ``gu``. Vocabulary hints go in repeated
  ``keyterm`` params (Nova-3 only, ~500 tokens total). Bad query -> HTTP 400 at
  upgrade; bad key -> 401.
- Client -> server: binary PCM16 frames; TEXT frames ``{"type":"KeepAlive"}``
  (every 3-5 s without audio, server closes after ~10 s) and
  ``{"type":"CloseStream"}`` (flush, Metadata, close).
- Server -> client: ``Results`` (``is_final``, ``speech_final``,
  ``channel.alternatives[0].transcript``) where each message covers only the
  current unfinalised segment (interims replace each other, finals append);
  ``UtteranceEnd`` closes the utterance; ``SpeechStarted``/``Metadata`` are
  informational; ``Error`` is non-fatal (fatal errors close the socket).

Utterance = is_final transcripts joined until ``speech_final`` or
``UtteranceEnd``; live line = those finals + the latest interim.
"""

import json
import logging
from typing import Any
from urllib.parse import urlencode

import websockets
from websockets.exceptions import InvalidStatus

from gujusub.cloud.base import CloudAuthError, CloudConfigError, CloudError, WebSocketSession
from gujusub.cloud.registry import provider_info

logger = logging.getLogger("cloud.deepgram")

INFO = provider_info("deepgram")

URL = "wss://api.deepgram.com/v1/listen"
MODEL = "nova-3"
LANGUAGES = {"gu": "gu", "auto": "gu", "en": "en"}  # no gu/en LID on Deepgram
MAX_KEYTERMS = 60
MAX_KEYTERM_CHARS = 500
BASE_PARAMS = (
    ("encoding", "linear16"),
    ("channels", "1"),
    ("interim_results", "true"),
    ("endpointing", "300"),
    ("utterance_end_ms", "1000"),
    ("vad_events", "true"),
    ("smart_format", "true"),
    ("punctuate", "true"),
)
KEEPALIVE = json.dumps({"type": "KeepAlive"})
CLOSE_STREAM = json.dumps({"type": "CloseStream"})
AUTH_HINTS = ("auth", "credential", "api key", "token", "unauthorized", "forbidden")


async def _connect(url: str, headers: dict[str, str]) -> Any:
    """Indirection so tests can swap in a fake socket."""
    return await websockets.connect(url, additional_headers=headers)


def keyterms(terms: tuple[str, ...]) -> list[str]:
    """Unique non-empty terms, capped at MAX_KEYTERMS and MAX_KEYTERM_CHARS total."""
    out: list[str] = []
    seen: set[str] = set()
    total = 0
    for term in terms:
        term = " ".join(str(term).split())
        if not term or term in seen:
            continue
        if len(out) >= MAX_KEYTERMS or total + len(term) > MAX_KEYTERM_CHARS:
            break
        seen.add(term)
        out.append(term)
        total += len(term)
    return out


def build_url(language: str, sample_rate: int, terms: tuple[str, ...] = ()) -> str:
    if language not in LANGUAGES:
        raise CloudConfigError(f"Deepgram: unsupported language {language!r}")
    params = [("model", MODEL), ("language", LANGUAGES[language]),
              ("sample_rate", str(sample_rate)), *BASE_PARAMS]
    params += [("keyterm", t) for t in keyterms(terms)]
    return f"{URL}?{urlencode(params)}"


class Session(WebSocketSession):
    keepalive_s = 4.0

    def __init__(self, creds, config):
        super().__init__(creds, config)
        self.lang = LANGUAGES.get(config.language, "gu")
        self.finals: list[str] = []  # is_final transcripts of the open utterance
        self.interim = ""  # latest non-final transcript (replaced, not appended)

    async def connect(self) -> Any:
        url = build_url(self.config.language, self.config.sample_rate, self.config.terms)
        headers = {"Authorization": f"Token {self.creds.key}"}
        self.finals, self.interim = [], ""  # a reconnect starts a fresh stream
        try:
            return await _connect(url, headers)
        except InvalidStatus as e:
            status = e.response.status_code
            if status in (401, 403):
                raise CloudAuthError("Deepgram rejected the API key") from None
            if status == 400:
                raise CloudConfigError("Deepgram rejected the stream settings (HTTP 400)") from None
            raise CloudError(f"Deepgram refused the connection (HTTP {status})") from None
        except CloudError:
            raise
        except Exception as e:
            raise CloudError(f"Deepgram connection failed: {type(e).__name__}") from None

    def encode_audio(self, pcm16: bytes) -> bytes:
        return pcm16

    async def keepalive(self) -> None:
        await self.ws.send(KEEPALIVE)

    async def finish(self) -> None:
        await self.ws.send(CLOSE_STREAM)

    async def handle_message(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            return
        msg = json.loads(raw)
        if not isinstance(msg, dict):
            return
        kind = msg.get("type")
        if kind == "Results":
            self._on_results(msg)
        elif kind == "UtteranceEnd":
            self._close_utterance()
        elif kind == "Error":
            self._on_error(msg)
        # Metadata, SpeechStarted and unknown types carry nothing for the caption

    def _on_results(self, msg: dict) -> None:
        alts = (msg.get("channel") or {}).get("alternatives") or [{}]
        text = str(alts[0].get("transcript") or "").strip()
        if msg.get("is_final"):
            if text:
                self.finals.append(text)
            self.interim = ""
            if msg.get("speech_final"):
                self._close_utterance()
                return
        elif text:
            self.interim = text
        else:
            return
        self.tracker.update(" ".join(self.finals), self.interim, lang=self.lang)

    def _close_utterance(self) -> None:
        text = " ".join(self.finals)
        self.finals, self.interim = [], ""
        self.tracker.finalize(text, lang=self.lang)

    def _on_error(self, msg: dict) -> None:
        detail = " ".join(str(msg.get(k) or "") for k in ("description", "message")).strip()
        logger.warning("Deepgram error %s: %s", msg.get("code") or msg.get("variant"), detail)
        if any(h in detail.lower() for h in AUTH_HINTS):
            raise CloudAuthError("Deepgram rejected the API key")
