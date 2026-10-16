"""Soniox real-time STT (stt-rt-v5) provider: Gujarati/English with built-in translation.

Wire protocol (researched 2026-10-16 from soniox.com/docs):

- ``wss://stt-rt.soniox.com/transcribe-websocket`` with ``Authorization: Bearer <key>``.
- First text frame is the JSON config (model, pcm_s16le 16 kHz mono, language hints,
  endpoint detection, optional ``context`` with vocabulary terms, optional one-way
  translation to English). Then binary frames of raw PCM16.
- Control frames: ``{"type": "keepalive"}`` while idle (server drops after 20 s without
  audio); an empty text frame ``""`` ends the stream, after which the server flushes the
  remaining tokens, sends ``{"finished": true}`` and closes.
- Responses carry ``tokens``: final tokens are sent once and never repeated; each response
  carries the full current set of non-final tokens (replace the previous ones). Token
  texts are concatenated verbatim (spaces are tokens of their own). ``<end>`` (endpoint)
  or ``<fin>`` (manual finalize) closes the segment. Tokens with ``translation_status``
  ``"translation"`` are the English translation (they carry ``source_language``);
  ``"original"``/``"none"`` are the spoken text.
- Errors: ``{"error_code": 401, "error_message": "..."}`` then close.
"""

import json
from typing import Any

import websockets
from websockets.exceptions import InvalidStatus

from gujusub.cloud.base import (
    CloudAuthError,
    CloudError,
    Credentials,
    SessionConfig,
    WebSocketSession,
    logger,
)

URL = "wss://stt-rt.soniox.com/transcribe-websocket"
MODEL = "stt-rt-v5"
MAX_TERMS = 300
MAX_CONFIG_CHARS = 9000  # context limit is ~10k chars; stay well under it
DOMAIN = "BAPS Swaminarayan sabha; Gujarati devotional discourse"
END_TOKENS = ("<end>", "<fin>")
AUTH_STATUSES = (401, 403)
LANGUAGE_HINTS = {"gu": ["gu"], "en": ["en"], "auto": ["gu", "en"]}


async def _connect(url: str, headers: dict[str, str]) -> Any:
    """Open the socket (module-level so tests can replace it)."""
    return await websockets.connect(url, additional_headers=headers)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _fit_terms(config: dict, terms: list[str], limit: int = MAX_CONFIG_CHARS) -> list[str]:
    """Longest prefix of ``terms`` keeping ``config`` + terms under ``limit`` chars."""
    size = len(_dumps({**config, "context": {**config["context"], "terms": []}}))
    kept: list[str] = []
    for term in terms:
        size += len(_dumps(term)) + (2 if kept else 0)  # ", " separator
        if size > limit:
            break
        kept.append(term)
    return kept


def build_config(language: str, translate: bool, terms: tuple[str, ...] = ()) -> dict:
    """The first (config) frame for a stream, as a dict."""
    hints = LANGUAGE_HINTS.get(language, LANGUAGE_HINTS["gu"])
    config: dict[str, Any] = {
        "model": MODEL,
        "audio_format": "pcm_s16le",
        "sample_rate": 16000,
        "num_channels": 1,
        "language_hints": hints,
        "enable_language_identification": language == "auto",
        "enable_endpoint_detection": True,
        "max_endpoint_delay_ms": 1500,
        "endpoint_sensitivity": 0.3,
        "endpoint_latency_adjustment_level": 2,
        "client_reference_id": "gujusub",
    }
    if translate and language != "en":
        config["translation"] = {"type": "one_way", "target_language": "en"}
    clean = [t.strip() for t in terms if t and t.strip()][:MAX_TERMS]
    if clean:
        config["context"] = {"general": [{"key": "domain", "value": DOMAIN}]}
        config["context"]["terms"] = _fit_terms(config, clean)
    return config


def _error_from_status(status: int | None, detail: str) -> CloudError:
    if status in AUTH_STATUSES:
        return CloudAuthError(f"Soniox rejected the API key ({status}): {detail}")
    return CloudError(f"Soniox error {status}: {detail}")


class Session(WebSocketSession):
    """One Soniox stream; see the module docstring for the protocol."""

    keepalive_s = 8

    def __init__(self, creds: Credentials, config: SessionConfig):
        super().__init__(creds, config)
        self._finished = False
        self._reset_segment()

    def _reset_segment(self) -> None:
        self._orig_final = ""
        self._orig_pending = ""
        self._trans_final = ""
        self._trans_pending = ""
        self._seg_lang: str | None = None

    # ---- hooks -------------------------------------------------------------
    async def connect(self) -> Any:
        headers = {"Authorization": f"Bearer {self.creds.key}"}
        try:
            ws = await _connect(URL, headers)
        except InvalidStatus as e:
            status = getattr(e.response, "status_code", None)
            raise _error_from_status(status, "handshake refused") from None
        except Exception as e:
            raise CloudError(f"could not reach Soniox ({type(e).__name__})") from None
        cfg = build_config(self.config.language, self.config.translate, self.config.terms)
        try:
            await ws.send(_dumps(cfg))
        except Exception as e:
            raise CloudError(f"Soniox config send failed ({type(e).__name__})") from None
        self._finished = False
        self._reset_segment()
        return ws

    def encode_audio(self, pcm16: bytes) -> bytes:
        return pcm16

    async def keepalive(self) -> None:
        await self.ws.send(_dumps({"type": "keepalive"}))

    async def finish(self) -> None:
        await self.ws.send("")

    async def handle_message(self, raw: str | bytes) -> None:
        msg = json.loads(raw)
        if msg.get("error_code") is not None:
            detail = str(msg.get("error_message") or msg.get("error_type") or "unknown error")
            raise _error_from_status(msg.get("error_code"), detail)
        self._orig_pending = ""
        self._trans_pending = ""
        for tok in msg.get("tokens") or []:
            self._take_token(tok)
        self._publish_partial()
        if msg.get("finished"):
            self._finished = True
            if not self._closing:
                self.tracker.finalize(self._orig_final, translation=self._translation(True),
                                      lang=self._lang())
                self._reset_segment()

    # ---- token assembly ------------------------------------------------------
    def _take_token(self, tok: dict) -> None:
        text = tok.get("text") or ""
        final = bool(tok.get("is_final"))
        if text in END_TOKENS:
            if final:
                self._close_segment()
            return
        is_translation = tok.get("translation_status") == "translation"
        if is_translation:
            if not self.config.translate:
                return
            if final:
                self._trans_final = self._append(self._trans_final, text)
            else:
                base = self._trans_final + self._trans_pending
                self._trans_pending += text.lstrip() if not base else text
            return
        lang = tok.get("language")
        if lang:
            self._seg_lang = lang
        if final:
            self._orig_final = self._append(self._orig_final, text)
        else:
            base = self._orig_final + self._orig_pending
            self._orig_pending += text.lstrip() if not base else text

    @staticmethod
    def _append(buffer: str, text: str) -> str:
        return buffer + text if buffer else text.lstrip()

    def _close_segment(self) -> None:
        self.tracker.finalize(self._orig_final, translation=self._translation(final_only=True),
                              lang=self._lang())
        self._reset_segment()

    def _publish_partial(self) -> None:
        self.tracker.update(self._orig_final, self._orig_pending,
                            translation=self._translation(), lang=self._lang())

    def _translation(self, final_only: bool = False) -> str | None:
        if not self.config.translate:
            return None
        text = self._trans_final if final_only else self._trans_final + self._trans_pending
        return text.strip() or None

    def _lang(self) -> str | None:
        """Detected language, used only in auto mode (fixed modes keep their language)."""
        if self.config.language != "auto":
            return None
        return self._seg_lang if self._seg_lang in ("gu", "en") else None

    # ---- base overrides ------------------------------------------------------
    async def _reconnect(self, cause: Exception) -> bool:
        if self._finished:  # server ended the stream on purpose: stop, do not reconnect
            if self.state != "error":
                self.state = "closed"
            logger.info("Soniox stream finished")
            return False
        return await super()._reconnect(cause)
