"""Per-connection pipeline for a cloud ASR backend (asr_backend != "local").

receiver keeps the raw PCM16 bytes; the worker forwards them to the provider
session; a reader task publishes ``session.events()`` through the server's
filtered() + publish(), so fillers, glossary, translation and the /ws/view
broadcast are the same as for local events. English that a translating
provider attaches (``event.translation``) goes through the glossary and
bypasses the local translator; otherwise the local translator runs.

Missing key / auth failure / a stream that dies: a status with the cloud
error goes to the control sockets and the mic socket is closed with 1011.

Imported lazily by server.ws_transcribe (server globals are read at call time
so tests can monkeypatch ``server.open_session``, ``server.keys`` etc.).
"""

import asyncio
import dataclasses
import logging

from fastapi import WebSocket, WebSocketDisconnect

from gujusub import server
from gujusub.cloud.base import CloudError, SessionConfig
from gujusub.streaming import TranscriptEvent

logger = logging.getLogger("server.cloud")


class CloudPipeline(server._Pipeline):
    """receiver (raw PCM16) -> worker (session.send_audio); reader publishes session events.

    Events go through filtered() and the same publish() as local ones; English
    a translating provider attaches (event.translation) bypasses the local
    translator, otherwise the local translator runs as for local events.
    """

    def __init__(self, ws: WebSocket, backend: str):
        super().__init__(ws)
        self.backend = backend
        self.info = server.provider_info(backend)
        self.session = None
        self.config: SessionConfig | None = None
        self.warning: str | None = None
        self.error: str | None = None
        self.closing = False
        self.provider_last: dict[int, str] = {}  # utterance_id -> latest provider English

    def make_transcriber(self) -> None:
        return None  # the provider transcribes; no VAD/engines needed

    def build_config(self) -> tuple[SessionConfig, str | None]:
        s = server.get_settings()
        language, warning = s.asr_mode, None
        if language == "auto" and not self.info.language_id:
            language = "gu"
            warning = f"{self.info.label} has no language ID; running Gujarati"
        terms = server.get_glossary().terms()
        asks = self.info.translates and s.translate and not server.no_translate
        return SessionConfig(language=language, translate=asks, terms=terms), warning

    def status(self) -> dict:
        backlog = round(self.queued_samples / server.SAMPLE_RATE, 3)
        lang = self.config.language if self.config else server.get_settings().asr_mode
        session = self.session
        state = "error" if self.error else (session.state if session else "connecting")
        provider_tx = self.config is not None and self.config.translate
        translate = (self.translating() or (provider_tx and self.provider_translating()))
        return {"type": "status", "asr": lang, "translate": translate and lang != "en",
                "lid": lang == "auto", "decode_ms": 0.0, "avg_decode_ms": 0.0,
                "window_s": 0.0, "decodes": 0, "lag_s": backlog, "backlog_s": backlog,
                "utterances": self.utterances, "asr_warning": self.warning, "lid_last": None,
                "backend": self.backend,
                "cloud": {"provider": self.backend, "state": state,
                          "error": self.error or (session.error if session else None)},
                "models": server.models_state(), "translator": server.translator_state()}

    async def receive(self) -> None:
        try:
            while True:
                data = await self.ws.receive_bytes()
                if len(data) % 2:
                    data = data[:-1]  # PCM16: never split a sample
                self.queued_samples += len(data) // 2
                self.audio.put_nowait(data)
        finally:
            self.audio.put_nowait(None)

    async def _work(self) -> None:
        done = False
        while not done:
            chunks = [await self.audio.get()]
            while not self.audio.empty():
                chunks.append(self.audio.get_nowait())
            done = chunks[-1] is None
            data = b"".join(c for c in chunks if c is not None)
            if not data:
                continue
            self.queued_samples -= len(data) // 2
            await self.session.send_audio(data)

    def with_provider_translation(self, event: TranscriptEvent) -> TranscriptEvent:
        """Partials without provider English keep the latest English of their utterance."""
        if self.config is None or not self.config.translate:
            return event
        uid = event.utterance_id
        if event.translation is None:
            event = dataclasses.replace(event, translation=self.provider_last.get(uid, ""))
        else:
            self.provider_last[uid] = event.translation
        if event.type == "final":
            self.provider_last.pop(uid, None)
        return event

    async def read_events(self) -> None:
        try:
            async for event in self.session.events():
                await self.publish(server.filtered(self.with_provider_translation(event)))
        except Exception:
            logger.exception("cloud event reader failed for %s", self.ws.client)
        if self.session.state == "error" and not self.closing:
            await self.fail(self.session.error or f"{self.info.label} stream failed")

    async def fail(self, message: str) -> None:
        """Report a cloud error on the control sockets and close the mic with 1011."""
        self.error = message
        server._last_cloud_error = (self.backend, message)
        logger.error("%s: %s", self.info.label, message)
        await server.send_controls(self.status())
        try:
            await self.ws.close(code=1011)
        except RuntimeError:
            pass  # client already gone

    async def start_session(self) -> bool:
        creds = server.keys.get(self.info) if server.keys is not None else None
        if creds is None:
            await self.fail(f"No API key saved for {self.info.label}")
            return False
        self.config, self.warning = self.build_config()
        try:
            self.session = server.open_session(self.backend, creds, self.config)
            await self.session.start()
        except Exception as e:
            if not isinstance(e, CloudError):
                logger.exception("could not start %s session", self.backend)
            message = str(e) if isinstance(e, CloudError) else (
                f"could not connect to {self.info.label} ({type(e).__name__})")
            if self.session is not None:
                try:
                    await self.session.close()
                except Exception:
                    logger.debug("close after failed start raised", exc_info=True)
            await self.fail(message)
            return False
        if server._last_cloud_error is not None and server._last_cloud_error[0] == self.backend:
            server._last_cloud_error = None
        logger.info("%s session started for %s", self.backend, self.ws.client)
        server.ensure_translator(self.backend)
        return True

    async def _run(self) -> None:
        if not await self.start_session():
            return
        worker = asyncio.create_task(self.work())
        reader = asyncio.create_task(self.read_events())
        translating = asyncio.create_task(self.translate_partials())
        reporting = asyncio.create_task(self.report_status())
        try:
            await self.receive()
        except WebSocketDisconnect:
            logger.info("client disconnected: %s", self.ws.client)
        finally:
            try:
                await worker  # forwards the audio already queued
            finally:
                self.closing = True
                try:
                    await self.session.close()  # drains the provider's last finals
                except Exception:
                    logger.exception("closing the %s session failed", self.backend)
                try:  # the reader publishes the drained events, then ends
                    await asyncio.wait_for(reader, server.CLOUD_READER_DRAIN_S)
                except Exception:
                    logger.warning("cloud event reader did not finish; cancelled")
                translating.cancel()
                reporting.cancel()
            await asyncio.wait([translating, reporting])
