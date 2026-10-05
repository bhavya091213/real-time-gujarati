"""WebSocket ASR server.

Protocol:
  /ws       client -> server: binary frames of PCM16 mono @ 16 kHz
            server -> client: JSON TranscriptEvents
              {"type": "partial"|"final", "utterance_id": n,
               "committed": "...", "tail": "...", "translation": "..."}
  /ws/view  receive-only: every TranscriptEvent from every /ws client is
            broadcast here (for the broadcast display page at /display)

Per-connection pipeline (see _Pipeline):
  receiver    only reads audio messages into an asyncio.Queue.
  worker      waits for audio, drains everything queued, and calls feed() once
              on the concatenation, so a backlog costs one executor hop instead
              of one per 8 ms message. Of the resulting events only the last
              partial per utterance is sent (older ones are already stale).
  translator  latest-only slot. Partials never wait for translation: they go
              out at once carrying the last translation known for their
              utterance. A partial whose committed text changed queues a
              translation (at most one start per PARTIAL_TRANSLATE_GAP_S; the
              slot keeps only the newest text); when it completes the latest
              partial is re-sent with the new translation (only if that
              utterance has no final yet).
              Finals are translated inline by the worker and sent exactly once,
              after their translation, so events stay in order.

Run:  python -m gujusub.server [--device cpu|mps] [--lang gu] [--port 8765]
"""

import argparse
import asyncio
import dataclasses
import logging
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from gujusub.asr_engine import ASREngine
from gujusub.fillers import clean_event
from gujusub.glossary import Glossary, load_glossary
from gujusub.streaming import StreamingConfig, StreamingTranscriber, TranscriptEvent
from gujusub.translator import Translator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("server")

STATIC = Path(__file__).parent / "static"

app = FastAPI()
engine: ASREngine | None = None  # loaded once in main()
translator: Translator | None = None  # loaded once in main(); None if --no-translate
filter_fillers = True  # set from --no-filter in main()
_glossary: Glossary | None = None  # lazily loaded; see get_glossary()
viewers: set[WebSocket] = set()  # connected /ws/view sockets
# Min seconds between partial-translation starts. Committed text grows on most
# ticks in continuous speech, so commit-change gating alone still translates
# ~55% of partials; the gap (latest-only, so the newest text wins) halves that.
PARTIAL_TRANSLATE_GAP_S = 1.0


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/display")
async def display() -> FileResponse:
    return FileResponse(STATIC / "display.html")


def get_glossary() -> Glossary:
    """BAPS spelling glossary (sub-millisecond; safe to call on the event loop)."""
    global _glossary
    if _glossary is None:
        _glossary = load_glossary()
    _glossary.maybe_reload()
    return _glossary


def filtered(event: TranscriptEvent) -> TranscriptEvent:
    """Fillers first, then glossary (glossary only for English captions)."""
    committed, tail = event.committed, event.tail
    if filter_fillers:
        committed, tail = clean_event(committed, tail, final=event.type == "final")
    if getattr(event, "lang", "gu") == "en":
        glossary = get_glossary()
        committed, tail = glossary.apply(committed), glossary.apply(tail)
    if (committed, tail) == (event.committed, event.tail):
        return event
    return dataclasses.replace(event, committed=committed, tail=tail)


def new_transcriber() -> StreamingTranscriber:
    """One transcriber per /ws connection (tests monkeypatch this)."""
    return StreamingTranscriber(engine, StreamingConfig())


def latest_events(events: list[TranscriptEvent]) -> list[TranscriptEvent]:
    """Drop partials superseded by a later event of the same utterance."""
    return [
        ev
        for i, ev in enumerate(events)
        if ev.type == "final"
        or not any(later.utterance_id == ev.utterance_id for later in events[i + 1:])
    ]


async def broadcast(payload: dict) -> None:
    dead = []
    for viewer in list(viewers):  # /ws/view can join or leave during the await
        try:
            await viewer.send_json(payload)
        except Exception:  # closed mid-send; dropped below
            dead.append(viewer)
    for viewer in dead:
        viewers.discard(viewer)


class _Pipeline:
    """receiver -> coalescing worker -> (latest-only) translator for one /ws."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.transcriber = new_transcriber()
        self.translator = translator
        self.loop = asyncio.get_running_loop()
        self.audio: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self.last_sent: dict | None = None  # most recent payload sent
        self.translations: dict[int, str] = {}  # utterance_id -> latest English
        self.requested: tuple[int, str] | None = None  # last (uid, committed) queued
        self.pending: tuple[int, str] | None = None  # latest-only slot
        self.wake = asyncio.Event()

    async def receive(self) -> None:
        try:
            while True:
                data = await self.ws.receive_bytes()
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self.audio.put_nowait(audio)
        finally:
            self.audio.put_nowait(None)  # tell the worker to stop after draining

    async def work(self) -> None:
        try:
            await self._work()
        except Exception:
            logger.exception("pipeline worker failed; closing %s", self.ws.client)
            try:
                await self.ws.close(code=1011)  # unblocks receive() so run() ends
            except RuntimeError:
                pass  # client already gone

    async def _work(self) -> None:
        done = False
        while not done:
            chunks = [await self.audio.get()]
            while not self.audio.empty():
                chunks.append(self.audio.get_nowait())
            done = chunks[-1] is None
            chunks = [c for c in chunks if c is not None]
            if not chunks:
                continue
            batch = np.concatenate(chunks)
            # feed() runs VAD + (sometimes) inference; keep it off the event loop
            events = await self.loop.run_in_executor(None, self.transcriber.feed, batch)
            for raw in latest_events(events):
                await self.publish(filtered(raw))

    async def publish(self, event: TranscriptEvent) -> None:
        uid = event.utterance_id
        text = f"{event.committed} {event.tail}".strip()
        if not text:
            if event.type == "final":
                self.translations.pop(uid, None)  # nothing published; still forget it
            return  # utterance was nothing but fillers
        payload = event.to_dict()
        if self.translator is not None:
            if event.type == "final":
                payload["translation"] = await self.translate(text)
                self.translations.pop(uid, None)
            else:
                payload["translation"] = self.translations.get(uid, "")
                if self.requested != (uid, event.committed):
                    self.requested = (uid, event.committed)
                    self.pending = (uid, text)
                    self.wake.set()
        await self.send(payload)

    async def translate(self, text: str) -> str:
        try:
            english = await self.loop.run_in_executor(None, self.translator.translate, text)
            return get_glossary().apply(english)
        except Exception:
            logger.exception("translation failed for %r", text)
            return ""

    async def translate_partials(self) -> None:
        while True:
            await self.wake.wait()
            self.wake.clear()
            if self.pending is None:
                continue
            uid, text = self.pending
            self.pending = None
            started = self.loop.time()
            try:
                english = await self.translate(text)
                await self.publish_translation(uid, english)
            except Exception:  # keep translating later partials
                logger.exception("partial translation update failed for %r", text)
            gap = PARTIAL_TRANSLATE_GAP_S - (self.loop.time() - started)
            if gap > 0:
                await asyncio.sleep(gap)  # newer text keeps landing in self.pending

    async def publish_translation(self, uid: int, english: str) -> None:
        """Re-send the latest partial of `uid` with a fresh translation."""
        last = self.last_sent
        if last is None or last["utterance_id"] != uid or last["type"] != "partial":
            return  # utterance already finalized (or nothing to update)
        if not english:
            return  # failed/dropped: partials keep the last good English
        self.translations[uid] = english
        if english != last.get("translation"):
            await self.send({**last, "translation": english})

    async def send(self, payload: dict) -> None:
        self.last_sent = payload
        try:
            await self.ws.send_json(payload)
        except Exception:  # mic client gone; viewers still get it
            logger.debug("send to closed /ws client dropped")
        await broadcast(payload)

    async def run(self) -> None:
        worker = asyncio.create_task(self.work())
        translating = asyncio.create_task(self.translate_partials())
        try:
            await self.receive()
        except WebSocketDisconnect:
            logger.info("client disconnected: %s", self.ws.client)
        finally:
            try:
                await worker  # drains queued audio; never cancel mid-feed()
            finally:  # unconditional: shutdown may cancel us mid-await
                translating.cancel()
            try:  # submit flush before the next await: a cancel lands there
                events = await self.loop.run_in_executor(None, self.transcriber.flush)
                for raw in latest_events(events):
                    await self.publish(filtered(raw))
            except Exception:
                logger.exception("flush failed for %s", self.ws.client)
            # reap; asyncio.wait never re-raises the child's CancelledError
            await asyncio.wait([translating])


@app.websocket("/ws")
async def ws_transcribe(ws: WebSocket) -> None:
    await ws.accept()
    logger.info("client connected: %s", ws.client)
    await _Pipeline(ws).run()


@app.websocket("/ws/view")
async def ws_view(ws: WebSocket) -> None:
    await ws.accept()
    viewers.add(ws)
    logger.info("viewer connected: %s (%d total)", ws.client, len(viewers))
    try:
        while True:
            await ws.receive_text()  # viewers send nothing; this detects close
    except WebSocketDisconnect:
        pass
    finally:
        viewers.discard(ws)
        logger.info("viewer disconnected: %s", ws.client)


def main() -> None:
    global engine, translator, filter_fillers
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--lang", default="gu")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-translate", action="store_true",
                        help="disable Gujarati->English translation")
    parser.add_argument("--no-filter", action="store_true",
                        help="show filler words (um, uh, ...) instead of hiding them")
    args = parser.parse_args()

    filter_fillers = not args.no_filter
    engine = ASREngine(lang=args.lang, device=args.device)
    if not args.no_translate:
        translator = Translator()
        translator.warmup()
    logger.info("warming up model ...")
    engine.warmup()
    logger.info("ready — mic: http://localhost:%d  display: http://localhost:%d/display",
                args.port, args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
