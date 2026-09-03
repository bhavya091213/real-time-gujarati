"""WebSocket ASR server.

Protocol:
  /ws       client -> server: binary frames of PCM16 mono @ 16 kHz
            server -> client: JSON TranscriptEvents
              {"type": "partial"|"final", "utterance_id": n,
               "committed": "...", "tail": "...", "translation": "..."}
  /ws/view  receive-only: every TranscriptEvent from every /ws client is
            broadcast here (for the broadcast display page at /display)

Run:  python -m gujusub.server [--device cpu|mps] [--lang gu] [--port 8765]
"""

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from gujusub.asr_engine import ASREngine
from gujusub.fillers import clean_event
from gujusub.streaming import StreamingConfig, StreamingTranscriber, TranscriptEvent
from gujusub.translator import Translator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("server")

STATIC = Path(__file__).parent / "static"

app = FastAPI()
engine: ASREngine | None = None  # loaded once in main()
translator: Translator | None = None  # loaded once in main(); None if --no-translate
filter_fillers = True  # set from --no-filter in main()
viewers: set[WebSocket] = set()  # connected /ws/view sockets


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/display")
async def display() -> FileResponse:
    return FileResponse(STATIC / "display.html")


def filtered(event: TranscriptEvent) -> TranscriptEvent:
    if not filter_fillers:
        return event
    committed, tail = clean_event(
        event.committed, event.tail, final=event.type == "final"
    )
    return TranscriptEvent(event.type, event.utterance_id, committed, tail)


async def broadcast(payload: dict) -> None:
    dead = []
    for viewer in viewers:
        try:
            await viewer.send_json(payload)
        except Exception:  # closed mid-send; dropped below
            dead.append(viewer)
    for viewer in dead:
        viewers.discard(viewer)


@app.websocket("/ws")
async def ws_transcribe(ws: WebSocket) -> None:
    await ws.accept()
    transcriber = StreamingTranscriber(engine, StreamingConfig())
    loop = asyncio.get_running_loop()
    logger.info("client connected: %s", ws.client)
    try:
        while True:
            data = await ws.receive_bytes()
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            # feed() runs VAD + (sometimes) inference; keep it off the event loop
            events = await loop.run_in_executor(None, transcriber.feed, audio)
            for raw in events:
                event = filtered(raw)
                text = f"{event.committed} {event.tail}".strip()
                if not text:
                    continue  # utterance was nothing but fillers
                payload = event.to_dict()
                if translator is not None:
                    payload["translation"] = await loop.run_in_executor(
                        None, translator.translate, text
                    )
                await ws.send_json(payload)
                await broadcast(payload)
    except WebSocketDisconnect:
        logger.info("client disconnected: %s", ws.client)
    finally:
        await loop.run_in_executor(None, transcriber.flush)


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
