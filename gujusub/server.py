"""WebSocket ASR server.

Protocol:
  /ws       client -> server: binary frames of PCM16 mono @ 16 kHz
            server -> client: JSON TranscriptEvents
              {"type": "partial"|"final", "utterance_id": n,
               "committed": "...", "tail": "...", "translation": "..."}
  /ws/view  receive-only: first {"type": "settings", <display keys>, "schema"},
            then every TranscriptEvent from every /ws client (for /display);
            later settings changes arrive as further "settings" messages.
  /ws/control  JSON both ways (mic page settings panel).
            server -> client on connect: {"type": "settings", <all keys>}, then
              one {"type": "status", ...}.
            client -> server: {"type": "get"} | {"type": "set", "settings": {...}}
            "set" -> sender and other control clients get the full "settings",
              viewers get the display-only "settings"; a bad request gets
              {"type": "error", "message": "..."} and changes nothing.
            While a /ws pipeline runs, every STATUS_INTERVAL_S:
              {"type": "status", "asr": "gu"|"en"|"auto"|null, "translate": bool,
               "lid": bool, "decode_ms", "avg_decode_ms", "window_s",
               "decodes", "lag_s", "backlog_s", "utterances",
               "asr_warning": str|null, "lid_last": {...}|null}
              (lag_s == backlog_s: seconds of audio queued for the worker;
               asr_warning explains a fallback, e.g. en engine not loaded;
               lid_last is the latest Auto-mode LID outcome: utterance_id,
               lang ("gu"|"en"|"unsure"), p, speech_s, since_onset_s, fallback).

Mode routing: settings.asr_mode picks the engine when VAD opens an utterance
(select_engine); it stays pinned until that utterance's final, and events
carry its `lang`. English events are never translated (translation "").
Auto: select_engine returns an AutoRoute (fresh LidDecider over the shared
LanguageID); the transcriber pins gu/en once LID decides (see streaming.py).
Utterances opened in Auto are never translated (D6), Gujarati included.

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

Run:  python -m gujusub.server [--device cpu|mps] [--engines gu,en] [--port 8765]
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from gujusub.engine import ASREngine
from gujusub.engines import EngineRegistry, load_engines, parse_engine_list
from gujusub.fillers import clean_event
from gujusub.glossary import Glossary, load_glossary
from gujusub.lid import LanguageID, LidConfig, LidDecider
from gujusub.settings import Settings, SettingsStore, display_payload, snapshot_payload
from gujusub.streaming import AutoRoute, StreamingConfig, StreamingTranscriber, TranscriptEvent
from gujusub.translator import Translator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("server")

STATIC = Path(__file__).parent / "static"

app = FastAPI()
engines: EngineRegistry | None = None  # loaded once in main()
translator: Translator | None = None  # loaded once in main(); None if --no-translate
language_id: LanguageID | None = None  # loaded in main() when the en engine is
lid_config = LidConfig()  # Auto-mode decision policy (lid_fallback "gu")
filter_fillers = True  # set from --no-filter in main()
_glossary: Glossary | None = None  # lazily loaded; see get_glossary()
viewers: set[WebSocket] = set()  # connected /ws/view sockets
viewer_membership_lock = asyncio.Lock()  # guards viewers + _joining; never held across a send
VIEWER_INIT_TIMEOUT_S = 2.0  # bound on each send to a still-connecting viewer
_joining: dict[WebSocket, list[dict]] = {}  # connecting viewer -> broadcasts to flush
controls: set[WebSocket] = set()  # connected /ws/control sockets
store: SettingsStore | None = None  # created in main(); tests inject one
CONTROL_MAX_MESSAGE_BYTES = 8 * 1024
CONTROL_SET_RATE = 10.0
CONTROL_SET_BURST = 20
_control_clock = time.monotonic
_control_buckets: dict[str, tuple[float, float]] = {}
_settings_update_lock = asyncio.Lock()
pipelines: set["_Pipeline"] = set()  # active /ws pipelines (for status)
STATUS_INTERVAL_S = 1.0
SAMPLE_RATE = 16000
_warned_modes: set[str] = set()  # asr_mode values already warned about
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
        committed, tail = clean_event(committed, tail, final=event.type == "final",
                                      lang=getattr(event, "lang", "gu"))
    if getattr(event, "lang", "gu") == "en":
        glossary = get_glossary()
        committed, tail = glossary.apply(committed), glossary.apply(tail)
    if (committed, tail) == (event.committed, event.tail):
        return event
    return dataclasses.replace(event, committed=committed, tail=tail)


def new_transcriber() -> StreamingTranscriber:
    """One transcriber per /ws connection (tests monkeypatch this)."""
    return StreamingTranscriber(config=StreamingConfig(),
                                engine_for_utterance=select_engine,
                                thresholds_provider=confidence_thresholds)


def confidence_thresholds() -> tuple[float, float]:
    """(conf_word_min, conf_utt_min) from the live settings, read per decode."""
    s = get_settings()
    return s.conf_word_min, s.conf_utt_min


def latest_events(events: list[TranscriptEvent]) -> list[TranscriptEvent]:
    """Drop partials superseded by a later event of the same utterance."""
    return [
        ev
        for i, ev in enumerate(events)
        if ev.type == "final"
        or not any(later.utterance_id == ev.utterance_id for later in events[i + 1:])
    ]


def get_settings() -> Settings:
    return store.get() if store is not None else Settings()


def idle_status() -> dict:
    return {"type": "status", "asr": None, "translate": False, "lid": False,
            "decode_ms": 0.0, "avg_decode_ms": 0.0, "window_s": 0.0,
            "decodes": 0, "lag_s": 0.0, "backlog_s": 0.0, "utterances": 0,
            "asr_warning": None, "lid_last": None}


def current_status() -> dict:
    for pipe in list(pipelines):
        return pipe.status()
    return idle_status()


def active_asr() -> tuple[str, str | None]:
    """(engine lang or "auto" to run, fallback warning or None) for asr_mode.

    en needs the English engine; auto needs it plus the LID model (both loaded
    with --engines gu,en). Otherwise falls back to gu, logging once per mode.
    """
    mode = get_settings().asr_mode
    if mode == "gu":
        return "gu", None
    has_en = engines is not None and "en" in engines
    if mode == "en":
        if has_en:
            return "en", None
        warning = ("English engine not loaded (start with --engines gu,en); "
                   "running Gujarati")
    elif mode == "auto":
        if has_en and language_id is not None:
            return "auto", None
        warning = ("Auto needs the English engine and language ID "
                   "(start with --engines gu,en); running Gujarati")
    else:
        warning = f"asr_mode {mode!r} not implemented; running Gujarati"
    if mode not in _warned_modes:
        _warned_modes.add(mode)
        logger.warning(warning)
    return "gu", warning


def effective_asr_mode() -> str:
    return active_asr()[0]


def select_engine() -> ASREngine | AutoRoute:
    """Engine (or Auto-mode LID route) for a new utterance (executor thread)."""
    if engines is None:
        raise RuntimeError("no ASR engines loaded (server.main() not run)")
    mode = effective_asr_mode()
    if mode == "auto":
        return AutoRoute(LidDecider(language_id.classify, lid_config), engines.get)
    return engines.get(mode)


def translation_allowed(event: TranscriptEvent) -> bool:
    """Whether this utterance's pinned route permits Gujarati translation."""
    return event.lang == "gu" and event.translate_allowed


def load_language_id() -> LanguageID:
    return LanguageID()


async def send_controls(payload: dict, exclude: WebSocket | None = None) -> None:
    dead = []
    for ctl in list(controls):
        if ctl is exclude:
            continue
        try:
            await ctl.send_json(payload)
        except Exception:  # closed mid-send; dropped below
            dead.append(ctl)
    for ctl in dead:
        controls.discard(ctl)


async def broadcast(payload: dict) -> None:
    async with viewer_membership_lock:
        recipients = list(viewers)
        for backlog in _joining.values():  # connecting viewers get it after their snapshot
            backlog.append(payload)
    dead = []
    for viewer in recipients:
        try:
            await viewer.send_json(payload)
        except Exception:  # closed mid-send; dropped below
            dead.append(viewer)
    if dead:
        async with viewer_membership_lock:
            viewers.difference_update(dead)


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
        self.translation_eligibility: dict[int, bool] = {}  # pinned route policy
        self.requested: tuple[int, str] | None = None  # last (uid, committed) queued
        self.pending: tuple[int, str] | None = None  # latest-only slot
        self.wake = asyncio.Event()
        self.queued_samples = 0  # audio waiting in self.audio (status backlog)
        self.utterances = 0  # finals published

    def enqueue(self, audio: np.ndarray) -> None:
        self.queued_samples += len(audio)
        self.audio.put_nowait(audio)

    def translating(self) -> bool:
        return self.translator is not None and get_settings().translate

    def status(self) -> dict:
        tx = self.transcriber
        backlog = round(self.queued_samples / SAMPLE_RATE, 3)
        lang, warning = active_asr()
        auto = lang == "auto"
        return {"type": "status", "asr": lang,
                "translate": self.translating() and lang == "gu", "lid": auto,
                "decode_ms": float(getattr(tx, "last_decode_ms", 0.0)),
                "avg_decode_ms": float(getattr(tx, "avg_decode_ms", 0.0)),
                "window_s": float(getattr(tx, "last_window_s", 0.0)),
                "decodes": int(getattr(tx, "decode_count", 0)),
                "lag_s": backlog, "backlog_s": backlog,
                "utterances": self.utterances, "asr_warning": warning,
                "lid_last": getattr(tx, "lid_last", None) if auto else None}

    async def report_status(self) -> None:
        while True:
            await asyncio.sleep(STATUS_INTERVAL_S)
            try:
                await send_controls(self.status())
            except Exception:  # never let status reporting kill the pipeline
                logger.exception("status report failed")

    async def receive(self) -> None:
        try:
            while True:
                data = await self.ws.receive_bytes()
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self.enqueue(audio)
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
            self.queued_samples -= len(batch)
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
                self.translation_eligibility.pop(uid, None)
            return  # utterance was nothing but fillers
        payload = event.to_dict()
        eligible = translation_allowed(event)
        if event.type == "final":
            self.utterances += 1
            self.translation_eligibility.pop(uid, None)
        else:
            self.translation_eligibility[uid] = eligible
        if self.translator is not None and (
                not get_settings().translate or not eligible):
            # translation off, already English, or Auto-origin: no executor call
            payload["translation"] = ""
            self.translations.pop(uid, None)
        elif self.translator is not None:
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
        if (not get_settings().translate
                or not self.translation_eligibility.get(uid, False)):
            return  # switched off or route-ineligible while translation was in flight
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
        pipelines.add(self)
        try:
            await self._run()
        finally:
            pipelines.discard(self)
            if not pipelines:
                await send_controls(idle_status())

    async def _run(self) -> None:
        worker = asyncio.create_task(self.work())
        translating = asyncio.create_task(self.translate_partials())
        reporting = asyncio.create_task(self.report_status())
        try:
            await self.receive()
        except WebSocketDisconnect:
            logger.info("client disconnected: %s", self.ws.client)
        finally:
            try:
                await worker  # drains queued audio; never cancel mid-feed()
            finally:  # unconditional: shutdown may cancel us mid-await
                translating.cancel()
                reporting.cancel()
            try:  # submit flush before the next await: a cancel lands there
                events = await self.loop.run_in_executor(None, self.transcriber.flush)
                for raw in latest_events(events):
                    await self.publish(filtered(raw))
            except Exception:
                logger.exception("flush failed for %s", self.ws.client)
            # reap; asyncio.wait never re-raises the child's CancelledError
            await asyncio.wait([translating, reporting])


@app.websocket("/ws")
async def ws_transcribe(ws: WebSocket) -> None:
    await ws.accept()
    logger.info("client connected: %s", ws.client)
    await _Pipeline(ws).run()


async def _send_bounded(ws: WebSocket, payload: dict) -> None:
    await asyncio.wait_for(ws.send_json(payload), VIEWER_INIT_TIMEOUT_S)


async def _join_viewer(ws: WebSocket) -> bool:
    """Send the settings snapshot, then any broadcasts that raced it, then join.

    The membership lock covers only the decisions (register as joining; join
    once the backlog is empty), never a send, so a stalled connecting display
    cannot block broadcast(). Broadcasts during the snapshot are queued in
    order and flushed after it, preserving snapshot-before-events ordering.
    """
    async with viewer_membership_lock:
        _joining[ws] = []
    try:
        await _send_bounded(
            ws, {"type": "settings", **display_payload(get_settings())})
        while True:
            async with viewer_membership_lock:
                backlog = _joining[ws]
                if not backlog:
                    del _joining[ws]
                    viewers.add(ws)
                    return True
                _joining[ws] = []
            for payload in backlog:
                await _send_bounded(ws, payload)
    except Exception as e:  # includes TimeoutError
        async with viewer_membership_lock:
            _joining.pop(ws, None)
        logger.warning("viewer %s failed during init (%r); dropping", ws.client, e)
        try:
            await ws.close()
        except Exception:
            pass
        return False


@app.websocket("/ws/view")
async def ws_view(ws: WebSocket) -> None:
    await ws.accept()
    if not await _join_viewer(ws):
        return  # stalled/gone viewer: not added, socket closed
    logger.info("viewer connected: %s (%d total)", ws.client, len(viewers))
    try:
        while True:
            await ws.receive_text()  # viewers send nothing; this detects close
    except WebSocketDisconnect:
        pass
    finally:
        async with viewer_membership_lock:
            viewers.discard(ws)
        logger.info("viewer disconnected: %s", ws.client)


def _allow_control_set(ws: WebSocket) -> bool:
    client = ws.client
    source_ip = getattr(client, "host", None)
    if source_ip is None and isinstance(client, tuple) and client:
        source_ip = client[0]
    key = str(source_ip)
    now = _control_clock()
    tokens, updated_at = _control_buckets.get(
        key, (float(CONTROL_SET_BURST), now))
    tokens = min(
        float(CONTROL_SET_BURST),
        tokens + max(0.0, now - updated_at) * CONTROL_SET_RATE,
    )
    if tokens < 1.0:
        _control_buckets[key] = (tokens, now)
        return False
    _control_buckets[key] = (tokens - 1.0, now)
    return True


async def handle_control(ws: WebSocket, text: str) -> bool:
    if len(text.encode("utf-8")) > CONTROL_MAX_MESSAGE_BYTES:
        await ws.close(code=1009)
        return False
    try:
        msg = json.loads(text)
    except ValueError:
        await ws.send_json({"type": "error", "message": "message is not valid JSON"})
        return True
    kind = msg.get("type") if isinstance(msg, dict) else None
    if kind == "get":
        if set(msg) != {"type"}:
            await ws.send_json({"type": "error", "message": "invalid get message"})
            return True
        await ws.send_json({"type": "settings", **snapshot_payload(get_settings())})
        return True
    if kind != "set":
        await ws.send_json({"type": "error", "message": f"unknown message type: {kind!r}"})
        return True
    if set(msg) != {"type", "settings"}:
        await ws.send_json({"type": "error", "message": "invalid set message"})
        return True
    if not _allow_control_set(ws):
        await ws.send_json({"type": "error", "message": "control update rate limit exceeded"})
        return True
    try:
        async with _settings_update_lock:
            previous = store.get()
            new = await asyncio.to_thread(store.update, msg["settings"])
    except ValueError as e:
        await ws.send_json({"type": "error", "message": str(e)})
        return True
    except OSError:
        logger.exception("could not save settings to %s", store.path)
        await ws.send_json({"type": "error", "message": "could not save settings"})
        return True
    full = {"type": "settings", **snapshot_payload(new)}
    await ws.send_json(full)
    if new == previous:
        return True
    await send_controls(full, exclude=ws)
    await broadcast({"type": "settings", **display_payload(new)})
    return True


@app.websocket("/ws/control")
async def ws_control(ws: WebSocket) -> None:
    await ws.accept()
    controls.add(ws)
    logger.info("control connected: %s (%d total)", ws.client, len(controls))
    try:
        await ws.send_json({"type": "settings", **snapshot_payload(get_settings())})
        await ws.send_json(current_status())
        while True:
            if not await handle_control(ws, await ws.receive_text()):
                return
    except WebSocketDisconnect:
        pass
    finally:
        controls.discard(ws)
        logger.info("control disconnected: %s", ws.client)


def _engine_list_arg(text: str) -> tuple[str, ...]:
    try:
        return parse_engine_list(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def main() -> None:
    global engines, translator, filter_fillers, store, language_id
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--lang", default="gu",
                        help="deprecated: the Gujarati slot is always 'gu'; use --engines")
    parser.add_argument("--engines", default="gu,en", type=_engine_list_arg,
                        help="ASR engines to load: gu,en (default) or gu (low RAM)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-translate", action="store_true",
                        help="disable Gujarati->English translation")
    parser.add_argument("--no-filter", action="store_true",
                        help="show filler words (um, uh, ...) instead of hiding them")
    args = parser.parse_args()

    filter_fillers = not args.no_filter
    store = SettingsStore()
    store.load()
    loaded = store.get()
    logger.info("settings: %s (asr_mode=%s translate=%s)", store.path,
                loaded.asr_mode, loaded.translate)
    if args.no_translate:
        logger.info("--no-translate: translation off regardless of settings")
    if args.lang != "gu":
        logger.warning("--lang %s ignored: use --engines (gu,en)", args.lang)
    started = time.perf_counter()
    engines = load_engines(args.engines, args.device)
    if not args.no_translate:
        translator = Translator()
        translator.warmup()
    engines.warmup()
    if "en" in engines:  # Auto mode needs gu + en + LID
        language_id = load_language_id()
        language_id.warmup()
    logger.info("engines %s loaded and warm in %.1f s",
                ",".join(engines.langs), time.perf_counter() - started)
    logger.info("ready — mic: http://localhost:%d  display: http://localhost:%d/display",
                args.port, args.port)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning",
        ws_max_size=64 * 1024,
    )


if __name__ == "__main__":
    main()
