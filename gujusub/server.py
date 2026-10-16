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
              {"type": "providers", "backends": [...], "catalog": [...],
               "keys": {<id>: {"set", "hint", "source", "region", "env"}}}, then
              one {"type": "status", ...}.
            client -> server: {"type": "get"} | {"type": "set", "settings": {...}}
              | {"type": "set_key", "provider": id, "key": str[, "region": str]}
              | {"type": "clear_key", "provider": id}
            set_key/clear_key -> every control client gets {"type": "keys",
              "keys": {...}} (masked hints only; the key is never sent back).
            Selecting a cloud asr_backend without a key is an error; selecting
              "local" when the models were skipped loads them once in the
              background (status "models": "loading" -> "ready"|"error").
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
              Every status also has "backend" ("local"|provider id), "cloud"
              (null, or {"provider", "state", "error"} for a cloud pipeline;
              state connecting|connected|reconnecting|closed|error) and
              "models" ("ready"|"loading"|"error"|"unloaded") and "translator"
              ("ready"|"loading"|"missing"; loaded on demand for cloud providers
              that do not translate).

Cloud backend (settings.asr_backend != "local"; see gujusub/cloud_pipeline.py):
the mic audio is streamed to the provider and its events take the same path.

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
                              [--backend local|soniox|...]
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

from gujusub.cloud.keys import KeyStore, validate_key, validate_region
from gujusub.cloud.registry import (
    BACKENDS,
    BY_ID,
    LOCAL,
    PROVIDERS,
    catalog,
    is_cloud,
    open_session,  # noqa: F401  (used by cloud_pipeline as server.open_session; tests patch it)
    provider_info,
)
from gujusub.engine import ASREngine
from gujusub.engines import (
    DEFAULT_GU_MODEL,
    GU_MODELS,
    EngineRegistry,
    load_engines,
    parse_engine_list,
)
from gujusub.fillers import clean_event
from gujusub.glossary import Glossary, load_glossary
from gujusub.lid import LanguageID, LidConfig, LidDecider
from gujusub.local_models import LocalModels, LocalSpec
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
no_translate = False  # --no-translate: also stops asking cloud providers for English
keys: KeyStore | None = None  # cloud API keys; created in main(); tests inject one
local_models: LocalModels | None = None  # runtime loader when started on a cloud backend
_last_cloud_error: tuple[str, str] | None = None  # (provider, message) for idle status
LOADING_MODELS_WARNING = "loading local models…"
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
CLOUD_READER_DRAIN_S = 5.0  # after session.close(): bound on publishing the last events


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


def models_state() -> str:
    """Local models: "ready" | "loading" | "error" | "unloaded" (cloud start)."""
    if engines is not None:
        return "ready"
    if local_models is not None and local_models.state in ("loading", "error"):
        return local_models.state
    return "unloaded"


def models_warning() -> str | None:
    """The background local load failed: it is not retried."""
    if engines is None and local_models is not None and local_models.state == "error":
        return (f"local models failed to load ({local_models.error}); "
                "restart the server to retry")
    return None


def translator_state() -> str:
    """Local translator: "ready" | "loading" (on-demand job) | "missing"."""
    if translator is not None:
        return "ready"
    if local_models is not None and local_models.translator_state == "loading":
        return "loading"
    return "missing"


def idle_status() -> dict:
    backend = get_settings().asr_backend
    cloud = None
    if _last_cloud_error is not None and _last_cloud_error[0] == backend:
        cloud = {"provider": backend, "state": "error", "error": _last_cloud_error[1]}
    return {"type": "status", "asr": None, "translate": False, "lid": False,
            "decode_ms": 0.0, "avg_decode_ms": 0.0, "window_s": 0.0,
            "decodes": 0, "lag_s": 0.0, "backlog_s": 0.0, "utterances": 0,
            "asr_warning": models_warning(), "lid_last": None,
            "backend": backend, "cloud": cloud, "models": models_state(),
            "translator": translator_state()}


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

    backend = LOCAL

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.transcriber = self.make_transcriber()
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
        self.waiting_models = False  # local models still loading in the background

    def make_transcriber(self) -> StreamingTranscriber | None:
        return new_transcriber()

    def enqueue(self, audio: np.ndarray) -> None:
        self.queued_samples += len(audio)
        self.audio.put_nowait(audio)

    def translating(self) -> bool:
        return self.translator is not None and get_settings().translate

    def provider_translating(self) -> bool:
        return get_settings().translate and not no_translate

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
                "utterances": self.utterances,
                "asr_warning": (LOADING_MODELS_WARNING if self.waiting_models
                                else models_warning() or warning),
                "lid_last": getattr(tx, "lid_last", None) if auto else None,
                "backend": LOCAL, "cloud": None, "models": models_state(),
                "translator": translator_state()}

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

    async def wait_for_models(self) -> bool:
        """Local models loading in the background: wait, then drop the stale audio.

        Returns False when the mic hung up while waiting; raises if the load failed.
        """
        if engines is not None or local_models is None or local_models.state != "loading":
            return True
        self.waiting_models = True
        try:
            ready = await local_models.wait()
        finally:
            self.waiting_models = False
        if not ready:
            raise RuntimeError(f"local models failed to load: {local_models.error}")
        self.translator = translator
        stale = []
        while not self.audio.empty():
            stale.append(self.audio.get_nowait())
        self.queued_samples = 0
        return all(chunk is not None for chunk in stale)

    async def _work(self) -> None:
        if not await self.wait_for_models():
            return
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
        if event.translation is not None:  # English from a translating cloud provider
            on = self.provider_translating() and eligible
            payload["translation"] = get_glossary().apply(event.translation) if on else ""
            self.translations.pop(uid, None)
        elif self.translator is not None and (
                not get_settings().translate or not eligible):
            # translation off, already English, or Auto-origin: no executor call
            payload["translation"] = ""
            self.translations.pop(uid, None)
        elif self.translator is None and translator_state() == "loading":
            payload["translation"] = ""  # on-demand translator not ready yet
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
    backend = get_settings().asr_backend
    if is_cloud(backend):
        from gujusub.cloud_pipeline import CloudPipeline  # imports this module

        await CloudPipeline(ws, backend).run()
    else:
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


def keys_status() -> dict[str, dict]:
    """Per-provider key status for the UI (masked hint only, never the key)."""
    if keys is None:
        return {p.id: {"set": False, "hint": "", "source": None, "region": None,
                       "env": p.key_env} for p in PROVIDERS}
    return keys.status(list(PROVIDERS))


def providers_payload() -> dict:
    return {"type": "providers", "backends": list(BACKENDS), "catalog": catalog(),
            "keys": keys_status()}


def backend_switch_error(partial: object) -> str | None:
    """A cloud backend can only be selected once it has credentials."""
    if not isinstance(partial, dict):
        return None
    backend = partial.get("asr_backend")
    if not isinstance(backend, str) or not is_cloud(backend):
        return None  # local, or invalid (the store reports that)
    if backend == get_settings().asr_backend:
        return None
    info = provider_info(backend)
    if keys is None or keys.get(info) is None:
        return f"No API key saved for {info.label}"
    return None


_background: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.ensure_future(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


def _local_models_done(loaded: LocalModels) -> None:
    """Runs on the event loop when the background load finishes."""
    global engines, translator, language_id
    if loaded.state == "ready":
        engines = loaded.engines
        if translator is None:
            translator = loaded.translator
        if language_id is None:
            language_id = loaded.language_id
        logger.info("local models ready (%s)", ",".join(engines.langs))
    else:
        logger.error("local models failed to load: %s", loaded.error)
    _spawn(send_controls(current_status()))


def ensure_local_models() -> bool:
    """Switched to "local" without the models in memory: load them once."""
    if engines is not None or local_models is None:
        return False
    need_translator = translator is None and local_models.translator_state != "loading"
    return local_models.start_background(
        asyncio.get_running_loop(), need_translator, _local_models_done)


def _translator_done(loaded: LocalModels) -> None:
    """Runs on the event loop when the on-demand translator job finishes."""
    global translator
    if loaded.translator_state == "ready":
        if translator is None:
            translator = loaded.translator
        for pipe in list(pipelines):  # running pipelines start translating now
            if pipe.translator is None:
                pipe.translator = translator
        logger.info("translator ready")
    _spawn(send_controls(current_status()))


def ensure_translator(backend: str) -> bool:
    """Cloud provider without built-in translation: load the local translator once."""
    if (translator is not None or local_models is None or no_translate
            or not get_settings().translate or not is_cloud(backend)
            or provider_info(backend).translates):
        return False
    return local_models.start_translator(asyncio.get_running_loop(), _translator_done)


_KEY_FIELDS = {"set_key": ({"type", "provider", "key"}, {"region"}),
               "clear_key": ({"type", "provider"}, set())}


async def handle_key_message(ws: WebSocket, msg: dict) -> None:
    """set_key / clear_key; the reply (and fan-out) is the masked key status."""
    global _last_cloud_error
    kind = msg["type"]
    required, optional = _KEY_FIELDS[kind]
    if not required <= set(msg) <= required | optional:
        await ws.send_json({"type": "error", "message": f"invalid {kind} message"})
        return
    if not _allow_control_set(ws):
        await ws.send_json({"type": "error", "message": "control update rate limit exceeded"})
        return
    provider = msg["provider"]
    info = BY_ID.get(provider) if isinstance(provider, str) else None
    if info is None:
        await ws.send_json({"type": "error", "message": f"unknown provider: {provider!r}"})
        return
    if keys is None:
        await ws.send_json({"type": "error", "message": "key storage unavailable"})
        return
    try:
        async with _settings_update_lock:
            if kind == "set_key":
                key = validate_key(msg["key"])
                region = validate_region(msg.get("region"))
                if info.needs_region and region is None:
                    raise ValueError(f"{info.label} needs a region (e.g. eastus)")
                await asyncio.to_thread(keys.set, info.id, key, region)
            else:
                await asyncio.to_thread(keys.clear, info.id)
    except ValueError as e:  # messages never contain the key
        await ws.send_json({"type": "error", "message": str(e)})
        return
    except OSError:
        logger.error("could not save API keys to %s", keys.path)
        await ws.send_json({"type": "error", "message": "could not save key"})
        return
    logger.info("API key %s for %s", "saved" if kind == "set_key" else "removed", info.id)
    if _last_cloud_error is not None and _last_cloud_error[0] == info.id:
        _last_cloud_error = None
    payload = {"type": "keys", "keys": keys_status()}
    await ws.send_json(payload)
    await send_controls(payload, exclude=ws)


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
    if kind in _KEY_FIELDS:
        await handle_key_message(ws, msg)
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
    switch_error = backend_switch_error(msg["settings"])
    if switch_error is not None:
        await ws.send_json({"type": "error", "message": switch_error})
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
    if new.asr_backend != previous.asr_backend:
        logger.info("asr_backend %s -> %s (applies to the next mic connection)",
                    previous.asr_backend, new.asr_backend)
        if new.asr_backend == LOCAL and ensure_local_models():
            await send_controls(current_status())  # models: "loading"
        elif ensure_translator(new.asr_backend):
            await send_controls(current_status())  # translator: "loading"
    return True


@app.websocket("/ws/control")
async def ws_control(ws: WebSocket) -> None:
    await ws.accept()
    controls.add(ws)
    logger.info("control connected: %s (%d total)", ws.client, len(controls))
    try:
        await ws.send_json({"type": "settings", **snapshot_payload(get_settings())})
        await ws.send_json(providers_payload())
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


def _make_local_models(args: argparse.Namespace) -> LocalModels:
    # late-binding wrappers: tests monkeypatch server.load_engines etc.
    return LocalModels(
        LocalSpec(langs=args.engines, device=args.device, gu_model=args.gu_model,
                  translate=not args.no_translate),
        load_engines=lambda *a, **kw: load_engines(*a, **kw),
        make_translator=lambda: Translator(),
        load_language_id=lambda: load_language_id(),
    )


def _start_cloud(backend: str, args: argparse.Namespace) -> None:
    global translator
    info = provider_info(backend)
    logger.info("backend %s: local ASR models not loaded", backend)
    if keys.get(info) is None:
        logger.warning("no API key for %s yet (set %s or add it on the mic page)",
                       info.label, info.key_env)
    if not args.no_translate and not info.translates:
        translator = Translator()
        translator.warmup()


def main() -> None:
    global engines, translator, filter_fillers, store, language_id
    global no_translate, keys, local_models
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--lang", default="gu",
                        help="deprecated: the Gujarati slot is always 'gu'; use --engines")
    parser.add_argument("--engines", default="gu,en", type=_engine_list_arg,
                        help="ASR engines to load: gu,en (default) or gu (low RAM)")
    parser.add_argument("--gu-model", default=DEFAULT_GU_MODEL, choices=GU_MODELS,
                        help="Gujarati ASR: full (600M multilingual IndicConformer, default) "
                             "or fast (Gujarati-only 120M int8: ~2x faster, ~2.3 GB less RAM)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-translate", action="store_true",
                        help="disable Gujarati->English translation")
    parser.add_argument("--no-filter", action="store_true",
                        help="show filler words (um, uh, ...) instead of hiding them")
    parser.add_argument("--backend", default=None, choices=BACKENDS,
                        help="speech engine: local models or a cloud provider "
                             "(saved in settings; default: the saved choice, else local)")
    args = parser.parse_args()

    filter_fillers = not args.no_filter
    no_translate = args.no_translate
    store = SettingsStore()
    store.load()
    if args.backend is not None:
        store.update({"asr_backend": args.backend})
    loaded = store.get()
    backend = loaded.asr_backend
    logger.info("settings: %s (asr_backend=%s asr_mode=%s translate=%s)", store.path,
                backend, loaded.asr_mode, loaded.translate)
    keys = KeyStore()
    keys.load()
    local_models = _make_local_models(args)
    if args.no_translate:
        logger.info("--no-translate: translation off regardless of settings")
    if args.lang != "gu":
        logger.warning("--lang %s ignored: use --engines (gu,en)", args.lang)
    started = time.perf_counter()
    if is_cloud(backend):
        _start_cloud(backend, args)
    else:
        local_models.load()
        engines, translator = local_models.engines, local_models.translator
        language_id = local_models.language_id
        logger.info("engines %s (gu model: %s) loaded and warm in %.1f s",
                    ",".join(engines.langs), args.gu_model, time.perf_counter() - started)
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
    # Run as the importable module so gujusub.cloud_pipeline (which imports
    # gujusub.server) sees the same globals that main() sets.
    from gujusub import server as _server

    _server.main()
