import pytest
from fastapi.testclient import TestClient

from gujusub import server
from gujusub.settings import SettingsStore
from gujusub.streaming import TranscriptEvent


@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture(autouse=True)
def settings_store(monkeypatch, tmp_path):
    """Every test gets a fresh store in tmp_path (never ~/.cache)."""
    store = SettingsStore(tmp_path / "settings.json")
    store.load()
    monkeypatch.setattr(server, "store", store)
    monkeypatch.setattr(server, "_control_buckets", {})  # fresh rate-limit state per test
    return store


def test_display_page_served(client):
    r = client.get("/display")
    assert r.status_code == 200
    assert "ws/view" in r.text


def test_filtered_strips_fillers():
    ev = TranscriptEvent("partial", 1, "um hello", "there uh")
    out = server.filtered(ev)
    assert (out.committed, out.tail) == ("hello", "there")
    assert out.utterance_id == 1


def test_filtered_respects_no_filter(monkeypatch):
    monkeypatch.setattr(server, "filter_fillers", False)
    ev = TranscriptEvent("partial", 1, "um hello", "")
    assert server.filtered(ev) is ev


def test_viewer_receives_broadcast(client):
    with client, client.websocket_connect("/ws/view") as viewer:
        assert len(server.viewers) == 1
        assert viewer.receive_json()["type"] == "settings"  # snapshot first
        payload = {"type": "final", "utterance_id": 1, "committed": "hi", "tail": ""}
        client.portal.call(server.broadcast, payload)
        assert viewer.receive_json() == payload
    assert len(server.viewers) == 0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("This is my project", False),
        ("આ મારું પ્રોજેક્ટ છે અને translation.", True),
        ("My name is ભવ્ય", False),
        ("", False),
        ("...", False),
    ],
)
def test_looks_untranslated(text, expected):
    from gujusub.translator import looks_untranslated

    assert looks_untranslated(text) is expected


# --- per-connection pipeline: coalescing + async translation -------------

import asyncio  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from gujusub.streaming import StreamingTranscriber  # noqa: E402
from tests.fakes import FakeEngine, FakeVAD  # noqa: E402

MSG = 128  # samples per mic-page message (8 ms)


def pcm(n: int = MSG) -> bytes:
    return np.full(n, 1000, dtype=np.int16).tobytes()


class ScriptedTranscriber:
    """feed() returns the next scripted event list (then []); records lengths.
    `delay` makes each feed slow so a burst queues up behind it."""

    def __init__(self, script=(), delay: float = 0.0, done_at: int | None = None):
        self.script = list(script)
        self.delay = delay
        self.done_at = done_at  # emit a "done" partial once this many samples fed
        self.feeds: list[int] = []
        self.flushed = 0

    def feed(self, audio):
        assert audio.dtype == np.float32
        self.feeds.append(len(audio))
        if self.delay:
            time.sleep(self.delay)
        if self.done_at is not None and sum(self.feeds) >= self.done_at:
            return [P(0, "done")]
        return self.script.pop(0) if self.script else []

    def flush(self):
        self.flushed += 1
        return []


class FakeTranslator:
    """Records calls; optional threading gate blocks translate() until set."""

    def __init__(self, gate: threading.Event | None = None):
        self.calls: list[str] = []
        self.gate = gate

    def translate(self, text: str) -> str:
        self.calls.append(text)
        if self.gate is not None:
            assert self.gate.wait(5), "test gate never opened"
        return f"EN<{text}>"


def P(uid, committed, tail=""):
    return TranscriptEvent("partial", uid, committed, tail)


def F(uid, text):
    return TranscriptEvent("final", uid, text, "")


@pytest.fixture
def wire(monkeypatch):
    """Install a transcriber/translator into the server; returns a setter."""

    def install(transcriber, translator=None):
        monkeypatch.setattr(server, "new_transcriber", lambda: transcriber)
        monkeypatch.setattr(server, "translator", translator)
        monkeypatch.setattr(server, "filter_fillers", False)
        monkeypatch.setattr(server, "PARTIAL_TRANSLATE_GAP_S", 0.0)
        return transcriber

    return install


def eventually(cond, timeout=2.0):
    """Handler teardown (flush) can trail the client's view of the close."""
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "condition not reached in time"
        time.sleep(0.01)


def recv_until(ws, pred, limit=20):
    seen = []
    for _ in range(limit):
        msg = ws.receive_json()
        seen.append(msg)
        if pred(msg):
            return seen
    raise AssertionError(f"condition never met: {seen}")


def test_burst_is_coalesced_in_order(client, wire):
    tx = wire(ScriptedTranscriber(delay=0.05, done_at=40 * MSG))
    with client, client.websocket_connect("/ws") as ws:
        for _ in range(40):
            ws.send_bytes(pcm())
        assert ws.receive_json()["committed"] == "done"
    assert sum(tx.feeds) == 40 * MSG  # every sample fed, none lost
    assert len(tx.feeds) < 40  # but in far fewer feed() calls
    eventually(lambda: tx.flushed == 1)


def test_coalesced_samples_keep_order(client, wire):
    got = []

    class Rec(ScriptedTranscriber):
        def feed(self, audio):
            got.append(audio.copy())
            return super().feed(audio)

    wire(Rec(delay=0.02, done_at=30 * MSG))
    with client, client.websocket_connect("/ws") as ws:
        for i in range(30):
            ws.send_bytes(np.full(MSG, i, dtype=np.int16).tobytes())
        assert ws.receive_json()["committed"] == "done"
    audio = np.concatenate(got) * 32768.0
    expected = np.repeat(np.arange(30), MSG).astype(np.float32)
    assert np.array_equal(np.round(audio), expected)


def test_real_transcriber_through_pipeline(client, wire):
    eng = FakeEngine(["હા", "હા જી", "હા જી"])
    st = StreamingTranscriber(eng, vad=FakeVAD([(0, 40)]))
    wire(st)
    with client, client.websocket_connect("/ws") as ws:
        ws.send_bytes(pcm(512 * 70))  # 40 speech frames + 30 silence -> final
        seen = recv_until(ws, lambda m: m["type"] == "final")
    assert seen[-1]["committed"] == "હા જી"
    assert all("translation" not in m for m in seen)  # translation disabled


def test_latest_events_keeps_last_partial_and_all_finals():
    evs = [P(1, "a"), P(1, "a b"), F(1, "a b c"), P(2, "x"), P(2, "x y")]
    assert server.latest_events(evs) == [F(1, "a b c"), P(2, "x y")]
    assert server.latest_events([]) == []


def test_translate_only_when_committed_changes(client, wire):
    tr = FakeTranslator()
    wire(
        ScriptedTranscriber([[P(1, "a", "x")], [P(1, "a", "y")], [P(1, "a", "z")],
                             [F(1, "a z")]]),
        tr,
    )
    with client, client.websocket_connect("/ws") as ws:
        for _ in range(3):
            ws.send_bytes(pcm())
            recv_until(ws, lambda m: m["type"] == "partial" and m["tail"] != "")
        ws.send_bytes(pcm())
        seen = recv_until(ws, lambda m: m["type"] == "final")
    partial_calls = [c for c in tr.calls if c != "a z"]
    assert partial_calls == ["a x"]  # same committed -> translated once
    assert tr.calls[-1] == "a z"  # final always translated
    assert seen[-1]["translation"] == "EN<a z>"


def test_committed_change_triggers_new_translation(client, wire):
    tr = FakeTranslator()
    wire(ScriptedTranscriber([[P(1, "a", "x")], [P(1, "a b", "y")], [F(1, "a b")]]), tr)
    with client, client.websocket_connect("/ws") as ws:
        ws.send_bytes(pcm())
        recv_until(ws, lambda m: m.get("translation") == "EN<a x>")
        ws.send_bytes(pcm())
        seen = recv_until(ws, lambda m: m.get("translation") == "EN<a b y>")
        assert seen[0]["translation"] == "EN<a x>"  # carries last known translation
        ws.send_bytes(pcm())
        final = recv_until(ws, lambda m: m["type"] == "final")[-1]
    assert tr.calls == ["a x", "a b y", "a b"]
    assert final["translation"] == "EN<a b>"


def test_partials_do_not_wait_for_slow_translator(client, wire):
    gate = threading.Event()
    tr = FakeTranslator(gate)
    wire(ScriptedTranscriber([[P(1, "a", "x")], [P(1, "a", "x y")]]), tr)
    try:
        with client, client.websocket_connect("/ws") as ws:
            ws.send_bytes(pcm())
            first = ws.receive_json()
            ws.send_bytes(pcm())
            second = ws.receive_json()
            assert tr.calls == ["a x"]  # translation started but still blocked
            assert (first["tail"], second["tail"]) == ("x", "x y")
            assert first["translation"] == second["translation"] == ""
            gate.set()
            update = ws.receive_json()  # follow-up with the finished translation
            assert update["translation"] == "EN<a x>"
            assert update["tail"] == "x y"  # re-sends the latest state, never older
            assert update["utterance_id"] == 1 and update["type"] == "partial"
    finally:
        gate.set()


def test_stale_translation_not_resent_after_final(client, wire):
    gate = threading.Event()
    tr = FakeTranslator(gate)
    wire(ScriptedTranscriber([[P(1, "a", "x")], [F(1, "a x")], [P(2, "b")]]), tr)
    try:
        with client, client.websocket_connect("/ws") as ws:
            ws.send_bytes(pcm())
            assert ws.receive_json()["type"] == "partial"
            ws.send_bytes(pcm())
            gate.set()
            final = recv_until(ws, lambda m: m["type"] == "final")
            assert final[-1]["translation"] == "EN<a x>"
            ws.send_bytes(pcm())
            after = recv_until(ws, lambda m: m["utterance_id"] == 2)
            assert len(after) == 1  # nothing from utterance 1 after its final
            assert after[0]["translation"] == ""  # never utterance 1's text
    finally:
        gate.set()


def test_events_broadcast_to_viewers(client, wire):
    wire(ScriptedTranscriber([[P(1, "a")]]), FakeTranslator())
    with client, client.websocket_connect("/ws/view") as viewer:
        assert viewer.receive_json()["type"] == "settings"  # snapshot first
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(pcm())
            sent = ws.receive_json()
            assert viewer.receive_json() == sent


def test_disconnect_flush_final_is_broadcast_to_viewer(wire):
    from starlette.websockets import WebSocketDisconnect

    class FlushFinal(ScriptedTranscriber):
        def flush(self):
            self.flushed += 1
            return [P(1, "stale"), F(1, "finished")]

    class ClosedMic:
        client = "closed mic"

        def __init__(self):
            self.send_attempts = 0

        async def receive_bytes(self):
            raise WebSocketDisconnect()

        async def send_json(self, payload):
            self.send_attempts += 1
            raise RuntimeError("mic socket is closed")

    class Viewer:
        def __init__(self):
            self.sent = []

        async def send_json(self, payload):
            self.sent.append(payload)

    tx = wire(FlushFinal())
    mic = ClosedMic()
    viewer = Viewer()

    async def run():
        server.viewers.add(viewer)
        try:
            await server._Pipeline(mic).run()
        finally:
            server.viewers.discard(viewer)

    asyncio.run(run())

    assert viewer.sent == [F(1, "finished").to_dict()]
    assert mic.send_attempts == 1
    assert tx.flushed == 1


@dataclasses.dataclass(frozen=True)
class RichEvent(TranscriptEvent):
    confidence: float = 0.5


def test_filtered_preserves_extra_fields():
    out = server.filtered(RichEvent("partial", 3, "um hello", "uh", confidence=0.9))
    assert isinstance(out, RichEvent)
    assert (out.committed, out.tail, out.confidence) == ("hello", "", 0.9)


def test_worker_failure_closes_socket(client, wire):
    class Boom(ScriptedTranscriber):
        def feed(self, audio):
            raise RuntimeError("decode exploded")

    tx = wire(Boom())
    from starlette.websockets import WebSocketDisconnect

    with client, client.websocket_connect("/ws") as ws:
        ws.send_bytes(pcm())
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 1011
    assert tx.feeds == []  # Boom raised before recording; socket closed, not hung


def test_partial_translations_are_throttled_latest_only(client, wire, monkeypatch):
    tr = FakeTranslator()
    script = [[P(1, "a", "x")], [P(1, "a b")], [P(1, "a b c")], [P(1, "a b c d")]]
    wire(ScriptedTranscriber(script), tr)
    monkeypatch.setattr(server, "PARTIAL_TRANSLATE_GAP_S", 0.5)
    with client, client.websocket_connect("/ws") as ws:
        for (ev,) in script:
            ws.send_bytes(pcm())
            recv_until(ws, lambda m, c=ev.committed: m["committed"] == c)
        recv_until(ws, lambda m: m["translation"] == "EN<a b c d>")
    assert tr.calls == ["a x", "a b c d"]  # intermediate commits skipped


# --- round-2 review fixes: direct-asyncio harness ---------------------------

from starlette.websockets import WebSocketDisconnect  # noqa: E402


class FakeMic:
    """In-process /ws client: test pushes bytes into `inbox` (None = hang up)."""

    client = "fake mic"

    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed_with: int | None = None
        self.changed = asyncio.Event()

    async def receive_bytes(self):
        data = await self.inbox.get()
        if data is None:
            raise WebSocketDisconnect()
        return data

    async def send_json(self, payload):
        self.sent.append(payload)
        self.changed.set()

    async def close(self, code=1000):
        self.closed_with = code
        self.inbox.put_nowait(None)

    async def wait_for(self, pred, timeout=2.0):
        async def loop():
            while not any(pred(m) for m in self.sent):
                self.changed.clear()
                await self.changed.wait()

        await asyncio.wait_for(loop(), timeout)


class RecordingViewer:
    def __init__(self, on_send=None):
        self.sent: list[dict] = []
        self.on_send = on_send

    async def send_json(self, payload):
        self.sent.append(payload)
        if self.on_send is not None:
            self.on_send(self)


def test_viewer_initialization_orders_concurrent_broadcast(
        settings_store, monkeypatch):
    class BlockingViewer:
        client = "blocking viewer"

        def __init__(self):
            self.sent: list[dict] = []
            self.snapshot_started = asyncio.Event()
            self.release_snapshot = asyncio.Event()
            self.disconnect = asyncio.Event()

        async def accept(self):
            pass

        async def send_json(self, payload):
            if not self.sent:
                self.snapshot_started.set()
                await self.release_snapshot.wait()
            self.sent.append(payload)

        async def receive_text(self):
            await self.disconnect.wait()
            raise WebSocketDisconnect()

    final = {"type": "final", "utterance_id": 1, "committed": "hi", "tail": ""}
    expected_snapshot = {
        "type": "settings",
        **server.display_payload(settings_store.get()),
    }
    viewer = BlockingViewer()
    monkeypatch.setattr(server, "viewer_membership_lock", asyncio.Lock())

    async def run():
        view_task = asyncio.create_task(server.ws_view(viewer))
        await asyncio.wait_for(viewer.snapshot_started.wait(), 2)
        broadcast_task = asyncio.create_task(server.broadcast(final))
        await asyncio.sleep(0)  # let broadcast reach the initialization boundary
        viewer.release_snapshot.set()
        await asyncio.wait_for(broadcast_task, 2)
        viewer.disconnect.set()
        await asyncio.wait_for(view_task, 2)

    asyncio.run(run())

    assert viewer.sent == [expected_snapshot, final]
    assert viewer not in server.viewers


def test_stalled_connecting_viewer_does_not_block_broadcast(monkeypatch):
    monkeypatch.setattr(server, "VIEWER_INIT_TIMEOUT_S", 0.6)
    monkeypatch.setattr(server, "viewer_membership_lock", asyncio.Lock())

    class StalledViewer:
        client = "stalled viewer"

        def __init__(self):
            self.never = asyncio.Event()
            self.closed = False

        async def accept(self):
            pass

        async def send_json(self, payload):
            await self.never.wait()

        async def receive_text(self):
            await self.never.wait()

        async def close(self):
            self.closed = True

    stalled, joined = StalledViewer(), RecordingViewer()

    async def run():
        server.viewers.add(joined)
        try:
            view_task = asyncio.create_task(server.ws_view(stalled))
            await asyncio.sleep(0.05)  # stalled viewer is mid-settings-send
            t0 = asyncio.get_running_loop().time()
            await asyncio.wait_for(server.broadcast({"n": 1}), 0.5)
            assert asyncio.get_running_loop().time() - t0 < 0.5
            assert joined.sent == [{"n": 1}]
            assert stalled not in server.viewers  # still connecting
            await asyncio.wait_for(view_task, 2)  # timeout fires, handler ends
            assert stalled not in server.viewers
            assert stalled.closed
        finally:
            server.viewers.clear()

    asyncio.run(run())


def test_broadcast_survives_viewer_join_and_leave_mid_send():
    joiner, leaver = RecordingViewer(), RecordingViewer()

    def churn(_viewer):
        if leaver in server.viewers:
            server.viewers.add(joiner)  # OBS reload: a new /ws/view connects
        else:
            server.viewers.discard(joiner)  # ... and later goes away again

    first = RecordingViewer(on_send=churn)

    async def run():
        server.viewers.update({first, leaver})
        try:
            await server.broadcast({"n": 1})  # join mid-iteration
            await server.broadcast({"n": 2})
            server.viewers.discard(leaver)
            await server.broadcast({"n": 3})  # leave mid-iteration
        finally:
            server.viewers.clear()

    asyncio.run(run())  # used to raise "Set changed size during iteration"
    assert {"n": 2} in joiner.sent  # the joiner gets the next broadcast


def test_viewer_churn_does_not_close_mic(wire):
    joiner = RecordingViewer()
    churner = RecordingViewer(on_send=lambda _v: server.viewers.add(joiner))
    wire(ScriptedTranscriber([[P(1, "a")], [F(1, "a b")]]))

    async def run():
        mic = FakeMic()
        server.viewers.add(churner)
        try:
            task = asyncio.create_task(server._Pipeline(mic).run())
            mic.inbox.put_nowait(pcm())
            await mic.wait_for(lambda m: m["type"] == "partial")
            mic.inbox.put_nowait(pcm())
            await mic.wait_for(lambda m: m["type"] == "final")
            mic.inbox.put_nowait(None)
            await asyncio.wait_for(task, 2)
        finally:
            server.viewers.clear()
        return mic

    mic = asyncio.run(run())
    assert mic.closed_with is None  # captions kept flowing; no 1011
    assert any(m["type"] == "final" for m in joiner.sent)


def test_partial_translation_task_survives_a_failed_publish(wire, monkeypatch):
    tr = FakeTranslator()
    wire(ScriptedTranscriber([[P(1, "a", "x")], [P(1, "a b", "y")]]), tr)
    original = server._Pipeline.publish_translation
    failures = []

    async def flaky(self, uid, english):
        if not failures:
            failures.append(english)
            raise RuntimeError("viewer set changed mid-broadcast")
        await original(self, uid, english)

    monkeypatch.setattr(server._Pipeline, "publish_translation", flaky)

    async def run():
        mic = FakeMic()
        task = asyncio.create_task(server._Pipeline(mic).run())
        mic.inbox.put_nowait(pcm())
        await mic.wait_for(lambda m: m["committed"] == "a")
        await asyncio.wait_for(_until(lambda: failures), 2)
        mic.inbox.put_nowait(pcm())
        await mic.wait_for(lambda m: m.get("translation") == "EN<a b y>")
        mic.inbox.put_nowait(None)
        await asyncio.wait_for(task, 2)

    asyncio.run(run())
    assert failures == ["EN<a x>"]
    assert tr.calls == ["a x", "a b y"]


async def _until(cond):
    while not cond():
        await asyncio.sleep(0.005)


class SequenceTranslator:
    """Returns scripted outputs in order ("" = failed/dropped translation)."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[str] = []

    def translate(self, text):
        self.calls.append(text)
        return self.outputs.pop(0)


def test_failed_partial_translation_keeps_last_good_english(wire):
    tr = SequenceTranslator(["hello", ""])
    script = [[P(1, "a", "x")], [P(1, "a b")], [P(1, "a b", "z")]]
    wire(ScriptedTranscriber(script), tr)

    async def run():
        mic = FakeMic()
        pipe = server._Pipeline(mic)
        task = asyncio.create_task(pipe.run())
        mic.inbox.put_nowait(pcm())
        await mic.wait_for(lambda m: m.get("translation") == "hello")
        mic.inbox.put_nowait(pcm())
        await mic.wait_for(lambda m: m["committed"] == "a b")
        # the second translation fails ("") and its publish has finished
        await asyncio.wait_for(_until(lambda: len(tr.calls) == 2 and not pipe.pending
                                      and not pipe.wake.is_set()), 2)
        await asyncio.sleep(0.05)
        mic.inbox.put_nowait(pcm())
        await mic.wait_for(lambda m: m["tail"] == "z")
        mic.inbox.put_nowait(None)
        await asyncio.wait_for(task, 2)
        return mic

    mic = asyncio.run(run())
    latest = [m for m in mic.sent if m["tail"] == "z"][-1]
    assert latest["translation"] == "hello"  # not blanked by the failed one


def _translate_tasks():
    return [
        t for t in asyncio.all_tasks()
        if "translate_partials" in getattr(t.get_coro(), "__qualname__", "")
    ]


def test_cancelled_handler_cancels_translation_task(wire):
    gate = threading.Event()

    class Blocking(ScriptedTranscriber):
        def feed(self, audio):
            gate.wait(5)
            return []

    wire(Blocking(), FakeTranslator())

    async def run():
        mic = FakeMic()
        task = asyncio.create_task(server._Pipeline(mic).run())
        mic.inbox.put_nowait(pcm())
        mic.inbox.put_nowait(None)  # handler now sits in `await worker`
        await asyncio.sleep(0.05)
        assert _translate_tasks()
        task.cancel()  # server shutdown
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)
        leaked = [t for t in _translate_tasks() if not t.done()]
        gate.set()
        return leaked

    try:
        assert asyncio.run(run()) == []
    finally:
        gate.set()


def test_flush_failure_does_not_escape_handler(wire):
    class BadFlush(ScriptedTranscriber):
        def flush(self):
            self.flushed += 1
            raise RuntimeError("flush exploded")

    tx = wire(BadFlush(), FakeTranslator())

    async def run():
        mic = FakeMic()
        mic.inbox.put_nowait(None)
        await asyncio.wait_for(server._Pipeline(mic).run(), 2)

    asyncio.run(run())
    assert tx.flushed == 1


def test_filler_only_final_drops_cached_translation(wire):
    tr = FakeTranslator()
    wire(ScriptedTranscriber([[P(1, "a", "x")], [F(1, "")]]), tr)

    async def run():
        mic = FakeMic()
        pipe = server._Pipeline(mic)
        task = asyncio.create_task(pipe.run())
        mic.inbox.put_nowait(pcm())
        await mic.wait_for(lambda m: m.get("translation") == "EN<a x>")
        mic.inbox.put_nowait(pcm())  # final with nothing left after filtering
        mic.inbox.put_nowait(None)
        await asyncio.wait_for(task, 2)
        return pipe

    assert asyncio.run(run()).translations == {}


# --- settings: snapshot push, control socket, model-run flags -------------

from gujusub.settings import Settings, control_payload, display_payload  # noqa: E402


class FakeControlSocket:
    """Small in-process control socket for deterministic protocol tests."""

    def __init__(self, host="192.0.2.1", port=1000):
        self.client = type("Client", (), {"host": host, "port": port})()
        self.sent: list[dict] = []
        self.closed_with: int | None = None

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=1000):
        self.closed_with = code


def test_view_gets_settings_snapshot_before_events(client, wire, settings_store):
    settings_store.update({"size": 50})
    wire(ScriptedTranscriber([[P(1, "a")]]))
    with client, client.websocket_connect("/ws/view") as viewer:
        first = viewer.receive_json()
        assert first == {"type": "settings", **display_payload(settings_store.get())}
        assert first["size"] == 50 and "asr_mode" not in first
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(pcm())
            assert viewer.receive_json() == ws.receive_json()


def test_control_snapshot_and_idle_status_on_connect(client):
    with client, client.websocket_connect("/ws/control") as ctl:
        snap = ctl.receive_json()
        assert snap == {"type": "settings", **control_payload(Settings()),
                        "defaults": control_payload(Settings())}
        status = ctl.receive_json()
        assert status["type"] == "status" and status["asr"] is None
        ctl.send_json({"type": "get"})
        assert ctl.receive_json() == snap
    assert server.controls == set()


def test_control_snapshot_carries_defaults(client):
    with client, client.websocket_connect("/ws/control") as ctl:
        snap = ctl.receive_json()
        assert snap["defaults"] == control_payload(Settings())
        assert "defaults" not in control_payload(Settings())
    with client, client.websocket_connect("/ws/view") as viewer:
        assert "defaults" not in viewer.receive_json()


def test_control_set_fans_out_and_persists(client, settings_store):
    with client, client.websocket_connect("/ws/view") as viewer, \
            client.websocket_connect("/ws/control") as a, \
            client.websocket_connect("/ws/control") as b:
        viewer.receive_json()
        for ctl in (a, b):
            ctl.receive_json()  # settings
            ctl.receive_json()  # idle status
        a.send_json({"type": "set", "settings": {"size": 40, "translate": False}})
        ack = a.receive_json()
        assert ack["type"] == "settings" and ack["size"] == 40
        assert ack["translate"] is False
        other = b.receive_json()
        assert other == ack
        shown = viewer.receive_json()
        assert shown["type"] == "settings" and shown["size"] == 40
        assert "translate" not in shown
    assert settings_store.get().size == 40
    assert SettingsStore(settings_store.path).load() == settings_store.get()


@pytest.mark.parametrize("msg", [
    {"type": "set", "settings": {"size": -1}},
    {"type": "set", "settings": {"size": 1e100}},
    {"type": "set", "settings": {"bogus": 1}},
    {"type": "set", "settings": "size=1"},
    {"type": "explode"},
    ["not", "an", "object"],
])
def test_control_invalid_message_errors_and_changes_nothing(
        client, settings_store, msg, monkeypatch):
    control_fanout = []
    viewer_fanout = []

    async def record_control_fanout(payload, exclude=None):
        control_fanout.append((payload, exclude))

    async def record_viewer_fanout(payload):
        viewer_fanout.append(payload)

    monkeypatch.setattr(server, "send_controls", record_control_fanout)
    monkeypatch.setattr(server, "broadcast", record_viewer_fanout)
    with client, client.websocket_connect("/ws/control") as ctl:
        ctl.receive_json()
        ctl.receive_json()
        ctl.send_json(msg)
        err = ctl.receive_json()
        assert err["type"] == "error" and err["message"]
        ctl.send_json({"type": "get"})
        assert ctl.receive_json()["type"] == "settings"  # socket still usable
    assert settings_store.get() == Settings()
    assert not settings_store.path.exists()
    assert control_fanout == []
    assert viewer_fanout == []


def test_control_malformed_json_errors(client):
    with client, client.websocket_connect("/ws/control") as ctl:
        ctl.receive_json()
        ctl.receive_json()
        ctl.send_text("{nope")
        assert ctl.receive_json()["type"] == "error"


def test_control_oversized_utf8_closes_1009_before_processing(
        client, settings_store, monkeypatch):
    calls = {"parse": 0, "update": 0, "controls": 0, "viewers": 0}

    def forbidden_parse(_text):
        calls["parse"] += 1
        raise AssertionError("oversized control message was parsed")

    def forbidden_update(_partial):
        calls["update"] += 1
        raise AssertionError("oversized control message reached the store")

    async def forbidden_controls(_payload, exclude=None):
        calls["controls"] += 1
        raise AssertionError("oversized control message reached control fan-out")

    async def forbidden_viewers(_payload):
        calls["viewers"] += 1
        raise AssertionError("oversized control message reached viewer fan-out")

    with client, client.websocket_connect("/ws/control") as ctl:
        ctl.receive_json()  # settings snapshot
        ctl.receive_json()  # idle status
        monkeypatch.setattr(server.json, "loads", forbidden_parse)
        monkeypatch.setattr(settings_store, "update", forbidden_update)
        monkeypatch.setattr(server, "send_controls", forbidden_controls)
        monkeypatch.setattr(server, "broadcast", forbidden_viewers)
        text = "é" * (server.CONTROL_MAX_MESSAGE_BYTES // 2 + 1)
        assert len(text) <= server.CONTROL_MAX_MESSAGE_BYTES
        assert len(text.encode("utf-8")) > server.CONTROL_MAX_MESSAGE_BYTES
        ctl.send_text(text)
        with pytest.raises(WebSocketDisconnect) as exc:
            ctl.receive_text()
        assert exc.value.code == 1009

    assert calls == {"parse": 0, "update": 0, "controls": 0, "viewers": 0}


@pytest.mark.parametrize("msg", [
    {"type": "get", "extra": True},
    {"type": "set", "settings": {"size": 40}, "extra": True},
])
def test_control_rejects_extra_top_level_fields(
        client, settings_store, monkeypatch, msg):
    async def forbidden_fanout(*_args, **_kwargs):
        raise AssertionError("invalid envelope was broadcast")

    def forbidden_update(_partial):
        raise AssertionError("invalid envelope reached the store")

    monkeypatch.setattr(settings_store, "update", forbidden_update)
    monkeypatch.setattr(server, "send_controls", forbidden_fanout)
    monkeypatch.setattr(server, "broadcast", forbidden_fanout)
    with client, client.websocket_connect("/ws/control") as ctl:
        ctl.receive_json()
        ctl.receive_json()
        ctl.send_json(msg)
        assert ctl.receive_json()["type"] == "error"
        ctl.send_json({"type": "get"})
        assert ctl.receive_json()["type"] == "settings"


def test_control_set_rate_limit_is_shared_by_source_ip_and_refills(
        monkeypatch):
    class MemoryStore:
        def __init__(self):
            self.current = Settings()
            self.updates: list[dict] = []

        def get(self):
            return self.current

        def update(self, partial):
            self.updates.append(partial)
            self.current = self.current.with_updates(partial)
            return self.current

    memory_store = MemoryStore()
    now = [1000.0]
    control_fanout = []
    viewer_fanout = []

    async def record_controls(payload, exclude=None):
        control_fanout.append((payload, exclude))

    async def record_viewers(payload):
        viewer_fanout.append(payload)

    monkeypatch.setattr(server, "store", memory_store)
    monkeypatch.setattr(server, "_control_clock", lambda: now[0])
    monkeypatch.setattr(server, "_control_buckets", {})
    monkeypatch.setattr(server, "send_controls", record_controls)
    monkeypatch.setattr(server, "broadcast", record_viewers)
    a = FakeControlSocket("198.51.100.7", 1001)
    b = FakeControlSocket("198.51.100.7", 1002)

    async def send(ws, size):
        await server.handle_control(
            ws, json.dumps({"type": "set", "settings": {"size": size}}))

    async def run():
        for i in range(server.CONTROL_SET_BURST):
            await send(a if i % 2 == 0 else b, 25 + i)
        await send(b, 99)
        assert b.sent[-1]["type"] == "error"

        now[0] += 1 / server.CONTROL_SET_RATE
        await send(a, 100)
        assert a.sent[-1]["type"] == "settings"
        await send(b, 101)
        assert b.sent[-1]["type"] == "error"

    asyncio.run(run())

    accepted = server.CONTROL_SET_BURST + 1
    assert len(memory_store.updates) == accepted
    assert len(control_fanout) == accepted
    assert len(viewer_fanout) == accepted
    assert memory_store.current.size == 100


def test_control_noop_acknowledges_sender_without_fanout_or_save(
        settings_store, monkeypatch):
    control_fanout = []
    viewer_fanout = []

    async def record_controls(payload, exclude=None):
        control_fanout.append((payload, exclude))

    async def record_viewers(payload):
        viewer_fanout.append(payload)

    monkeypatch.setattr(server, "_control_buckets", {})
    monkeypatch.setattr(server, "send_controls", record_controls)
    monkeypatch.setattr(server, "broadcast", record_viewers)
    ws = FakeControlSocket("203.0.113.10")

    asyncio.run(server.handle_control(
        ws, json.dumps({"type": "set", "settings": {"size": 64}})))

    assert len(ws.sent) == 1
    assert ws.sent[0]["type"] == "settings" and ws.sent[0]["size"] == 64
    assert control_fanout == []
    assert viewer_fanout == []
    assert not settings_store.path.exists()


def test_control_mutations_run_off_loop_and_are_serialized(monkeypatch):
    class BlockingStore:
        def __init__(self):
            self.current = Settings()
            self.thread_ids: list[int] = []
            self.active = 0
            self.max_active = 0
            self.guard = threading.Lock()
            self.first_started = threading.Event()
            self.second_started = threading.Event()
            self.release_first = threading.Event()

        def get(self):
            return self.current

        def update(self, partial):
            with self.guard:
                self.thread_ids.append(threading.get_ident())
                ordinal = len(self.thread_ids)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                (self.first_started if ordinal == 1 else self.second_started).set()
            try:
                if ordinal == 1 and not self.release_first.wait(1):
                    raise TimeoutError("event loop could not release blocked save")
                self.current = self.current.with_updates(partial)
                return self.current
            finally:
                with self.guard:
                    self.active -= 1

    async def ignore_fanout(*_args, **_kwargs):
        pass

    blocking_store = BlockingStore()
    monkeypatch.setattr(server, "store", blocking_store)
    monkeypatch.setattr(server, "_control_buckets", {})
    monkeypatch.setattr(server, "send_controls", ignore_fanout)
    monkeypatch.setattr(server, "broadcast", ignore_fanout)
    a = FakeControlSocket("203.0.113.20", 2001)
    b = FakeControlSocket("203.0.113.21", 2002)

    async def run():
        loop_thread = threading.get_ident()
        first = asyncio.create_task(server.handle_control(
            a, json.dumps({"type": "set", "settings": {"size": 40}})))
        await asyncio.wait_for(
            asyncio.to_thread(blocking_store.first_started.wait, 0.5), 1)
        second = asyncio.create_task(server.handle_control(
            b, json.dumps({"type": "set", "settings": {"size": 41}})))
        await asyncio.sleep(0.05)
        assert not blocking_store.second_started.is_set()
        blocking_store.release_first.set()
        await asyncio.wait_for(asyncio.gather(first, second), 2)
        return loop_thread

    loop_thread = asyncio.run(run())

    assert blocking_store.second_started.is_set()
    assert blocking_store.max_active == 1
    assert all(thread_id != loop_thread for thread_id in blocking_store.thread_ids)
    assert blocking_store.current.size == 41


def test_main_configures_explicit_websocket_message_limit(
        settings_store, monkeypatch):
    run_calls = []

    class DummyEngine:
        def __init__(self, **_kwargs):
            pass

        def warmup(self):
            pass

    monkeypatch.setattr("sys.argv", ["gujusub.server", "--no-translate"])
    monkeypatch.setattr(server, "SettingsStore", lambda: settings_store)
    monkeypatch.setattr(server, "load_engines",
                        lambda langs, device: EngineRegistry(
                            {lang: DummyEngine() for lang in langs}))
    monkeypatch.setattr(server, "engines", None)
    monkeypatch.setattr(server, "translator", None)
    monkeypatch.setattr(server.uvicorn, "run",
                        lambda *args, **kwargs: run_calls.append((args, kwargs)))

    server.main()

    assert len(run_calls) == 1
    assert run_calls[0][1]["ws_max_size"] == 64 * 1024


def test_translate_off_never_calls_translator(client, wire, settings_store):
    settings_store.update({"translate": False})
    tr = FakeTranslator()
    wire(ScriptedTranscriber([[P(1, "a", "x")], [P(1, "a b", "y")], [F(1, "a b")]]), tr)
    with client, client.websocket_connect("/ws") as ws:
        seen = []
        for _ in range(3):
            ws.send_bytes(pcm())
            seen.append(ws.receive_json())
    assert tr.calls == []
    assert [m["translation"] for m in seen] == ["", "", ""]
    assert seen[-1]["type"] == "final"


def test_translate_flag_read_per_event(wire, settings_store):
    tr = FakeTranslator()
    wire(ScriptedTranscriber([[F(1, "a")], [F(2, "b")]]), tr)

    async def run():
        mic = FakeMic()
        task = asyncio.create_task(server._Pipeline(mic).run())
        mic.inbox.put_nowait(pcm())
        await mic.wait_for(lambda m: m["utterance_id"] == 1)
        settings_store.update({"translate": False})
        mic.inbox.put_nowait(pcm())
        await mic.wait_for(lambda m: m["utterance_id"] == 2)
        mic.inbox.put_nowait(None)
        await asyncio.wait_for(task, 2)
        return mic.sent

    sent = asyncio.run(run())
    assert tr.calls == ["a"]
    assert [m["translation"] for m in sent] == ["EN<a>", ""]


def test_auto_asr_mode_warns_once_and_runs_gu(wire, settings_store, caplog, monkeypatch):
    settings_store.update({"asr_mode": "auto"})
    monkeypatch.setattr(server, "_warned_modes", set())
    wire(ScriptedTranscriber())

    async def run():
        pipe = server._Pipeline(FakeMic())
        return [pipe.status(), pipe.status()]

    statuses = asyncio.run(run())
    assert [st["asr"] for st in statuses] == ["gu", "gu"]
    assert "not implemented" in statuses[0]["asr_warning"]
    assert caplog.text.count("asr_mode 'auto' not implemented") == 1


class CountingTranscriber(ScriptedTranscriber):
    decode_count = 3
    last_decode_ms = 180.0
    avg_decode_ms = 150.5
    last_window_s = 4.0


def test_status_shape_while_pipeline_active(monkeypatch, wire, settings_store):
    monkeypatch.setattr(server, "STATUS_INTERVAL_S", 0.01)
    wire(CountingTranscriber([[F(1, "a")]]), FakeTranslator())
    ctl = RecordingViewer()
    server.controls.add(ctl)

    async def run():
        mic = FakeMic()
        task = asyncio.create_task(server._Pipeline(mic).run())
        mic.inbox.put_nowait(pcm())
        await mic.wait_for(lambda m: m["type"] == "final")
        await asyncio.sleep(0.05)
        mic.inbox.put_nowait(None)
        await asyncio.wait_for(task, 2)

    try:
        asyncio.run(run())
    finally:
        server.controls.discard(ctl)
    statuses = [m for m in ctl.sent if m["type"] == "status"]
    assert statuses[-1] == server.idle_status()  # pipeline ended -> idle once
    live = [m for m in statuses if m["asr"] is not None]
    assert live
    last = live[-1]
    assert set(last) == {"type", "asr", "translate", "lid", "decode_ms",
                         "avg_decode_ms", "window_s", "decodes", "lag_s",
                         "backlog_s", "utterances", "asr_warning"}
    assert last["asr_warning"] is None
    assert last["asr"] == "gu" and last["translate"] is True and last["lid"] is False
    assert (last["decode_ms"], last["avg_decode_ms"]) == (180.0, 150.5)
    assert (last["window_s"], last["decodes"]) == (4.0, 3)
    assert last["utterances"] == 1
    assert last["lag_s"] == last["backlog_s"] >= 0


def test_backlog_counts_queued_audio(wire):
    wire(ScriptedTranscriber())

    async def run():
        pipe = server._Pipeline(FakeMic())
        pipe.enqueue(np.zeros(16000, dtype=np.float32))
        pipe.enqueue(np.zeros(8000, dtype=np.float32))
        return pipe.status()["backlog_s"]

    assert asyncio.run(run()) == 1.5


def test_status_reports_translate_off_without_translator(wire):
    wire(CountingTranscriber())  # translator None, as with --no-translate

    async def run():
        return server._Pipeline(FakeMic()).status()

    assert asyncio.run(run())["translate"] is False


def test_idle_status_shape():
    st = server.idle_status()
    assert st["type"] == "status" and st["asr"] is None
    assert st["translate"] is False and st["lid"] is False
# --- BAPS glossary hook ----------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LangEvent(TranscriptEvent):
    lang: str = "gu"


def test_filtered_applies_glossary_to_english_events():
    ev = LangEvent("final", 1, "um he gave prasad.", "to pramuk swami", lang="en")
    out = server.filtered(ev)
    assert out.committed == "he gave Prasad."
    assert out.tail == "to Pramukh Swami Maharaj"
    assert out.lang == "en"


def test_filtered_leaves_gujarati_mode_events_alone(monkeypatch):
    monkeypatch.setattr(server, "filter_fillers", False)
    ev = LangEvent("final", 1, "prasad", "pramuk swami", lang="gu")
    assert server.filtered(ev) is ev
    plain = TranscriptEvent("final", 1, "prasad", "")
    assert server.filtered(plain) is plain


def test_translation_text_gets_glossary():
    class T:
        def translate(self, text):
            return "he gave prasad."

    async def run():
        pipe = server._Pipeline.__new__(server._Pipeline)
        pipe.loop = asyncio.get_running_loop()
        pipe.translator = T()
        return await pipe.translate("x")

    assert asyncio.run(run()) == "he gave Prasad."


# --- mode routing (unit 3.2) ------------------------------------------------

from gujusub.engines import EngineRegistry  # noqa: E402


def _lang_engine(lang, script):
    eng = FakeEngine(script)
    eng.lang = lang
    return eng


@pytest.fixture
def registry(monkeypatch):
    def install(*langs):
        engines = {lang: _lang_engine(lang, [f"{lang}-text"]) for lang in langs}
        monkeypatch.setattr(server, "engines", EngineRegistry(engines))
        monkeypatch.setattr(server, "_warned_modes", set())
        return engines
    return install


def test_select_engine_follows_asr_mode(registry, settings_store):
    loaded = registry("gu", "en")
    assert server.select_engine() is loaded["gu"]
    settings_store.update({"asr_mode": "en"})
    assert server.select_engine() is loaded["en"]


def test_en_mode_without_en_engine_falls_back_to_gu(
        registry, settings_store, wire, caplog):
    loaded = registry("gu")
    settings_store.update({"asr_mode": "en"})
    assert server.select_engine() is loaded["gu"]
    assert server.select_engine() is loaded["gu"]
    assert caplog.text.count("English engine not loaded") == 1  # warned once
    wire(CountingTranscriber())

    async def run():
        return server._Pipeline(FakeMic()).status()

    status = asyncio.run(run())
    assert status["asr"] == "gu"
    assert "English engine not loaded" in status["asr_warning"]


def test_status_in_en_mode_reports_en_and_translate_off(
        registry, settings_store, wire):
    registry("gu", "en")
    settings_store.update({"asr_mode": "en", "translate": True})
    wire(CountingTranscriber(), FakeTranslator())

    async def run():
        return server._Pipeline(FakeMic()).status()

    status = asyncio.run(run())
    assert (status["asr"], status["translate"], status["asr_warning"]) == ("en", False, None)


def test_idle_status_has_no_warning():
    assert server.idle_status()["asr_warning"] is None


def test_en_events_are_never_translated(client, wire):
    tr = FakeTranslator()
    en_p = TranscriptEvent("partial", 1, "hello", "there", lang="en")
    en_f = TranscriptEvent("final", 1, "hello there", "", lang="en")
    wire(ScriptedTranscriber([[en_p], [en_f], [F(2, "ગુ")]]), tr)
    with client, client.websocket_connect("/ws") as ws:
        seen = []
        for _ in range(3):
            ws.send_bytes(pcm())
            seen.append(ws.receive_json())
    assert [(m["lang"], m["translation"]) for m in seen] == [
        ("en", ""), ("en", ""), ("gu", "EN<ગુ>")]
    assert tr.calls == ["ગુ"]  # never the English text


def test_filtered_preserves_lang_and_uses_english_filler_rules(monkeypatch):
    monkeypatch.setattr(server, "filter_fillers", True)
    ev = TranscriptEvent("final", 1, "So, um today we begin", "", lang="en")
    out = server.filtered(ev)
    assert out.lang == "en" and out.committed == "So, today we begin"
    gu = server.filtered(TranscriptEvent("final", 1, "so today we begin", ""))
    assert gu.lang == "gu" and gu.committed == "today we begin"


def test_control_set_en_switches_next_utterance(client, wire, registry):
    registry("gu", "en")
    tr = FakeTranslator()
    # utterance 1: frames 0-39, final after 19 silent frames; utterance 2 from 70
    st = StreamingTranscriber(engine_for_utterance=server.select_engine,
                              vad=FakeVAD([(0, 40), (70, 110)]))
    wire(st, tr)
    with client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(pcm(512 * 70))
            first = recv_until(ws, lambda m: m["type"] == "final")
            with client.websocket_connect("/ws/control") as ctl:
                ctl.receive_json()  # settings snapshot
                ctl.receive_json()  # status
                ctl.send_json({"type": "set", "settings": {"asr_mode": "en"}})
                assert recv_until(ctl, lambda m: m["type"] == "settings")[-1][
                    "asr_mode"] == "en"
            ws.send_bytes(pcm(512 * 70))
            second = recv_until(ws, lambda m: m["type"] == "final")
    assert {m["lang"] for m in first} == {"gu"}
    assert first[-1]["committed"] == "gu-text"
    assert first[-1]["translation"] == "EN<gu-text>"
    assert {m["lang"] for m in second} == {"en"}
    assert second[-1]["committed"] == "en-text"
    assert {m["translation"] for m in second} == {""}
    assert "en-text" not in tr.calls


def test_main_loads_requested_engines(settings_store, monkeypatch):
    requested = []

    class DummyEngine:
        def warmup(self):
            pass

    def fake_load(langs, device):
        requested.append((langs, device))
        return EngineRegistry({lang: DummyEngine() for lang in langs})

    monkeypatch.setattr("sys.argv", ["gujusub.server", "--no-translate", "--engines", "gu"])
    monkeypatch.setattr(server, "SettingsStore", lambda: settings_store)
    monkeypatch.setattr(server, "load_engines", fake_load)
    monkeypatch.setattr(server, "engines", None)
    monkeypatch.setattr(server, "translator", None)
    monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: None)
    server.main()
    assert requested == [(("gu",), "cpu")]
    assert server.engines.langs == ("gu",)
