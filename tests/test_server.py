import pytest
from fastapi.testclient import TestClient

from gujusub import server
from gujusub.streaming import TranscriptEvent


@pytest.fixture
def client():
    return TestClient(server.app)


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
