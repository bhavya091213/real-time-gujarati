import asyncio
import json

import pytest

from gujusub.cloud import base
from gujusub.cloud.base import (
    CloudAuthError,
    CloudError,
    CloudSession,
    Credentials,
    SegmentTracker,
    SessionConfig,
    WebSocketSession,
)

# --- SegmentTracker ---------------------------------------------------------


@pytest.fixture
def tracked():
    events = []
    return events, SegmentTracker(events.append)


def test_tracker_numbers_utterances(tracked):
    events, tr = tracked
    tr.update("", "hello")
    tr.update("hello", "there")
    tr.finalize("hello there")
    tr.update("", "next")
    tr.finalize("next one")
    assert [(e.type, e.utterance_id, e.committed, e.tail) for e in events] == [
        ("partial", 1, "", "hello"), ("partial", 1, "hello", "there"),
        ("final", 1, "hello there", ""), ("partial", 2, "", "next"),
        ("final", 2, "next one", "")]


def test_tracker_dedupes_identical_partials(tracked):
    events, tr = tracked
    tr.update("a", "b")
    tr.update(" a ", "b ")  # same after strip
    tr.update("a", "b", translation="x")  # translation change is news
    assert len(events) == 2


def test_tracker_ignores_empty_updates_and_drops_empty_final(tracked):
    events, tr = tracked
    tr.update("", "  ")
    tr.finalize("")
    tr.finalize("   ")
    assert events == [] and tr.utterance_id == 0  # no id consumed


def test_tracker_final_without_partial_opens_an_utterance(tracked):
    events, tr = tracked
    tr.finalize("whole thing")
    assert [(e.type, e.utterance_id) for e in events] == [("final", 1)]


def test_tracker_empty_final_closes_an_open_partial(tracked):
    events, tr = tracked
    tr.update("", "dangling")
    tr.finalize("")
    assert events[-1].type == "final" and events[-1].utterance_id == 1


def test_tracker_carries_lang_and_translation():
    events = []
    tr = SegmentTracker(events.append, lang="en")
    tr.update("hi", lang="xx")  # unknown -> default
    tr.finalize("આવો", translation="welcome", lang="gu")
    assert (events[0].lang, events[0].translation) == ("en", None)
    assert (events[1].lang, events[1].translation) == ("gu", "welcome")


# --- WebSocketSession with a fake connection --------------------------------


class Closed(Exception):
    pass


class FakeConn:
    def __init__(self, on_send=None):
        self.sent = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.closed = False
        self.on_send = on_send

    async def send(self, data):
        if self.closed:
            raise Closed("send on closed socket")
        self.sent.append(data)
        if self.on_send is not None:
            self.on_send(self, data)

    async def recv(self):
        item = await self.inbox.get()
        if isinstance(item, Exception):
            raise item
        return item

    def push(self, **msg):
        self.inbox.put_nowait(json.dumps(msg))

    async def close(self):
        self.closed = True


def _eos(conn, data):
    if data == "EOS":  # provider flushes the pending text, then hangs up
        conn.push(final="flushed")
        conn.inbox.put_nowait(Closed("bye"))


class Session(WebSocketSession):
    def __init__(self, connects, config=None):
        super().__init__(Credentials("k" * 12), config or SessionConfig())
        self.connects = list(connects)  # FakeConn | Exception | asyncio.Event gate pairs
        self.connect_calls = 0

    async def connect(self):
        self.connect_calls += 1
        item = self.connects.pop(0)
        if isinstance(item, tuple):  # (gate, conn): wait before "connecting"
            gate, item = item
            await gate.wait()
        if isinstance(item, Exception):
            raise item
        return item

    def encode_audio(self, pcm16):
        return b"A" + pcm16

    async def handle_message(self, raw):
        msg = json.loads(raw)
        if "error" in msg:
            raise CloudError(msg["error"])
        if "final" in msg:
            self.tracker.finalize(msg["final"])
        else:
            self.tracker.update(msg.get("committed", ""), msg.get("pending", ""))

    async def finish(self):
        await self.ws.send("EOS")


@pytest.fixture(autouse=True)
def fast_reconnect(monkeypatch):
    monkeypatch.setattr(base, "RECONNECT_BACKOFF_S", (0.0,))


async def _collect(session):
    return [e async for e in session.events()]


async def _until(cond):
    for _ in range(400):
        if cond():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")


def test_session_start_send_events_close_ordering():
    async def run():
        conn = FakeConn(on_send=_eos)
        s = Session([conn])
        assert isinstance(s, CloudSession) and s.state == "connecting"
        await s.start()
        assert s.state == "connected"
        collector = asyncio.create_task(_collect(s))
        await s.send_audio(b"\x01\x00")
        conn.push(pending="hel")
        await _until(lambda: s.tracker.utterance_id == 1)
        await s.close()
        events = await asyncio.wait_for(collector, 2)
        return conn, s, events

    conn, s, events = asyncio.run(run())
    assert conn.sent == [b"A\x01\x00", "EOS"]  # audio, then end-of-stream
    assert conn.closed and s.state == "closed"
    assert [(e.type, e.utterance_id, e.committed or e.tail) for e in events] == [
        ("partial", 1, "hel"), ("final", 1, "flushed")]


def test_close_finalises_pending_partial_when_provider_sends_nothing():
    async def run():
        conn = FakeConn(on_send=lambda c, d: d == "EOS" and c.inbox.put_nowait(Closed()))
        s = Session([conn])
        await s.start()
        conn.push(committed="half", pending="said")
        await _until(lambda: s.tracker.open)
        await s.close()
        await s.close()  # idempotent
        return await asyncio.wait_for(_collect(s), 2)

    events = asyncio.run(run())
    assert [e.type for e in events] == ["partial", "final"]
    assert events[-1].committed == "half said"  # closed with what was last shown


def test_reconnect_after_recv_failure_resends_buffered_audio():
    async def run():
        first, second = FakeConn(), FakeConn()
        gate = asyncio.Event()
        s = Session([first, (gate, second)])
        await s.start()
        collector = asyncio.create_task(_collect(s))
        first.push(pending="before drop")
        first.inbox.put_nowait(Closed("network"))
        await _until(lambda: s.state == "reconnecting" and s.connect_calls == 2)
        await s.send_audio(b"\x02\x00")  # buffered while the socket is down
        await s.send_audio(b"\x03\x00")
        assert second.sent == []
        gate.set()
        await _until(lambda: s.state == "connected" and len(second.sent) == 2)
        second.push(final="after")
        await _until(lambda: s.tracker.utterance_id == 2)
        second.on_send = _eos
        await s.close()
        return first, second, s, await asyncio.wait_for(collector, 2)

    first, second, s, events = asyncio.run(run())
    assert first.closed
    assert second.sent[:2] == [b"A\x02\x00", b"A\x03\x00"]  # in order
    assert s.state == "closed"
    pairs = [(e.type, e.utterance_id) for e in events]
    assert pairs[:3] == [("partial", 1), ("final", 1), ("final", 2)]  # drop closes utt 1


def test_auth_error_on_reconnect_ends_with_error_state():
    async def run():
        first = FakeConn()
        s = Session([first, CloudAuthError("key rejected")])
        await s.start()
        collector = asyncio.create_task(_collect(s))
        first.inbox.put_nowait(Closed("network"))
        events = await asyncio.wait_for(collector, 2)  # events() ends by itself
        await s.close()
        return s, events

    s, events = asyncio.run(run())
    assert s.state == "error" and s.error == "key rejected"
    assert events == []


def test_gives_up_after_max_reconnects(monkeypatch):
    async def run():
        first = FakeConn()
        s = Session([first, *[OSError("down")] * base.MAX_RECONNECTS])
        await s.start()
        first.inbox.put_nowait(Closed("network"))
        await asyncio.wait_for(_collect(s), 2)
        return s

    s = asyncio.run(run())
    assert s.state == "error" and "gave up" in s.error
    assert s.connect_calls == 1 + base.MAX_RECONNECTS


def test_provider_error_message_fails_the_session():
    async def run():
        conn = FakeConn()
        s = Session([conn])
        await s.start()
        conn.push(error="quota exceeded")
        await asyncio.wait_for(_collect(s), 2)
        return s

    s = asyncio.run(run())
    assert (s.state, s.error) == ("error", "quota exceeded")


def test_audio_before_start_is_dropped():
    async def run():
        s = Session([])
        await s.send_audio(b"\x00\x00")  # not connected: nothing to send to
        return s

    assert asyncio.run(run()).ws is None


def test_reconnect_buffer_is_bounded(monkeypatch):
    s = Session([])
    s.state = "reconnecting"
    chunk = b"\x00" * base.BYTES_PER_SECOND  # 1 s
    for _ in range(5):
        asyncio.run(s.send_audio(chunk))
    assert s._pending_bytes <= base.BUFFER_WHILE_RECONNECTING_S * base.BYTES_PER_SECOND
