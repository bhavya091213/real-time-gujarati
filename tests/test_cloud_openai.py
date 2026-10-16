"""OpenAI Realtime transcription client against a fake socket (no network)."""

import asyncio
import base64
import json

import numpy as np
import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from gujusub.cloud import openai as oa
from gujusub.cloud.base import (
    CloudAuthError,
    CloudConfigError,
    CloudError,
    Credentials,
    SessionConfig,
)

KEY = "sk-test-0000"


class FakeSocket:
    def __init__(self, incoming=()):
        self.sent: list[str] = []
        self.incoming: asyncio.Queue = asyncio.Queue()
        for msg in incoming:
            self.incoming.put_nowait(msg)
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        return await self.incoming.get()

    async def close(self):
        self.closed = True


@pytest.fixture
def connect(monkeypatch):
    calls = []

    def install(sock=None, exc=None):
        async def fake(url, headers):
            calls.append((url, headers))
            if exc is not None:
                raise exc
            return sock

        monkeypatch.setattr(oa, "_connect", fake)
        return calls

    return install


def make(language="gu", terms=()):
    return oa.Session(Credentials(KEY), SessionConfig(language=language, terms=terms))


def collect(session, messages):
    events = []
    session.tracker._emit = events.append
    for m in messages:
        asyncio.run(session.handle_message(json.dumps(m)))
    return [(e.type, e.utterance_id, e.committed + e.tail, e.lang) for e in events]


# ---- handshake -------------------------------------------------------------

@pytest.mark.parametrize("language", ["gu", "en", "auto"])
def test_connect_sends_session_update(connect, language):
    sock = FakeSocket()
    calls = connect(sock)
    session = make(language, ("Swaminarayan", "Akshar"))
    ws = asyncio.run(session.connect())
    assert ws is sock
    url, headers = calls[0]
    assert url == "wss://api.openai.com/v1/realtime?intent=transcription"
    assert headers == {"Authorization": f"Bearer {KEY}"}
    msg = json.loads(sock.sent[0])
    assert msg["type"] == "session.update"
    s = msg["session"]
    assert s["type"] == "transcription"
    inp = s["audio"]["input"]
    assert inp["format"] == {"type": "audio/pcm", "rate": 24000}
    assert inp["turn_detection"] == {"type": "server_vad", "threshold": 0.5,
                                     "prefix_padding_ms": 300, "silence_duration_ms": 500}
    assert inp["noise_reduction"] == {"type": "near_field"}
    tr = inp["transcription"]
    assert tr["model"] == oa.MODEL == "gpt-4o-transcribe"
    if language == "auto":
        assert "language" not in tr
    else:
        assert tr["language"] == language
    assert tr["prompt"].endswith("Terms: Swaminarayan, Akshar")


def test_prompt_without_terms_and_length_cap():
    assert oa.build_prompt("gu") == "Gujarati devotional discourse."
    terms = tuple(f"term{i:04d}" for i in range(500)) + ("term0000", " ")
    prompt = oa.build_prompt("gu", terms)
    assert len(prompt) <= oa.MAX_PROMPT_CHARS
    assert prompt.startswith("Gujarati devotional discourse. Terms: term0000, term0001")
    assert prompt.count("term0000") == 1


@pytest.mark.parametrize("status,error", [(401, CloudAuthError), (403, CloudAuthError),
                                          (500, CloudError)])
def test_connect_http_status(connect, status, error):
    connect(exc=InvalidStatus(Response(status, "x", Headers())))
    with pytest.raises(error) as info:
        asyncio.run(make().connect())
    assert KEY not in str(info.value)
    if error is CloudError:
        assert not isinstance(info.value, CloudAuthError)


def test_connect_network_failure(connect):
    connect(exc=OSError("unreachable"))
    with pytest.raises(CloudError, match="could not reach OpenAI"):
        asyncio.run(make().connect())


# ---- audio -----------------------------------------------------------------

def pcm(values) -> bytes:
    return np.asarray(values, dtype="<i2").tobytes()


def test_resampler_length_and_monotone_ramp():
    r = oa.Resampler(16000, 24000)
    ramp = np.arange(16000) * 2 - 16000
    out = np.frombuffer(r(pcm(ramp)), dtype="<i2")
    assert out.size == 24000
    assert np.all(np.diff(out.astype(int)) >= 0)
    assert out[0] >= ramp[0] and out[-1] == ramp[-1]


def test_resampler_continuous_across_chunks():
    ramp = np.arange(16000) * 2 - 16000
    whole = oa.Resampler()(pcm(ramp))
    r = oa.Resampler()
    parts = b"".join(r(pcm(c)) for c in np.array_split(ramp, [1, 7, 160, 1601, 9999]))
    assert parts == whole
    out = np.frombuffer(parts, dtype="<i2").astype(int)
    assert np.all(np.diff(out) > 0)  # no step back or repeat at a boundary
    assert r(b"") == b""


def test_encode_audio_append_message():
    session = make()
    msg = json.loads(session.encode_audio(pcm([0, 300, 600, 900])))
    assert msg["type"] == "input_audio_buffer.append"
    samples = np.frombuffer(base64.b64decode(msg["audio"]), dtype="<i2")
    # first chunk: previous sample taken as the first one, so output lags one input sample
    assert samples.tolist() == [0, 100, 300, 500, 700, 900]


def test_finish_sends_nothing(connect):
    sock = FakeSocket()
    connect(sock)
    session = make()
    session.ws = asyncio.run(session.connect())
    asyncio.run(session.finish())
    assert len(sock.sent) == 1  # only the session.update


# ---- transcripts -----------------------------------------------------------

def delta(item, text):
    return {"type": oa.DELTA, "item_id": item, "delta": text}


def completed(item, text):
    return {"type": oa.COMPLETED, "item_id": item, "transcript": text}


def started(item):
    return {"type": "input_audio_buffer.speech_started", "item_id": item, "audio_start_ms": 0}


def test_delta_accumulates_then_completes():
    events = collect(make(), [
        {"type": "session.created"}, started("a"), delta("a", "jay "), delta("a", "swami"),
        {"type": "input_audio_buffer.speech_stopped", "item_id": "a"},
        {"type": "input_audio_buffer.committed", "item_id": "a"},
        completed("a", "jay swaminarayan"),
    ])
    assert events == [("partial", 1, "jay", "gu"), ("partial", 1, "jay swami", "gu"),
                      ("final", 1, "jay swaminarayan", "gu")]


def test_interleaved_items_finalise_in_speech_order():
    events = collect(make("en"), [
        started("a"), delta("a", "one"), started("b"), delta("b", "two"),
        completed("b", "two."), delta("a", " more"), completed("a", "one more."),
    ])
    assert events == [("partial", 1, "one", "en"), ("partial", 1, "one more", "en"),
                      ("final", 1, "one more.", "en"), ("final", 2, "two.", "en")]


def test_completed_without_deltas_and_failed_turn():
    events = collect(make("auto"), [
        started("a"), started("b"), {"type": oa.FAILED, "item_id": "a", "error": {}},
        completed("b", "kem cho"),
    ])
    assert events == [("final", 1, "kem cho", "gu")]  # auto -> gu; empty fail dropped


def test_stuck_turn_forced_out():
    msgs = [started(i) for i in "abcde"] + [delta("a", "held"), completed("e", "late")]
    events = collect(make(), msgs)
    assert ("final", 1, "held", "gu") in events


def test_connect_resets_item_buffers(connect):
    connect(FakeSocket())
    session = make()
    collect(session, [started("a"), delta("a", "x")])
    asyncio.run(session.connect())
    assert session.items == {} and session.done == {}


# ---- errors ----------------------------------------------------------------

@pytest.mark.parametrize("err,exc", [
    ({"type": "invalid_request_error", "code": "invalid_api_key",
      "message": f"Incorrect API key provided: {KEY}"}, CloudAuthError),
    ({"type": "authentication_error", "code": None, "message": "nope"}, CloudAuthError),
    ({"type": "invalid_request_error", "code": "x", "message": "Missing API key"},
     CloudAuthError),
    ({"type": "invalid_request_error", "code": "invalid_value",
      "message": "Invalid language 'gu'", "param": "session.audio.input.transcription"},
     CloudConfigError),
    ({"type": "invalid_request_error", "code": "invalid_value", "message": "bad value",
      "param": "session.audio.input.transcription.language"}, CloudConfigError),
    ({"type": "server_error", "code": "server_error", "message": "oops"}, CloudError),
])
def test_error_events(err, exc):
    with pytest.raises(exc) as info:
        asyncio.run(make().handle_message(json.dumps({"type": "error", "error": err})))
    assert type(info.value) is exc
    assert KEY not in str(info.value)


def test_error_fails_running_session(connect):
    sock = FakeSocket([json.dumps({"type": "error", "error": {"code": "invalid_api_key",
                                                              "message": "bad"}})])
    connect(sock)

    async def run():
        session = make()
        await session.start()
        events = [e async for e in session.events()]
        return session, events

    session, events = asyncio.run(run())
    assert events == [] and session.state == "error"
    assert "rejected the API key" in session.error
