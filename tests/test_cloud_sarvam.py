"""Sarvam Saaras v3 realtime client against a fake socket (no network)."""

import asyncio
import base64
import json
import urllib.parse

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

from gujusub.cloud import sarvam
from gujusub.cloud.base import (
    CloudAuthError,
    CloudConfigError,
    CloudError,
    Credentials,
    SessionConfig,
)


class FakeSocket:
    def __init__(self):
        self.sent: list[str] = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        item = await self.inbox.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        self.closed = True

    def push(self, **msg):
        self.inbox.put_nowait(json.dumps(msg))


@pytest.fixture
def fake(monkeypatch):
    sock = FakeSocket()
    sock.calls = []

    async def connect(url, headers):
        sock.calls.append((url, headers))
        return sock

    monkeypatch.setattr(sarvam, "_connect", connect)
    return sock


def make(language="gu", terms=()):
    return sarvam.Session(Credentials("sk-test"),
                          SessionConfig(language=language, terms=tuple(terms)))


def query(url):
    parsed = urllib.parse.urlsplit(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == sarvam.ENDPOINT
    return {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}


async def drain(session, n):
    out = []
    async for ev in session.events():
        out.append(ev)
        if len(out) == n:
            break
    return out


def closed(code):
    return ConnectionClosedError(Close(code, "bye"), None)


def status_error(code):
    return InvalidStatus(Response(code, "x", Headers(), b""))


# ---- handshake -------------------------------------------------------------
@pytest.mark.parametrize("language,code", [("gu", "gu-IN"), ("en", "en-IN"), ("auto", "auto")])
def test_url_params_and_header(fake, language, code):
    async def run():
        s = make(language)
        await s.start()
        await s.close()
    asyncio.run(run())
    url, headers = fake.calls[0]
    q = query(url)
    assert q == {"language_code": code, "model": "saaras:v3-realtime", "stream_type": "fast",
                 "mode": "transcribe", "encoding": "linear16", "sample_rate": "16000",
                 "endpointing": "vad", "silence_duration_ms": "600"}
    assert headers == {"API-SUBSCRIPTION-KEY": "sk-test"}


def test_keyterms_joined_and_truncated():
    q = query(sarvam.build_url("gu", ("Swaminarayan", " Akshar ", "Swaminarayan", "")))
    assert q["keyterms"] == "Swaminarayan, Akshar"
    many = tuple(f"term{i:04d}" for i in range(200))
    hints = query(sarvam.build_url("gu", many))["keyterms"]
    assert len(hints) <= sarvam.MAX_KEYTERMS_CHARS
    assert all(t in many for t in hints.split(", "))  # no term cut in half
    assert len(hints) > sarvam.MAX_KEYTERMS_CHARS - 20


def test_unknown_language_is_config_error():
    with pytest.raises(CloudConfigError):
        sarvam.build_url("hi")


@pytest.mark.parametrize("status,exc", [(401, CloudAuthError), (403, CloudAuthError),
                                        (400, CloudConfigError), (500, CloudError)])
def test_handshake_status_mapping(monkeypatch, status, exc):
    async def connect(url, headers):
        raise status_error(status)
    monkeypatch.setattr(sarvam, "_connect", connect)
    with pytest.raises(exc) as info:
        asyncio.run(make().start())
    if exc is CloudError:
        assert not isinstance(info.value, (CloudAuthError, CloudConfigError))


def test_classify_close():
    assert isinstance(sarvam._classify_close(1003), CloudAuthError)
    assert isinstance(sarvam._classify_close(4000), CloudConfigError)
    assert sarvam._classify_close(1011) is None
    assert sarvam._classify_close(None) is None


# ---- client messages -------------------------------------------------------
def test_audio_ping_and_end_messages(fake):
    pcm = bytes(range(256)) * 4

    async def run():
        s = make()
        await s.start()
        await s.send_audio(pcm)
        await s.keepalive()
        await s.close()
    asyncio.run(run())
    msgs = [json.loads(m) for m in fake.sent]
    assert msgs[0] == {"event": "audio_input", "audio": base64.b64encode(pcm).decode()}
    assert msgs[1] == {"event": "ping"}
    assert msgs[-1] == {"event": "end"}
    assert sarvam.Session.keepalive_s == 5
    assert fake.closed


# ---- server messages -------------------------------------------------------
def test_partial_then_final_events(fake):
    async def run():
        s = make("gu")
        await s.start()
        fake.push(event="session.begin", request_id="r1", config={})
        fake.push(event="vad.speech_start", utterance_idx=0, confidence=0.9)
        fake.push(event="transcript.partial", utterance_idx=0, text="જય")
        fake.push(event="transcript.partial", utterance_idx=0, text="જય સ્વામિનારાયણ")
        fake.push(event="transcript.final", utterance_idx=0, text="જય સ્વામિનારાયણ")
        fake.push(event="vad.speech_end", utterance_idx=0, confidence=0.9)
        fake.push(event="pong")
        fake.push(event="transcript.partial", utterance_idx=1, text="બીજું")
        fake.push(event="transcript.final", utterance_idx=1, text="બીજું વાક્ય")
        evs = await drain(s, 5)
        await s.close()
        return evs
    evs = asyncio.run(run())
    assert [(e.type, e.utterance_id, e.committed, e.tail) for e in evs] == [
        ("partial", 1, "", "જય"),
        ("partial", 1, "", "જય સ્વામિનારાયણ"),
        ("final", 1, "જય સ્વામિનારાયણ", ""),
        ("partial", 2, "", "બીજું"),
        ("final", 2, "બીજું વાક્ય", ""),
    ]
    assert all(e.lang == "gu" and e.translation is None for e in evs)


def test_auto_language_mapping(fake):
    async def run():
        s = make("auto")
        await s.start()
        fake.push(event="transcript.final", utterance_idx=0, text="hello", language="en-IN")
        fake.push(event="transcript.final", utterance_idx=1, text="કેમ છો", language="gu-IN")
        fake.push(event="transcript.final", utterance_idx=2, text="नमस्ते", language="hi-IN")
        evs = await drain(s, 3)
        await s.close()
        return evs
    evs = asyncio.run(run())
    assert [e.lang for e in evs] == ["en", "gu", "gu"]  # unknown -> default (gu)
    assert [e.utterance_id for e in evs] == [1, 2, 3]


def test_session_end_stops_without_reconnect(fake):
    async def run():
        s = make()
        await s.start()
        fake.push(event="transcript.partial", utterance_idx=0, text="અધૂરું")
        fake.push(event="session.end", request_id="r1", total_utterances=1)
        fake.inbox.put_nowait(closed(1000))
        evs = [e async for e in s.events()]
        return s, evs
    s, evs = asyncio.run(run())
    assert [(e.type, e.tail or e.committed) for e in evs] == [("partial", "અધૂરું"),
                                                              ("final", "")]
    assert len(fake.calls) == 1 and s.state == "closed" and s.error is None


# ---- errors ----------------------------------------------------------------
@pytest.mark.parametrize("msg,exc", [
    ({"code": "unauthorized", "is_fatal": False, "status_code": 401}, CloudAuthError),
    ({"code": "forbidden", "is_fatal": True, "status_code": 403}, CloudAuthError),
    ({"code": "internal", "is_fatal": True, "status_code": 500}, CloudError),
])
def test_fatal_error_fails_session(fake, msg, exc):
    s = make()
    with pytest.raises(exc):
        asyncio.run(s.handle_message(json.dumps({"event": "error", "message": "m", **msg})))

    async def run():
        s2 = make()
        await s2.start()
        fake.push(event="error", message="boom", **msg)
        return s2, [e async for e in s2.events()]
    s2, evs = asyncio.run(run())
    assert evs == [] and s2.state == "error" and "Sarvam" in s2.error


def test_non_fatal_error_is_logged_and_ignored(fake, caplog):
    async def run():
        s = make()
        await s.start()
        fake.push(event="error", code="invalid_config", is_fatal=False,
                  message="bad vad", status_code=400)
        fake.push(event="transcript.final", utterance_idx=0, text="ચાલુ")
        evs = await drain(s, 1)
        await s.close()
        return s, evs
    s, evs = asyncio.run(run())
    assert evs[0].committed == "ચાલુ" and s.state == "closed"
    assert "invalid_config" in caplog.text


@pytest.mark.parametrize("code,exc", [(1003, CloudAuthError), (4000, CloudConfigError)])
def test_terminal_close_codes_fail_without_reconnect(fake, code, exc):
    async def run():
        s = make()
        await s.start()
        fake.inbox.put_nowait(closed(code))
        return s, [e async for e in s.events()]
    s, evs = asyncio.run(run())
    assert evs == [] and s.state == "error" and len(fake.calls) == 1
    assert s.error == str(sarvam._classify_close(code))


def test_other_close_reconnects(fake, monkeypatch):
    monkeypatch.setattr("gujusub.cloud.base.RECONNECT_BACKOFF_S", (0.0,))

    async def run():
        s = make()
        await s.start()
        fake.inbox.put_nowait(closed(1011))
        fake.push(event="transcript.final", utterance_idx=0, text="ફરી")
        evs = await drain(s, 1)
        await s.close()
        return evs
    evs = asyncio.run(run())
    assert evs[0].committed == "ફરી" and len(fake.calls) == 2
