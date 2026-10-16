"""Deepgram provider against a fake socket (no network)."""

import asyncio
import json
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from gujusub.cloud import deepgram
from gujusub.cloud.base import (
    CloudAuthError,
    CloudConfigError,
    CloudError,
    Credentials,
    SessionConfig,
)


class FakeSocket:
    def __init__(self):
        self.sent: list = []
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


def run(coro):
    return asyncio.run(coro)


def install(monkeypatch, sock=None, status=None):
    calls = []

    async def fake_connect(url, headers):
        calls.append((url, headers))
        if status is not None:
            raise InvalidStatus(Response(status, "x", Headers(), b""))
        return sock

    monkeypatch.setattr(deepgram, "_connect", fake_connect)
    return calls


def params(url):
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == "wss://api.deepgram.com/v1/listen"
    return parse_qs(parts.query)


def results(text, is_final=False, speech_final=False):
    return json.dumps({"type": "Results", "is_final": is_final, "speech_final": speech_final,
                       "channel": {"alternatives": [{"transcript": text, "confidence": 0.9}]}})


def make(language="gu", terms=()):
    config = SessionConfig(language=language, terms=terms)
    return deepgram.Session(Credentials("secret-key"), config)


def drain(session):
    out = []
    while not session._queue.empty():
        ev = session._queue.get_nowait()
        if ev is not None:
            out.append((ev.type, ev.utterance_id, ev.committed, ev.tail, ev.lang))
    return out


@pytest.mark.parametrize("language,expected", [("gu", "gu"), ("auto", "gu"), ("en", "en")])
def test_url_params_and_header(monkeypatch, language, expected):
    calls = install(monkeypatch, FakeSocket())
    s = make(language)
    run(s.connect())
    url, headers = calls[0]
    q = params(url)
    assert q["model"] == ["nova-3"] and q["language"] == [expected]
    assert q["encoding"] == ["linear16"] and q["sample_rate"] == ["16000"]
    assert q["channels"] == ["1"] and q["interim_results"] == ["true"]
    assert q["endpointing"] == ["300"] and q["utterance_end_ms"] == ["1000"]
    for flag in ("vad_events", "smart_format", "punctuate"):
        assert q[flag] == ["true"]
    assert "keyterm" not in q
    assert headers == {"Authorization": "Token secret-key"}
    assert s.tracker.default_lang == expected


def test_keyterms_repeated_deduped_and_capped(monkeypatch):
    calls = install(monkeypatch, FakeSocket())
    run(make(terms=("Jai Shri Krishna", "  ", "satsang", "satsang")).connect())
    assert params(calls[0][0])["keyterm"] == ["Jai Shri Krishna", "satsang"]
    many = tuple(f"term{i:03d}" for i in range(200))
    capped = deepgram.keyterms(many)
    assert len(capped) == deepgram.MAX_KEYTERMS
    long = tuple("x" * 120 + str(i) for i in range(10))
    assert sum(map(len, deepgram.keyterms(long))) <= deepgram.MAX_KEYTERM_CHARS


def test_unsupported_language_is_config_error():
    with pytest.raises(CloudConfigError):
        deepgram.build_url("fr", 16000)


@pytest.mark.parametrize("status,exc", [(401, CloudAuthError), (403, CloudAuthError),
                                        (400, CloudConfigError), (500, CloudError)])
def test_upgrade_status_errors(monkeypatch, status, exc):
    install(monkeypatch, status=status)
    with pytest.raises(exc) as info:
        run(make().connect())
    assert "secret-key" not in str(info.value)
    if exc is CloudError:
        assert not isinstance(info.value, (CloudAuthError, CloudConfigError))


def test_start_401_raises_auth_error(monkeypatch):
    install(monkeypatch, status=401)
    with pytest.raises(CloudAuthError):
        run(make().start())


def test_audio_raw_keepalive_and_closestream_frames(monkeypatch):
    sock = FakeSocket()
    install(monkeypatch, sock)

    async def go():
        s = make()
        await s.start()
        await s.send_audio(b"\x01\x02\x03\x04")
        await s.keepalive()
        close = asyncio.create_task(s.close())
        await asyncio.sleep(0)
        sock.inbox.put_nowait(ConnectionError("closed by server"))
        await close
        return s

    s = run(go())
    assert deepgram.Session.keepalive_s == 4
    assert sock.sent[0] == b"\x01\x02\x03\x04"
    assert isinstance(sock.sent[1], str) and json.loads(sock.sent[1]) == {"type": "KeepAlive"}
    assert isinstance(sock.sent[2], str) and json.loads(sock.sent[2]) == {"type": "CloseStream"}
    assert sock.closed and s.state == "closed"


def test_interim_replaces_final_accumulates_speech_final_closes():
    s = make()

    async def go():
        for raw in (results("હું"), results("હું ઘરે"), results("હું ઘરે જાઉં છું.", is_final=True),
                    results("કાલે"), results("કાલે મળીશું.", is_final=True, speech_final=True),
                    results("next")):
            await s.handle_message(raw)

    run(go())
    assert drain(s) == [
        ("partial", 1, "", "હું", "gu"),
        ("partial", 1, "", "હું ઘરે", "gu"),
        ("partial", 1, "હું ઘરે જાઉં છું.", "", "gu"),
        ("partial", 1, "હું ઘરે જાઉં છું.", "કાલે", "gu"),
        ("final", 1, "હું ઘરે જાઉં છું. કાલે મળીશું.", "", "gu"),
        ("partial", 2, "", "next", "gu"),
    ]
    assert s.finals == [] and s.interim == "next"


def test_utterance_end_finalises_and_next_id_increments():
    s = make("en")

    async def go():
        await s.handle_message(results("hello there", is_final=True))
        await s.handle_message(json.dumps({"type": "UtteranceEnd", "last_word_end": 1.2}))
        await s.handle_message(json.dumps({"type": "UtteranceEnd", "last_word_end": 1.2}))
        await s.handle_message(results("again", is_final=True, speech_final=True))

    run(go())
    assert drain(s) == [
        ("partial", 1, "hello there", "", "en"),
        ("final", 1, "hello there", "", "en"),
        ("final", 2, "again", "", "en"),
    ]


def test_empty_transcripts_and_noise_ignored():
    s = make()

    async def go():
        await s.handle_message(results(""))
        await s.handle_message(results("", is_final=True))
        await s.handle_message(results("", is_final=True, speech_final=True))
        await s.handle_message(json.dumps({"type": "SpeechStarted", "timestamp": 2.1}))
        await s.handle_message(json.dumps({"type": "Metadata", "request_id": "r"}))
        await s.handle_message(b"\x00binary")
        await s.handle_message(json.dumps({"type": "Results", "channel": {"alternatives": []}}))

    run(go())
    assert drain(s) == []
    assert s.tracker.utterance_id == 0


def test_error_messages(caplog):
    s = make()
    run(s.handle_message(json.dumps({"type": "Error", "code": "X", "description": "slow down"})))
    with pytest.raises(CloudAuthError):
        run(s.handle_message(json.dumps({"type": "Error", "message": "Invalid credentials"})))


def test_auth_error_from_reader_fails_session(monkeypatch):
    sock = FakeSocket()
    install(monkeypatch, sock)

    async def go():
        s = make()
        await s.start()
        sock.inbox.put_nowait(json.dumps({"type": "Error", "description": "Unauthorized token"}))
        events = [ev async for ev in s.events()]
        await s.close()
        return s, events

    s, events = run(go())
    assert events == [] and s.state == "error" and "rejected" in s.error
