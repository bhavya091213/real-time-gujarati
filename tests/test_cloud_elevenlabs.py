"""ElevenLabs Scribe v2 Realtime client against a fake socket (no network)."""

import asyncio
import base64
import json
import urllib.parse

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

from gujusub.cloud import elevenlabs
from gujusub.cloud.base import CloudAuthError, CloudError, Credentials, SessionConfig


class FakeSocket:
    def __init__(self):
        self.sent: list[str] = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.closed = False

    async def send(self, data):
        self.sent.append(data)
        if json.loads(data)["commit"]:  # the service flushes and closes after the commit
            self.inbox.put_nowait(ConnectionError("closed by server"))

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
    calls: list[tuple[str, dict]] = []
    sock = FakeSocket()

    async def connect(url, headers):
        calls.append((url, headers))
        return sock

    monkeypatch.setattr(elevenlabs, "_connect", connect)
    sock.calls = calls
    return sock


def make(language="gu", terms=()):
    return elevenlabs.Session(Credentials("sk-test"),
                              SessionConfig(language=language, terms=tuple(terms)))


def query(url):
    parsed = urllib.parse.urlsplit(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == elevenlabs.ENDPOINT
    return urllib.parse.parse_qs(parsed.query)


async def drain(session, n):
    out = []
    async for ev in session.events():
        out.append(ev)
        if len(out) == n:
            break
    return out


# ---- handshake -------------------------------------------------------------
@pytest.mark.parametrize("language,code", [("gu", ["guj"]), ("en", ["eng"]), ("auto", None)])
def test_url_language_and_fixed_params(fake, language, code):
    async def run():
        s = make(language)
        await s.start()
        await s.close()
    asyncio.run(run())
    url, headers = fake.calls[0]
    q = query(url)
    assert q.get("language_code") == code
    assert q["model_id"] == ["scribe_v2_realtime"]
    assert q["audio_format"] == ["pcm_16000"]
    assert q["commit_strategy"] == ["vad"]
    assert q["vad_silence_threshold_secs"] == ["0.8"]
    assert q["include_timestamps"] == ["true"]
    assert q["include_language_detection"] == ["true"]
    assert "keyterms" not in q
    assert headers == {"xi-api-key": "sk-test"}


def test_keyterms_repeated_filtered_and_capped():
    long_term = "x" * 21
    terms = ["Swaminarayan", long_term, "  ", "Akshar Purushottam", "Swaminarayan"]
    terms += [f"t{i}" for i in range(80)]
    q = query(elevenlabs.build_url("gu", tuple(terms)))
    kt = q["keyterms"]
    assert kt[:2] == ["Swaminarayan", "Akshar Purushottam"]
    assert long_term not in kt
    assert len(kt) == 50 and len(set(kt)) == 50
    assert all(len(t) <= 20 for t in kt)


def _invalid(status):
    return InvalidStatus(Response(status, "nope", Headers()))


@pytest.mark.parametrize("status", [401, 403])
def test_handshake_auth_rejection(monkeypatch, status):
    async def connect(url, headers):
        raise _invalid(status)
    monkeypatch.setattr(elevenlabs, "_connect", connect)
    with pytest.raises(CloudAuthError):
        asyncio.run(make().connect())


def test_handshake_other_status_is_cloud_error(monkeypatch):
    async def connect(url, headers):
        raise _invalid(500)
    monkeypatch.setattr(elevenlabs, "_connect", connect)
    with pytest.raises(CloudError) as info:
        asyncio.run(make().connect())
    assert not isinstance(info.value, CloudAuthError)


# ---- client messages -------------------------------------------------------
def test_audio_message_shape(fake):
    pcm = bytes(range(256)) * 4

    async def run():
        s = make()
        await s.start()
        await s.send_audio(pcm)
        await s.close()
    asyncio.run(run())
    msg = json.loads(fake.sent[0])
    assert msg == {"message_type": "input_audio_chunk",
                   "audio_base_64": base64.b64encode(pcm).decode(),
                   "commit": False, "sample_rate": 16000}


def test_finish_sends_commit(fake):
    async def run():
        s = make()
        await s.start()
        await s.close()
    asyncio.run(run())
    msg = json.loads(fake.sent[-1])
    assert msg["message_type"] == "input_audio_chunk"
    assert msg["audio_base_64"] == "" and msg["commit"] is True
    assert msg["sample_rate"] == 16000
    assert fake.closed


def test_keepalive_sends_silent_chunk(fake):
    async def run():
        s = make()
        s.keepalive_s = 0.01
        await s.start()
        await asyncio.sleep(0.05)
        await s.close()
    asyncio.run(run())
    first = json.loads(fake.sent[0])
    assert first["commit"] is False
    audio = base64.b64decode(first["audio_base_64"])
    assert audio == bytes(3200)  # 100 ms of 16 kHz PCM16 silence
    assert elevenlabs.Session.keepalive_s == 5.0


# ---- server messages -------------------------------------------------------
def test_partials_then_committed_final_with_language(fake):
    async def run():
        s = make("auto")
        await s.start()
        fake.push(message_type="session_started", session_id="abc", config={})
        fake.push(message_type="partial_transcript", text="hello")
        fake.push(message_type="partial_transcript", text="hello there")
        fake.push(message_type="committed_transcript", text="hello there")
        fake.push(message_type="committed_transcript_with_timestamps", text="hello there",
                  language_code="eng", words=[])
        fake.push(message_type="partial_transcript", text="કેમ છો")
        fake.push(message_type="committed_transcript", text="કેમ છો")
        fake.push(message_type="committed_transcript_with_timestamps", text="કેમ છો",
                  language_code="guj", words=[])
        evs = await drain(s, 5)
        await s.close()
        return evs, s
    evs, s = asyncio.run(run())
    assert [(e.type, e.utterance_id, e.committed, e.tail) for e in evs] == [
        ("partial", 1, "", "hello"),
        ("partial", 1, "", "hello there"),
        ("final", 1, "hello there", ""),
        ("partial", 2, "", "કેમ છો"),
        ("final", 2, "કેમ છો", ""),
    ]
    assert evs[2].lang == "en" and evs[4].lang == "gu"
    assert all(e.translation is None for e in evs)
    assert s.state == "closed"


def test_plain_committed_transcript_alone_is_ignored(fake):
    async def run():
        s = make()
        await s.start()
        fake.push(message_type="committed_transcript", text="ignored")
        fake.push(message_type="committed_transcript_with_timestamps", text="kept",
                  language_code="gu", words=[])
        evs = await drain(s, 1)
        await s.close()
        return evs
    evs = asyncio.run(run())
    assert [(e.type, e.committed, e.lang) for e in evs] == [("final", "kept", "gu")]


def test_unknown_language_code_falls_back_to_session_default(fake):
    async def run():
        s = make("en")
        await s.start()
        fake.push(message_type="committed_transcript_with_timestamps", text="hola",
                  language_code="spa", words=[])
        evs = await drain(s, 1)
        await s.close()
        return evs
    assert asyncio.run(run())[0].lang == "en"


def test_warning_is_logged_not_fatal(fake, caplog):
    async def run():
        s = make()
        await s.start()
        fake.push(message_type="warning", warning="slow down")
        fake.push(message_type="partial_transcript", text="still here")
        evs = await drain(s, 1)
        await s.close()
        return evs, s
    evs, s = asyncio.run(run())
    assert evs[0].tail == "still here" and s.state == "closed"
    assert "slow down" in caplog.text


@pytest.mark.parametrize("kind,auth", [("auth_error", True), ("unaccepted_terms", True),
                                       ("quota_exceeded", False), ("rate_limited", False),
                                       ("session_time_limit_exceeded", False),
                                       ("transcriber_error", False)])
def test_error_messages_fail_session(fake, kind, auth):
    async def run():
        s = make()
        await s.start()
        fake.push(message_type=kind, error="details here")
        evs = [e async for e in s.events()]
        await s.close()
        return evs, s
    evs, s = asyncio.run(run())
    assert evs == []
    assert s.state == "error"
    assert kind in s.error and "details here" in s.error

    with pytest.raises(CloudAuthError if auth else CloudError) as info:
        asyncio.run(make().handle_message(json.dumps({"message_type": kind, "error": "x"})))
    assert isinstance(info.value, CloudAuthError) == auth
