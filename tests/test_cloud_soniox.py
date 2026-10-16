"""Soniox provider against a fake socket (no network)."""

import asyncio
import json

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedOK, InvalidStatus
from websockets.http11 import Response

from gujusub.cloud import soniox
from gujusub.cloud.base import CloudAuthError, CloudError, Credentials, SessionConfig


class FakeSocket:
    """Queued recv messages; recv raises ConnectionClosedOK once drained and closed."""

    def __init__(self, messages=()):
        self.sent: list = []
        self._inbox: asyncio.Queue = asyncio.Queue()
        for m in messages:
            self.push(m)
        self.closed = False

    def push(self, msg) -> None:
        self._inbox.put_nowait(msg if isinstance(msg, str) else json.dumps(msg))

    def end(self) -> None:
        self._inbox.put_nowait(None)

    async def send(self, data) -> None:
        self.sent.append(data)
        if data == "":  # end of stream: the server flushes, says finished, closes
            self.push({"tokens": [], "finished": True})
            self.end()

    async def recv(self):
        msg = await self._inbox.get()
        if msg is None:
            raise ConnectionClosedOK(None, None)
        return msg

    async def close(self) -> None:
        self.closed = True


def tok(text, final=True, lang="gu", status="original", **extra):
    return {"text": text, "is_final": final, "language": lang,
            "translation_status": status, **extra}


def tr(text, final=True):
    return tok(text, final, lang="en", status="translation", source_language="gu")


@pytest.fixture
def fake(monkeypatch):
    holder = {"sock": FakeSocket(), "calls": []}

    async def fake_connect(url, headers):
        holder["calls"].append((url, headers))
        return holder["sock"]

    monkeypatch.setattr(soniox, "_connect", fake_connect)
    return holder


def run(coro):
    return asyncio.run(coro)


async def collect(session, sock, messages):
    """Start, feed messages, end the stream, and return (events, sent frames)."""
    await session.start()
    for m in messages:
        sock.push(m)
    await asyncio.sleep(0.01)  # let the reader drain the queued messages
    await session.close()  # sends "", the fake answers finished + close
    return [ev async for ev in session.events()]


def make(language="gu", translate=True, terms=()):
    return soniox.Session(Credentials("sk-test"),
                          SessionConfig(language=language, translate=translate, terms=terms))


# ---- config frame -----------------------------------------------------------
def test_connect_sends_auth_header_and_config(fake):
    async def go():
        s = make("gu", translate=True)
        await s.start()
        await s.close()
    run(go())
    url, headers = fake["calls"][0]
    assert url == soniox.URL
    assert headers == {"Authorization": "Bearer sk-test"}
    cfg = json.loads(fake["sock"].sent[0])
    assert cfg["model"] == "stt-rt-v5"
    assert (cfg["audio_format"], cfg["sample_rate"], cfg["num_channels"]) == ("pcm_s16le", 16000, 1)
    assert cfg["language_hints"] == ["gu"]
    assert cfg["enable_language_identification"] is False
    assert cfg["enable_endpoint_detection"] is True
    assert cfg["max_endpoint_delay_ms"] == 1500
    assert cfg["endpoint_sensitivity"] == 0.3
    assert cfg["endpoint_latency_adjustment_level"] == 2
    assert cfg["translation"] == {"type": "one_way", "target_language": "en"}
    assert "context" not in cfg


@pytest.mark.parametrize("language,hints,lid", [
    ("gu", ["gu"], False), ("en", ["en"], False), ("auto", ["gu", "en"], True),
])
def test_language_hints(language, hints, lid):
    cfg = soniox.build_config(language, translate=False)
    assert cfg["language_hints"] == hints
    assert cfg["enable_language_identification"] is lid
    assert "translation" not in cfg


def test_translation_only_when_requested():
    assert "translation" in soniox.build_config("gu", True)
    assert "translation" in soniox.build_config("auto", True)
    assert "translation" not in soniox.build_config("gu", False)
    assert "translation" not in soniox.build_config("en", True)  # en -> en is pointless


def test_terms_go_into_context():
    cfg = soniox.build_config("gu", True, ("Swaminarayan", " ", "Mahant Swami"))
    assert cfg["context"]["general"] == [{"key": "domain", "value": soniox.DOMAIN}]
    assert cfg["context"]["terms"] == ["Swaminarayan", "Mahant Swami"]


def test_terms_budget_caps_count_and_size():
    many = tuple(f"t{i}" for i in range(1000))
    assert len(soniox.build_config("gu", True, many)["context"]["terms"]) == 300
    long = tuple("ભગવાન સ્વામિનારાયણ " * 4 + str(i) for i in range(300))
    cfg = soniox.build_config("auto", True, long)
    text = json.dumps(cfg, ensure_ascii=False)
    assert len(text) <= soniox.MAX_CONFIG_CHARS
    assert 0 < len(cfg["context"]["terms"]) < 300
    assert cfg["context"]["terms"] == list(long[: len(cfg["context"]["terms"])])
    # the fit is tight: one more term would overflow
    nxt = long[len(cfg["context"]["terms"])]
    assert len(text) + len(json.dumps(nxt, ensure_ascii=False)) + 2 > soniox.MAX_CONFIG_CHARS


# ---- connect errors -----------------------------------------------------------
def _refuse(monkeypatch, exc):
    async def boom(url, headers):
        raise exc
    monkeypatch.setattr(soniox, "_connect", boom)


@pytest.mark.parametrize("status", [401, 403])
def test_handshake_auth_failure(monkeypatch, status):
    _refuse(monkeypatch, InvalidStatus(Response(status, "Unauthorized", Headers())))
    with pytest.raises(CloudAuthError):
        run(make().start())


def test_handshake_other_failure(monkeypatch):
    _refuse(monkeypatch, InvalidStatus(Response(500, "Server Error", Headers())))
    with pytest.raises(CloudError) as ei:
        run(make().start())
    assert not isinstance(ei.value, CloudAuthError)
    _refuse(monkeypatch, OSError("dns"))
    with pytest.raises(CloudError):
        run(make().start())


# ---- control frames -----------------------------------------------------------
def test_audio_keepalive_and_finish_frames(fake):
    async def go():
        s = make()
        assert s.keepalive_s == 8
        await s.start()
        await s.send_audio(b"\x01\x02" * 10)
        await s.keepalive()
        await s.close()
    run(go())
    sent = fake["sock"].sent
    assert sent[1] == b"\x01\x02" * 10
    assert json.loads(sent[2]) == {"type": "keepalive"}
    assert sent[3] == ""
    assert fake["sock"].closed


# ---- token assembly -----------------------------------------------------------
def test_partials_finals_and_endpoint(fake):
    msgs = [
        {"tokens": [tok(" જય", False)]},
        {"tokens": [tok(" જય"), tok(" ", False), tok("સ્વામિ", False)]},
        {"tokens": [tok(" "), tok("સ્વામિનારાયણ"), tok("<end>")]},
        {"tokens": [tok(" બીજું", False)]},
        {"tokens": [tok(" બીજું"), tok("<fin>")]},
    ]
    events = run(collect(make(translate=False), fake["sock"], msgs))
    shape = [(e.type, e.utterance_id, e.committed, e.tail) for e in events]
    assert shape == [
        ("partial", 1, "", "જય"),
        ("partial", 1, "જય", "સ્વામિ"),
        ("final", 1, "જય સ્વામિનારાયણ", ""),  # endpoint in the same message: no partial
        ("partial", 2, "", "બીજું"),
        ("final", 2, "બીજું", ""),
    ]
    assert all(e.translation is None and e.lang == "gu" for e in events)


def test_nonfinals_are_replaced_each_message(fake):
    msgs = [
        {"tokens": [tok("કથા", False), tok(" ", False), tok("ચા", False)]},
        {"tokens": [tok("કથા", False)]},
    ]
    events = run(collect(make(translate=False), fake["sock"], msgs))
    assert [e.tail for e in events if e.type == "partial"] == ["કથા ચા", "કથા"]
    assert (events[-1].type, events[-1].utterance_id) == ("final", 1)  # close() flushes


def test_translation_attached_when_requested(fake):
    msgs = [
        {"tokens": [tok("જય"), tr(" Glory", False)]},
        {"tokens": [tr(" Glory"), tr(" be", False)]},
        {"tokens": [tr(" "), tr("be"), tok("<end>")]},
    ]
    events = run(collect(make(translate=True), fake["sock"], msgs))
    assert [e.translation for e in events] == ["Glory", "Glory be", "Glory be"]
    final = events[-1]
    assert (final.type, final.committed, final.translation) == ("final", "જય", "Glory be")


def test_translation_ignored_when_off(fake):
    msgs = [{"tokens": [tok("જય"), tr(" Glory"), tok("<end>")]}]
    events = run(collect(make(translate=False), fake["sock"], msgs))
    assert [(e.type, e.committed, e.translation) for e in events] == [
        ("final", "જય", None)]


def test_translation_status_none_is_original_text(fake):
    msgs = [{"tokens": [tok("Akshardham", status="none", lang="en"), tok("<end>")]}]
    events = run(collect(make(), fake["sock"], msgs))
    assert events[-1].committed == "Akshardham"
    assert events[-1].translation is None


def test_language_detection_in_auto_mode(fake):
    msgs = [
        {"tokens": [tok("Welcome", lang="en"), tok(" everyone", lang="en"), tok("<end>")]},
        {"tokens": [tok("જય", lang="gu"), tr(" Glory"), tok("<end>")]},
    ]
    events = run(collect(make("auto"), fake["sock"], msgs))
    finals = [e for e in events if e.type == "final"]
    assert [(e.committed, e.lang, e.translation) for e in finals] == [
        ("Welcome everyone", "en", None), ("જય", "gu", "Glory")]


def test_fixed_language_ignores_token_language(fake):
    msgs = [{"tokens": [tok("ok", lang="en"), tok("<end>")]}]
    events = run(collect(make("gu"), fake["sock"], msgs))
    assert {e.lang for e in events} == {"gu"}


def test_unsolicited_finished_finalizes_and_stops(fake):
    async def go():
        s = make(translate=False)
        await s.start()
        fake["sock"].push({"tokens": [tok("અધૂરું")]})
        fake["sock"].push({"tokens": [], "finished": True})
        fake["sock"].end()
        events = [ev async for ev in s.events()]  # ends without close()
        return s, events
    s, events = run(go())
    assert [(e.type, e.committed) for e in events] == [("partial", "અધૂરું"), ("final", "અધૂરું")]
    assert s.state == "closed" and s.error is None
    assert len(fake["calls"]) == 1  # no reconnect


# ---- provider errors ------------------------------------------------------------
@pytest.mark.parametrize("code,auth", [(401, True), (403, True), (402, False), (500, False)])
def test_error_message_fails_session(fake, code, auth):
    async def go():
        s = make()
        await s.start()
        fake["sock"].push({"tokens": [], "error_code": code, "error_type": "x",
                           "error_message": "nope"})
        events = [ev async for ev in s.events()]
        return s, events
    s, events = run(go())
    assert events == []
    assert s.state == "error" and "nope" in s.error
    assert ("API key" in s.error) is auth


def test_handle_message_raises_auth_error():
    s = make()
    with pytest.raises(CloudAuthError):
        run(s.handle_message(json.dumps({"tokens": [], "error_code": 401,
                                         "error_message": "bad key"})))
    with pytest.raises(CloudError):
        run(s.handle_message(json.dumps({"tokens": [], "error_code": 429,
                                         "error_message": "limit"})))
