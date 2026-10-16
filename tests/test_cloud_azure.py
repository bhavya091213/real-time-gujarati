"""Azure Speech SDK session against a fake ``speechsdk`` module (no SDK, no network)."""

import asyncio
import builtins
import threading
import types

import pytest

from gujusub.cloud import azure
from gujusub.cloud.base import (
    CloudAuthError,
    CloudConfigError,
    CloudError,
    CloudSession,
    Credentials,
    SessionConfig,
)

RECOGNIZED, NO_MATCH = "RecognizedSpeech", "NoMatch"
ERROR, END_OF_STREAM = "Error", "EndOfStream"


class Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, cb):
        self.callbacks.append(cb)

    def fire(self, evt):
        """Invoke the callbacks from a worker thread, like the SDK does."""
        t = threading.Thread(target=lambda: [cb(evt) for cb in self.callbacks])
        t.start()
        t.join()


class Future:
    def __init__(self, action=None):
        self.action = action

    def get(self):
        if self.action:
            self.action()


def make_sdk():
    rec = types.SimpleNamespace(speech_config=None, recognizer=None, stream=None,
                                phrases=[], started=False, stopped=False, on_start=None)

    class SpeechConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.properties = {}
            self.speech_recognition_language = None
            rec.speech_config = self

        def set_property(self, key, value):
            self.properties[key] = value

    class AudioStreamFormat:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class PushAudioInputStream:
        def __init__(self, stream_format):
            self.format = stream_format
            self.writes: list[bytes] = []
            self.closed = False
            rec.stream = self

        def write(self, data):
            self.writes.append(bytes(data))

        def close(self):
            self.closed = True

    class AudioConfig:
        def __init__(self, stream):
            self.stream = stream

    class AutoDetectSourceLanguageConfig:
        def __init__(self, languages):
            self.languages = languages

    class SpeechRecognizer:
        def __init__(self, speech_config, audio_config, auto_detect_source_language_config=None):
            self.speech_config = speech_config
            self.audio_config = audio_config
            self.lid = auto_detect_source_language_config
            self.recognizing, self.recognized = Signal(), Signal()
            self.canceled, self.session_stopped = Signal(), Signal()
            rec.recognizer = self

        def start_continuous_recognition_async(self):
            rec.started = True
            return Future(rec.on_start)

        def stop_continuous_recognition_async(self):
            def stop():
                rec.stopped = True
                self.session_stopped.fire(types.SimpleNamespace())
            return Future(stop)

    class PhraseListGrammar:
        @staticmethod
        def from_recognizer(reco):
            return types.SimpleNamespace(addPhrase=rec.phrases.append)

    class AutoDetectSourceLanguageResult:
        def __init__(self, result):
            self.language = result.language

    sdk = types.SimpleNamespace(
        SpeechConfig=SpeechConfig,
        SpeechRecognizer=SpeechRecognizer,
        PhraseListGrammar=PhraseListGrammar,
        AutoDetectSourceLanguageResult=AutoDetectSourceLanguageResult,
        PropertyId=types.SimpleNamespace(SpeechServiceConnection_LanguageIdMode="LidMode"),
        ResultReason=types.SimpleNamespace(RecognizedSpeech=RECOGNIZED, NoMatch=NO_MATCH),
        CancellationReason=types.SimpleNamespace(Error=ERROR, EndOfStream=END_OF_STREAM),
        audio=types.SimpleNamespace(AudioStreamFormat=AudioStreamFormat,
                                    PushAudioInputStream=PushAudioInputStream,
                                    AudioConfig=AudioConfig),
        languageconfig=types.SimpleNamespace(
            AutoDetectSourceLanguageConfig=AutoDetectSourceLanguageConfig),
    )
    return sdk, rec


@pytest.fixture
def fake(monkeypatch):
    sdk, rec = make_sdk()
    monkeypatch.setattr(azure, "_sdk", lambda: sdk)
    return rec


def make(language="gu", region="eastus", terms=()):
    return azure.Session(Credentials("k3y", region), SessionConfig(language=language, terms=terms))


def result(text, reason=RECOGNIZED, language="gu-IN"):
    return types.SimpleNamespace(result=types.SimpleNamespace(text=text, reason=reason,
                                                              language=language))


def cancel(details, reason=ERROR):
    return types.SimpleNamespace(cancellation_details=types.SimpleNamespace(
        reason=reason, error_details=details))


async def drain():
    for _ in range(5):
        await asyncio.sleep(0)


async def collect(session):
    return [ev async for ev in session.events()]


def test_constructor_does_no_io_and_matches_protocol(fake):
    s = make()
    assert isinstance(s, CloudSession)
    assert (s.state, s.error, fake.speech_config) == ("connecting", None, None)


@pytest.mark.parametrize("language,locale", [("gu", "gu-IN"), ("en", "en-IN")])
def test_single_language_config(fake, language, locale):
    async def run():
        s = make(language)
        await s.start()
        assert s.state == "connected"
        await s.close()

    asyncio.run(run())
    cfg = fake.speech_config
    assert cfg.kwargs == {"subscription": "k3y", "region": "eastus"}
    assert cfg.speech_recognition_language == locale
    assert cfg.properties == {}
    assert fake.recognizer.lid is None
    assert fake.stream.format.kwargs == {"samples_per_second": 16000, "bits_per_sample": 16,
                                         "channels": 1}
    assert fake.recognizer.audio_config.stream is fake.stream
    assert fake.started


def test_auto_uses_endpoint_and_continuous_lid(fake):
    asyncio.run(make("auto", region="westeurope").start())
    cfg = fake.speech_config
    assert cfg.kwargs == {"subscription": "k3y",
                          "endpoint": "wss://westeurope.stt.speech.microsoft.com/speech/universal/v2"}
    assert cfg.properties == {"LidMode": "Continuous"}
    assert cfg.speech_recognition_language is None
    assert fake.recognizer.lid.languages == ["gu-IN", "en-IN"]


def test_phrase_list_filtered_and_capped(fake):
    terms = ("Ahmedabad", " ", "", *[f"t{i}" for i in range(600)])
    asyncio.run(make(terms=terms).start())
    assert fake.phrases[0] == "Ahmedabad"
    assert len(fake.phrases) == azure.MAX_PHRASES


def test_no_terms_no_phrase_list(fake):
    asyncio.run(make().start())
    assert fake.phrases == []


@pytest.mark.parametrize("region", [None, "", "  "])
def test_missing_region_is_config_error(fake, region):
    s = make(region=region)
    with pytest.raises(CloudConfigError, match="region"):
        asyncio.run(s.start())
    assert s.state == "error"
    assert not fake.started


def test_missing_sdk_is_config_error(monkeypatch):
    real_import = builtins.__import__

    def no_azure(name, *args, **kwargs):
        if name.startswith("azure"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_azure)
    with pytest.raises(CloudConfigError, match="pip install azure-cognitiveservices-speech"):
        azure._sdk()
    with pytest.raises(CloudConfigError):
        asyncio.run(make().start())


def test_partials_and_finals_with_ids(fake):
    async def run():
        s = make()
        await s.start()
        reco = fake.recognizer
        reco.recognizing.fire(result("કેમ"))
        reco.recognizing.fire(result("કેમ છો"))
        reco.recognized.fire(result("કેમ છો?"))
        reco.recognized.fire(result("", reason=NO_MATCH))
        reco.recognizing.fire(result("સારું"))
        reco.recognized.fire(result("સારું છે"))
        await drain()
        await s.close()
        return await collect(s), s.state

    events, state = asyncio.run(run())
    got = [(e.type, e.utterance_id, e.committed, e.tail, e.lang) for e in events]
    assert got == [
        ("partial", 1, "", "કેમ", "gu"),
        ("partial", 1, "", "કેમ છો", "gu"),
        ("final", 1, "કેમ છો?", "", "gu"),
        ("partial", 2, "", "સારું", "gu"),
        ("final", 2, "સારું છે", "", "gu"),
    ]
    assert all(e.translation is None for e in events)
    assert state == "closed"


def test_auto_detected_language_per_utterance(fake):
    async def run():
        s = make("auto")
        await s.start()
        reco = fake.recognizer
        reco.recognized.fire(result("hello there", language="en-IN"))
        reco.recognizing.fire(result("આવો", language="gu-IN"))
        reco.recognized.fire(result("આવો", language="gu-IN"))
        reco.recognized.fire(result("??", language="Unknown"))
        await drain()
        await s.close()
        return await collect(s)

    events = asyncio.run(run())
    assert [(e.type, e.utterance_id, e.lang) for e in events] == [
        ("final", 1, "en"), ("partial", 2, "gu"), ("final", 2, "gu"), ("final", 3, "gu")]


@pytest.mark.parametrize("details", [
    "WebSocket upgrade failed: Authentication error (401). Please check subscription information",
    "Unauthorized",
])
def test_auth_cancel_at_start_raises_cloud_auth_error(fake, details):
    s = make()
    fake.on_start = lambda: fake.recognizer.canceled.fire(cancel(details))
    with pytest.raises(CloudAuthError):
        asyncio.run(s.start())
    assert s.state == "error"
    assert details in s.error


def test_other_cancel_at_start_raises_cloud_error(fake):
    s = make()
    fake.on_start = lambda: fake.recognizer.canceled.fire(cancel("Connection was closed (1006)"))
    with pytest.raises(CloudError) as info:
        asyncio.run(s.start())
    assert not isinstance(info.value, CloudAuthError)
    assert s.state == "error"


def test_cancel_mid_stream_finalizes_and_ends_events(fake):
    async def run():
        s = make()
        await s.start()
        fake.recognizer.recognizing.fire(result("અધૂરું"))
        fake.recognizer.canceled.fire(cancel("Service timeout"))
        fake.recognizer.canceled.fire(cancel("", reason=END_OF_STREAM))  # ignored
        events = await asyncio.wait_for(collect(s), 1)
        await s.close()
        return s, events

    s, events = asyncio.run(run())
    assert [(e.type, e.tail, e.committed) for e in events] == [
        ("partial", "અધૂરું", ""), ("final", "", "")]
    assert s.state == "error"
    assert "Service timeout" in s.error


def test_send_audio_writes_bytes_only_while_connected(fake):
    async def run():
        s = make()
        await s.send_audio(b"early")  # before start: dropped
        await s.start()
        await s.send_audio(b"\x01\x02")
        await s.send_audio(b"\x03\x04")
        await s.close()
        await s.send_audio(b"late")  # after close: dropped

    asyncio.run(run())
    assert fake.stream.writes == [b"\x01\x02", b"\x03\x04"]


def test_close_stops_recognizer_flushes_pending_and_ends_events(fake):
    async def run():
        s = make()
        await s.start()
        fake.recognizer.recognizing.fire(result("છેલ્લું"))
        reader = asyncio.create_task(collect(s))
        await drain()
        await s.close()
        await s.close()  # idempotent
        fake.recognizer.recognized.fire(result("too late"))  # ignored after end
        await drain()
        return s, await asyncio.wait_for(reader, 1)

    s, events = asyncio.run(run())
    assert fake.stream.closed and fake.stopped
    assert [(e.type, e.utterance_id) for e in events] == [("partial", 1), ("final", 1)]
    assert s.state == "closed"


def test_session_stopped_ends_events(fake):
    async def run():
        s = make()
        await s.start()
        fake.recognizer.session_stopped.fire(types.SimpleNamespace())
        return await asyncio.wait_for(collect(s), 1)

    assert asyncio.run(run()) == []
