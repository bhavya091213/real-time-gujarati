"""Per-utterance engine selection (unit 3.2): pinning, lang, per-lang window."""

import pytest
from fakes import FakeEngine, FakeVAD, frames, ramp

from gujusub.engines import EngineNotLoaded, EngineRegistry, parse_engine_list
from gujusub.streaming import StreamingConfig, StreamingTranscriber, TranscriptEvent

INTERVAL_FRAMES = 15
SR = 16000


def lang_engine(lang, script=("x",), **kw):
    eng = FakeEngine(script, **kw)
    eng.lang = lang
    return eng


class Mode:
    """Mutable mode provider; counts how often it is consulted."""

    def __init__(self, engines, lang="gu"):
        self.engines, self.lang, self.calls = engines, lang, 0

    def __call__(self):
        self.calls += 1
        return self.engines[self.lang]


def test_engine_pinned_at_utterance_start_not_mid_utterance():
    gu, en = lang_engine("gu", ["ગુ"]), lang_engine("en", ["hello"])
    mode = Mode({"gu": gu, "en": en})
    # utterance 1: frames 0..59, silence 60..89, utterance 2: frames 90..
    tr = StreamingTranscriber(engine_for_utterance=mode,
                              vad=FakeVAD([(0, 60), (90, 10_000)]))
    events = []
    for i in range(150):
        if i == 30:
            mode.lang = "en"  # mid-utterance switch: must not apply yet
        events.extend(tr.feed(frames(1)))
    events.extend(tr.flush())
    by_id = {}
    for e in events:
        by_id.setdefault(e.utterance_id, []).append(e)
    assert {e.lang for e in by_id[1]} == {"gu"}
    assert {e.committed + e.tail for e in by_id[1]} == {"ગુ"}
    assert {e.lang for e in by_id[2]} == {"en"}
    assert by_id[2][-1].committed == "hello"
    assert mode.calls == 2  # consulted once per utterance, at its start


def test_single_engine_path_sets_lang_from_engine():
    eng = lang_engine("en", ["hi there"])
    tr = StreamingTranscriber(eng, vad=FakeVAD([(0, 10_000)]))
    events = [e for _ in range(1 + INTERVAL_FRAMES) for e in tr.feed(frames(1))]
    assert events and all(e.lang == "en" for e in events)


def test_engine_without_lang_defaults_to_gu():
    eng = FakeEngine(["a"])  # no .lang attribute
    tr = StreamingTranscriber(eng, vad=FakeVAD([(0, 10_000)]))
    events = [e for _ in range(1 + INTERVAL_FRAMES) for e in tr.feed(frames(1))]
    assert events[0].lang == "gu"


def test_transcriber_requires_an_engine_source():
    with pytest.raises(ValueError):
        StreamingTranscriber(vad=FakeVAD([]))


def test_event_lang_defaults_and_serializes():
    assert TranscriptEvent("partial", 1, "a", "").lang == "gu"
    ev = TranscriptEvent("final", 2, "hi", "", lang="en")
    assert ev.to_dict()["lang"] == "en"


def test_config_default_window_by_lang():
    cfg = StreamingConfig()
    assert cfg.max_window_s == 5.0
    assert cfg.max_window_s_by_lang == {"en": 4.0}


@pytest.mark.parametrize(("lang", "window_s"), [("en", 4.0), ("gu", 5.0)])
def test_window_bound_follows_pinned_engine_lang(lang, window_s):
    eng = lang_engine(lang, word_s=0.4)
    tr = StreamingTranscriber(eng, vad=FakeVAD([(0, 100_000)]))
    for i in range(300):  # 9.6 s, below the 12 s cap
        tr.feed(ramp(i, 1))
    partial_max = max(eng.audio_lengths)
    bound = int((window_s + 0.48) * SR) + 512
    assert partial_max <= bound
    assert partial_max > int((window_s - 0.5) * SR)  # really uses its own window


# --- registry --------------------------------------------------------------

def test_registry_get_and_langs():
    gu, en = lang_engine("gu"), lang_engine("en")
    reg = EngineRegistry({"gu": gu, "en": en})
    assert reg.get("en") is en and reg.get("gu") is gu
    assert "en" in reg and "auto" not in reg
    assert reg.langs == ("gu", "en")


def test_registry_missing_engine_raises_clear_error():
    reg = EngineRegistry({"gu": lang_engine("gu")})
    with pytest.raises(EngineNotLoaded, match="'en'.*--engines"):
        reg.get("en")


def test_registry_warmup_all():
    warmed = []

    class E:
        def __init__(self, lang):
            self.lang = lang

        def warmup(self):
            warmed.append(self.lang)

    EngineRegistry({"gu": E("gu"), "en": E("en")}).warmup()
    assert warmed == ["gu", "en"]


@pytest.mark.parametrize(("text", "expected"), [
    ("gu,en", ("gu", "en")), ("gu", ("gu",)), (" en , gu ", ("en", "gu")),
    ("gu,gu,en", ("gu", "en")),
])
def test_parse_engine_list(text, expected):
    assert parse_engine_list(text) == expected


@pytest.mark.parametrize("text", ["", "en", "gu,fr", "gu,auto"])
def test_parse_engine_list_rejects(text):
    with pytest.raises(ValueError):
        parse_engine_list(text)
