"""Auto mode (unit 4.2): LID gates the engine choice for each utterance."""

import logging

import numpy as np
import pytest
from fakes import FakeClock, FakeEngine, FakeVAD

from gujusub.lid import LidConfig, LidDecider
from gujusub.streaming import VAD_FRAME, AutoRoute, StreamingTranscriber

SR = 16000
FRAME_S = VAD_FRAME / SR
SPEECH, SILENCE = 0.1, 0.0


def gu(p):
    return {"gu": p, "en": 1.0 - p}


def en(p):
    return {"gu": 1.0 - p, "en": p}


class ScriptedLid:
    """classify(): scripted posteriors per call; records each window it saw."""

    def __init__(self, script):
        self.script = list(script)
        self.windows: list[np.ndarray] = []

    def __call__(self, audio):
        self.windows.append(audio.copy())
        return self.script[min(len(self.windows), len(self.script)) - 1]


def lang_engine(lang, text, **kw):
    eng = FakeEngine([text], **kw)
    eng.lang = lang
    return eng


class Setup:
    def __init__(self, script, *, speech, config=None, mode="auto", clock=None):
        self.lid = ScriptedLid(script)
        self.config = config or LidConfig()
        self.engines = {
            "gu": lang_engine("gu", "ગુ વાત", clock=clock, decode_ms=100),
            "en": lang_engine("en", "hello there", clock=clock, decode_ms=100),
        }
        self.mode = mode
        self.vad = FakeVAD(speech)
        self.tr = StreamingTranscriber(
            engine_for_utterance=self.provider, vad=self.vad, clock=clock or FakeClock()
        )
        self.events = []  # (frame index, event)

    def provider(self):
        if self.mode == "auto":
            return AutoRoute(LidDecider(self.lid, self.config), self.engines.__getitem__)
        return self.engines[self.mode]

    def run(self, n_frames, start=0):
        for i in range(start, start + n_frames):
            value = SPEECH if self.vad._is_speech(i) else SILENCE
            for ev in self.tr.feed(np.full(VAD_FRAME, value, dtype=np.float32)):
                self.events.append((i, ev))
        return self

    def flush(self):
        self.events.extend((None, ev) for ev in self.tr.flush())
        return self


ONSET = 10  # first speech frame in most tests


def frames_for(seconds):
    return int(np.ceil(seconds * SR / VAD_FRAME))


def test_decided_at_second_window_pins_en_and_partials_at_once():
    s = Setup([en(0.85), en(0.9)], speech=[(ONSET, 10_000)]).run(ONSET + 80)
    decide_frame = ONSET + frames_for(1.5) - 1
    assert s.events, "expected events after the decision"
    first_frame, first = s.events[0]
    assert first_frame == decide_frame and first.type == "partial"
    assert {ev.lang for _, ev in s.events} == {"en"}
    assert s.engines["gu"].calls == 0
    assert len(s.lid.windows) == 2


def test_decider_sees_only_speech_frames_from_onset():
    # onset frame 10, speech 10-14, a 6-frame gap (no endpoint), speech again from 21
    s = Setup([gu(0.9), gu(0.9)], speech=[(ONSET, 15), (21, 10_000)]).run(80)
    first, second = s.lid.windows
    assert len(first) == SR and len(second) == int(1.5 * SR)
    assert np.all(second == np.float32(SPEECH))  # no preroll, no gap frames
    assert {ev.lang for _, ev in s.events} == {"gu"}


def test_flip_flop_emits_nothing_until_decided():
    s = Setup([gu(0.85), en(0.85), en(0.85)], speech=[(ONSET, 10_000)]).run(ONSET + 80)
    decide_frame = ONSET + frames_for(2.0) - 1
    assert min(i for i, _ in s.events) == decide_frame
    assert {ev.lang for _, ev in s.events} == {"en"}
    assert s.engines["gu"].calls == 0


def test_endpoint_while_unsure_drops_the_utterance(caplog):
    caplog.set_level(logging.INFO)
    # 0.8 s of speech then silence: no scheduled check; final check is weak
    s = Setup([gu(0.7)], speech=[(ONSET, ONSET + 25)]).run(ONSET + 60).flush()
    assert s.events == []
    assert len(s.lid.windows) == 1 and len(s.lid.windows[0]) == 25 * VAD_FRAME
    assert s.engines["gu"].calls == s.engines["en"].calls == 0
    assert "dropped" in caplog.text and "0.70" in caplog.text
    assert s.tr.lid_last["lang"] == "unsure"


def test_endpoint_final_check_strong_emits_final_in_that_lang():
    s = Setup([en(0.97)], speech=[(ONSET, ONSET + 25)]).run(ONSET + 60)
    assert [(ev.type, ev.lang, ev.committed) for _, ev in s.events] == [
        ("final", "en", "hello there")
    ]


def test_max_s_without_decision_falls_back_to_gu_with_warning(caplog):
    s = Setup([gu(0.7)], speech=[(ONSET, 10_000)])
    s.run(ONSET + 110)
    timeout_frame = ONSET + frames_for(3.0) - 1
    assert len(s.lid.windows) == 5
    assert min(i for i, _ in s.events) == timeout_frame
    assert {ev.lang for _, ev in s.events} == {"gu"}
    assert "LID undecided, defaulting to gu" in caplog.text
    assert s.tr.lid_last["lang"] == "gu" and s.tr.lid_last["fallback"] is True


def test_max_s_without_decision_and_no_fallback_drops():
    s = Setup([gu(0.7)], speech=[(ONSET, 10_000)], config=LidConfig(lid_fallback="none"))
    s.run(ONSET + 150).flush()
    assert s.events == []
    assert s.engines["gu"].calls == s.engines["en"].calls == 0


def test_first_partial_within_one_frame_of_a_1_5_s_decision():
    clock = FakeClock()
    s = Setup([en(0.85), en(0.95)], speech=[(ONSET, 10_000)], clock=clock)
    s.run(ONSET + 80)
    first_frame, _ = s.events[0]
    since_onset_s = (first_frame - ONSET + 1) * FRAME_S
    assert since_onset_s <= 1.5 + FRAME_S
    last = s.tr.lid_last
    assert last["lang"] == "en" and last["p"] == pytest.approx(0.95)
    assert last["speech_s"] == pytest.approx(1.5)
    assert last["since_onset_s"] == pytest.approx(since_onset_s)
    assert last["utterance_id"] == 1


def test_decision_logged_with_posteriors_and_timing(caplog):
    caplog.set_level(logging.INFO)
    Setup([gu(0.85), gu(0.9)], speech=[(ONSET, 10_000)]).run(ONSET + 60)
    assert "LID gu" in caplog.text and "0.90" in caplog.text and "1.50 s" in caplog.text


def test_each_utterance_gets_its_own_decision():
    s = Setup(
        [en(0.9), en(0.9), gu(0.9), gu(0.9)], speech=[(ONSET, ONSET + 60), (ONSET + 90, 10_000)]
    )
    s.run(ONSET + 170)
    by_id = {}
    for _, ev in s.events:
        by_id.setdefault(ev.utterance_id, set()).add(ev.lang)
    assert by_id == {1: {"en"}, 2: {"gu"}}


def test_mode_change_gu_to_auto_applies_at_next_utterance():
    s = Setup([en(0.9), en(0.9)], speech=[(ONSET, ONSET + 60), (ONSET + 90, 10_000)], mode="gu")
    s.run(ONSET + 30)
    s.mode = "auto"  # mid-utterance: utterance 1 stays gu, no LID
    s.run(140, start=ONSET + 30)
    langs = {ev.utterance_id: ev.lang for _, ev in s.events}
    assert langs == {1: "gu", 2: "en"}
    assert len(s.lid.windows) == 2  # only utterance 2 was classified


def test_lid_last_is_none_before_any_decision():
    s = Setup([gu(0.9)], speech=[])
    assert s.tr.lid_last is None
