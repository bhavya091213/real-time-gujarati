"""tools/replay.py core: model-free, deterministic via an injected clock."""

import importlib.util
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from gujusub.streaming import VAD_FRAME, StreamingConfig
from tests.fakes import FakeVAD

_SPEC = importlib.util.spec_from_file_location(
    "replay", Path(__file__).resolve().parents[1] / "tools" / "replay.py"
)
replay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(replay)

SUMMARY_KEYS = {
    "audio_s", "decodes", "decodes_per_s", "decode_p50_ms", "decode_p95_ms",
    "decode_max_ms", "translate_calls", "translate_p50_ms", "max_lag_s",
    "end_lag_s", "rtf", "final_text",
}  # fmt: skip


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class CostEngine:
    """Engine whose decode 'costs' cost_s on the injected clock."""

    def __init__(self, clock, cost_s=0.0, text="ab cd ef gh"):
        self.clock, self.cost_s, self.text = clock, cost_s, text
        self.calls = 0

    def warmup(self):
        pass

    def transcribe(self, audio):
        self.calls += 1
        self.clock.now += self.cost_s
        return self.text


class CostTranslator:
    def __init__(self, clock, cost_s=0.1):
        self.clock, self.cost_s, self.texts = clock, cost_s, []

    def translate(self, text):
        self.clock.now += self.cost_s
        self.texts.append(text)
        return "english"


def audio_of(frames):
    return np.full(frames * VAD_FRAME, 0.1, dtype=np.float32)


def run(clock, engine, frames=60, speech=((0, 60),), translator=None, **kw):
    cfg = StreamingConfig(endpoint_ms=600)
    return replay.run_replay(
        audio_of(frames), engine, translator=translator, config=cfg,
        vad=FakeVAD(list(speech)), chunk_ms=32, clock=clock, **kw,
    )  # fmt: skip


def test_summary_has_all_keys():
    c = Clock()
    out = run(c, CostEngine(c))
    assert SUMMARY_KEYS <= set(out["summary"])
    assert out["events"]
    assert out["summary"]["audio_s"] == pytest.approx(60 * VAD_FRAME / 16000)


def test_zero_cost_engine_has_zero_lag():
    c = Clock()
    s = run(c, CostEngine(c, 0.0))["summary"]
    assert s["max_lag_s"] == 0 and s["end_lag_s"] == 0 and s["rtf"] == 0
    assert s["decodes"] > 0


def test_lag_equals_decode_cost_when_decodes_are_sparse():
    c = Clock()
    s = run(c, CostEngine(c, 0.05))["summary"]
    assert s["max_lag_s"] == pytest.approx(0.05)
    assert s["decode_p50_ms"] == pytest.approx(50)
    assert s["decode_max_ms"] == pytest.approx(50)


def test_lag_accumulates_when_decode_slower_than_realtime():
    c = Clock()
    s = run(c, CostEngine(c, 1.0), frames=80, speech=((0, 80),))["summary"]
    assert s["max_lag_s"] > 1.0
    assert s["rtf"] > 1.0
    assert s["end_lag_s"] > 1.0


def test_decode_counts_match_engine_and_rate():
    c = Clock()
    eng = CostEngine(c)
    s = run(c, eng)["summary"]
    assert s["decodes"] == eng.calls
    assert s["decodes_per_s"] == pytest.approx(eng.calls / s["audio_s"])


def test_translate_policy_and_counts():
    c = Clock()
    tr = CostTranslator(c, 0.1)
    out = run(c, CostEngine(c), translator=tr)
    s = out["summary"]
    assert s["translate_calls"] == len(tr.texts) >= 1
    assert s["translate_p50_ms"] == pytest.approx(100)
    # a final is always translated
    assert any(e["type"] == "final" and e["translation"] == "english" for e in out["events"])
    assert s["max_lag_s"] >= 0.1


def test_no_translator_means_no_translate_calls():
    c = Clock()
    s = run(c, CostEngine(c))["summary"]
    assert s["translate_calls"] == 0


def test_final_text_joined_and_event_records():
    c = Clock()
    out = run(c, CostEngine(c, text="hello world"), speech=((0, 20),))
    assert out["summary"]["final_text"] == "hello world"
    rec = out["events"][0]
    assert {"t", "type", "utterance_id", "window_s", "decode_ms", "lag_s",
            "committed", "tail"} <= set(rec)  # fmt: skip


def test_rtf_paced_sleeps_to_arrival_time():
    c = Clock()
    slept = []

    def sleep(dt):
        slept.append(dt)
        c.now += dt

    run(c, CostEngine(c), frames=10, speech=((0, 10),), rtf=1, sleep=sleep)
    assert sum(slept) == pytest.approx(10 * VAD_FRAME / 16000)


def test_loop_audio_and_json_roundtrip(tmp_path):
    a = audio_of(10)
    assert len(replay.loop_to(a, 1.0)) == 16000
    assert len(replay.loop_to(a, 0.0)) == len(a)
    c = Clock()
    out = run(c, CostEngine(c))
    p = tmp_path / "o.json"
    replay.write_json(out, p)
    assert json.loads(p.read_text())["summary"]["decodes"] == out["summary"]["decodes"]


def test_percentile_helper():
    assert replay.percentile([], 50) == 0.0
    assert replay.percentile([1, 2, 3, 4], 50) == 2
    assert replay.percentile([1, 2, 3, 4], 100) == 4


def test_cli_wav_with_fake_registry(tmp_path, monkeypatch, capsys):
    wav = tmp_path / "x.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.full(16000, 3000, dtype=np.int16)).tobytes())
    c = Clock()
    monkeypatch.setitem(replay.ENGINES, "gu", lambda: CostEngine(c))
    monkeypatch.setattr(replay, "load_audio", lambda p: np.full(16000, 0.1, dtype=np.float32))
    monkeypatch.setattr(replay, "make_vad", lambda: FakeVAD([(0, 100)]))
    rc = replay.main([str(wav), "--quiet"])
    assert rc == 0
    assert "max lag" in capsys.readouterr().out


def test_partial_translations_respect_gap_and_commit_change():
    c = Clock()
    tr = CostTranslator(c, 0.0)
    t = replay._Translations(tr, c)
    ev = replay.TranscriptEvent
    gap = replay.PARTIAL_TRANSLATE_GAP_S
    assert t(ev("partial", 1, "a", "x"), 0.0) == "english"
    assert t(ev("partial", 1, "a b", "x"), gap / 2) == ""  # changed, but inside the gap
    assert t(ev("partial", 1, "a b", "x"), gap + 0.1) == "english"  # changed, gap passed
    assert t(ev("partial", 1, "a b", "y"), gap * 3) == ""  # committed unchanged
    assert t(ev("final", 1, "a b c", ""), gap * 3) == "english"  # finals always
    assert len(tr.texts) == 3


def test_english_events_are_not_translated():
    c = Clock()
    tr = CostTranslator(c, 0.0)
    t = replay._Translations(tr, c)
    ev = replay.TranscriptEvent("final", 1, "hello there", "", lang="en")
    assert t(ev, 0.0) == ""
    assert tr.texts == []


def test_display_keeps_lang_and_english_openers():
    ev = replay.TranscriptEvent("final", 1, "So, um today we begin", "", lang="en")
    shown = replay._display(ev, filter_fillers=True)
    assert (shown.lang, shown.committed) == ("en", "So, today we begin")


# --- Auto mode (unit 4.2) ---------------------------------------------------


class FixedLid:
    def __init__(self, posteriors):
        self.posteriors, self.calls = posteriors, 0

    def classify(self, audio):
        self.calls += 1
        return self.posteriors

    def warmup(self):
        pass


def lang_cost_engine(clock, lang, text):
    eng = CostEngine(clock, text=text)
    eng.lang = lang
    return eng


def test_auto_replay_reports_lid_decisions_and_never_translates():
    c = Clock()
    engines = {"gu": lang_cost_engine(c, "gu", "ગુ"), "en": lang_cost_engine(c, "en", "hi there")}
    lid = FixedLid({"gu": 0.1, "en": 0.9})
    tr = CostTranslator(c, 0.0)
    cfg = StreamingConfig(endpoint_ms=600)
    out = replay.run_replay(
        audio_of(90), engines, translator=tr, config=cfg, vad=FakeVAD([(5, 80)]),
        chunk_ms=32, clock=c, lid=lid.classify,
    )  # fmt: skip
    (dec,) = out["summary"]["lid"]
    assert (dec["utterance_id"], dec["lang"], dec["p"]) == (1, "en", 0.9)
    assert dec["speech_s"] == pytest.approx(1.5)
    assert dec["since_onset_s"] == pytest.approx(47 * VAD_FRAME / 16000)
    assert dec["first_event_s"] == pytest.approx(dec["since_onset_s"])
    assert {e["lang"] for e in out["events"]} == {"en"}
    assert engines["gu"].calls == 0
    assert tr.texts == [] and out["summary"]["final_text"] == "hi there"
    assert "LID" in replay.format_summary(out["summary"])


def test_cli_auto_builds_both_engines_and_lid(tmp_path, monkeypatch, capsys):
    c = Clock()
    monkeypatch.setitem(replay.ENGINES, "gu", lambda: lang_cost_engine(c, "gu", "ગુ"))
    monkeypatch.setitem(replay.ENGINES, "en", lambda: lang_cost_engine(c, "en", "hi"))
    monkeypatch.setattr(replay, "load_lid", lambda: FixedLid({"gu": 0.95, "en": 0.05}))
    monkeypatch.setattr(replay, "load_audio", lambda p: np.full(48000, 0.1, dtype=np.float32))
    monkeypatch.setattr(replay, "make_vad", lambda: FakeVAD([(0, 1000)]))
    assert replay.main([str(tmp_path / "x.wav"), "--engine", "auto", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "LID u1: gu p=0.95" in out
