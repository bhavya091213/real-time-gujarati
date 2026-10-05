"""Confidence trimming: per-word trim, utterance gate, and the streaming hook."""

import numpy as np
import pytest
from fakes import FakeVAD, frames

from gujusub import server
from gujusub.confidence import apply_confidence, trim_words, utterance_ok
from gujusub.engine import Transcript, Word
from gujusub.settings import Settings, SettingsStore
from gujusub.streaming import StreamingTranscriber

INTERVAL_FRAMES = 15  # 480 ms partial cadence
ENDPOINT_FRAMES = 19  # 600 ms of silence finalizes


def w(text: str, conf: float | None) -> Word:
    return Word(text, 0.0, 0.1, conf)


def tr(*pairs) -> Transcript:
    words = tuple(w(t, c) for t, c in pairs)
    return Transcript(" ".join(x.text for x in words), words)


# --- pure functions ---------------------------------------------------------


def test_trim_words_drops_below_threshold_and_keeps_order():
    words = (w("a", 0.9), w("b", 0.2), w("c", 0.5), w("d", 0.49))
    assert [x.text for x in trim_words(words, 0.5)] == ["a", "c"]


def test_trim_words_keeps_unknown_confidence():
    assert [x.text for x in trim_words((w("a", None), w("b", 0.1)), 0.5)] == ["a"]


def test_trim_words_zero_threshold_is_identity():
    words = (w("a", 0.01), w("b", 0.0))
    assert trim_words(words, 0.0) == words


def test_utterance_ok_off_when_threshold_zero():
    assert utterance_ok((), 0.0)
    assert utterance_ok((w("a", 0.01),), 0.0)


def test_utterance_ok_requires_mean_above_threshold():
    assert utterance_ok((w("a", 0.9), w("b", 0.6)), 0.7)
    assert not utterance_ok((w("a", 0.9), w("b", 0.4)), 0.7)


def test_utterance_ok_keeps_single_high_conf_word_by_default():
    assert utterance_ok((w("Amen", 0.99),), 0.5)


def test_utterance_ok_drops_single_low_conf_word():
    assert not utterance_ok((w("Amen", 0.2),), 0.5)


def test_utterance_ok_min_words_param_still_enforced():
    assert not utterance_ok((w("a", 0.99),), 0.5, min_words=2)
    assert utterance_ok((w("a", 0.99), w("b", 0.99)), 0.5, min_words=2)
    assert not utterance_ok((), 0.5, min_words=0)  # nothing kept never passes


def test_utterance_ok_ignores_unknown_confidence_in_mean():
    assert utterance_ok((w("a", None), w("b", 0.8)), 0.7)
    assert utterance_ok((w("a", None), w("b", None)), 0.7)


def test_apply_identity_when_off():
    t = tr(("a", 0.1), ("b", 0.2))
    assert apply_confidence(t, 0.0, 0.0) is t


def test_apply_identity_when_nothing_trimmed():
    t = Transcript("a  b", (w("a", 0.9), w("b", 0.9)))  # original text preserved
    assert apply_confidence(t, 0.5, 0.5) is t


def test_apply_rebuilds_text_from_kept_words():
    out = apply_confidence(tr(("મારું", 0.9), ("અતાર", 0.3), ("છે", 0.95)), 0.5, 0.0)
    assert out.text == "મારું છે"
    assert [x.text for x in out.words] == ["મારું", "છે"]


def test_apply_returns_empty_when_gate_fails():
    assert apply_confidence(tr(("a", 0.6), ("b", 0.55)), 0.5, 0.8) == Transcript.empty()


def test_apply_without_words_is_passthrough():
    t = Transcript("no alignment", ())
    assert apply_confidence(t, 0.9, 0.9) is t


# --- streaming hook ---------------------------------------------------------


class ConfEngine:
    """Returns scripted Transcripts (last one repeats)."""

    lang = "gu"

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def transcribe(self, audio):
        return self.transcribe_detailed(audio).text

    def transcribe_detailed(self, audio):
        out = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return out


def feed(t, n):
    out = []
    for _ in range(n):
        out.extend(t.feed(frames(1)))
    return out


def run(engine, provider, speech_frames=1 + 3 * INTERVAL_FRAMES):
    st = StreamingTranscriber(
        engine, vad=FakeVAD([(0, speech_frames)]), thresholds_provider=provider
    )
    return feed(st, speech_frames + ENDPOINT_FRAMES + 2)


def test_low_conf_tail_word_never_commits():
    flicker = tr(("a", 0.9), ("b", 0.9), ("zz", 0.2))
    engine = ConfEngine([flicker])  # repeated: without trimming "zz" commits
    events = run(engine, lambda: (0.5, 0.0))
    assert all("zz" not in f"{e.committed} {e.tail}" for e in events)
    assert events[-1].type == "final" and events[-1].committed == "a b"


def test_without_trimming_the_flicker_word_commits():
    engine = ConfEngine([tr(("a", 0.9), ("b", 0.9), ("zz", 0.2))])
    events = run(engine, lambda: (0.0, 0.0))
    assert events[-1].committed == "a b zz"


def test_all_low_utterance_emits_no_events():
    engine = ConfEngine([tr(("x", 0.3), ("y", 0.35))])
    assert run(engine, lambda: (0.2, 0.6)) == []
    assert engine.calls > 0  # it decoded, then the gate dropped everything


def test_default_provider_is_off():
    engine = ConfEngine([tr(("x", 0.01), ("y", 0.01))])
    st = StreamingTranscriber(engine, vad=FakeVAD([(0, 50)]))
    events = feed(st, 50 + ENDPOINT_FRAMES + 2)
    assert events[-1].committed == "x y"


def test_provider_consulted_per_decode():
    engine = ConfEngine([tr(("a", 0.9), ("b", 0.4))])
    thresholds = [(0.0, 0.0)]
    calls = []

    def provider():
        calls.append(1)
        return thresholds[0]

    st = StreamingTranscriber(engine, vad=FakeVAD([(0, 10_000)]), thresholds_provider=provider)
    first = feed(st, 1 + INTERVAL_FRAMES)
    assert first[-1].tail == "a b"
    thresholds[0] = (0.5, 0.0)  # settings change mid-run
    second = feed(st, INTERVAL_FRAMES)
    assert f"{second[-1].committed} {second[-1].tail}".strip() == "a"
    assert len(calls) == engine.calls


# --- server provider ----------------------------------------------------------


def test_server_thresholds_follow_settings(monkeypatch, tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.load()
    monkeypatch.setattr(server, "store", store)
    store.update({"conf_word_min": 0.3, "conf_utt_min": 0.6})
    assert server.confidence_thresholds() == (0.3, 0.6)
    store.update({"conf_word_min": 0.0})
    assert server.confidence_thresholds() == (0.0, 0.6)


def test_server_transcriber_uses_settings_provider(monkeypatch):
    monkeypatch.setattr(server, "store", None)
    monkeypatch.setattr("gujusub.streaming.SileroVAD", lambda: FakeVAD([]))
    st = server.new_transcriber()
    assert st._thresholds_provider is server.confidence_thresholds
    defaults = Settings()
    assert server.confidence_thresholds() == (defaults.conf_word_min, defaults.conf_utt_min)


def test_settings_defaults_in_range():
    s = Settings()
    assert 0.0 <= s.conf_word_min < 1.0 and 0.0 <= s.conf_utt_min < 1.0


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_settings_reject_out_of_range(bad):
    with pytest.raises(ValueError):
        Settings().with_updates({"conf_word_min": bad})


# --- calibration tool (model-free parts) -------------------------------------


def _calibrate_module():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "calibrate_confidence",
        Path(__file__).resolve().parents[1] / "tools" / "calibrate_confidence.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_calibration_wer():
    cal = _calibrate_module()
    assert cal.wer("a b c".split(), "a b c".split()) == 0.0
    assert cal.wer("a b c d".split(), "a x c".split()) == 0.5  # 1 sub + 1 del
    assert cal.wer([], []) == 0.0 and cal.wer([], ["x"]) == 1.0


def test_calibration_synthetic_levels():
    cal = _calibrate_module()
    base = (0.1 * np.sin(np.arange(16_000) / 5)).astype(np.float32)
    out = cal.synthetic(base)
    rms = lambda a: float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))  # noqa: E731
    assert abs(20 * np.log10(rms(out["noise"])) + 30) < 0.5
    assert abs(20 * np.log10(np.abs(out["tremolo"]).max()) + 12) < 0.01
    snr = 20 * np.log10(rms(base) / rms(out["noisy"] - base))
    assert abs(snr) < 0.5
    assert all(a.dtype == np.float32 for a in out.values())
