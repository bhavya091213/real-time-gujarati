"""Tests for gujusub.lid: the decision policy (pure) and LanguageID with a fake classifier."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gujusub import lid
from gujusub.lid import LanguageID, LidConfig, LidDecider

SR = 16000
N_LABELS = 107
EN_IDX, GU_IDX = 20, 31


# ---------------------------------------------------------------- LidDecider


class ScriptedClassifier:
    """Returns pre-scripted posteriors per call and records the audio length seen."""

    def __init__(self, script: list[dict[str, float]]):
        self.script = list(script)
        self.lengths: list[int] = []

    def __call__(self, audio: np.ndarray) -> dict[str, float]:
        self.lengths.append(len(audio))
        return self.script[len(self.lengths) - 1]


def gu(p: float) -> dict[str, float]:
    return {"gu": p, "en": 1.0 - p}


def en(p: float) -> dict[str, float]:
    return {"gu": 1.0 - p, "en": p}


def secs(s: float) -> np.ndarray:
    return np.zeros(int(round(s * SR)), dtype=np.float32)


def tone(s: float, amplitude: float = 0.1) -> np.ndarray:
    t = np.arange(int(round(s * SR)), dtype=np.float32) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def feed_steps(decider: LidDecider, total_s: float, step_s: float = 0.25) -> list[str]:
    out = []
    for _ in range(int(round(total_s / step_s))):
        out.append(decider.feed(secs(step_s)))
    return out


def test_unsure_before_first_window_and_no_call():
    clf = ScriptedClassifier([])
    d = LidDecider(clf)
    assert d.feed(secs(0.99)) == "unsure"
    assert clf.lengths == []
    assert not d.done


def test_decides_on_second_consecutive_confident_window():
    clf = ScriptedClassifier([gu(0.85), gu(0.90)])
    d = LidDecider(clf)
    assert d.feed(secs(1.0)) == "unsure"
    assert not d.done
    assert d.feed(secs(0.5)) == "gu"
    assert d.done and d.lang == "gu"
    assert clf.lengths == [SR, int(1.5 * SR)]


def test_strong_single_window_at_first_check_does_not_decide():
    # near-silence scores as confident English, so the 1.0 s window alone never commits
    clf = ScriptedClassifier([en(0.99), gu(0.6)])
    d = LidDecider(clf)
    assert d.feed(secs(1.0)) == "unsure"
    assert d.feed(secs(0.5)) == "unsure"
    assert not d.done


def test_strong_single_window_decides_from_second_check():
    clf = ScriptedClassifier([gu(0.6), en(0.97)])
    d = LidDecider(clf)
    assert d.feed(secs(1.0)) == "unsure"
    assert d.feed(secs(0.5)) == "en"
    assert d.lang == "en"


def test_two_confident_windows_from_the_first_check_still_decide():
    clf = ScriptedClassifier([en(0.99), en(0.85)])
    d = LidDecider(clf)
    assert d.feed(secs(1.5)) == "en"


def test_last_posteriors_and_decision_time_are_recorded():
    clf = ScriptedClassifier([gu(0.85), gu(0.9)])
    d = LidDecider(clf)
    assert d.posteriors is None and d.decided_s is None
    d.feed(secs(1.5))
    assert d.posteriors == gu(0.9)
    assert d.decided_s == 1.5


def test_final_check_strong_before_guard_stays_unsure():
    clf = ScriptedClassifier([gu(0.97)])
    d = LidDecider(clf)
    d.feed(secs(0.8))
    assert d.final_check() == "unsure"
    assert d.done and d.lang is None
    assert clf.lengths == [int(0.8 * SR)]


def test_final_check_continues_a_confident_streak():
    clf = ScriptedClassifier([en(0.85), en(0.82)])
    d = LidDecider(clf)
    d.feed(secs(1.2))
    assert d.final_check() == "en"
    assert clf.lengths == [SR, int(1.2 * SR)]


def test_final_check_weak_stays_unsure():
    clf = ScriptedClassifier([gu(0.85)])
    d = LidDecider(clf)
    d.feed(secs(0.8))
    assert d.final_check() == "unsure"
    assert d.done and d.lang is None


def test_final_check_without_new_audio_does_not_reclassify():
    clf = ScriptedClassifier([gu(0.7), gu(0.7)])
    d = LidDecider(clf)
    d.feed(secs(1.5))
    assert d.final_check() == "unsure"
    assert len(clf.lengths) == 2


def test_final_check_with_no_audio_is_unsure():
    clf = ScriptedClassifier([])
    assert LidDecider(clf).final_check() == "unsure"
    assert clf.lengths == []


def test_final_check_after_decision_keeps_it():
    clf = ScriptedClassifier([en(0.9), en(0.9)])
    d = LidDecider(clf)
    d.feed(secs(1.5))
    assert d.final_check() == "en"
    assert len(clf.lengths) == 2


def test_flip_flop_stays_unsure_then_times_out():
    script = [gu(0.85), en(0.85), gu(0.85), en(0.85), gu(0.85)]
    clf = ScriptedClassifier(script)
    d = LidDecider(clf)
    results = feed_steps(d, 3.0)
    assert set(results) == {"unsure"}
    assert d.done and d.lang is None
    assert len(clf.lengths) == 5  # 1.0, 1.5, 2.0, 2.5, 3.0 s


def test_weak_window_breaks_the_streak():
    clf = ScriptedClassifier([gu(0.85), gu(0.6), gu(0.85), gu(0.85)])
    d = LidDecider(clf)
    results = feed_steps(d, 2.5)
    assert results[-1] == "gu"
    assert results.count("gu") == 1
    assert len(clf.lengths) == 4


def test_timeout_is_final_and_stops_classifying():
    clf = ScriptedClassifier([gu(0.6)] * 5)
    d = LidDecider(clf)
    feed_steps(d, 3.0)
    assert d.done and d.lang is None
    assert d.feed(secs(1.0)) == "unsure"
    assert len(clf.lengths) == 5


def test_decision_is_sticky_after_done():
    clf = ScriptedClassifier([en(0.99), en(0.99)])
    d = LidDecider(clf)
    d.feed(secs(1.5))
    assert d.feed(secs(2.0)) == "en"
    assert len(clf.lengths) == 2


def test_big_chunk_evaluates_each_checkpoint_on_its_prefix():
    clf = ScriptedClassifier([gu(0.85), gu(0.85)])
    d = LidDecider(clf)
    assert d.feed(secs(2.2)) == "gu"
    assert clf.lengths == [SR, int(1.5 * SR)]


def test_buffer_capped_at_max_seconds():
    clf = ScriptedClassifier([gu(0.5)] * 5)
    d = LidDecider(clf)
    d.feed(secs(10.0))
    assert max(clf.lengths) == 3 * SR
    assert d.done and d.lang is None


def test_reset_starts_a_new_utterance():
    clf = ScriptedClassifier([en(0.99), en(0.99), gu(0.99), gu(0.99)])
    d = LidDecider(clf)
    assert d.feed(secs(1.5)) == "en"
    d.reset()
    assert not d.done and d.lang is None and d.posteriors is None
    assert d.feed(secs(0.5)) == "unsure"
    assert d.feed(secs(1.0)) == "gu"
    assert clf.lengths == [SR, int(1.5 * SR), SR, int(1.5 * SR)]


def test_custom_config_thresholds():
    cfg = LidConfig(first_s=0.5, step_s=0.5, max_s=1.0, p_min=0.6, p_strong=0.99)
    clf = ScriptedClassifier([en(0.7), en(0.7)])
    d = LidDecider(clf, cfg)
    assert d.feed(secs(1.0)) == "en"
    assert clf.lengths == [SR // 2, SR]


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        LidConfig(p_min=0.97, p_strong=0.95)
    with pytest.raises(ValueError):
        LidConfig(first_s=4.0, max_s=3.0)
    with pytest.raises(ValueError):
        LidConfig(step_s=0.0)
    with pytest.raises(ValueError):
        LidConfig(lid_fallback="en")


def test_config_defaults_for_auto_mode():
    cfg = LidConfig()
    assert cfg.strong_from_s == 1.5
    assert cfg.lid_fallback == "gu"
    assert LidConfig(lid_fallback="none").lid_fallback == "none"


# ---------------------------------------------------------------- LanguageID


class FakeSpeechBrain:
    """Mimics EncoderClassifier.classify_batch: log-probs over 107 labels."""

    def __init__(self, gu_logit: float, en_logit: float, other: float = -5.0):
        labels = {f"x{i}: Lang{i}": i for i in range(N_LABELS)}
        labels = {k: v for k, v in labels.items() if v not in (EN_IDX, GU_IDX)}
        labels["en: English"] = EN_IDX
        labels["gu: Gujarati"] = GU_IDX
        self.hparams = SimpleNamespace(label_encoder=SimpleNamespace(lab2ind=labels))
        logits = torch.full((1, N_LABELS), other)
        logits[0, GU_IDX] = gu_logit
        logits[0, EN_IDX] = en_logit
        self._out = torch.log_softmax(logits, dim=-1)
        self.calls: list[tuple] = []

    def classify_batch(self, wavs):
        self.calls.append(tuple(wavs.shape))
        return self._out, None, None, None


def test_language_id_renormalises_over_gu_and_en():
    fake = FakeSpeechBrain(gu_logit=2.0, en_logit=0.0)
    model = LanguageID(classifier=fake)
    probs = model.classify(tone(2.0))
    assert set(probs) == {"gu", "en"}
    assert math.isclose(probs["gu"] + probs["en"], 1.0, rel_tol=1e-6)
    assert math.isclose(probs["gu"], 1 / (1 + math.exp(-2.0)), rel_tol=1e-5)
    assert fake.calls == [(1, 2 * SR)]


def test_language_id_looks_up_label_indices():
    model = LanguageID(classifier=FakeSpeechBrain(0.0, 0.0))
    assert model.indices == {"gu": GU_IDX, "en": EN_IDX}


def test_language_id_missing_label_raises():
    fake = FakeSpeechBrain(0.0, 0.0)
    del fake.hparams.label_encoder.lab2ind["gu: Gujarati"]
    with pytest.raises(RuntimeError, match="gu"):
        LanguageID(classifier=fake)


def test_language_id_short_audio_is_uninformative():
    fake = FakeSpeechBrain(5.0, 0.0)
    probs = LanguageID(classifier=fake).classify(secs(0.05))
    assert probs == {"gu": 0.5, "en": 0.5}
    assert fake.calls == []


def test_language_id_rejects_2d_audio():
    model = LanguageID(classifier=FakeSpeechBrain(0.0, 0.0))
    with pytest.raises(ValueError):
        model.classify(np.zeros((2, SR), dtype=np.float32))


def test_language_id_restores_torch_threads():
    before = torch.get_num_threads()
    LanguageID(classifier=FakeSpeechBrain(0.0, 0.0), threads=1).classify(tone(1.0))
    assert torch.get_num_threads() == before


def test_warmup_runs_one_classification():
    fake = FakeSpeechBrain(0.0, 0.0)
    LanguageID(classifier=fake).warmup()
    assert len(fake.calls) == 1


def test_decider_accepts_language_id_classify():
    fake = FakeSpeechBrain(gu_logit=10.0, en_logit=0.0)
    d = LidDecider(LanguageID(classifier=fake).classify)
    assert d.feed(tone(1.0)) == "unsure"
    assert d.feed(tone(0.5)) == "gu"


def test_language_id_near_silence_is_uninformative_without_model_call():
    fake = FakeSpeechBrain(gu_logit=0.0, en_logit=10.0)
    model = LanguageID(classifier=fake)
    # 0.005 peak sine -> RMS ~ -49 dBFS, below the -45 dBFS default floor
    assert model.classify(tone(1.0, amplitude=0.005)) == {"gu": 0.5, "en": 0.5}
    assert model.classify(secs(1.0)) == {"gu": 0.5, "en": 0.5}
    assert fake.calls == []
    probs = model.classify(tone(1.0, amplitude=0.02))  # ~ -37 dBFS: classified
    assert probs["en"] > 0.99 and len(fake.calls) == 1


def test_language_id_rms_floor_is_configurable():
    fake = FakeSpeechBrain(gu_logit=10.0, en_logit=0.0)
    model = LanguageID(classifier=fake, min_rms_dbfs=-20.0)
    assert model.classify(tone(1.0, amplitude=0.05)) == {"gu": 0.5, "en": 0.5}
    assert fake.calls == []


def test_savedir_follows_model_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GUJUSUB_MODEL_DIR", str(tmp_path))
    assert lid.savedir() == tmp_path / "lid"
