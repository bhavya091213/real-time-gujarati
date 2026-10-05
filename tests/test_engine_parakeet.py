"""Tests for gujusub.engine_parakeet with a fake ``onnx_asr`` module (no model loaded)."""

import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from gujusub import engine_parakeet
from gujusub.engine import ASREngine, Transcript

SR = 16000


@dataclass
class FakeResult:
    text: str
    timestamps: list[float] | None
    tokens: list[str] | None
    logprobs: list[float] | None


class FakeModel:
    def __init__(self, result: FakeResult):
        self.result = result
        self.calls: list[np.ndarray] = []

    def with_timestamps(self) -> "FakeModel":
        return self

    def recognize(self, audio, **kwargs):
        self.calls.append(audio)
        return self.result


def _install_fake(monkeypatch: pytest.MonkeyPatch, result: FakeResult) -> dict:
    captured: dict = {}
    model = FakeModel(result)

    def load_model(name, path=None, **kwargs):
        captured.update(name=name, path=path, **kwargs)
        return model

    monkeypatch.setitem(sys.modules, "onnx_asr", types.SimpleNamespace(load_model=load_model))
    monkeypatch.setattr(engine_parakeet, "prepare_model_dir", lambda: Path("/fake/parakeet"))
    monkeypatch.setattr(engine_parakeet, "ort_session_options", lambda: "SESS_OPTS")
    captured["model"] = model
    return captured


HELLO = FakeResult(
    text=" Hello world.",
    timestamps=[0.0, 0.08, 0.16, 0.48, 0.56],
    tokens=["▁Hel", "lo", "▁wor", "ld", "."],
    logprobs=[-0.1, -0.01, -0.2, -0.05, -0.3],
)


def test_load_model_gets_int8_sess_options_and_path(monkeypatch):
    captured = _install_fake(monkeypatch, HELLO)
    engine = engine_parakeet.ParakeetEngine()
    assert captured["name"] == "nemo-parakeet-tdt-0.6b-v2"
    assert captured["quantization"] == "int8"
    assert captured["sess_options"] == "SESS_OPTS"
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert Path(captured["path"]) == Path("/fake/parakeet")
    assert engine.lang == "en"
    assert isinstance(engine, ASREngine)


def test_words_grouped_on_sentencepiece_marker(monkeypatch):
    _install_fake(monkeypatch, HELLO)
    tr = engine_parakeet.ParakeetEngine().transcribe_detailed(np.zeros(SR, np.float32))
    assert tr.text == "Hello world."
    assert [w.text for w in tr.words] == ["Hello", "world."]
    hello, world = tr.words
    assert hello.start_s == pytest.approx(0.0)
    assert hello.end_s == pytest.approx(0.08 + engine_parakeet.TOKEN_FRAME_S)
    assert world.start_s == pytest.approx(0.16)
    assert world.end_s == pytest.approx(0.56 + engine_parakeet.TOKEN_FRAME_S)


def test_word_conf_is_exp_of_weakest_token_logprob(monkeypatch):
    _install_fake(monkeypatch, HELLO)
    tr = engine_parakeet.ParakeetEngine().transcribe_detailed(np.zeros(SR, np.float32))
    assert tr.words[0].conf == pytest.approx(math.exp(-0.1))
    assert tr.words[1].conf == pytest.approx(math.exp(-0.3))


def test_transcribe_returns_text(monkeypatch):
    _install_fake(monkeypatch, HELLO)
    assert engine_parakeet.ParakeetEngine().transcribe(np.zeros(SR, np.float32)) == "Hello world."


def test_no_tokens_is_empty(monkeypatch):
    _install_fake(monkeypatch, FakeResult("", [], [], []))
    tr = engine_parakeet.ParakeetEngine().transcribe_detailed(np.zeros(SR, np.float32))
    assert tr == Transcript.empty()


def test_all_words_below_floor_is_empty(monkeypatch):
    low = math.log(0.01)
    _install_fake(monkeypatch, FakeResult(" Uh huh", [0.0, 0.4], ["▁Uh", "▁huh"], [low, low]))
    tr = engine_parakeet.ParakeetEngine().transcribe_detailed(np.zeros(SR, np.float32))
    assert tr == Transcript.empty()


def test_one_confident_word_keeps_transcript(monkeypatch):
    low = math.log(0.01)
    _install_fake(monkeypatch, FakeResult(" Uh yes", [0.0, 0.4], ["▁Uh", "▁yes"], [low, -0.1]))
    tr = engine_parakeet.ParakeetEngine().transcribe_detailed(np.zeros(SR, np.float32))
    assert [w.text for w in tr.words] == ["Uh", "yes"]


def test_missing_logprobs_gives_full_conf(monkeypatch):
    _install_fake(monkeypatch, FakeResult(" Hi", [0.0], ["▁Hi"], None))
    tr = engine_parakeet.ParakeetEngine().transcribe_detailed(np.zeros(SR, np.float32))
    assert tr.words[0].conf == 1.0


def test_too_short_audio_skips_model(monkeypatch):
    captured = _install_fake(monkeypatch, HELLO)
    tr = engine_parakeet.ParakeetEngine().transcribe_detailed(np.zeros(100, np.float32))
    assert tr == Transcript.empty()
    assert captured["model"].calls == []


def test_rejects_2d_audio(monkeypatch):
    _install_fake(monkeypatch, HELLO)
    with pytest.raises(ValueError):
        engine_parakeet.ParakeetEngine().transcribe_detailed(np.zeros((2, SR), np.float32))


def test_warmup_runs_1_4_8_second_passes(monkeypatch):
    captured = _install_fake(monkeypatch, HELLO)
    engine_parakeet.ParakeetEngine().warmup()
    assert [len(a) / SR for a in captured["model"].calls] == [1.0, 4.0, 8.0]


def test_warmup_single_pass(monkeypatch):
    captured = _install_fake(monkeypatch, HELLO)
    engine_parakeet.ParakeetEngine().warmup(seconds=1.0)
    assert [len(a) / SR for a in captured["model"].calls] == [1.0]


def test_prepare_model_dir_fetches_only_int8_files(monkeypatch, tmp_path):
    seen: dict = {}
    snap = tmp_path / "snap"
    snap.mkdir()

    def fake_download(repo_id, allow_patterns=None):
        seen.update(repo_id=repo_id, allow_patterns=allow_patterns)
        return str(snap)

    monkeypatch.setattr(engine_parakeet, "snapshot_download", fake_download)
    monkeypatch.setattr(engine_parakeet, "default_root", lambda: tmp_path / "root")
    out = engine_parakeet.prepare_model_dir()
    assert seen["repo_id"] == "istupakov/parakeet-tdt-0.6b-v2-onnx"
    assert not any(p.endswith("model.onnx") and "int8" not in p for p in seen["allow_patterns"])
    assert "encoder-model.int8.onnx" in seen["allow_patterns"]
    assert out == tmp_path / "root" / "istupakov--parakeet-tdt-0.6b-v2-onnx"


def test_words_grouped_on_leading_space(monkeypatch):
    """onnx-asr rewrites the vocab's "▁" to a space: ' G', 'ood', ' m', 'or', 'ning'."""
    result = FakeResult(
        " Good morning.", [0.0, 0.08, 0.24, 0.32, 0.4, 0.48],
        [" G", "ood", " m", "or", "ning", "."], [-0.03, 0.0, 0.0, 0.0, 0.0, -0.01],
    )
    _install_fake(monkeypatch, result)
    tr = engine_parakeet.ParakeetEngine().transcribe_detailed(np.zeros(SR, np.float32))
    assert [w.text for w in tr.words] == ["Good", "morning."]
    assert tr.words[0].conf == pytest.approx(math.exp(-0.03))
    assert tr.words[1].start_s == pytest.approx(0.24)
