"""Tests for gujusub.engine_gu_fast with a fake ORT session and preprocessor (no model loaded)."""

import json
from pathlib import Path

import numpy as np
import pytest

from gujusub import engine_gu_fast
from gujusub.engine import ASREngine, Transcript
from gujusub.engine_gu_fast import (
    BLANK_TOKEN,
    FP32_FILE,
    INT8_FILE,
    FastGujaratiEngine,
    ensure_int8,
    load_vocab,
    log_softmax,
    read_config,
)

SR = 16000
# ids: 0 <unk>, 1 "▁મા", 2 "રું", 3 "▁છે", 4 <blk>
VOCAB_LINES = ["<unk> 0", "▁મા 1", "રું 2", "▁છે 3", "<blk> 4"]


def _model_dir(tmp_path: Path, subsampling: int = 4) -> Path:
    (tmp_path / "vocab.txt").write_text("\n".join(VOCAB_LINES) + "\n", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "nemo-conformer-ctc", "subsampling_factor": subsampling})
    )
    (tmp_path / FP32_FILE).write_bytes(b"fp32")
    return tmp_path


class FakePreprocessor:
    """10 ms hop: features (1, 80, n_samples // 160)."""

    def __call__(self, wav, lens):
        n = int(lens[0]) // 160
        return np.zeros((1, 80, n), np.float32), np.array([n], np.int64)


class FakeSession:
    """Emits a fixed token path over the frames: ▁મા ▁મા રું blank blank ▁છે ... blank."""

    def __init__(self, subsampling: int = 4):
        self.subsampling = subsampling
        self.calls: list[dict] = []

    def run(self, outputs, feeds):
        self.calls.append(feeds)
        n_feat = int(feeds["length"][0])
        n = (n_feat - 1) // self.subsampling + 1
        logits = np.full((1, n, 5), -10.0, np.float32)
        path = [1, 1, 2, 4, 4, 3] + [4] * max(0, n - 6)
        for t, tok in enumerate(path[:n]):
            logits[0, t, tok] = 2.0 if t != 2 else 0.5  # weaker frame for રું
        return [logits]


@pytest.fixture
def fake_engine(monkeypatch, tmp_path):
    model_dir = _model_dir(tmp_path)
    session = FakeSession()
    monkeypatch.setattr(engine_gu_fast, "_load_session", lambda path: session)
    monkeypatch.setattr(engine_gu_fast, "_load_preprocessor", lambda n: FakePreprocessor())
    monkeypatch.setattr(engine_gu_fast, "ensure_int8", lambda d: d / INT8_FILE)
    engine = FastGujaratiEngine(model_dir)
    return engine, session


# --- vocab / config -----------------------------------------------------------


def test_load_vocab_orders_by_id_and_finds_blank():
    vocab, blank = load_vocab(VOCAB_LINES)
    assert vocab[0] == "<unk>" and vocab[4] == BLANK_TOKEN and blank == 4
    assert vocab[1] == "▁મા"


def test_load_vocab_skips_blank_lines_and_crlf():
    vocab, blank = load_vocab(["<unk> 0\r\n", "", "<blk> 1\r\n"])
    assert vocab == ["<unk>", BLANK_TOKEN] and blank == 1


@pytest.mark.parametrize(
    "lines",
    [["<unk> 0", "<blk> 2"], ["<unk>", "<blk> 1"], ["<unk> 0", "x 1"], ["<unk> x"]],
)
def test_load_vocab_rejects_gaps_malformed_and_missing_blank(lines):
    with pytest.raises(ValueError):
        load_vocab(lines)


def test_read_config_defaults_when_missing(tmp_path):
    assert read_config(tmp_path) == (4, 80)
    (tmp_path / "config.json").write_text('{"subsampling_factor": 8, "features_size": 128}')
    assert read_config(tmp_path) == (8, 128)


def test_log_softmax_normalises_and_is_idempotent():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], np.float32)
    lp = log_softmax(logits)
    assert np.exp(lp).sum(axis=-1) == pytest.approx([1.0, 1.0])
    assert log_softmax(lp) == pytest.approx(lp, abs=1e-6)


# --- int8 build ---------------------------------------------------------------


def test_ensure_int8_builds_once_via_temp_then_rename(tmp_path):
    model_dir = _model_dir(tmp_path)
    calls: list[tuple[Path, Path]] = []

    def quantize(src: Path, dst: Path) -> None:
        calls.append((src, dst))
        dst.write_bytes(b"int8")

    out = ensure_int8(model_dir, quantize)
    assert out == model_dir / INT8_FILE and out.read_bytes() == b"int8"
    assert calls == [(model_dir / FP32_FILE, model_dir / (INT8_FILE + ".tmp"))]
    assert not (model_dir / (INT8_FILE + ".tmp")).exists()
    # second call: already built, quantizer not invoked
    assert ensure_int8(model_dir, quantize) == out
    assert len(calls) == 1


def test_ensure_int8_requires_fp32_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        ensure_int8(tmp_path, lambda s, d: None)


# --- engine -------------------------------------------------------------------


def test_engine_is_gujarati_and_implements_protocol(fake_engine):
    engine, _ = fake_engine
    assert engine.lang == "gu"
    assert isinstance(engine, ASREngine)
    assert engine.frame_s == pytest.approx(0.04)


def test_frame_duration_follows_config(monkeypatch, tmp_path):
    model_dir = _model_dir(tmp_path, subsampling=8)
    monkeypatch.setattr(engine_gu_fast, "_load_session", lambda path: FakeSession(8))
    monkeypatch.setattr(engine_gu_fast, "_load_preprocessor", lambda n: FakePreprocessor())
    monkeypatch.setattr(engine_gu_fast, "ensure_int8", lambda d: d / INT8_FILE)
    assert FastGujaratiEngine(model_dir).frame_s == pytest.approx(0.08)


def test_transcribe_detailed_words_times_and_confidence(fake_engine):
    engine, session = fake_engine
    tr = engine.transcribe_detailed(np.zeros(SR * 2, np.float32))
    assert tr.text == "મારું છે"  # "મારું છે"
    assert [w.text for w in tr.words] == ["મારું", "છે"]
    first, second = tr.words
    assert first.start_s == pytest.approx(0.0) and first.end_s == pytest.approx(3 * 0.04)
    assert second.start_s == pytest.approx(5 * 0.04) and second.end_s == pytest.approx(6 * 0.04)
    # word conf = weakest token's best frame prob: રું frame has logit 0.5 vs -10 elsewhere
    assert first.conf < second.conf
    weak_frame = log_softmax(np.array([-10.0, -10.0, 0.5, -10.0, -10.0]))
    assert first.conf == pytest.approx(float(np.exp(weak_frame[2])), rel=1e-3)
    feeds = session.calls[-1]
    assert feeds["audio_signal"].shape == (1, 80, 200) and feeds["length"].tolist() == [200]


def test_transcribe_returns_text_only(fake_engine):
    engine, _ = fake_engine
    assert engine.transcribe(np.zeros(SR, np.float32)) == "મારું છે"


def test_short_audio_is_empty_without_running_model(fake_engine):
    engine, session = fake_engine
    assert engine.transcribe_detailed(np.zeros(SR // 20, np.float32)) == Transcript.empty()
    assert session.calls == []


def test_rejects_non_1d_audio(fake_engine):
    engine, _ = fake_engine
    with pytest.raises(ValueError):
        engine.transcribe_detailed(np.zeros((2, SR), np.float32))


def test_warmup_runs_default_passes_and_single_pass(fake_engine):
    engine, session = fake_engine
    engine.warmup()
    assert len(session.calls) == len(engine_gu_fast.WARMUP_PASSES)
    engine.warmup(seconds=1.0)
    assert len(session.calls) == len(engine_gu_fast.WARMUP_PASSES) + 1
    assert session.calls[-1]["length"].tolist() == [100]
