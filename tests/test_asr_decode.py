import numpy as np
import pytest

from gujusub import asr_engine
from gujusub.asr_engine import ctc_words

BLANK = 4
VOCAB = ["▁ka", "m", "▁cha", "e", "<blank>"]
FRAME = 0.08


def logprobs_for(path, peak=0.9):
    """(T, V) log-prob matrix whose argmax follows ``path``."""
    v = len(VOCAB)
    rest = (1.0 - peak) / (v - 1)
    probs = np.full((len(path), v), rest, dtype=np.float32)
    probs[np.arange(len(path)), path] = peak
    return np.log(probs)


def reference_greedy(lp, vocab, blank):
    idx = lp.argmax(-1)
    collapsed = [int(x) for i, x in enumerate(idx) if i == 0 or x != idx[i - 1]]
    return "".join(vocab[x] for x in collapsed if x != blank).replace("▁", " ").strip()


def test_repeats_collapse_and_words_split_on_marker():
    path = [0, 0, 1, BLANK, 2, 2, 3, BLANK]
    t = ctc_words(logprobs_for(path), VOCAB, BLANK, FRAME)
    assert t.text == "kam chae"
    assert [w.text for w in t.words] == ["kam", "chae"]


def test_blank_separates_repeated_token():
    path = [1, BLANK, 1, 1]
    t = ctc_words(logprobs_for(path), VOCAB, BLANK, FRAME)
    assert t.text == "mm"
    assert [w.text for w in t.words] == ["mm"]


def test_times_from_frames_and_monotone():
    path = [BLANK, 0, 0, 1, BLANK, BLANK, 2, 3]
    t = ctc_words(logprobs_for(path), VOCAB, BLANK, FRAME)
    first, second = t.words
    assert first.start_s == pytest.approx(1 * FRAME)
    assert first.end_s == pytest.approx(4 * FRAME)
    assert second.start_s == pytest.approx(6 * FRAME)
    assert second.end_s == pytest.approx(8 * FRAME)
    starts = [w.start_s for w in t.words]
    assert starts == sorted(starts)
    assert all(w.start_s < w.end_s for w in t.words)


def test_confidence_token_max_word_min():
    lp = logprobs_for([0, 0, 1])
    lp[0, 0] = np.log(0.5)  # "▁ka" frames: 0.5 then 0.9 -> token conf 0.9
    lp[2, 1] = np.log(0.6)  # "m": 0.6 -> word conf = min(0.9, 0.6)
    (w,) = ctc_words(lp, VOCAB, BLANK, FRAME).words
    assert w.conf == pytest.approx(0.6, rel=1e-5)
    assert 0.0 < w.conf <= 1.0


def test_all_blank_is_empty():
    t = ctc_words(logprobs_for([BLANK] * 5), VOCAB, BLANK, FRAME)
    assert t.text == ""
    assert t.words == ()


def test_zero_frames_is_empty():
    t = ctc_words(np.zeros((0, len(VOCAB)), dtype=np.float32), VOCAB, BLANK, FRAME)
    assert t.text == "" and t.words == ()


def test_leading_continuation_token_forms_word():
    t = ctc_words(logprobs_for([1, 3, BLANK, 0]), VOCAB, BLANK, FRAME)
    assert t.text == "me ka"
    assert [w.text for w in t.words] == ["me", "ka"]


def test_bare_marker_token_does_not_emit_empty_word():
    vocab = ["▁", "a", "<blank>"]
    lp = np.log(np.full((4, 3), 0.05, dtype=np.float32))
    for f, tok in enumerate([0, 1, BLANK - 2, 0]):
        lp[f, tok] = np.log(0.9)
    t = ctc_words(lp, vocab, 2, FRAME)
    assert t.text == "a"
    assert [w.text for w in t.words] == ["a"]


def test_text_matches_reference_on_random_matrices():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lp = rng.normal(size=(int(rng.integers(1, 40)), len(VOCAB))).astype(np.float32)
        lp[:, BLANK] += 1.0
        lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
        t = ctc_words(lp, VOCAB, BLANK, FRAME)
        assert t.text == reference_greedy(lp, VOCAB, BLANK)
        assert " ".join(w.text for w in t.words) == t.text
        assert all(0.0 < w.conf <= 1.0 for w in t.words)


def test_ort_shim_injects_session_options(monkeypatch):
    calls = []

    class FakeOrt:
        SessionOptions = object

        @staticmethod
        def InferenceSession(*a, **k):
            calls.append(k)
            return "sess"

    sentinel = object()
    shim = asr_engine._OrtShim(FakeOrt, lambda: sentinel)
    assert shim.InferenceSession("p", providers=["CPU"]) == "sess"
    assert calls[-1]["sess_options"] is sentinel
    explicit = object()
    shim.InferenceSession("p", sess_options=explicit)
    assert calls[-1]["sess_options"] is explicit
    assert shim.SessionOptions is object  # other attributes forward
