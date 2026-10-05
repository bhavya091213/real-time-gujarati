import numpy as np
import pytest
from fakes import FakeClock, FakeEngine, FakeVAD, PlainEngine, frames, ramp

from gujusub import streaming
from gujusub.streaming import (
    VAD_FRAME,
    VAD_FRAME_MS,
    StreamingConfig,
    StreamingTranscriber,
    TranscriptEvent,
)

# 480 ms interval = 15 frames; 600 ms endpoint = ceil(600/32)=19 frames (18.75).
INTERVAL_FRAMES = 15
ENDPOINT_FRAMES = 19


def feed_all(tr, n_frames):
    """Feed n frames one at a time; return all events."""
    out = []
    for _ in range(n_frames):
        out.extend(tr.feed(frames(1)))
    return out


def make(engine, speech):
    vad = FakeVAD(speech)
    return StreamingTranscriber(engine, vad=vad), vad


def test_frame_constants():
    assert VAD_FRAME == 512 and VAD_FRAME_MS == 32


def test_partial_cadence_every_480ms():
    engine = FakeEngine(["a"])
    tr, _ = make(engine, [(0, 10_000)])
    events = feed_all(tr, 1 + 3 * INTERVAL_FRAMES)  # onset frame + 3 intervals
    assert engine.calls == 3
    assert [e.type for e in events] == ["partial"] * 3


def test_no_decode_before_interval_elapses():
    engine = FakeEngine(["a"])
    tr, _ = make(engine, [(0, 10_000)])
    assert feed_all(tr, INTERVAL_FRAMES) == []  # onset + 14 frames
    assert engine.calls == 0


def test_local_agreement_commits_common_prefix_of_last_two():
    engine = FakeEngine(["a b c", "a b d", "a b d e"])
    tr, _ = make(engine, [(0, 10_000)])
    events = feed_all(tr, 1 + 3 * INTERVAL_FRAMES)
    got = [(e.committed, e.tail) for e in events]
    assert got == [("", "a b c"), ("a b", "d"), ("a b d", "e")]


def test_committed_never_shrinks_within_utterance():
    engine = FakeEngine(["a b c", "a b c", "a x y", "a x y"])
    tr, _ = make(engine, [(0, 10_000)])
    events = feed_all(tr, 1 + 4 * INTERVAL_FRAMES)
    lengths = [len(e.committed.split()) for e in events]
    assert lengths == sorted(lengths)
    assert events[1].committed == "a b c"
    assert events[2].committed == "a b c"  # diverging hypothesis does not retract
    assert events[2].tail == ""  # hyp shorter than committed -> empty tail


@pytest.mark.parametrize("final_hyp", ["a", "a b x", ""])
def test_final_decode_cannot_retract_committed_words(final_hyp):
    engine = FakeEngine(["a b c", "a b c", final_hyp])
    tr, _ = make(engine, [(0, 10_000)])
    partials = feed_all(tr, 1 + 2 * INTERVAL_FRAMES)
    assert partials[-1].committed == "a b c"

    finals = tr.flush()

    assert len(finals) == 1
    assert finals[0].type == "final"
    assert finals[0].committed == "a b c"


def test_endpoint_emits_final_and_increments_utterance_id():
    engine = FakeEngine(lambda n: "hello world")
    # speech frames 0..19, then silence, then new speech.
    tr, vad = make(engine, [(0, 20), (200, 300)])
    events = feed_all(tr, 20 + ENDPOINT_FRAMES)
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1
    assert (finals[0].committed, finals[0].tail) == ("hello world", "")
    assert finals[0].utterance_id == 1
    assert vad.resets == 1
    # final re-decode covers speech + trailing silence frames
    assert engine.audio_lengths[-1] == (20 + ENDPOINT_FRAMES) * VAD_FRAME
    # next utterance gets a new id
    tr.feed(frames(200 - (20 + ENDPOINT_FRAMES) + 1))
    events = feed_all(tr, INTERVAL_FRAMES)
    assert events and events[-1].utterance_id == 2


def test_silence_below_endpoint_does_not_finalize():
    engine = FakeEngine(lambda n: "hi")
    tr, _ = make(engine, [(0, 20)])
    events = feed_all(tr, 20 + ENDPOINT_FRAMES - 1)
    assert all(e.type == "partial" for e in events)


def test_hard_cap_forces_final_at_12s():
    engine = FakeEngine(lambda n: "words")
    tr, _ = make(engine, [(0, 100_000)])
    cap_frames = int(12.0 * 16000) // VAD_FRAME  # 375 frames incl. onset
    events = feed_all(tr, cap_frames - 1)
    assert [e for e in events if e.type == "final"] == []
    events = feed_all(tr, 6)
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 1
    # No alignment is available, so trimming is deferred to preserve text.
    assert engine.audio_lengths[-1] == int(12.0 * SR)


# ---- bounded window (unit 1.3) ----

SR = 16000
WORD_S = 0.4
INTERVAL_S = 0.48
MAX_WINDOW_SAMPLES = int((5.0 + INTERVAL_S) * SR) + VAD_FRAME


def feed_ramp(tr, n_frames, start=0):
    out = []
    for i in range(start, start + n_frames):
        out.extend(tr.feed(ramp(i, 1)))
    return out


def by_utterance(events):
    utts: dict[int, list] = {}
    for e in events:
        utts.setdefault(e.utterance_id, []).append(e)
    return utts


def assert_monotone(events):
    for utt in by_utterance(events).values():
        for prev, cur in zip(utt, utt[1:], strict=False):
            assert cur.committed.startswith(prev.committed), (prev, cur)


def test_bounded_window_30s_no_word_lost_or_duplicated():
    engine = FakeEngine(word_s=WORD_S)
    tr, _ = make(engine, [(0, 100_000)])
    n = 938  # 30.016 s
    events = feed_ramp(tr, n) + tr.flush()
    assert max(engine.audio_lengths) <= MAX_WINDOW_SAMPLES
    assert_monotone(events)
    finals = [e for e in events if e.type == "final"]
    assert len(finals) == 3  # 12 s cap twice, then flush
    all_words = " ".join(f.committed for f in finals).split()
    expected = [f"w{i}" for i in range(75)]  # midpoints < 30.016 s
    assert all_words == expected
    # every partial's committed text survives into its utterance's final
    finals_by_id = {f.utterance_id: f.committed for f in finals}
    for e in events:
        if e.type == "partial":
            assert finals_by_id[e.utterance_id].startswith(e.committed)
    # partial committed grows past the window: prefix carries dropped words
    longest = max(len(e.committed.split()) for e in events if e.type == "partial")
    assert longest > int(5.0 / WORD_S)


def test_seam_residue_of_cut_word_is_not_duplicated():
    # Real CTC re-hears the tail of the word just before a cut; the fake
    # models that by echoing words up to 0.3 s before the window start.
    engine = FakeEngine(word_s=WORD_S, echo_s=0.3)
    tr, _ = make(engine, [(0, 100_000)])
    events = feed_ramp(tr, 300) + tr.flush()  # 9.6 s, several trims
    assert_monotone(events)
    assert events[-1].committed.split() == [f"w{i}" for i in range(24)]


def test_repeated_words_across_aligned_trim_survive_partial_and_final():
    def word_text(i):
        return {10: "go", 11: "go", 12: "home"}.get(i, f"w{i}")

    engine = FakeEngine(word_s=WORD_S, word_text=word_text)
    config = StreamingConfig(min_context_s=0.1)
    tr = StreamingTranscriber(engine, config=config, vad=FakeVAD([(0, 100_000)]))

    partials = feed_ramp(tr, 1 + 11 * INTERVAL_FRAMES)
    trim_call = next(
        i
        for i, (before, after) in enumerate(
            zip(engine.audio_lengths, engine.audio_lengths[1:], strict=False), start=1
        )
        if after < before
    )
    first_post_trim = partials[trim_call]
    assert (first_post_trim.committed + " " + first_post_trim.tail).split()[-3:] == [
        "go",
        "go",
        "home",
    ]

    final = tr.flush()[0]
    assert final.committed.split()[-3:] == ["go", "go", "home"]


def test_shorter_seam_overlap_beats_longest_periodic_match():
    def word_text(i):
        return "go" if 8 <= i <= 11 else "home" if i == 12 else f"w{i}"

    engine = FakeEngine(word_s=WORD_S, echo_s=0.3, word_text=word_text)
    config = StreamingConfig(max_window_s=5.5, min_context_s=0.1, trim_margin_s=1.5)
    tr = StreamingTranscriber(engine, config=config, vad=FakeVAD([(0, 100_000)]))

    partials = feed_ramp(tr, 1 + 12 * INTERVAL_FRAMES)
    trim_call = next(
        i
        for i, (before, after) in enumerate(
            zip(engine.audio_lengths, engine.audio_lengths[1:], strict=False), start=1
        )
        if after < before
    )
    expected = [word_text(i) for i in range(14)]
    first_post_trim = partials[trim_call]
    assert (first_post_trim.committed + " " + first_post_trim.tail).split() == expected

    final = tr.flush()[0]
    assert final.committed.split() == expected


def test_equal_seam_scores_prefer_shortest_overlap():
    def word_text(i):
        return "a" if i % 2 == 0 else "b"

    engine = FakeEngine(word_s=0.15, echo_s=0.12, word_text=word_text)
    config = StreamingConfig(max_window_s=4.5, min_context_s=0.1)
    tr = StreamingTranscriber(engine, config=config, vad=FakeVAD([(0, 100_000)]))

    partials = feed_ramp(tr, 1 + 10 * INTERVAL_FRAMES)
    trim_call = next(
        i
        for i, (before, after) in enumerate(
            zip(engine.audio_lengths, engine.audio_lengths[1:], strict=False), start=1
        )
        if after < before
    )
    expected = [word_text(i) for i in range(32)]
    first_post_trim = partials[trim_call]
    assert (first_post_trim.committed + " " + first_post_trim.tail).split() == expected

    final = tr.flush()[0]
    assert final.committed.split() == expected


def test_trim_keeps_min_context():
    engine = FakeEngine(word_s=WORD_S)
    tr, _ = make(engine, [(0, 100_000)])
    feed_ramp(tr, 300)  # 9.6 s, below the cap
    lengths = engine.audio_lengths
    trimmed = [b for a, b in zip(lengths, lengths[1:], strict=False) if b < a]
    assert trimmed, "window was never trimmed"
    assert min(trimmed) >= int(2.0 * SR)


def test_aligned_trim_without_a_retained_word_is_deferred():
    engine = FakeEngine(word_s=20.0)  # one word longer than any window
    tr, _ = make(engine, [(0, 100_000)])
    events = feed_ramp(tr, 300) + tr.flush()
    assert max(engine.audio_lengths) > MAX_WINDOW_SAMPLES
    assert max(engine.audio_lengths) < int(12.0 * SR)
    assert_monotone(events)


def test_plain_engine_defers_trim_until_12s_hard_cap():
    inner = FakeEngine(lambda n: f"x{n // 16000}")
    tr, _ = make(PlainEngine(inner), [(0, 100_000)])
    cap_frames = int(12.0 * SR) // VAD_FRAME
    events = feed_all(tr, cap_frames + 10)
    assert max(inner.audio_lengths) > MAX_WINDOW_SAMPLES
    assert max(inner.audio_lengths) == int(12.0 * SR)
    assert_monotone(events)
    assert [event.type for event in events].count("final") == 1


def test_short_plain_engine_utterance_final_unchanged():
    tr, _ = make(PlainEngine(FakeEngine(lambda n: "hello world")), [(0, 20)])
    events = feed_all(tr, 20 + ENDPOINT_FRAMES)
    assert [(e.type, e.committed) for e in events][-1] == ("final", "hello world")


def _decodes_in_10s(decode_ms):
    clock = FakeClock()
    engine = FakeEngine(["a"], clock=clock, decode_ms=decode_ms)
    tr = StreamingTranscriber(engine, vad=FakeVAD([(0, 100_000)]), clock=clock)
    feed_all(tr, 312)  # ~10 s, under the cap
    return engine.calls, tr


def test_adaptive_interval_backs_off_when_decode_is_slow():
    fast_calls, fast = _decodes_in_10s(10.0)
    slow_calls, slow = _decodes_in_10s(1000.0)
    assert fast_calls == 20
    assert slow_calls < fast_calls
    assert slow_calls <= int(10.0 / 1.2) + 1
    assert slow.last_decode_ms == pytest.approx(1000.0)
    assert slow.avg_decode_ms == pytest.approx(1000.0)
    assert fast.decode_count == fast_calls


def test_adaptive_interval_never_below_configured():
    calls, tr = _decodes_in_10s(0.0)
    assert calls == 20
    assert tr.last_decode_ms == 0.0


def test_stats_before_any_decode():
    tr, _ = make(FakeEngine(), [])
    assert tr.decode_count == 0 and tr.last_decode_ms == 0.0 and tr.avg_decode_ms == 0.0


# ---- adaptive interval robustness (F-04) ----

CAP_SAMPLES = 2000 * SR // 1000 + VAD_FRAME  # cap, rounded up to whole frames


def _scripted_decode_ms(schedule, default):
    """decode_ms callable returning schedule[i] for the i-th decode."""
    calls = iter(schedule)
    return lambda _n: next(calls, default)


def _partial_gaps(engine):
    """Audio (samples) between successive decodes (no trim: plain frames)."""
    lengths = engine.audio_lengths
    return [b - a for a, b in zip(lengths, lengths[1:], strict=False)]


def test_one_slow_decode_does_not_stall_partials():
    clock = FakeClock()
    engine = FakeEngine(["a"], clock=clock, decode_ms=_scripted_decode_ms([20_000.0], 10.0))
    tr = StreamingTranscriber(engine, vad=FakeVAD([(0, 100_000)]), clock=clock)
    feed_all(tr, 312)  # ~10 s, under the 12 s cap: no final yet
    assert engine.calls >= 3  # partials resume after the outlier
    assert _partial_gaps(engine)[0] <= CAP_SAMPLES
    assert tr.last_decode_ms == pytest.approx(10.0)  # counters still raw


def test_slow_estimate_does_not_carry_into_next_utterance():
    clock = FakeClock()
    engine = FakeEngine(["a"], clock=clock, decode_ms=1000.0)
    tr = StreamingTranscriber(engine, vad=FakeVAD([(0, 40), (200, 400)]), clock=clock)
    feed_all(tr, 200)  # utterance 1 ends with a 1000 ms final decode
    calls_before = engine.calls
    events = feed_all(tr, INTERVAL_FRAMES + 1)  # onset + one base interval
    assert engine.calls == calls_before + 1
    assert [(e.type, e.utterance_id) for e in events] == [("partial", 2)]


def test_sustained_slow_decodes_back_off_but_stay_capped():
    clock = FakeClock()
    engine = FakeEngine(["a"], clock=clock, decode_ms=5000.0)
    tr = StreamingTranscriber(engine, vad=FakeVAD([(0, 100_000)]), clock=clock)
    feed_all(tr, 312)
    gaps = _partial_gaps(engine)
    assert engine.calls < 20  # backed off vs the 480 ms base cadence
    assert gaps and all(INTERVAL_FRAMES * VAD_FRAME < g <= CAP_SAMPLES for g in gaps)


def test_max_interval_ms_config_is_respected():
    assert StreamingConfig().max_interval_ms == 2000
    clock = FakeClock()
    engine = FakeEngine(["a"], clock=clock, decode_ms=5000.0)
    cfg = StreamingConfig(max_interval_ms=1000)
    tr = StreamingTranscriber(engine, cfg, vad=FakeVAD([(0, 100_000)]), clock=clock)
    feed_all(tr, 312)
    gaps = _partial_gaps(engine)
    assert gaps and all(g <= 1000 * SR // 1000 + VAD_FRAME for g in gaps)


def test_config_defaults_for_window():
    cfg = StreamingConfig()
    assert (cfg.max_window_s, cfg.min_context_s, cfg.trim_margin_s) == (5.0, 2.0, 0.2)
    assert cfg.max_utterance_s == 12.0


def test_preroll_not_stale_after_final():
    engine = FakeEngine(word_s=WORD_S)
    # utterance 1 frames 0..39, endpoint, then utterance 2 starts at frame 60
    tr, _ = make(engine, [(0, 40), (60, 100_000)])
    calls_before = None
    for i in range(61 + INTERVAL_FRAMES):
        if i == 60:
            calls_before = engine.calls
        tr.feed(ramp(i, 1))
    assert engine.calls == calls_before + 1
    # utterance 1 finalised at frame 58; preroll is frame 59 + onset 60 only
    # (no stale frames left over from before utterance 1's onset)
    assert engine.audio_lengths[-1] == (61 + INTERVAL_FRAMES - 59) * VAD_FRAME


def test_preroll_included_before_vad_trigger():
    engine = FakeEngine(lambda n: "x")
    # 30 silent frames then speech; keep speech to end of audio, flush decodes.
    tr, _ = make(engine, [(30, 100_000)])
    feed_all(tr, 30 + 1)  # 30 silence + onset frame
    tr.flush()
    preroll_frames = 320 // VAD_FRAME_MS  # 10, includes the onset frame itself
    assert preroll_frames == 10
    assert engine.audio_lengths[-1] == preroll_frames * VAD_FRAME


@pytest.mark.parametrize("n_speech", [1, 3, 7])
def test_short_utterance_dropped_as_noise(n_speech):
    engine = FakeEngine(lambda n: "blip")
    tr, _ = make(engine, [(0, n_speech)])
    events = feed_all(tr, n_speech + ENDPOINT_FRAMES + 5)
    # Pinned: the 480 ms partial cadence keeps ticking during the endpoint
    # silence, so a partial can still fire; but no final is produced.
    assert [e for e in events if e.type == "final"] == []
    assert engine.calls == 1  # the partial only; no final decode


def test_empty_hypothesis_emits_no_event():
    engine = FakeEngine([""])
    tr, _ = make(engine, [(0, 10_000)])
    assert feed_all(tr, 1 + 2 * INTERVAL_FRAMES) == []
    assert engine.calls == 2


def test_empty_final_text_emits_no_event():
    engine = FakeEngine([""])
    tr, _ = make(engine, [(0, 20)])
    assert feed_all(tr, 20 + ENDPOINT_FRAMES) == []
    assert engine.calls >= 1


def test_default_vad_loads_silero_via_loader(monkeypatch):
    calls = []

    class _Model:
        def reset_states(self):
            calls.append("reset")

        def __call__(self, tensor, sr):
            calls.append(("call", tuple(tensor.shape), sr))

            class _R:
                def item(self_inner):
                    return 0.9

            return _R()

    monkeypatch.setattr(streaming, "_load_silero_vad", lambda: calls.append("load") or _Model())
    tr = StreamingTranscriber(FakeEngine())
    assert calls == ["load"]
    assert isinstance(tr.vad, streaming.SileroVAD)
    assert tr.vad.speech_prob(np.zeros(VAD_FRAME, dtype=np.float32)) == 0.9
    tr.vad.reset()
    assert calls[-1] == "reset"


def test_injected_vad_skips_silero(monkeypatch):
    def boom():
        raise AssertionError("silero must not load when vad injected")

    monkeypatch.setattr(streaming, "_load_silero_vad", boom)
    StreamingTranscriber(FakeEngine(), vad=FakeVAD([]))


def test_event_to_dict_shape():
    ev = TranscriptEvent("partial", 3, "a b", "c")
    assert ev.to_dict() == {
        "type": "partial",
        "utterance_id": 3,
        "committed": "a b",
        "tail": "c",
        "lang": "gu",
    }


def test_feed_rejects_non_float32():
    tr, _ = make(FakeEngine(), [])
    with pytest.raises(ValueError):
        tr.feed(np.zeros(VAD_FRAME, dtype=np.float64))


def test_feed_buffers_partial_frames():
    engine = FakeEngine(["a"])
    tr, vad = make(engine, [(0, 10_000)])
    tr.feed(np.zeros(VAD_FRAME - 1, dtype=np.float32))
    assert vad.index == 0
    tr.feed(np.zeros(1, dtype=np.float32))
    assert vad.index == 1


def test_flush_without_utterance_is_empty():
    tr, _ = make(FakeEngine(["a"]), [])
    feed_all(tr, 5)
    assert tr.flush() == []


def test_flush_finalizes_in_progress_utterance_once():
    engine = FakeEngine(lambda n: "partial words")
    tr, vad = make(engine, [(0, 10_000)])
    feed_all(tr, 20)
    events = tr.flush()
    assert [(e.type, e.committed, e.tail) for e in events] == [("final", "partial words", "")]
    assert vad.resets == 1
    assert tr.flush() == []


def test_flush_short_utterance_returns_nothing():
    engine = FakeEngine(lambda n: "x")
    tr, _ = make(engine, [(0, 10_000)])
    feed_all(tr, 3)
    assert tr.flush() == []
    assert engine.calls == 0


# ---- R2-02: cached seam overlap re-checked against each decode ----


def _just_trimmed(engine, prefix, seam_reference):
    """Transcriber whose open utterance looks like it was just trimmed."""
    tr, _ = make(engine, [(0, 10_000)])
    tr.feed(frames(1))  # onset opens the utterance
    utt = tr._utt
    utt.prefix = list(prefix)
    utt.seam_reference = list(seam_reference)
    utt.seam_overlap = None
    return tr


def test_cached_seam_overlap_not_applied_once_residue_disappears():
    # First post-trim decode re-hears "home" (residue) -> overlap 1 cached.
    # Later decodes no longer start with "home"; "is" must not be stripped.
    engine = FakeEngine(["home is", "is where it", "is where it", "is where it"])
    tr = _just_trimmed(engine, ["go", "home"], ["is"])
    partials = feed_all(tr, 3 * INTERVAL_FRAMES)
    assert (partials[0].committed + " " + partials[0].tail).split() == ["go", "home", "is"]
    assert (partials[1].committed + " " + partials[1].tail).split() == [
        "go", "home", "is", "where", "it",
    ]
    assert partials[-1].committed == "go home is where it"
    assert tr.flush()[0].committed == "go home is where it"


def test_cached_seam_overlap_still_applied_while_residue_persists():
    engine = FakeEngine(["home is", "home is where", "home is where", "home is where"])
    tr = _just_trimmed(engine, ["go", "home"], ["is"])
    partials = feed_all(tr, 3 * INTERVAL_FRAMES)
    assert partials[-1].committed == "go home is where"
    assert tr.flush()[0].committed == "go home is where"


# ---- R2-01: final disagreeing inside committed keeps the on-screen tail ----


def test_final_with_substitution_inside_committed_keeps_new_words():
    engine = FakeEngine(["a b c", "a b c", "a B c d e f"])
    tr, _ = make(engine, [(0, 10_000)])
    feed_all(tr, 1 + 2 * INTERVAL_FRAMES)
    assert tr.flush()[0].committed == "a b c d e f"


def test_final_with_insertion_falls_back_to_last_partial_tail():
    engine = FakeEngine(["a b c", "a b c d", "a x b c d"])
    tr, _ = make(engine, [(0, 10_000)])
    partials = feed_all(tr, 1 + 2 * INTERVAL_FRAMES)
    assert (partials[-1].committed, partials[-1].tail) == ("a b c", "d")
    assert tr.flush()[0].committed == "a b c d"


def test_final_mismatch_without_usable_partial_keeps_committed_only():
    # Last partial diverged from committed, so its tail is not trusted.
    engine = FakeEngine(["a b c", "a b c", "a y z q", "a x b c d"])
    tr, _ = make(engine, [(0, 10_000)])
    partials = feed_all(tr, 1 + 3 * INTERVAL_FRAMES)
    assert partials[-1].committed == "a b c"
    assert tr.flush()[0].committed == "a b c"


def _bounded_run(provider):
    engine = FakeEngine(word_s=WORD_S)  # every word conf 0.9
    tr = StreamingTranscriber(engine, vad=FakeVAD([(0, 100_000)]),
                              thresholds_provider=provider)
    return feed_ramp(tr, 400) + tr.flush()


@pytest.mark.parametrize("thresholds", [(0.5, 0.0), (0.0, 0.0)])
def test_bounded_window_unchanged_when_every_word_passes_confidence(thresholds):
    assert _bounded_run(lambda: thresholds) == _bounded_run(None)


def test_recommended_defaults_keep_finals_and_one_word_partials():
    gated, off = _bounded_run(lambda: (0.5, 0.7)), _bounded_run(None)
    finals = [e for e in gated if e.type == "final"]
    assert finals == [e for e in off if e.type == "final"]
    # min_words defaults to 1: first (one-word) partials are not held back
    assert any(len(f"{e.committed} {e.tail}".split()) == 1 for e in gated)
    assert len(gated) == len(off)


def test_bounded_window_gate_drops_everything_below_utterance_threshold():
    engine = FakeEngine(word_s=WORD_S)  # every word conf 0.9
    tr = StreamingTranscriber(engine, vad=FakeVAD([(0, 100_000)]),
                              thresholds_provider=lambda: (0.0, 0.95))
    assert feed_ramp(tr, 400) + tr.flush() == []
