"""Real-time streaming transcription on top of an offline ASR model.

Algorithm: Silero-VAD-gated growing window with LocalAgreement-2 commits
(whisper_streaming style, Machacek et al. 2023).

- VAD gates the stream: nothing is transcribed during silence.
- While speech is active, the whole current utterance buffer is re-decoded
  every `interval_ms` of new audio (audio clock, so it is deterministic in
  tests). The longest common prefix of the last two hypotheses is "committed"
  and never retracted; the rest is an unstable tail.
- A pause of `endpoint_ms` ends the utterance and emits a final decode.
- Bounded window: once the buffered audio exceeds `max_window_s`, aligned
  audio through a committed word is dropped only when another aligned word
  remains as post-cut context. The dropped words move into a frozen prefix.
  The first post-trim decode uses that retained hypothesis to distinguish CTC
  residue from a genuine repeated word before selecting a 1-5 word overlap;
  the selection is reused until the next trim. Without alignment evidence,
  trimming is deferred and the 12-second utterance cap remains the bound.
- Adaptive interval: the re-decode cadence is max(interval_ms,
  1.2 x recent decode time), clamped to max_interval_ms, so a slow machine
  decodes less often instead of falling behind. "Recent decode time" is the
  low median of this utterance's last few partial decodes (see
  _interval_samples).
- Engine per utterance: with `engine_for_utterance`, the engine is chosen once
  when VAD opens an utterance and pinned for its whole life (one language per
  utterance; a mode switch applies from the next utterance). Events carry the
  pinned engine's `lang`, and the window bound is max_window_s_by_lang[lang]
  (falling back to max_window_s).
- Confidence trimming: every decode (partial and final) passes through
  confidence.apply_confidence with thresholds from `thresholds_provider`
  (read per decode) BEFORE LocalAgreement, so a low-confidence word never
  commits and an all-low decode counts as an empty hypothesis.
- Auto mode: the provider may return an AutoRoute instead of an engine. The
  utterance then opens with no engine; only VAD speech frames from the onset
  frame on (no preroll, no pauses) are fed to the LID decider, and nothing is
  decoded or emitted while it is unsure. On a decision the engine is pinned
  and a partial pass runs at once over the whole buffered utterance. If the
  utterance ends undecided, one last check runs on all its speech; still
  unsure -> dropped. If max_s passes undecided, LidConfig.lid_fallback applies
  ("gu": pin Gujarati with a warning; "none": drop the utterance).
"""

import collections
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from gujusub.asr_engine import SAMPLE_RATE, ASREngine
from gujusub.confidence import apply_confidence
from gujusub.engine import Transcript, Word

logger = logging.getLogger("streaming")

VAD_FRAME = 512  # samples per silero window @ 16 kHz (32 ms)
VAD_FRAME_MS = VAD_FRAME * 1000 // SAMPLE_RATE


class VAD(Protocol):
    """Minimal voice-activity interface the transcriber depends on."""

    def speech_prob(self, frame: np.ndarray) -> float:
        """Speech probability in [0, 1] for one VAD_FRAME float32 frame."""
        ...

    def reset(self) -> None:
        """Clear any internal state (called at each utterance end)."""
        ...


def _load_silero_vad():
    # Lazy: keeps torch/silero out of import time (and out of tests).
    from silero_vad import load_silero_vad

    return load_silero_vad()


class SileroVAD:
    """Adapter from the Silero model to the VAD protocol."""

    def __init__(self):
        self._model = _load_silero_vad()

    def speech_prob(self, frame: np.ndarray) -> float:
        import torch

        return self._model(torch.from_numpy(frame), SAMPLE_RATE).item()

    def reset(self) -> None:
        self._model.reset_states()


@dataclass(frozen=True)
class TranscriptEvent:
    type: str  # "partial" | "final"
    utterance_id: int
    committed: str  # stable text, never retracted within an utterance
    tail: str  # unstable text, may change on the next update
    lang: str = "gu"  # language of the engine that decoded this utterance

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "utterance_id": self.utterance_id,
            "committed": self.committed,
            "tail": self.tail,
            "lang": self.lang,
        }


@dataclass(frozen=True)
class StreamingConfig:
    vad_start_prob: float = 0.5  # speech starts above this
    vad_end_prob: float = 0.35  # silence counted below this (hysteresis)
    endpoint_ms: int = 600  # this much silence finalizes the utterance
    preroll_ms: int = 320  # audio kept from before speech onset
    interval_ms: int = 480  # re-decode cadence while speech is active
    min_speech_ms: int = 250  # shorter blips are dropped as noise
    max_utterance_s: float = 12.0  # force a final after this much audio
    max_window_s: float = 5.0  # trim the decode window beyond this
    # per-engine override of max_window_s (Parakeet: ~325 ms per 4 s window)
    max_window_s_by_lang: dict[str, float] = field(default_factory=lambda: {"en": 4.0})
    min_context_s: float = 2.0  # audio always kept in the window after a trim
    trim_margin_s: float = 0.2  # gap required after the cut word's end
    adaptive_factor: float = 1.2  # interval >= factor x recent decode time
    max_interval_ms: int = 2000  # adaptive interval never exceeds this (audio time)


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b, strict=False):  # prefix of the shorter list
        if x != y:
            break
        out.append(x)
    return out


class LidPolicy(Protocol):
    """What the transcriber needs from gujusub.lid.LidDecider."""

    config: object  # has .lid_fallback ("gu" | "none")
    lang: str | None
    done: bool
    posteriors: dict[str, float] | None
    decided_s: float | None

    @property
    def speech_s(self) -> float: ...

    def feed(self, chunk: np.ndarray) -> str: ...

    def final_check(self) -> str: ...


@dataclass(frozen=True)
class AutoRoute:
    """Auto mode for one utterance: a fresh LID decider + lang -> engine lookup."""

    decider: LidPolicy
    engine_for: Callable[[str], ASREngine]


THRESHOLDS_OFF = (0.0, 0.0)  # (word_min, utt_min): confidence trimming disabled

DECODE_HISTORY = 5  # partial decodes in the adaptive-interval estimate


@dataclass
class _UtteranceState:
    engine: ASREngine | None  # pinned at utterance start (Auto: on the LID decision)
    lang: str  # the pinned engine's language
    max_window_s: float  # window bound for the pinned engine
    frames: list = field(default_factory=list)  # window: VAD_FRAME np arrays
    route: AutoRoute | None = None  # Auto mode, LID still undecided
    onset_frame: int = 0  # index in frames of the VAD onset frame
    silence_frames: int = 0
    samples_since_infer: int = 0
    committed: list = field(default_factory=list)  # committed words (window)
    prev_hyp: list = field(default_factory=list)  # previous hypothesis words
    prefix: list = field(default_factory=list)  # frozen words of dropped audio
    dropped_frames: int = 0  # audio dropped from the front of the window
    words: tuple = ()  # Word times of the last decode (window-relative)
    decoded_since_trim: bool = False
    seam_reference: list = field(default_factory=list)  # expected post-cut hypothesis
    seam_overlap: int | None = 0  # selected overlap; None until first post-trim decode
    recent_decode_ms: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=DECODE_HISTORY)
    )  # this utterance's last partial decode times, for the adaptive interval

    @property
    def total_frames(self) -> int:
        return self.dropped_frames + len(self.frames)

    def committed_text(self) -> str:
        return " ".join(self.prefix + self.committed)


SEAM_MAX_WORDS = 5
LID_LANGS = ("gu", "en")


def _fmt_posteriors(posteriors: dict[str, float] | None) -> str:
    if not posteriors:
        return "no classification"
    return " ".join(f"{k}={v:.2f}" for k, v in sorted(posteriors.items()))


def _seam_overlaps(prefix: list[str], hyp: list[str]) -> list[int]:
    """Matching prefix-tail overlap lengths, from shortest to longest."""
    return [
        n
        for n in range(1, min(SEAM_MAX_WORDS, len(prefix), len(hyp)) + 1)
        if prefix[-n:] == hyp[:n]
    ]


def _select_seam_overlap(utt: _UtteranceState, hyp: list[str]) -> int:
    """Select a text overlap only when the trim reference disambiguates it."""
    if utt.seam_overlap is None:
        raw_agreement = len(_common_prefix(utt.seam_reference, hyp))
        scored = (
            (len(_common_prefix(utt.seam_reference, hyp[n:])), n)
            for n in _seam_overlaps(utt.prefix, hyp)
        )
        best_agreement, best_overlap = max(
            scored,
            key=lambda candidate: (candidate[0], -candidate[1]),
            default=(raw_agreement, 0),
        )
        utt.seam_overlap = best_overlap if best_agreement > raw_agreement else 0
    # The selection is cached until the next trim, but the residue it strips
    # can vanish from later decodes; strip only while it is still there.
    n = utt.seam_overlap
    return n if n and utt.prefix[-n:] == hyp[:n] else 0


def _final_suffix(utt: _UtteranceState, hyp: list[str]) -> list[str]:
    """Words the final decode adds after the (never retracted) committed words.

    On a mismatch inside committed, keep the final's new words if it still
    ends the committed span on the last committed word (a substitution);
    otherwise fall back to the last partial's tail, which viewers already saw.
    """
    committed = utt.committed
    n = len(committed)
    if hyp[:n] == committed:
        return hyp[n:]
    if len(hyp) > n and hyp[n - 1] == committed[-1]:
        return hyp[n:]
    if utt.prev_hyp[:n] == committed:
        return utt.prev_hyp[n:]
    return []


def _decode(engine: ASREngine, audio: np.ndarray) -> Transcript:
    detailed = getattr(engine, "transcribe_detailed", None)
    if detailed is None:
        return Transcript(engine.transcribe(audio), ())
    return detailed(audio)


def _aligned_words(tr: Transcript, hyp: list[str]) -> tuple[Word, ...]:
    """Word times only if they line up 1:1 with the hypothesis text."""
    if [w.text for w in tr.words] == hyp:
        return tr.words
    return ()


class StreamingTranscriber:
    """Consumes 16 kHz mono float32 audio, emits TranscriptEvents.

    Not thread-safe: feed() from a single thread (the server uses one
    transcriber per connection inside a single executor task).
    """

    def __init__(
        self,
        engine: ASREngine | None = None,
        config: StreamingConfig | None = None,
        vad: VAD | None = None,
        clock: Callable[[], float] = time.perf_counter,
        *,
        engine_for_utterance: Callable[[], ASREngine] | None = None,
        thresholds_provider: Callable[[], tuple[float, float]] | None = None,
    ):
        if engine is None and engine_for_utterance is None:
            raise ValueError("need an engine or an engine_for_utterance provider")
        self.engine = engine  # single-engine path (tests, replay)
        self._engine_for_utterance = engine_for_utterance or (lambda: engine)
        self.config = config or StreamingConfig()
        self._thresholds_provider = thresholds_provider or (lambda: THRESHOLDS_OFF)
        self.vad = vad if vad is not None else SileroVAD()
        self._clock = clock
        self.decode_count = 0
        self.last_decode_ms = 0.0
        self.last_window_s = 0.0
        self._total_decode_ms = 0.0
        self._pending = np.empty(0, dtype=np.float32)
        preroll_frames = max(1, self.config.preroll_ms // VAD_FRAME_MS)
        self._preroll: collections.deque = collections.deque(maxlen=preroll_frames)
        self._utt: _UtteranceState | None = None
        self._utt_id = 0
        self.lid_last: dict | None = None  # latest Auto-mode LID outcome (status)

    def feed(self, audio: np.ndarray) -> list[TranscriptEvent]:
        """Ingest a chunk of audio (any length) and return any new events."""
        if audio.dtype != np.float32:
            raise ValueError(f"expected float32 audio, got {audio.dtype}")
        events: list[TranscriptEvent] = []
        self._pending = np.concatenate([self._pending, audio])
        while len(self._pending) >= VAD_FRAME:
            frame = self._pending[:VAD_FRAME]
            self._pending = self._pending[VAD_FRAME:]
            events.extend(self._process_frame(frame))
        return events

    def flush(self) -> list[TranscriptEvent]:
        """Finalize any in-progress utterance (e.g. on disconnect)."""
        if self._utt is None:
            return []
        return self._finalize()

    def _process_frame(self, frame: np.ndarray) -> list[TranscriptEvent]:
        prob = self.vad.speech_prob(frame)
        cfg = self.config

        if self._utt is None:
            self._preroll.append(frame)
            if prob >= cfg.vad_start_prob:
                self._utt_id += 1
                self._utt = self._open_utterance(list(self._preroll))
                self._preroll.clear()  # never reuse stale pre-onset audio
                if self._utt.route is not None:
                    self._utt.route.decider.feed(frame)  # the onset frame is speech
            return []

        utt = self._utt
        utt.frames.append(frame)
        utt.samples_since_infer += VAD_FRAME
        utt.silence_frames = 0 if prob >= cfg.vad_end_prob else utt.silence_frames + 1

        if utt.silence_frames * VAD_FRAME_MS >= cfg.endpoint_ms:
            return self._finalize()
        if utt.total_frames * VAD_FRAME >= cfg.max_utterance_s * SAMPLE_RATE:
            return self._finalize()
        if utt.route is not None:
            return self._lid_step(frame, prob)
        if utt.engine is None:
            return []  # Auto mode dropped this utterance
        if utt.samples_since_infer >= self._interval_samples():
            return self._partial_pass()
        return []

    def _open_utterance(self, preroll: list) -> _UtteranceState:
        """Pin the engine (and its language and window) for a new utterance."""
        choice = self._engine_for_utterance()
        if isinstance(choice, AutoRoute):
            return _UtteranceState(engine=None, lang="", max_window_s=self.config.max_window_s,
                                   frames=preroll, route=choice,
                                   onset_frame=len(preroll) - 1)
        utt = _UtteranceState(engine=None, lang="", max_window_s=0.0, frames=preroll)
        self._pin(utt, choice)
        return utt

    def _pin(self, utt: _UtteranceState, engine: ASREngine) -> None:
        cfg = self.config
        utt.engine = engine
        utt.lang = getattr(engine, "lang", "gu")
        utt.max_window_s = cfg.max_window_s_by_lang.get(utt.lang, cfg.max_window_s)
        utt.route = None

    def _lid_step(self, frame: np.ndarray, prob: float) -> list[TranscriptEvent]:
        """Auto mode, undecided: feed speech frames to LID; decode once it decides."""
        utt = self._utt
        decider = utt.route.decider
        if prob < self.config.vad_start_prob:
            return []  # pauses and near-silence would read as English
        decider.feed(frame)
        if decider.lang is not None:
            self._lid_resolve(utt, decider.lang, "decided")
            return self._partial_pass()
        if not decider.done:
            return []
        if decider.config.lid_fallback == "gu":
            logger.warning("LID undecided, defaulting to gu (u%d, %s)",
                           self._utt_id, _fmt_posteriors(decider.posteriors))
            self._lid_resolve(utt, "gu", "fallback")
            return self._partial_pass()
        self._lid_resolve(utt, None, "dropped (undecided at max_s)")
        return []

    def _lid_resolve(self, utt: _UtteranceState, lang: str | None, how: str) -> None:
        """Record + log the LID outcome; pin `lang`'s engine or drop (None)."""
        decider = utt.route.decider
        posteriors = decider.posteriors or {}
        since_onset_s = (len(utt.frames) - utt.onset_frame) * VAD_FRAME / SAMPLE_RATE
        speech_s = decider.decided_s or decider.speech_s  # window that decided
        p = posteriors.get(lang) if lang else max(posteriors.values(), default=0.0)
        self.lid_last = {
            "utterance_id": self._utt_id,
            "lang": lang or "unsure",
            "p": round(float(p or 0.0), 3),
            "speech_s": round(speech_s, 3),
            "since_onset_s": round(since_onset_s, 3),
            "fallback": how == "fallback",
        }
        logger.info("LID %s %s (u%d) after %.2f s speech, %.2f s since onset: %s",
                    lang or "unsure", how, self._utt_id, speech_s,
                    since_onset_s, _fmt_posteriors(decider.posteriors))
        if lang is None:
            utt.route = None  # engine stays None: nothing is decoded or emitted
            return
        self._pin(utt, utt.route.engine_for(lang))

    @property
    def avg_decode_ms(self) -> float:
        return self._total_decode_ms / self.decode_count if self.decode_count else 0.0

    def _interval_samples(self) -> int:
        # Low median of the last DECODE_HISTORY partial decodes in this
        # utterance: a single outlier (GC pause, thermal blip) is ignored as
        # soon as one normal decode follows, whereas an EMA would stay inflated
        # for several ticks. The history resets per utterance so a slow final
        # decode never throttles the next one, and the cap bounds the gap
        # between partials even when every decode is slow.
        cfg = self.config
        recent = self._utt.recent_decode_ms
        estimate = statistics.median_low(recent) if recent else 0.0
        ms = max(cfg.interval_ms, cfg.adaptive_factor * estimate)
        ms = min(ms, max(cfg.interval_ms, cfg.max_interval_ms))
        return int(ms * SAMPLE_RATE // 1000)

    def _timed_decode(self, engine: ASREngine, audio: np.ndarray) -> Transcript:
        t0 = self._clock()
        result = _decode(engine, audio)
        ms = (self._clock() - t0) * 1000
        self.decode_count += 1
        self.last_decode_ms = ms
        self.last_window_s = len(audio) / SAMPLE_RATE
        self._total_decode_ms += ms
        return apply_confidence(result, *self._thresholds_provider())

    def _audio(self) -> np.ndarray:
        return np.concatenate(self._utt.frames)

    def _maybe_trim(self) -> None:
        """Drop front audio once the window exceeds max_window_s (see module doc)."""
        utt = self._utt
        cfg = self.config
        frame_s = VAD_FRAME / SAMPLE_RATE
        window_s = len(utt.frames) * frame_s
        if window_s <= utt.max_window_s or not utt.decoded_since_trim:
            return  # nothing decoded since the last trim: keep the audio
        cutoff_s = window_s - cfg.min_context_s
        agreed = len(_common_prefix(utt.prev_hyp, utt.committed))
        # A safe seam needs an aligned committed cut word and at least one
        # retained aligned word to serve as post-cut provenance.
        eligible = min(agreed, len(utt.words) - 1)
        k = next(
            (
                i
                for i in range(eligible - 1, -1, -1)
                if utt.words[i].end_s + cfg.trim_margin_s <= cutoff_s
            ),
            None,
        )
        if k is None:
            return
        cut_s = utt.words[k].end_s
        # CTC spikes mark a word's start; its sound runs on after end_s, so
        # cut mid-gap to avoid re-hearing the word's tail.
        cut_s = (cut_s + utt.words[k + 1].start_s) / 2
        n_drop = round(cut_s / frame_s)
        keep = k + 1
        if n_drop <= 0:
            return
        moved = utt.committed[:keep]
        utt.prefix = utt.prefix + moved
        utt.committed = utt.committed[len(moved):]
        utt.prev_hyp = utt.prev_hyp[keep:]
        utt.frames = utt.frames[n_drop:]
        utt.dropped_frames += n_drop
        utt.words = ()
        utt.decoded_since_trim = False
        utt.seam_reference = list(utt.prev_hyp)
        utt.seam_overlap = None

    def _partial_pass(self) -> list[TranscriptEvent]:
        utt = self._utt
        utt.samples_since_infer = 0
        self._maybe_trim()
        result = self._timed_decode(utt.engine, self._audio())
        utt.recent_decode_ms.append(self.last_decode_ms)
        hyp = result.text.split()
        words = _aligned_words(result, hyp)
        seam = _select_seam_overlap(utt, hyp)
        hyp = hyp[seam:]
        utt.words = words[seam:]
        utt.decoded_since_trim = True
        # LocalAgreement-2: commit the common prefix of the last two
        # hypotheses, extending (never shrinking) what is already committed.
        agreed = _common_prefix(utt.prev_hyp, hyp)
        if len(agreed) > len(utt.committed):
            utt.committed = agreed
        utt.prev_hyp = hyp
        if not hyp:
            return []
        return [
            TranscriptEvent(
                type="partial",
                utterance_id=self._utt_id,
                committed=utt.committed_text(),
                tail=" ".join(hyp[len(utt.committed):]),
                lang=utt.lang,
            )
        ]

    def _finalize(self) -> list[TranscriptEvent]:
        utt = self._utt
        self._utt = None
        self.vad.reset()
        speech_frames = utt.total_frames - utt.silence_frames
        if speech_frames * VAD_FRAME_MS < self.config.min_speech_ms:
            return []  # noise blip
        if utt.route is not None:  # Auto mode, ended undecided: one last look
            lang = utt.route.decider.final_check()
            decided = lang in LID_LANGS
            self._lid_resolve(utt, lang if decided else None,
                              "decided at end" if decided else "dropped")
        if utt.engine is None:
            return []
        hyp = self._timed_decode(utt.engine, np.concatenate(utt.frames)).text.split()
        hyp = hyp[_select_seam_overlap(utt, hyp):]
        text = " ".join(utt.prefix + utt.committed + _final_suffix(utt, hyp))
        if not text:
            return []
        return [
            TranscriptEvent(
                type="final", utterance_id=self._utt_id, committed=text, tail="",
                lang=utt.lang,
            )
        ]
