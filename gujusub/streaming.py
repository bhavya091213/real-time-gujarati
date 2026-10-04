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
"""

import collections
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from gujusub.asr_engine import SAMPLE_RATE, ASREngine
from gujusub.engine import Transcript, Word

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

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "utterance_id": self.utterance_id,
            "committed": self.committed,
            "tail": self.tail,
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


DECODE_HISTORY = 5  # partial decodes in the adaptive-interval estimate


@dataclass
class _UtteranceState:
    frames: list = field(default_factory=list)  # window: VAD_FRAME np arrays
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
        engine: ASREngine,
        config: StreamingConfig | None = None,
        vad: VAD | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.engine = engine
        self.config = config or StreamingConfig()
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
                self._utt = _UtteranceState(frames=list(self._preroll))
                self._preroll.clear()  # never reuse stale pre-onset audio
                self._utt_id += 1
            return []

        utt = self._utt
        utt.frames.append(frame)
        utt.samples_since_infer += VAD_FRAME
        utt.silence_frames = 0 if prob >= cfg.vad_end_prob else utt.silence_frames + 1

        if utt.silence_frames * VAD_FRAME_MS >= cfg.endpoint_ms:
            return self._finalize()
        if utt.total_frames * VAD_FRAME >= cfg.max_utterance_s * SAMPLE_RATE:
            return self._finalize()
        if utt.samples_since_infer >= self._interval_samples():
            return self._partial_pass()
        return []

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

    def _timed_decode(self, audio: np.ndarray) -> Transcript:
        t0 = self._clock()
        result = _decode(self.engine, audio)
        ms = (self._clock() - t0) * 1000
        self.decode_count += 1
        self.last_decode_ms = ms
        self.last_window_s = len(audio) / SAMPLE_RATE
        self._total_decode_ms += ms
        return result

    def _audio(self) -> np.ndarray:
        return np.concatenate(self._utt.frames)

    def _maybe_trim(self) -> None:
        """Drop front audio once the window exceeds max_window_s (see module doc)."""
        utt = self._utt
        cfg = self.config
        frame_s = VAD_FRAME / SAMPLE_RATE
        window_s = len(utt.frames) * frame_s
        if window_s <= cfg.max_window_s or not utt.decoded_since_trim:
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
        result = self._timed_decode(self._audio())
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
            )
        ]

    def _finalize(self) -> list[TranscriptEvent]:
        utt = self._utt
        self._utt = None
        self.vad.reset()
        speech_frames = utt.total_frames - utt.silence_frames
        if speech_frames * VAD_FRAME_MS < self.config.min_speech_ms:
            return []  # noise blip
        hyp = self._timed_decode(np.concatenate(utt.frames)).text.split()
        hyp = hyp[_select_seam_overlap(utt, hyp):]
        text = " ".join(utt.prefix + utt.committed + _final_suffix(utt, hyp))
        if not text:
            return []
        return [
            TranscriptEvent(
                type="final", utterance_id=self._utt_id, committed=text, tail=""
            )
        ]
