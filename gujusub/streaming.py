"""Real-time streaming transcription on top of an offline ASR model.

Algorithm: Silero-VAD-gated growing window with LocalAgreement-2 commits
(whisper_streaming style, Machacek et al. 2023).

- VAD gates the stream: nothing is transcribed during silence.
- While speech is active, the whole current utterance buffer is re-decoded
  every `interval_ms` of new audio (audio clock, so it is deterministic in
  tests). The longest common prefix of the last two hypotheses is "committed"
  and never retracted; the rest is an unstable tail.
- A pause of `endpoint_ms` ends the utterance and emits a final decode.
"""

import collections
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from silero_vad import load_silero_vad

from gujusub.asr_engine import SAMPLE_RATE, ASREngine

logger = logging.getLogger(__name__)

VAD_FRAME = 512  # samples per silero window @ 16 kHz (32 ms)
VAD_FRAME_MS = VAD_FRAME * 1000 // SAMPLE_RATE


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
    max_utterance_s: float = 12.0  # force-finalize to bound decode cost


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b, strict=False):  # prefix of the shorter list
        if x != y:
            break
        out.append(x)
    return out


@dataclass
class _UtteranceState:
    frames: list = field(default_factory=list)  # np arrays of VAD_FRAME samples
    silence_frames: int = 0
    samples_since_infer: int = 0
    committed: list = field(default_factory=list)  # committed words
    prev_hyp: list = field(default_factory=list)  # previous hypothesis words


class StreamingTranscriber:
    """Consumes 16 kHz mono float32 audio, emits TranscriptEvents.

    Not thread-safe: feed() from a single thread (the server uses one
    transcriber per connection inside a single executor task).
    """

    def __init__(self, engine: ASREngine, config: StreamingConfig | None = None):
        self.engine = engine
        self.config = config or StreamingConfig()
        self.vad = load_silero_vad()
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
        prob = self.vad(torch.from_numpy(frame), SAMPLE_RATE).item()
        cfg = self.config

        if self._utt is None:
            self._preroll.append(frame)
            if prob >= cfg.vad_start_prob:
                self._utt = _UtteranceState(frames=list(self._preroll))
                self._utt_id += 1
            return []

        utt = self._utt
        utt.frames.append(frame)
        utt.samples_since_infer += VAD_FRAME
        utt.silence_frames = 0 if prob >= cfg.vad_end_prob else utt.silence_frames + 1

        if utt.silence_frames * VAD_FRAME_MS >= cfg.endpoint_ms:
            return self._finalize()
        if len(utt.frames) * VAD_FRAME >= cfg.max_utterance_s * SAMPLE_RATE:
            return self._finalize()
        if utt.samples_since_infer >= cfg.interval_ms * SAMPLE_RATE // 1000:
            return self._partial_pass()
        return []

    def _audio(self) -> np.ndarray:
        return np.concatenate(self._utt.frames)

    def _partial_pass(self) -> list[TranscriptEvent]:
        utt = self._utt
        utt.samples_since_infer = 0
        hyp = self.engine.transcribe(self._audio()).split()
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
                committed=" ".join(utt.committed),
                tail=" ".join(hyp[len(utt.committed):]),
            )
        ]

    def _finalize(self) -> list[TranscriptEvent]:
        utt = self._utt
        self._utt = None
        self.vad.reset_states()
        speech_frames = len(utt.frames) - utt.silence_frames
        if speech_frames * VAD_FRAME_MS < self.config.min_speech_ms:
            return []  # noise blip
        text = self.engine.transcribe(np.concatenate(utt.frames))
        if not text:
            return []
        return [
            TranscriptEvent(
                type="final", utterance_id=self._utt_id, committed=text, tail=""
            )
        ]
