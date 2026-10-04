"""Model-free fakes for streaming tests."""

from collections.abc import Callable, Sequence

import numpy as np

from gujusub.asr_engine import SAMPLE_RATE
from gujusub.engine import Transcript, Word
from gujusub.streaming import VAD_FRAME


class FakeClock:
    """Manual clock (seconds) to inject as the transcriber's timer."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeEngine:
    """Scripted ASR engine. `script` is a list of hypotheses returned on
    successive calls (the last one repeats) or a callable audio_len -> text.

    With `word_s`, the script is ignored and the text is derived from the
    audio itself: feed `ramp()` audio (sample value = absolute sample index),
    word i ("w<i>") covers [i*word_s, (i+1)*word_s) seconds of the stream and
    is recognised iff its midpoint lies inside the window. Word times are
    relative to the window start, like a real aligning engine. `word_text`
    can map an index to different text for repeated-word cases.

    With `clock` + `decode_ms` (ms, or callable audio_len -> ms), each decode
    advances the fake clock to simulate decode cost.
    """

    def __init__(
        self,
        script: Sequence[str] | Callable[[int], str] = ("",),
        *,
        word_s: float | None = None,
        clock: FakeClock | None = None,
        decode_ms: float | Callable[[int], float] = 0.0,
        echo_s: float = 0.0,
        word_text: Callable[[int], str] | None = None,
    ):
        self.script = script
        self.word_s = word_s
        self.clock = clock
        self.decode_ms = decode_ms
        self.echo_s = echo_s  # also "hear" words whose midpoint is this far before the window
        self.word_text = word_text or (lambda i: f"w{i}")
        self.audio_lengths: list[int] = []

    @property
    def calls(self) -> int:
        return len(self.audio_lengths)

    def transcribe(self, audio: np.ndarray) -> str:
        assert audio.dtype == np.float32 and audio.ndim == 1
        idx = len(self.audio_lengths)
        self.audio_lengths.append(len(audio))
        if self.clock is not None:
            ms = self.decode_ms(len(audio)) if callable(self.decode_ms) else self.decode_ms
            self.clock.advance(ms / 1000)
        if self.word_s is not None:
            return self._timed(audio).text
        if callable(self.script):
            return self.script(len(audio))
        return self.script[min(idx, len(self.script) - 1)]

    def transcribe_detailed(self, audio: np.ndarray) -> Transcript:
        text = self.transcribe(audio)
        if self.word_s is None:
            return Transcript(text, ())
        return self._timed(audio)

    def _timed(self, audio: np.ndarray) -> Transcript:
        start = float(audio[0]) / SAMPLE_RATE  # ramp audio encodes the clock
        end = start + len(audio) / SAMPLE_RATE
        first = max(0, int(start / self.word_s) - 1)
        words = tuple(
            Word(self.word_text(i), i * self.word_s - start, (i + 1) * self.word_s - start, 0.9)
            for i in range(first, int(end / self.word_s) + 1)
            if start - self.echo_s <= (i + 0.5) * self.word_s < end
        )
        return Transcript(" ".join(w.text for w in words), words)


class PlainEngine:
    """Engine with only transcribe() (no word alignment), wrapping FakeEngine."""

    def __init__(self, inner: FakeEngine):
        self.inner = inner

    def transcribe(self, audio: np.ndarray) -> str:
        return self.inner.transcribe(audio)


class FakeVAD:
    """Speech probability by frame index. `speech` is a list of
    (start, end) frame ranges (end exclusive) or a callable index -> bool.
    The frame counter is NOT reset by reset(); it tracks the audio clock."""

    def __init__(self, speech: Sequence[tuple[int, int]] | Callable[[int], bool]):
        self.speech = speech
        self.index = 0
        self.resets = 0

    def _is_speech(self, i: int) -> bool:
        if callable(self.speech):
            return self.speech(i)
        return any(a <= i < b for a, b in self.speech)

    def speech_prob(self, frame: np.ndarray) -> float:
        p = 1.0 if self._is_speech(self.index) else 0.0
        self.index += 1
        return p

    def reset(self) -> None:
        self.resets += 1


def frames(n: int) -> np.ndarray:
    """Audio of n VAD frames (non-silent constant so lengths are checkable)."""
    return np.full(n * VAD_FRAME, 0.1, dtype=np.float32)


def ramp(start_frame: int, n: int) -> np.ndarray:
    """n VAD frames whose sample values are their absolute sample index."""
    start = start_frame * VAD_FRAME
    return np.arange(start, start + n * VAD_FRAME, dtype=np.float32)
