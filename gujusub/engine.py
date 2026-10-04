"""Engine-neutral ASR types and the protocol every ASR engine implements."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Word:
    """One recognised word with its time span (seconds into the window) and confidence (0, 1]."""

    text: str
    start_s: float
    end_s: float
    conf: float


@dataclass(frozen=True)
class Transcript:
    """Decoded text plus per-word breakdown (``words`` empty if the engine cannot align)."""

    text: str
    words: tuple[Word, ...]

    @classmethod
    def empty(cls) -> "Transcript":
        return cls("", ())


@runtime_checkable
class ASREngine(Protocol):
    """Transcribes 16 kHz mono float32 audio windows."""

    lang: str

    def transcribe(self, audio: np.ndarray) -> str: ...

    def transcribe_detailed(self, audio: np.ndarray) -> Transcript: ...

    def warmup(self) -> None: ...
