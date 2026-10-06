"""Spoken language ID (Gujarati vs English) and the per-utterance decision policy.

Model: SpeechBrain ECAPA-TDNN trained on VoxLingua107
(``speechbrain/lang-id-voxlingua107-ecapa``, Apache-2.0, ~86 MB). It scores
107 languages; we keep only ``gu`` and ``en`` and renormalise between them,
since the operator has told us the speech is one of the two.

``LidDecider`` is the policy (grill decision D7): accumulate speech from the
start of an utterance, classify at 1.0 s and every +0.5 s up to 3.0 s, and
commit to a language on two consecutive confident windows (p >= 0.80) or a
single strong one (p >= 0.95) from 1.5 s on. Until then, and for good if 3.0 s
passes without a decision, the answer is ``"unsure"`` (the caller shows nothing
or, in Auto mode, applies ``LidConfig.lid_fallback``).

Near-silence scores as confident English with this model, so callers must feed
VAD-gated speech from speech onset, ``classify`` returns 0.5/0.5 for windows
quieter than ``min_rms_dbfs``, and no single window decides at the first check.
"""

import logging
import math
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gujusub.model_dir import default_root
from gujusub.threads import perf_cores

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
REPO_ID = "speechbrain/lang-id-voxlingua107-ecapa"
LANGS = ("gu", "en")
UNSURE = "unsure"
MIN_SAMPLES = SAMPLE_RATE // 10  # <100 ms: not enough signal to say anything
WARMUP_SECONDS = 1.0
WARMUP_SEED = 0
WARMUP_AMPLITUDE = 0.01
MIN_RMS_DBFS = -45.0  # quieter windows are near-silence: no opinion
FALLBACKS = ("gu", "none")

Posteriors = dict[str, float]
Classify = Callable[[np.ndarray], Posteriors]


def savedir() -> Path:
    """Where SpeechBrain keeps its hyperparams/checkpoint links for the LID model."""
    return default_root() / "lid"


def _label_indices(lab2ind: dict[str, int]) -> dict[str, int]:
    """Map our codes to classifier indices; VoxLingua labels look like ``"gu: Gujarati"``."""
    indices: dict[str, int] = {}
    for code in LANGS:
        matches = [i for label, i in lab2ind.items() if label.split(":")[0].strip() == code]
        if len(matches) != 1:
            raise RuntimeError(f"LID label for {code!r} not found in classifier labels")
        indices[code] = matches[0]
    return indices


def _rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20 * math.log10(rms) if rms > 0 else -math.inf


def _uninformative() -> Posteriors:
    return {code: 1.0 / len(LANGS) for code in LANGS}


class LanguageID:
    """Loads the VoxLingua107 classifier once; ``classify`` gives P(gu), P(en).

    Calls are serialised with a lock. Torch's intra-op thread count is set to
    ``threads`` (default: half the performance cores) for the duration of each
    call and restored afterwards, so LID does not starve ASR/translation.
    """

    def __init__(
        self, classifier=None, threads: int | None = None, min_rms_dbfs: float = MIN_RMS_DBFS
    ) -> None:
        self._lock = threading.Lock()
        self.min_rms_dbfs = min_rms_dbfs
        self._threads = threads if threads is not None else max(1, perf_cores() // 2)
        self._clf = classifier if classifier is not None else self._load()
        self.indices = _label_indices(self._clf.hparams.label_encoder.lab2ind)

    @staticmethod
    def _load():
        from speechbrain.inference.classifiers import EncoderClassifier  # deferred: heavy

        target = savedir()
        logger.info("loading %s into %s ...", REPO_ID, target)
        return EncoderClassifier.from_hparams(
            source=REPO_ID, savedir=str(target), run_opts={"device": "cpu"}
        )

    @contextmanager
    def _torch_threads(self):
        import torch

        previous = torch.get_num_threads()
        torch.set_num_threads(self._threads)
        try:
            yield torch
        finally:
            torch.set_num_threads(previous)

    def classify(self, audio: np.ndarray) -> Posteriors:
        """Posteriors over ``{"gu", "en"}`` (sum to 1) for 1-D float32 16 kHz audio."""
        if audio.ndim != 1:
            raise ValueError(f"expected 1-D audio, got shape {audio.shape}")
        if len(audio) < MIN_SAMPLES:
            return _uninformative()
        wav = np.ascontiguousarray(audio, dtype=np.float32)
        if _rms_dbfs(wav) < self.min_rms_dbfs:
            return _uninformative()  # near-silence reads as English; say nothing
        with self._lock, self._torch_threads() as torch, torch.inference_mode():
            log_probs = self._clf.classify_batch(torch.from_numpy(wav).unsqueeze(0))[0][0]
            picked = torch.stack([log_probs[self.indices[code]] for code in LANGS])
            probs = torch.softmax(picked.float(), dim=0).tolist()
        return dict(zip(LANGS, probs, strict=True))

    def warmup(self) -> None:
        """One throwaway classification on low-level noise (first call is slow)."""
        rng = np.random.default_rng(WARMUP_SEED)
        noise = rng.standard_normal(int(SAMPLE_RATE * WARMUP_SECONDS)).astype(np.float32)
        self.classify(noise * WARMUP_AMPLITUDE)


@dataclass(frozen=True)
class LidConfig:
    """Decision thresholds; times are seconds of speech since utterance start."""

    first_s: float = 1.0
    step_s: float = 0.5
    max_s: float = 3.0
    p_min: float = 0.80
    p_strong: float = 0.95
    strong_from_s: float = 1.5  # a single p_strong window decides only from here on
    lid_fallback: str = "gu"  # Auto, undecided (max_s or endpoint): "gu" = pin gu, "none" = drop

    def __post_init__(self) -> None:
        if self.step_s <= 0 or self.first_s <= 0:
            raise ValueError("first_s and step_s must be positive")
        if self.first_s > self.max_s:
            raise ValueError("first_s must not exceed max_s")
        if not 0.5 <= self.p_min <= self.p_strong <= 1.0:
            raise ValueError("need 0.5 <= p_min <= p_strong <= 1.0")
        if self.lid_fallback not in FALLBACKS:
            raise ValueError(f"lid_fallback must be one of {FALLBACKS}")


def _samples(seconds: float) -> int:
    return int(round(seconds * SAMPLE_RATE))


class LidDecider:
    """Per-utterance policy over a ``classify(audio) -> {"gu": p, "en": p}`` callable.

    Feed speech chunks with ``feed``; it returns ``"gu"``/``"en"`` once decided
    and ``"unsure"`` otherwise. ``done`` turns true on a decision or when
    ``max_s`` passes without one (``lang`` stays ``None``). ``final_check``
    classifies whatever is buffered once more (utterance ended undecided).
    ``posteriors`` is the latest classification and ``decided_s`` the speech
    seconds at the decision. Call ``reset`` at the start of each utterance.
    Not thread-safe; use one per audio stream.
    """

    def __init__(self, classify: Classify, config: LidConfig | None = None) -> None:
        self._classify = classify
        self.config = config or LidConfig()
        self._max_samples = _samples(self.config.max_s)
        self.reset()

    def reset(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._buffered = 0
        self._next_check = _samples(self.config.first_s)
        self._checked = 0  # samples covered by the latest classification
        self._streak: tuple[str, int] = ("", 0)
        self.lang: str | None = None
        self.done = False
        self.posteriors: Posteriors | None = None
        self.decided_s: float | None = None

    @property
    def speech_s(self) -> float:
        return self._buffered / SAMPLE_RATE

    def final_check(self) -> str:
        """One last classification of all buffered speech; then ``done``."""
        if not self.done and self._buffered > self._checked:
            self._judge(
                self._classify(self._audio()),
                self._buffered,
                strong_ok=self._buffered >= _samples(self.config.strong_from_s),
            )
        self.done = True
        return self.lang or UNSURE

    def _audio(self) -> np.ndarray:
        audio = np.concatenate(self._chunks) if self._chunks else np.empty(0, np.float32)
        self._chunks = [audio]
        return audio

    def feed(self, chunk: np.ndarray) -> str:
        if not self.done:
            room = self._max_samples - self._buffered
            if room > 0 and len(chunk):
                piece = np.asarray(chunk, dtype=np.float32)[:room]
                self._chunks = [*self._chunks, piece]
                self._buffered += len(piece)
            self._run_checks()
        return self.lang or UNSURE

    def _run_checks(self) -> None:
        if self._buffered < self._next_check:
            return
        audio = self._audio()
        strong_from = _samples(self.config.strong_from_s)
        while not self.done and self._next_check <= self._buffered:
            n = self._next_check
            self._judge(self._classify(audio[:n]), n, strong_ok=n >= strong_from)
            self._next_check += _samples(self.config.step_s)
            if not self.done and self._next_check > self._max_samples:
                self.done = True  # timed out: final "unsure"
                logger.debug("LID: no decision within %.1f s", self.config.max_s)

    def _judge(self, posteriors: Posteriors, n: int, *, strong_ok: bool) -> None:
        self.posteriors = posteriors
        self._checked = n
        best = max(LANGS, key=lambda code: posteriors.get(code, 0.0))
        p = posteriors.get(best, 0.0)
        if math.isnan(p) or p < self.config.p_min:
            self._streak = ("", 0)
            return
        count = self._streak[1] + 1 if self._streak[0] == best else 1
        self._streak = (best, count)
        if (strong_ok and p >= self.config.p_strong) or count >= 2:
            self.lang = best
            self.done = True
            self.decided_s = n / SAMPLE_RATE
