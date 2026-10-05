"""English ASR: NVIDIA Parakeet-TDT 0.6B v2 (int8 ONNX) via onnx-asr.

Model: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 (CC-BY-4.0), ONNX
export ``istupakov/parakeet-tdt-0.6b-v2-onnx``. Only the int8 graphs plus the
vocab/preprocessor/config are fetched (~660 MB); the int8 files carry their
weights inline, so no external-data sidecars are involved. The snapshot is
still materialised into real files (see model_dir.py) and onnx-asr is pointed
at that directory, so loads are offline after the first run.
"""

import logging
import math
import threading
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from gujusub.engine import Transcript, Word
from gujusub.model_dir import default_root, materialize_snapshot
from gujusub.threads import ort_session_options

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MODEL_NAME = "nemo-parakeet-tdt-0.6b-v2"
REPO_ID = "istupakov/parakeet-tdt-0.6b-v2-onnx"
QUANTIZATION = "int8"
MODEL_FILES = (
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "nemo128.onnx",
    "vocab.txt",
    "config.json",
)
PROVIDERS = ["CPUExecutionProvider"]
# A token starting with either marker begins a new word. onnx-asr rewrites the
# vocab's sentencepiece "\u2581" to a plain space; accept both.
SP_MARKER = "\u2581"
WORD_MARKERS = (SP_MARKER, " ")
TOKEN_FRAME_S = 0.08  # encoder frame (10 ms hop x 8 subsampling); a token's nominal span
SILENCE_CONF = 0.05  # every word below this -> treat the window as silence
WARMUP_SEED = 0
# (seconds, amplitude) of noise windows; the first decodes after load are slow.
WARMUP_PASSES = ((1.0, 0.001), (4.0, 0.01), (8.0, 0.01))

_Token = tuple[str, float, float]  # (piece, start_s, logprob)


def prepare_model_dir() -> Path:
    """Fetch only the int8 model files and return a symlink-free directory."""
    snapshot = Path(snapshot_download(REPO_ID, allow_patterns=list(MODEL_FILES)))
    return materialize_snapshot(snapshot, default_root() / REPO_ID.replace("/", "--"))


def _word(pieces: list[_Token]) -> Word:
    text = "".join(p for p, _, _ in pieces).replace(SP_MARKER, "").strip()
    conf = math.exp(min(lp for _, _, lp in pieces))
    return Word(text, pieces[0][1], pieces[-1][1] + TOKEN_FRAME_S, min(1.0, conf))


def tdt_words(
    tokens: list[str], timestamps: list[float], logprobs: list[float]
) -> tuple[Word, ...]:
    """Group sentencepiece tokens into words; conf = exp(weakest token log-prob)."""
    groups: list[list[_Token]] = []
    for tok in zip(tokens, timestamps, logprobs, strict=True):
        if tok[0].startswith(WORD_MARKERS) or not groups:
            groups.append([tok])
        else:
            groups[-1].append(tok)
    words = (_word(g) for g in groups)
    return tuple(w for w in words if w.text)


class ParakeetEngine:
    """Loads Parakeet-TDT once and transcribes 16 kHz mono float32 audio (English).

    Inference is serialized with a lock (ORT sessions plus the decoder state
    loop are driven from one Python call; keep it one window at a time).
    """

    lang = "en"

    def __init__(self) -> None:
        import onnx_asr  # deferred: heavy, and tests inject a fake

        self._lock = threading.Lock()
        model_dir = prepare_model_dir()
        logger.info("loading %s (%s) from %s ...", MODEL_NAME, QUANTIZATION, model_dir)
        model = onnx_asr.load_model(
            MODEL_NAME,
            path=model_dir,
            quantization=QUANTIZATION,
            sess_options=ort_session_options(),
            providers=PROVIDERS,
        )
        self._model = model.with_timestamps()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a 1-D float32 numpy array sampled at 16 kHz."""
        return self.transcribe_detailed(audio).text

    def transcribe_detailed(self, audio: np.ndarray) -> Transcript:
        """Text plus per-word timing/confidence; silence/low-confidence -> empty."""
        if audio.ndim != 1:
            raise ValueError(f"expected 1-D audio, got shape {audio.shape}")
        if len(audio) < SAMPLE_RATE // 10:  # <100 ms: not enough signal
            return Transcript.empty()
        with self._lock:
            result = self._model.recognize(np.ascontiguousarray(audio, dtype=np.float32))
        tokens = list(result.tokens or [])
        timestamps = list(result.timestamps or [0.0] * len(tokens))
        logprobs = list(result.logprobs) if result.logprobs is not None else [0.0] * len(tokens)
        words = tdt_words(tokens, timestamps, logprobs)
        if not words or all(w.conf < SILENCE_CONF for w in words):
            return Transcript.empty()
        return Transcript((result.text or "").strip(), words)

    def warmup(self, seconds: float | None = None) -> None:
        """Throwaway decodes (1 s, 4 s, 8 s of noise); ``seconds`` = one pass of that length."""
        passes = ((seconds, 0.01),) if seconds is not None else WARMUP_PASSES
        rng = np.random.default_rng(WARMUP_SEED)
        for secs, amplitude in passes:
            noise = rng.standard_normal(int(SAMPLE_RATE * secs)).astype(np.float32)
            self.transcribe_detailed(noise * amplitude)
