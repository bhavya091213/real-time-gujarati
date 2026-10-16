"""Fast Gujarati ASR: AI4Bharat's Gujarati-only IndicConformer (~120M, CTC) as int8 ONNX.

Model: ``ai4bharat/indicconformer_stt_gu_hybrid_ctc_rnnt_large`` (MIT), ONNX export
``OpenVoiceOS/ai4bharat-indicconformer-gu-onnx`` (fp32 graph + vocab, ~460 MB
fetched). The export's own ``model.int8.onnx`` uses signed ``ConvInteger``, which
onnxruntime's CPU provider does not implement, so the first load quantises the
fp32 graph's MatMuls to dynamic int8 (~177 MB) beside it and loads that.

Features come from onnx-asr's NumPy NeMo preprocessor (80 mel, 10 ms hop); the
encoder subsamples x4, so one output frame is 40 ms. Decoding reuses
``asr_engine.ctc_words``, so word confidence means the same as with the full
model (weakest token's best frame probability) and the 0.5/0.7 gates apply.

Measured 2026-10-09 on an M1 Pro against the full 600M fp32 model: 5 s window
113 ms vs 207 ms, 8 s window 142 ms vs 267 ms, about 0.55 GB vs 2.9 GB RSS,
load 0.4 s vs 6-11 s. Accuracy was checked on one clip only (see wiki).
"""

import json
import logging
import os
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from gujusub.asr_engine import ctc_words
from gujusub.engine import Transcript
from gujusub.model_dir import default_root, materialize_snapshot
from gujusub.threads import ort_session_options

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
REPO_ID = "OpenVoiceOS/ai4bharat-indicconformer-gu-onnx"
FP32_FILE = "model.onnx"
FP32_DATA_FILE = "model.onnx_data"
VOCAB_FILE = "vocab.txt"
CONFIG_FILE = "config.json"
MODEL_FILES = (FP32_FILE, FP32_DATA_FILE, VOCAB_FILE, CONFIG_FILE)
INT8_FILE = "model.gujusub-int8.onnx"  # built locally; not part of the snapshot
BLANK_TOKEN = "<blk>"
HOP_S = 0.01  # preprocessor hop: one feature frame per 10 ms
DEFAULT_SUBSAMPLING = 4
DEFAULT_FEATURES = 80
PROVIDERS = ["CPUExecutionProvider"]
WARMUP_SEED = 0
# (seconds, amplitude) of noise windows; the first decodes after load are slow.
WARMUP_PASSES = ((1.0, 0.001), (4.0, 0.01)) * 2 + ((8.0, 0.01),)

Quantizer = Callable[[Path, Path], None]


def prepare_model_dir() -> Path:
    """Fetch the fp32 graph, weights and vocab; return a symlink-free directory."""
    snapshot = Path(snapshot_download(REPO_ID, allow_patterns=list(MODEL_FILES)))
    return materialize_snapshot(snapshot, default_root() / REPO_ID.replace("/", "--"))


def _quantize_matmuls(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic  # deferred: slow import

    quantize_dynamic(
        str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"]
    )


def ensure_int8(model_dir: Path, quantize: Quantizer = _quantize_matmuls) -> Path:
    """Return the int8 graph in ``model_dir``, building it from the fp32 one on first use.

    The result is written under a temporary name and renamed into place, so an
    existing file is never written through (the materialised files are
    hardlinks of the hub cache; writing into one corrupts the cache).
    """
    target = model_dir / INT8_FILE
    if target.is_file():
        return target
    source = model_dir / FP32_FILE
    if not source.is_file():
        raise FileNotFoundError(f"{source} is missing; cannot build {INT8_FILE}")
    tmp = target.with_name(target.name + ".tmp")
    logger.info("quantising %s -> %s (one-time, MatMul int8) ...", source.name, target.name)
    quantize(source, tmp)
    os.replace(tmp, target)
    return target


def load_vocab(lines: Iterable[str]) -> tuple[list[str], int]:
    """Parse ``vocab.txt`` ("piece id" per line) into (pieces by id, blank id)."""
    by_id: dict[int, str] = {}
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            continue
        piece, sep, idx = line.rpartition(" ")
        if not sep or not idx.isdigit():
            raise ValueError(f"malformed vocab line: {line!r}")
        by_id[int(idx)] = piece
    if sorted(by_id) != list(range(len(by_id))):
        raise ValueError("vocab ids are not contiguous from 0")
    vocab = [by_id[i] for i in range(len(by_id))]
    if BLANK_TOKEN not in vocab:
        raise ValueError(f"vocab has no {BLANK_TOKEN!r} token")
    return vocab, vocab.index(BLANK_TOKEN)


def read_config(model_dir: Path) -> tuple[int, int]:
    """(subsampling factor, feature size) from the export's config.json, with defaults."""
    path = model_dir / CONFIG_FILE
    cfg = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return int(cfg.get("subsampling_factor", DEFAULT_SUBSAMPLING)), int(
        cfg.get("features_size", DEFAULT_FEATURES)
    )


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax; a no-op (up to rounding) on already normalised log-probs."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _load_session(model_path: Path):
    import onnxruntime as ort  # deferred: heavy

    return ort.InferenceSession(
        str(model_path), sess_options=ort_session_options(), providers=PROVIDERS
    )


def _load_preprocessor(features_size: int):
    from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy  # deferred

    return NemoPreprocessorNumpy(f"nemo{features_size}")


class FastGujaratiEngine:
    """Loads the Gujarati-only int8 model once and transcribes 16 kHz mono float32 audio.

    Inference is serialised with a lock (one ORT session, one window at a time).
    """

    lang = "gu"

    def __init__(self, model_dir: Path | None = None) -> None:
        self._lock = threading.Lock()
        model_dir = prepare_model_dir() if model_dir is None else Path(model_dir)
        subsampling, features = read_config(model_dir)
        self.frame_s = HOP_S * subsampling
        self.subsampling = subsampling
        self.vocab, self.blank_id = load_vocab(
            (model_dir / VOCAB_FILE).read_text(encoding="utf-8").splitlines()
        )
        model_path = ensure_int8(model_dir)
        logger.info("loading %s (%s) from %s ...", REPO_ID, INT8_FILE, model_dir)
        self._preprocessor = _load_preprocessor(features)
        self._session = _load_session(model_path)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a 1-D float32 numpy array sampled at 16 kHz."""
        return self.transcribe_detailed(audio).text

    def transcribe_detailed(self, audio: np.ndarray) -> Transcript:
        """Text plus per-word timing and confidence (greedy CTC, see ``ctc_words``)."""
        if audio.ndim != 1:
            raise ValueError(f"expected 1-D audio, got shape {audio.shape}")
        if len(audio) < SAMPLE_RATE // 10:  # <100 ms: not enough signal
            return Transcript.empty()
        wav = np.ascontiguousarray(audio, dtype=np.float32)[None, :]
        lengths = np.array([wav.shape[1]], dtype=np.int64)
        with self._lock:
            logprobs = self._logprobs(wav, lengths)
        return ctc_words(logprobs, self.vocab, self.blank_id, self.frame_s)

    def _logprobs(self, wav: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """(T, V) normalised log-probs for one window, trimmed to the valid frames."""
        feats, feat_lens = self._preprocessor(wav, lengths)
        (out,) = self._session.run(["logprobs"], {"audio_signal": feats, "length": feat_lens})
        n_frames = (int(feat_lens[0]) - 1) // self.subsampling + 1
        return log_softmax(np.asarray(out[0][:n_frames], dtype=np.float32))

    def warmup(self, seconds: float | None = None) -> None:
        """Throwaway decodes so the first real window isn't slow (see asr_engine)."""
        passes = ((seconds, 0.01),) if seconds is not None else WARMUP_PASSES
        rng = np.random.default_rng(WARMUP_SEED)
        for secs, amplitude in passes:
            noise = rng.standard_normal(int(SAMPLE_RATE * secs)).astype(np.float32)
            self.transcribe_detailed(noise * amplitude)
