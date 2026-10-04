"""Wrapper around ai4bharat/indic-conformer-600m-multilingual for repeated inference."""

import importlib.util
import json
import logging
import threading
from pathlib import Path

import numpy as np
import torch

from gujusub.engine import Transcript, Word
from gujusub.model_dir import local_model_dir
from gujusub.threads import ort_session_options

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
REMOTE_CODE_FILE = "model_onnx.py"
CONFIG_FILE = "config.json"
WORD_MARKER = "\u2581"  # sentencepiece "▁": starts a new word
WARMUP_SEED = 0
# (seconds, amplitude) of noise windows; the first decodes after load are slow.
WARMUP_PASSES = ((1.0, 0.001), (4.0, 0.01)) * 2 + ((8.0, 0.01),)


class _OrtShim:
    """Stand-in for the ``onnxruntime`` module inside the remote model code.

    model_onnx.py builds every session as ``ort.InferenceSession(path,
    providers=[...])`` with no SessionOptions. Rebinding its module-global
    ``ort`` to this shim (before the model is instantiated) injects our
    thread settings; every other attribute forwards to the real module.
    """

    def __init__(self, ort_module, make_options):
        self._ort = ort_module
        self._make_options = make_options

    def InferenceSession(self, *args, **kwargs):  # noqa: N802 - mirrors ORT API
        if "sess_options" not in kwargs and len(args) < 2:
            kwargs["sess_options"] = self._make_options()
        return self._ort.InferenceSession(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._ort, name)


def ctc_words(
    logprobs: np.ndarray, vocab: list[str], blank_id: int, frame_s: float
) -> Transcript:
    """Greedy CTC decode of one (T, V) log-prob matrix into text plus words.

    ``text`` is built exactly like model_onnx.py's ``_ctc_decode`` (argmax,
    collapse repeats, drop blanks, join, "▁" -> space, strip). A token's
    confidence is its best frame probability; a word's is its weakest token.
    Times are encoder frame index x ``frame_s``.
    """
    n_frames = logprobs.shape[0]
    if n_frames == 0:
        return Transcript.empty()
    path = logprobs.argmax(axis=-1)
    best = np.exp(logprobs[np.arange(n_frames), path])
    bounds = np.flatnonzero(np.diff(path)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [n_frames]))
    tokens = [
        (vocab[int(path[s])], int(s), int(e), min(1.0, float(best[s:e].max())))
        for s, e in zip(starts, ends, strict=True)
        if int(path[s]) != blank_id
    ]
    text = "".join(t[0] for t in tokens).replace(WORD_MARKER, " ").strip()
    return Transcript(text, _group_words(tokens, frame_s))


def _group_words(tokens: list[tuple[str, int, int, float]], frame_s: float) -> tuple[Word, ...]:
    words: list[Word] = []
    current: tuple[str, int, int, float] | None = None
    for piece, start_f, end_f, conf in tokens:
        if WORD_MARKER in piece or current is None:
            if current is not None and current[0]:
                words.append(_to_word(current, frame_s))
            current = (piece.replace(WORD_MARKER, ""), start_f, end_f, conf)
        else:
            current = (current[0] + piece, current[1], end_f, min(current[3], conf))
    if current is not None and current[0]:
        words.append(_to_word(current, frame_s))
    return tuple(words)


def _to_word(group: tuple[str, int, int, float], frame_s: float) -> Word:
    text, start_f, end_f, conf = group
    return Word(text, start_f * frame_s, end_f * frame_s, conf)


def prepare_model_dir() -> Path:
    """Download the model and return a directory of real files (see model_dir.py)."""
    return local_model_dir(MODEL_ID)


def _load_remote_model(model_dir: Path) -> torch.nn.Module:
    """Instantiate the repo's IndicASRModel from a materialised directory.

    Equivalent to ``AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)``
    but pointed at ``model_dir`` instead of the symlinked hub cache, which
    onnxruntime >= 1.24 refuses to load external data from.
    """
    spec = importlib.util.spec_from_file_location(
        "gujusub_indic_conformer_remote", model_dir / REMOTE_CODE_FILE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REMOTE_CODE_FILE} from {model_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ort = _OrtShim(module.ort, ort_session_options)  # thread settings, see _OrtShim

    cfg = json.loads((model_dir / CONFIG_FILE).read_text())
    cfg.pop("auto_map", None)
    config = module.IndicASRConfig(ts_folder=str(model_dir), **cfg)
    return module.IndicASRModel(config)


class ASREngine:
    """Loads the IndicConformer model once and transcribes 16 kHz mono float32 audio.

    Inference is serialized with a lock: the underlying NeMo modules are not
    safe for concurrent forward passes.
    """

    def __init__(self, lang: str = "gu", decoding: str = "ctc", device: str | None = None):
        self.lang = lang
        self.decoding = decoding
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._lock = threading.Lock()

        logger.info("loading %s on %s ...", MODEL_ID, self.device)
        model_dir = prepare_model_dir()
        self.model = _load_remote_model(model_dir)
        try:
            self.model = self.model.to(self.device)
        except Exception:
            logger.warning("could not move model to %s, falling back to cpu", self.device)
            self.device = "cpu"
        self.model.eval()

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a 1-D float32 numpy array sampled at 16 kHz."""
        return self.transcribe_detailed(audio).text

    def transcribe_detailed(self, audio: np.ndarray) -> Transcript:
        """Like ``transcribe`` but with per-word timing and confidence (CTC only)."""
        if audio.ndim != 1:
            raise ValueError(f"expected 1-D audio, got shape {audio.shape}")
        if len(audio) < SAMPLE_RATE // 10:  # <100 ms: not enough signal
            return Transcript.empty()
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        if self.decoding != "ctc":
            with self._lock, torch.inference_mode():
                text = self.model(wav, self.lang, self.decoding)
            if isinstance(text, (list, tuple)):
                text = text[0] if text else ""
            return Transcript((text or "").strip(), ())
        with self._lock, torch.inference_mode():
            logprobs = self._ctc_logprobs(wav)
        cfg = self.model.config
        return ctc_words(logprobs, self.model.vocab[self.lang], cfg.BLANK_ID, cfg.FRAME_DURATION_MS)

    def _ctc_logprobs(self, wav: torch.Tensor) -> np.ndarray:
        """(T, V) language-masked log-probs; mirrors model_onnx.py ``_ctc_decode``."""
        encoder_outputs, _ = self.model.encode(wav)
        logits = self.model.models["ctc_decoder"].run(
            ["logprobs"], {"encoder_output": encoder_outputs}
        )[0]
        masked = logits[:, :, self.model.language_masks[self.lang]]
        return torch.from_numpy(masked).log_softmax(dim=-1)[0].numpy()

    def warmup(self, seconds: float | None = None) -> None:
        """Run throwaway decodes so the first real window isn't slow.

        Default: low-level noise at 1 s / 4 s (twice) then 8 s. ``seconds``
        runs a single pass of that length instead (cheap smoke check).
        """
        passes = ((seconds, 0.01),) if seconds is not None else WARMUP_PASSES
        rng = np.random.default_rng(WARMUP_SEED)
        for secs, amplitude in passes:
            noise = rng.standard_normal(int(SAMPLE_RATE * secs)).astype(np.float32)
            self.transcribe_detailed(noise * amplitude)
