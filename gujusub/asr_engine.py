"""Wrapper around ai4bharat/indic-conformer-600m-multilingual for repeated inference."""

import importlib.util
import json
import logging
import threading
from pathlib import Path

import numpy as np
import torch

from gujusub.model_dir import local_model_dir

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
REMOTE_CODE_FILE = "model_onnx.py"
CONFIG_FILE = "config.json"


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
        if audio.ndim != 1:
            raise ValueError(f"expected 1-D audio, got shape {audio.shape}")
        if len(audio) < SAMPLE_RATE // 10:  # <100 ms: not enough signal
            return ""
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        with self._lock, torch.inference_mode():
            text = self.model(wav, self.lang, self.decoding)
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        return (text or "").strip()

    def warmup(self, seconds: float = 2.0) -> None:
        self.transcribe(np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32))
