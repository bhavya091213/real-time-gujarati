"""Wrapper around ai4bharat/indic-conformer-600m-multilingual for repeated inference."""

import logging
import threading

import numpy as np
import torch
from transformers import AutoModel

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"


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
        self.model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
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
