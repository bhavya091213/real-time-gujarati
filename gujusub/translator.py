"""Gujarati -> English translation via IndicTrans2 dist-200M on CTranslate2.

Uses the CT2 conversion of AI4Bharat's rotary IndicTrans2 distilled model
(adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M). CTranslate2 replaces
the transformers modeling code (which is incompatible with transformers 5.x)
and is considerably faster on CPU.
"""

import logging
import threading
from pathlib import Path

import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download

# it2_compat must be imported before IndicTransToolkit (see its docstring).
# isort: off
from gujusub import it2_compat  # noqa: F401
from IndicTransToolkit.processor import IndicProcessor

# isort: on

logger = logging.getLogger(__name__)

CT2_REPO = "adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M"
CT2_SUBDIR = "indic-en-200m-ct2/ctranslate2_model"

GUJARATI_BLOCK = range(0x0A80, 0x0B00)


def looks_untranslated(text: str) -> bool:
    """True if the model copied the Gujarati source through instead of
    translating (a known failure on long or garbled ASR input)."""
    # Count every Gujarati code point (vowel signs are not isalpha()) against
    # letters from other scripts.
    gujarati = sum(ord(c) in GUJARATI_BLOCK for c in text)
    other = sum(c.isalpha() and ord(c) not in GUJARATI_BLOCK for c in text)
    if gujarati + other == 0:
        return False
    return gujarati / (gujarati + other) > 0.5


class Translator:
    """Loads the CT2 model once and translates single sentences.

    Serialized with a lock so it can be shared across connections.
    """

    def __init__(self, src_lang: str = "guj_Gujr", tgt_lang: str = "eng_Latn"):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._lock = threading.Lock()

        logger.info("loading %s ...", CT2_REPO)
        root = Path(snapshot_download(CT2_REPO)) / CT2_SUBDIR
        self.model = ctranslate2.Translator(
            str(root), device="cpu", compute_type="int8"
        )
        self.sp_src = spm.SentencePieceProcessor(model_file=str(root / "vocab" / "model.SRC"))
        self.sp_tgt = spm.SentencePieceProcessor(model_file=str(root / "vocab" / "model.TGT"))
        self.processor = IndicProcessor(inference=True)

    def translate(self, text: str) -> str:
        """Translate one sentence/fragment; returns '' for empty input."""
        text = text.strip()
        if not text:
            return ""
        with self._lock:
            tagged = self.processor.preprocess_batch(
                [text], src_lang=self.src_lang, tgt_lang=self.tgt_lang
            )[0]
            src_tag, tgt_tag, body = tagged.split(" ", 2)
            tokens = [src_tag, tgt_tag] + self.sp_src.encode(body, out_type=str)
            results = self.model.translate_batch(
                [tokens], beam_size=1, max_decoding_length=128
            )
            out_tokens = results[0].hypotheses[0]
            decoded = self.sp_tgt.decode(out_tokens)
            result = self.processor.postprocess_batch([decoded], lang=self.tgt_lang)[0]
        if looks_untranslated(result):
            logger.warning("translation passed source through, dropping: %r", result)
            return ""
        return result

    def warmup(self) -> None:
        self.translate("મારું નામ ભવ્ય છે")
