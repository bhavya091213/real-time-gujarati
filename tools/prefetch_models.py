"""Download all models (Gujarati ASR, English ASR, language ID, translator) and prepare the
ASR model directories, so the first server start is fast and works offline.

Run:  python tools/prefetch_models.py            # download + materialise
      python tools/prefetch_models.py --verify   # also load every model and run a
                                                 # tiny inference (catches onnxruntime
                                                 # external-data and version problems)
"""

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gujusub import engine_parakeet, lid  # noqa: E402
from gujusub.asr_engine import MODEL_ID, prepare_model_dir  # noqa: E402
from gujusub.translator import CT2_REPO  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "test-guju.m4a"
SAMPLE_SECONDS = 3
LID_SECONDS = 2


def download() -> None:
    print(f"  {MODEL_ID}")
    print(f"    -> {prepare_model_dir()}")
    print(f"  {engine_parakeet.REPO_ID} ({engine_parakeet.QUANTIZATION})")
    print(f"    -> {engine_parakeet.prepare_model_dir()}")
    print(f"  {lid.REPO_ID}")
    print(f"    -> {snapshot_download(lid.REPO_ID)}")
    print(f"  {CT2_REPO}")
    print(f"    -> {snapshot_download(CT2_REPO)}")


def verify() -> None:
    import huggingface_hub
    import onnxruntime
    import transformers

    from gujusub.asr_engine import ASREngine
    from gujusub.translator import Translator

    print(
        f"  onnxruntime {onnxruntime.__version__}, "
        f"huggingface_hub {huggingface_hub.__version__}, "
        f"transformers {transformers.__version__}"
    )
    engine = ASREngine(device="cpu")
    engine.warmup(seconds=1.0)
    print("  ASR: loaded and ran on 1 s of silence")

    verify_english()
    verify_lid()

    translator = Translator()
    sample = "કેમ છો"
    print(f"  translator: {sample!r} -> {translator.translate(sample)!r}")


def verify_english() -> None:
    import onnx_asr
    from transcribe_file import SAMPLE_RATE, load_mono_16k  # sibling tool

    print(f"  onnx-asr {onnx_asr.__version__}")
    engine = engine_parakeet.ParakeetEngine()
    engine.warmup(seconds=1.0)
    print("  English ASR: loaded and ran on 1 s of silence")
    clip = load_mono_16k(SAMPLE).numpy()[: SAMPLE_RATE * SAMPLE_SECONDS]
    # Gujarati audio through an English model: junk text is expected, we only
    # check that decoding runs end to end.
    print(f"  English ASR on {SAMPLE.name} [0-{SAMPLE_SECONDS}s]: {engine.transcribe(clip)!r}")


def verify_lid() -> None:
    import speechbrain
    from transcribe_file import SAMPLE_RATE, load_mono_16k  # sibling tool

    print(f"  speechbrain {speechbrain.__version__}")
    model = lid.LanguageID()
    model.warmup()
    print(f"  LID: loaded, label indices {model.indices}")
    clip = load_mono_16k(SAMPLE).numpy()[: SAMPLE_RATE * LID_SECONDS]
    probs = model.classify(clip)
    shown = ", ".join(f"{code}={p:.3f}" for code, p in probs.items())
    verdict = "ok" if probs["gu"] > probs["en"] else "UNEXPECTED (want gu > en)"
    print(f"  LID on {SAMPLE.name} [0-{LID_SECONDS}s]: {shown} -> {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="load models and run a tiny inference"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.verify:
        verify()
    else:
        download()


if __name__ == "__main__":
    main()
