"""Transcribe an audio file once, offline — a quick check that the ASR works.

Run:  python tools/transcribe_file.py samples/test-guju.m4a [--lang gu] [--translate]
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gujusub.asr_engine import SAMPLE_RATE, ASREngine  # noqa: E402
from gujusub.fillers import clean_text  # noqa: E402


def load_mono_16k(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(wav)
    return wav


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--lang", default="gu")
    parser.add_argument("--translate", action="store_true", help="also print English")
    parser.add_argument("--raw", action="store_true", help="keep filler words")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    engine = ASREngine(lang=args.lang)
    text = engine.transcribe(load_mono_16k(args.path).numpy())
    if not args.raw:
        text = clean_text(text, final=True)
    print(text)

    if args.translate:
        from gujusub.translator import Translator

        print(Translator().translate(text))


if __name__ == "__main__":
    main()
