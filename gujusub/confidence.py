"""Confidence trimming: drop low-confidence words, gate whole low-confidence decodes.

Applied to every decode (partial and final) before LocalAgreement, so a
low-confidence flicker word never reaches the committed text, and a decode
that is mostly garbage (noise, music, wrong language) becomes empty.

Confidence semantics differ by engine (see calibration-3.4.md):
- IndicConformer (gu): word conf = min over its tokens of the CTC frame max-prob.
- Parakeet (en): word conf = exp(min token logprob) of the TDT decoder; it is
  ~1.0 on clean speech and often high on junk too, so the gate is weak for en.

Thresholds of 0 disable each stage. Words without a confidence (None) are kept
and ignored by the mean. A transcript without word alignment passes unchanged.
"""

from collections.abc import Sequence

from gujusub.engine import Transcript, Word

MIN_WORDS = 1  # the utterance gate needs this many surviving words


def _keep(word: Word, word_min: float) -> bool:
    return word.conf is None or word.conf >= word_min


def trim_words(words: Sequence[Word], word_min: float) -> tuple[Word, ...]:
    """Words with conf >= word_min (unknown conf is kept), in order."""
    return tuple(w for w in words if _keep(w, word_min))


def utterance_ok(words: Sequence[Word], utt_min: float, min_words: int = MIN_WORDS) -> bool:
    """Gate on the kept words: at least `min_words` (and at least one) survive and
    their mean known confidence is >= utt_min. utt_min <= 0 disables the gate.

    The default of 1 keeps a genuine one-word utterance ("Amen") when its
    confidence clears utt_min; raise `min_words` to also reject lone-word junk.
    """
    if utt_min <= 0:
        return True
    if len(words) < max(1, min_words):
        return False
    known = [w.conf for w in words if w.conf is not None]
    return not known or sum(known) / len(known) >= utt_min


def apply_confidence(transcript: Transcript, word_min: float, utt_min: float) -> Transcript:
    """Trimmed transcript (text rebuilt from kept words) or empty if the gate fails.

    Returns `transcript` itself when nothing changes, so its text is preserved.
    """
    if not transcript.words or (word_min <= 0 and utt_min <= 0):
        return transcript
    kept = trim_words(transcript.words, word_min)
    if not utterance_ok(kept, utt_min):
        return Transcript.empty()
    if len(kept) == len(transcript.words):
        return transcript
    return Transcript(" ".join(w.text for w in kept), kept)
