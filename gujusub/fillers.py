"""Filler-word suppression for caption display.

Strips hesitation sounds and discourse fillers from transcript text before
it is displayed or translated. Two tiers:

- ALWAYS: pure hesitations ("um", "અં") that carry no meaning and are
  dropped wherever they appear as standalone words.
- CONTEXTUAL: words that are fillers only in some positions ("like",
  "એટલે"). These are dropped only when the position suggests filler use:
  at the very start of an utterance, immediately repeated (a stutter),
  adjacent to another dropped filler, or dangling at the end of a *final*
  utterance. "I like this" is left untouched.

This is a pure display transform: it runs on the text of each
TranscriptEvent after the streaming commit logic, so it never affects
LocalAgreement stability.
"""

import unicodedata

# Hesitation sounds: meaningless in any position.
ALWAYS: frozenset[str] = frozenset({
    # Latin (code-switching and translated output)
    "um", "umm", "uh", "uhh", "uhm", "er", "erm", "hmm", "hmmm", "hm",
    "mm", "mmm", "ah", "eh", "huh",
    # Gujarati hesitations as the ASR tends to spell them
    "અં", "અઅ", "ઉં", "હં", "હમ્મ", "હમ",
})

# Discourse fillers: meaningful in normal positions, dropped only when the
# position marks them as filler (see _drop_contextual).
CONTEXTUAL: frozenset[str] = frozenset({
    "like", "so", "well", "right", "okay", "ok", "basically", "actually",
    "એટલે",  # so / that is
    "મતલબ",  # meaning / I mean
})

# Multi-word contextual fillers, matched before single tokens.
PHRASES: tuple[tuple[str, ...], ...] = (
    ("you", "know"),
    ("i", "mean"),
    ("sort", "of"),
    ("kind", "of"),
    ("એટલે", "કે"),  # that is to say
)

def _is_punct(ch: str) -> bool:
    # Category-based rather than \W: Gujarati matras and anusvara are
    # combining marks (Mn/Mc), which regex \w does not consider word chars.
    return unicodedata.category(ch)[0] in "PSZ"


def _norm(token: str) -> str:
    """Lowercase and strip surrounding punctuation for matching."""
    start, end = 0, len(token)
    while start < end and _is_punct(token[start]):
        start += 1
    while end > start and _is_punct(token[end - 1]):
        end -= 1
    return token[start:end].lower()


def _spans(norm: list[str]) -> list[tuple[int, int]]:
    """Split token indices into (start, end) spans, grouping filler phrases."""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(norm):
        for phrase in PHRASES:
            if tuple(norm[i:i + len(phrase)]) == phrase:
                spans.append((i, i + len(phrase)))
                i += len(phrase)
                break
        else:
            spans.append((i, i + 1))
            i += 1
    return spans


def _is_contextual(words: tuple[str, ...]) -> bool:
    return words in PHRASES or (len(words) == 1 and words[0] in CONTEXTUAL)


def _keep_mask(tokens: list[str], final: bool) -> list[bool]:
    norm = [_norm(t) for t in tokens]
    spans = _spans(norm)
    drop = [False] * len(spans)
    kept_any = False  # whether any earlier span survived
    prev_hesitation = False  # previous span was a dropped ALWAYS filler
    for idx, (start, end) in enumerate(spans):
        words = tuple(norm[start:end])
        is_always = len(words) == 1 and words[0] in ALWAYS
        drop[idx] = is_always or (
            _is_contextual(words)
            and _drop_contextual(spans, norm, idx, kept_any, prev_hesitation)
        )
        kept_any = kept_any or not drop[idx]
        prev_hesitation = is_always

    if final:  # a contextual filler dangling at the end: "... so", "... so um"
        for idx in range(len(spans) - 1, -1, -1):
            if drop[idx]:
                continue
            start, end = spans[idx]
            if not _is_contextual(tuple(norm[start:end])):
                break
            drop[idx] = True

    keep = [True] * len(tokens)
    for (start, end), d in zip(spans, drop, strict=True):
        if d:
            for j in range(start, end):
                keep[j] = False
    return keep


def _drop_contextual(
    spans: list[tuple[int, int]],
    norm: list[str],
    idx: int,
    kept_any: bool,
    prev_hesitation: bool,
) -> bool:
    """A contextual filler is dropped only in filler-like positions."""
    if not kept_any:  # opens the utterance ("like I said ...")
        return True
    if prev_hesitation:  # rides along with a hesitation ("um like ...")
        return True
    start, end = spans[idx]
    if idx + 1 < len(spans):  # stutter: "like like this" -> drop the first
        nstart, nend = spans[idx + 1]
        if norm[start:end] == norm[nstart:nend]:
            return True
    return False


def clean_text(text: str, *, final: bool = False) -> str:
    """Return `text` with filler words removed.

    `final=True` additionally trims a contextual filler dangling at the end,
    which is only safe once the utterance is complete.
    """
    tokens = text.split()
    if not tokens:
        return ""
    keep = _keep_mask(tokens, final)
    return " ".join(t for t, k in zip(tokens, keep, strict=True) if k)


def clean_event(committed: str, tail: str, *, final: bool = False) -> tuple[str, str]:
    """Filter a committed/tail pair jointly so phrases spanning the boundary
    ("you" | "know") are still caught. Returns new (committed, tail)."""
    committed_tokens = committed.split()
    tail_tokens = tail.split()
    tokens = committed_tokens + tail_tokens
    if not tokens:
        return "", ""
    keep = _keep_mask(tokens, final)
    n = len(committed_tokens)
    new_committed = " ".join(t for t, k in zip(tokens[:n], keep[:n], strict=True) if k)
    new_tail = " ".join(t for t, k in zip(tokens[n:], keep[n:], strict=True) if k)
    return new_committed, new_tail
