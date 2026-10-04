# Design decisions

## Streaming: VAD + LocalAgreement-2 (pre-2026-09-01)

Offline CTC model, so real-time is faked by re-decoding the growing utterance
buffer every 480 ms and committing only the common prefix of the last two
hypotheses (whisper_streaming, Machacek et al. 2023). Committed text never
retracts, which is what makes it safe to show on air.

## Filler suppression is a display transform, not an ASR change (2026-09-01)

`fillers.clean_event` runs on each `TranscriptEvent` *after* the commit logic
and *before* translation. Reasons:

- It cannot destabilise LocalAgreement (the streaming state never sees it).
- Cleaner Gujarati in → cleaner English out of the translator.
- `--no-filter` restores raw output for debugging.

### Two tiers

- **ALWAYS**: pure hesitations (`um uh er hmm mm ah`, Gujarati `અં અઅ ઉં હં હમ હમ્મ`)
  — dropped wherever they stand alone.
- **CONTEXTUAL**: real words that are fillers only by position
  (`like so well right okay basically actually`, phrases `you know / I mean /
  sort of / kind of`, Gujarati `એટલે`, `એટલે કે`, `મતલબ`). Dropped only when they
  open an utterance, follow a hesitation ("um like"), stutter ("like like" →
  drop the first), or dangle at the end of a **final** utterance (trailing
  hesitations are skipped when checking, so "…so um" trims both).
- `હવે` ("now") was deliberately **excluded** — too common as a real word.

Committed and tail are filtered *jointly* so a phrase spanning the boundary
("you | know") is still caught, then split back. Utterances that were entirely
fillers are suppressed server-side (no event sent).

## Translator pass-through guard (2026-09-01)

The distilled IndicTrans2 model occasionally copies Gujarati source through
instead of translating (seen live: Gujarati text ending in "translation.").
`translator.looks_untranslated()` treats output with >50 % Gujarati code
points (counting vowel signs, which are not `isalpha()`) as failed and returns
`""`. The display page then keeps the last good English for that utterance.
Threshold was chosen so "My name is ભવ્ય" (a name) still passes.

## Separate receive-only `/ws/view` socket (2026-09-01)

The output page must not own the microphone. `/ws` is one-per-mic-client and
carries audio; `/ws/view` fans out every event to any number of viewers (OBS,
a preview browser, a confidence monitor). Viewers reconnect with 1 s → 10 s
backoff so they survive server restarts during a long event.

## Display page choices (2026-09-01 / 02)

- **Single self-contained HTML, no CDN** — must work offline at a venue.
- **Hard two-line clamp**: block height fixed at `2 × size × line-height`,
  bottom-anchored, overflow hidden. Newest words always visible. In "Both"
  mode each language gets one line.
- **Auto-fit**: text shrinks (binary search) from `size` down to `minsize`
  to fit the clamp; at the floor it wraps/scrolls instead. Block height never
  changes, so nothing jumps.
- **`-webkit-text-stroke` + `paint-order: stroke fill`** for outlines — without
  paint-order the stroke eats thin Gujarati glyphs.
- **Left alignment default** (2026-09-02): centred text shifts every time a
  word is appended and is hard to read while streaming.
- **Caption bar**: optional vertical bar left of the text, height = height of
  text currently on screen, fades with the text. Signals "these are live
  captions" on the keyed feed.
- **Settings precedence**: URL params > localStorage > defaults. "Copy URL"
  serialises everything so OBS can load a fully configured page. A `schema`
  number in localStorage lets a changed default override stored values once
  (v2 forced `align` to left).
- **Chrome auto-hides** (gear, fullscreen, status dot, cursor) after 3 s so
  captures are clean; the status dot never shows in an idle capture.

## Mic page device selector (2026-09-02)

`enumerateDevices()` is called at load, after first `getUserMedia` (labels
only appear post-permission), and on `devicechange`. Choice persisted in
localStorage. Changing the device while streaming restarts the capture
(hot-swap). `stop()` now releases tracks so the browser mic indicator clears.

## WS1 latency work (2026-10-04)

- **Bounded re-decode window with seam dedupe.** Decode cost is about
  30 + 47 ms per second of window, and lag built up from roughly 7 s of
  continuous speech (12 s windows cost 600+ ms per 480 ms tick). The window
  is trimmed beyond `max_window_s` (5 s), whisper_streaming style: aligned,
  committed words before the cut become a prefix when at least one aligned
  word remains. On the first post-trim decode, a matching 1-5 word hypothesis
  head is dropped only if that produces a strictly longer common prefix with
  the retained pre-trim hypothesis; that choice is reused until the next trim.
  Without alignment evidence the trim is deferred, preserving ambiguous
  repeats while the 12 s utterance cap remains the safety bound. A 4 s cap
  would meet a 250 ms decode target; left at 5.
- **Adaptive interval.** Re-decode every `max(interval_ms, 1.2 x recent
  decode)`, clamped to `max_interval_ms` (2 s of audio), so a slow machine
  falls behind less instead of queueing decodes. "Recent" is the low median
  of the current utterance's last 5 partial decodes: unlike an EMA, one
  outlier (e.g. a 20 s stall) stops mattering after a single normal decode.
  The history resets per utterance so a slow final cannot throttle the next
  utterance, and the 2 s cap keeps partials flowing even when every decode is
  slow (an uncapped interval let one slow decode suppress all partials until
  the final). `last_decode_ms` / `avg_decode_ms` stay raw for stats.
- **Translation off the ASR path.** Partials go out at once with the last
  known translation; a latest-only slot translates the newest text, at most
  one start per 1 s (`PARTIAL_TRANSLATE_GAP_S`), and the latest partial is
  re-sent. Finals are translated inline and sent once. Commit-change gating
  alone still translated ~55% of partials; the gap halves that.
- **ORT intra-op = performance cores, spinning off.** Measured: the default
  (all cores incl. E-cores, spinning on) contended with CT2 translation.
- **Display: rAF coalescing + bounded fit passes** (1-6 layout passes per
  block instead of 1-9; one render per frame at most).
- **Word-level confidence** is exposed by `Transcript.words` but unused; it
  is there for later trimming/commit decisions.
