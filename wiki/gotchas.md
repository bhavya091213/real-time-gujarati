# Gotchas

Things that cost time. If one is fixed permanently, move it to
[decisions.md](decisions.md) or delete it.

## Python regex `\w` does not match Gujarati vowel signs

Anusvara (`ં`), matras (`ે`, `ા`, …) and virama are combining marks
(category Mn/Mc). `\w` / `str.isalnum()` / `str.isalpha()` all say **no**, so
`re.sub(r"\W+$", "", "અં")` strips the anusvara and returns `અ`.
`fillers._norm` therefore strips edge punctuation by Unicode category
(P/S/Z) instead, and `translator.looks_untranslated` counts code points in
the Gujarati block (U+0A80–U+0AFF) rather than `isalpha()` letters.

## Translator can return the Gujarati source (with a period)

See decisions → pass-through guard. Diagnostic tell: Latin script and/or a
trailing period in a "Gujarati" caption means it came from the *translator*,
because the CTC ASR never emits punctuation or Latin letters.

## Settings: pitfalls (WS2)

- Page `DEFAULTS` (display.html, index.html fallback) must match `settings.py`
  defaults; the server snapshot wins, but a mismatch shows as a flash on load.
- `schema` is not settable (rejected by `set`); Reset strips it.
- `asr_mode` `en`/`auto` fall back to `gu` (with a one-time warning) when the `en` engine (and, for auto, LID) is not loaded, e.g. `--engines gu`.
- `status` messages are per mic connection and untagged: with several mics open
  each pipeline sends its own, and the strip shows whichever arrived last.

## Display shows Gujarati when you expected English

Check, in order: Content mode in the panel (`mode` param) is not "Gujarati
only"; server was not started with `--no-translate`; translator pass-through
(above).

## FastAPI TestClient `portal` is `None` outside a `with client:` block

Needed to call async `server.broadcast()` from a sync test — wrap in
`with client, client.websocket_connect(...)`.

## Filler list is a guess until validated on real speech

The Gujarati hesitations in `fillers.ALWAYS` are how we *expect*
IndicConformer to spell them. After a real session, grep the raw output
(`--no-filter`) for what it actually produces and add those spellings.

## `tools/mic_client.py` shows filler-stripped text but no bar/clamp

It is a debugging tool, not the broadcast path. Use `/display` for output.

## Run everything from the repo root

Since the 2026-09-03 reorg the code is a package. `python gujusub/server.py`
fails with `ModuleNotFoundError: gujusub`; use `python -m gujusub.server`.
`tools/*.py` add the repo root to `sys.path` themselves so they can be run
directly.

## `# isort: off` must be the whole comment

Ruff only honours `# isort: off` / `# isort: on` when the comment is exactly
that. `# isort: off  -- reason` is ignored, and ruff will happily sort
`gujusub.it2_compat` below `IndicTransToolkit`, which breaks server startup
with `ImportError: cannot import name 'PreTrainedTokenizerBase'`. Put the
reason on its own comment line (see `translator.py`).

## Windows: IndicTransToolkit has no wheel

PyPI ships macOS arm64 and manylinux wheels only (checked 2026-10-04,
v1.1.1). On Windows pip builds the Cython extension from the sdist, which
needs the MSVC Build Tools ("Desktop development with C++"). Without them
`install-windows.cmd` fails inside `pip install -r requirements.txt` with
"Microsoft Visual C++ 14.0 or greater is required". `--device mps` is also
macOS-only; Windows runs on CPU.

## onnxruntime >= 1.24 rejects the symlinked Hugging Face cache

`encoder.onnx` keeps its weights in ~370 external-data files beside it.
huggingface_hub stores every snapshot file as a symlink into `blobs/`, and
newer onnxruntime validates that external data resolves inside the model's
own directory, so session creation fails with
`External data path validation failed for initializer ...`. Older
onnxruntime (1.20) did not check, which is why one machine worked and a
fresh install did not. Fix (2026-10-04): `gujusub/model_dir.py` hardlinks
the snapshot into `~/.cache/gujusub/<repo>` (override: `GUJUSUB_MODEL_DIR`)
and `asr_engine.py` loads from there. Do not load the ASR model straight
from the hub cache again.

## ORT defaults are the slowest config here

ONNX Runtime's default intra-op threads include the E-cores and spin-wait.
The worst measured setup was 8 threads with spinning on: idle ORT threads
fought CT2 for cores. `gujusub/threads.py` sets performance cores and spinning
off; do not remove that without re-measuring.

## CTC word times run early

Word `end_s` from the CTC engine is 0.1-0.3 s before the sound ends. Never cut
the window exactly at `end_s` (it duplicated words, "hun hun"); cut midway
between a word's end and the next word's start.

## `FRAME_DURATION_MS` warning is harmless

The ASR model logs a `FRAME_DURATION_MS` warning on load and in replay output.
It does not affect results.

## Partials keep ticking during endpoint silence

Until the 600 ms endpoint fires, partial decodes continue on the silent
tail, so `replay.py` and the server show events during pauses.

## `filtered()` must use `dataclasses.replace`

`server.filtered()` copies `TranscriptEvent` with `dataclasses.replace`;
rebuilding by hand drops any field added later.


## Out-of-range saved settings silently revert to defaults

`settings.py` clamps numeric display settings to the panel's ranges (size
24–160, pad 0–25, bottom 0–50, outline 0–12, …). A `settings.json` value
outside those ranges is dropped on load with a warning and the default is
used, so an edited file can "lose" a value without an error on screen.

## Parakeet / onnx-asr quirks

- onnx-asr turns the vocab's "▁" into a leading space (tokens `' G'`, `'ood'`),
  so words split on a leading space or "▁" (both handled and tested).
- Timestamps are token *starts* on an 80 ms grid; word end = last token start
  + 0.08 s.
- Parakeet confidence is ~1.0 on clean speech and can stay high on
  wrong-language audio, so the confidence gate is weak for `en` (the 0.05
  silence floor also does not reject it).

## Saved settings predating the confidence defaults keep the gate off

A `settings.json` written before `conf_word_min`/`conf_utt_min` defaulted to
0.5/0.7 contains `0.0` for both, and the store loads saved values over the
defaults, so the gate stays off. Delete the keys from the file (or move the
file aside) or set the sliders on the mic page.

## `--lang` is a no-op

The Gujarati slot is always `gu`; use `--engines` and the Speech language
setting. The flag is accepted only so old launch scripts keep working.

## baps.org blocks fetches

Glossary entries could not be checked against the BAPS site; 195 of 667 are
verified from other sources (Wikipedia etc.), the rest are `verified: false`.

## Glossary is not package-data

`gujusub/data/glossary_baps.json` is read via `importlib.resources`, but
`pyproject.toml` has no package-data config. Running from source works; an
installed wheel would need package-data added.

## Language ID (Auto mode)

- **Near-silence scores as confident English** (P(en) 0.89-0.96). Feeding the raw
  clip from t=0 makes the decider commit to `en` on Gujarati audio. Hence the
  speech-only feed from VAD onset, the -45 dBFS RMS guard in `classify`, and no
  single-window decision before 1.5 s. Do not feed preroll or silence to the decider.
- **speechbrain `expect_len` warning is harmless.** Ignore it.
- **Cold import of speechbrain takes ~17 s** on first run (warm start is quick).
  Model load itself is ~1 s.
- **LID and confidence defaults are provisional.** There are no user clips in
  `samples/` yet, so thresholds (0.80 / 0.95 / 1.5 s / 3 s, RMS guard) were set on
  one Gujarati clip plus synthetic English. Calibration and a bake-off are unit 4.3.
