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

## Display page: saved settings beat new defaults

`localStorage` overrides `DEFAULTS`, and a "Copy URL" link overrides both.
Changing a default in code does nothing for an existing browser. Bump
`SCHEMA` in `display.html` and add a migration line in `loadFromStorage`, and
tell the user to re-copy the OBS URL.

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
