# Session log

Newest first. Each entry: what changed, why, what's still open.

## 2026-09-03 — repository organisation, git init

**Done**
- Flat scripts → `gujusub/` package (`static/` moved inside it so the server
  finds its pages via `__file__`). Imports are absolute (`from gujusub.x`).
- `tests/`, `tools/` (`mic_client.py`, new `transcribe_file.py` replacing the
  hard-coded `inference.py`), `samples/` (was `audio-files/`).
- Added `.gitignore`, `README.md`, `pyproject.toml` (pytest + ruff config),
  `requirements-dev.txt`; updated `CLAUDE.md` and this wiki.
- `git init` + initial commit.

**Open threads** — unchanged from the previous entry (validate Gujarati
filler spellings, echo-cancellation toggles, ProPresenter test).

## 2026-09-01 → 2026-09-02 — filler suppression, broadcast display page, mic selector

**Starting point:** working ASR + translation server, one mic page that showed
scrolling Gujarati/English. No broadcast output, fillers shown verbatim.

**Done**
- `fillers.py` + `test_fillers.py`: two-tier, position-aware filler removal
  (English + Gujarati). Wired into `server.py` before translation;
  `--no-filter` flag. See [decisions.md](decisions.md).
- `server.py`: `/display` route, `/ws/view` receive-only broadcast, events that
  are all-filler are dropped.
- `static/display.html` (new, built by a subagent to spec, then iterated):
  aspect-ratio stage, fullscreen, hard two-line clamp, EN/GU/Both modes,
  full typography + outline (`paint-order`) + background/key presets,
  auto-hiding chrome, URL-serialised settings, reconnecting WS client.
  Iterations: auto-fit with `minsize` floor; left-align default with schema
  migration; caption bar that tracks text height and fades with it.
- `translator.py`: `looks_untranslated()` guard after a live screen showed the
  translator echoing Gujarati.
- `static/index.html`: audio-input selector (persisted, hot-swap, labels after
  permission), proper mic release on stop.
- `test_server.py`: display route, filter, `/ws/view` broadcast, guard.
  Suite: 34 passing.
- Installed `pytest`, `httpx` into `.venv` (test-only).

**Observed live**
- One utterance rendered as Gujarati + "translation." in English-only mode →
  translator pass-through, now guarded.
- Centre alignment unreadable while streaming → left default.

**Open threads**
- Validate Gujarati hesitation spellings against real ASR output; extend
  `fillers.ALWAYS`.
- Expose `echoCancellation` / `noiseSuppression` toggles on the mic page for
  line-level feeds.
- ~~Consider `git init`~~ — done 2026-09-03.
- `mic_client.py` has not been updated for the bar/clamp (intentional; debug
  tool only).
- Possible: per-language font size in "Both" mode; ProPresenter-specific
  testing not yet done.

## 2026-10-04 — get server running again

**Starting point:** `python -m gujusub.server` crashed on import:
`ImportError: cannot import name 'PreTrainedTokenizerBase' from
'transformers.tokenization_utils'` (transformers 5.16.1, IndicTransToolkit 1.1.1).

**Cause:** in `gujusub/translator.py` the `it2_compat` shim was imported
*after* `IndicTransToolkit.processor`, so the shim never ran before the
collator's broken import. The `# isort: off` marker had trailing text on the
same line, so ruff did not honour it and had sorted the shim below the
toolkit import (likely during the 2026-09-03 package reorg).

**Done**
- Moved `from gujusub import it2_compat` above the IndicTransToolkit import;
  bare `# isort: off` marker with the explanation on its own comment line.
- `ruff check .` clean, 34 tests pass, server boots in ~10 s with cached
  models and serves `/` and `/display`.

- Added `install-osx.sh` and `install-windows.cmd` (venv, deps, import
  check, model prefetch; `--dev`, `--skip-models`). macOS script verified on
  the reuse-.venv path (`--dev`, tests green); a fresh-.venv run was started
  in a scratch copy but had not finished downloading torch when the PR went
  up. The Windows script is untested on a real Windows machine. README rewritten around the installers with a
  Windows prerequisites section (Build Tools needed, see gotchas).

**Open threads**
- Run `install-windows.cmd` on an actual Windows box and fix whatever breaks.
- Earlier threads unchanged from 2026-09-02.
