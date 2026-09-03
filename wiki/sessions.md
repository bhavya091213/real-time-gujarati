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
