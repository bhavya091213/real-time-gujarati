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
  the reuse-.venv path (`--dev`, tests green) and on a fresh .venv in a
  scratch copy (resolved transformers 5.18.0 / torch 2.14.1, import check
  passed). The Windows script is untested on a real Windows machine. README rewritten around the installers with a
  Windows prerequisites section (Build Tools needed, see gotchas).

- Mac mini fresh install failed in onnxruntime with `External data path
  validation failed for initializer`. Cause: symlinked HF cache + newer
  onnxruntime (see gotchas). Added `gujusub/model_dir.py` (hardlink
  materialisation, 7 unit tests), rewired `asr_engine.py` to load the remote
  `model_onnx.py` from the materialised dir, added `onnxruntime` to
  `requirements.txt` (it was missing), and `tools/prefetch_models.py`
  (`--verify` loads both models). Both installers now prefetch *and* run the
  verify step. Pushed straight to main at the user's request without an
  end-to-end load run on this machine (unit tests + lint only).

**Open threads**
- Confirm the Mac mini install passes end to end after this change.
- Run `install-windows.cmd` on an actual Windows box and fix whatever breaks.
- Earlier threads unchanged from 2026-09-02.

## 2026-10-04 - WS1 latency/perf (branch ws1-perf)

**Why**: lag built up in continuous speech (decode cost grows with window; ~7 s
of nonstop speech was enough), and ORT/CT2 contended for cores.

**Done**: engine protocol + word timings (`engine.py`); bounded 5 s window with
seam dedupe and adaptive interval (`streaming.py`); ORT/CT2 thread policy
(`threads.py`); server pipeline with coalescing worker and latest-only
translation (1 s partial gap); display.html rAF coalescing; `tools/replay.py`
(+ model-free tests). 122 tests pass, ruff clean.

**Numbers** (60 s looped sample, `--no-endpoint --translate`, load avg ~4-5):
before (unit 1.6, load ~10): decode p95 630 ms, max lag 2.32 s, end lag 0.81 s,
52 translate calls, RTF 0.755. After: decode p50/p95 ~148/200 ms (max 236-743),
max lag 0.58-0.79 s, end lag ~0.38 s, 39 translate calls, RTF ~0.355. Natural
6.6 s sample: final text identical to pre-WS1, max lag ~0.36 s, RTF ~0.28.
Baselines were taken under heavier machine load, so the gain is partly noise.

**Open threads**: `max_window_s` 4 s option; CoreML EP and skipping the 26
unused ORT sessions as speedups; `feed()` still decodes every 480 ms inside
big batches; rare fuzzy seam fragment; display not verified on real GPU/OBS
with Gujarati script.

**Review (same day):** Codex (gpt-5.5 hunt, gpt-5.6-sol adjudicate) found 5
issues, then a Claude round found 4 more; all fixed before landing: disconnect
final was dropped; final could retract committed words (now prefix + committed +
anchored suffix, fallback to last partial's tail); seam dedupe ate real repeats
(now evidence-gated, re-checked on every use); adaptive interval unbounded (now
per-utterance lower-median estimate, capped at `max_interval_ms` 2 s); display
ignored empty translation; `broadcast` iterated the live viewer set across an
await; empty translations were stored and blanked the English line; translator
task cancel/await on shutdown; filler-only finals leaked a translation entry.
Deferred (low): one decode per batched feed() under overload + lag metric;
replay clock/batching parity; torch imported via streaming import; cache
`perf_cores()`; test hardening; late translation re-send after a final.
Tests: 146.
