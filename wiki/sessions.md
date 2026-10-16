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

**Review (WS3):** Codex found 2 issues: malformed operator glossary overrides could crash caption processing (fixed: schema validation + fallback, 357 tests); 'committed words not retracted when confidence later drops' rejected as by-design.

## 2026-10-05 - WS4: Auto mode (language ID)

Added `gujusub/lid.py` (SpeechBrain ECAPA restricted to gu/en, `LidDecider` policy),
`AutoRoute` in the streaming pipeline, Auto enabled on the mic page with a
`lid_last` status readout, `replay --engine auto`. Policy and rationale are in
decisions.md. Measured on M1 Pro: Gujarati decided at 1.5 s of speech, caption
+2.1 s after onset; English +1.9 s; 17-28 ms per LID call; +~250 MB RSS.
Review (Codex) fixes: F-01 `final_check` honours `strong_from_s`; F-02 translation
eligibility pinned per utterance. Tests: 416. User-approved deviation: fallback to
Gujarati after 3 s undecided.
Open: user clips -> calibration and LID bake-off (unit 4.3); deferred review items
from triage-ws2/ws3/ws4; per-language confidence thresholds (en gate is weak).

## 2026-10-05 - WS3: English engine, mode routing, glossary, confidence

Branch ws3-english. Parakeet-TDT 0.6B int8 (onnx-asr) as the English engine
(`engine_parakeet.py`); engine registry and `--engines gu,en` with the engine
pinned per utterance (`engines.py`); `asr_mode` gu/en live from the mic page,
no translation for English; BAPS glossary (667 entries, 195 verified,
`GUJUSUB_GLOSSARY` override, hot reload); confidence trimming before
LocalAgreement with defaults 0.5/0.7 and `tools/calibrate_confidence.py`.
Numbers: Parakeet 106 ms/1 s ... 324 ms/4 s ... 401 ms/5 s (target 250 ms at
5 s missed, so the English window is 4 s); both engines ~3.6 GB RSS; glossary
~0.06 ms per 50-word line. Single-word minimum for the gate is 1.
Open: calibrate on real user clips (only test-guju.m4a exists); Auto
language mode next (units 4.x); per-language confidence thresholds (en gate is
weak); deferred review items; glossary package-data for wheels.

## 2026-10-05 - WS2: server-side settings + mic-page panel

Settings moved to the server (`gujusub/settings.py`, `/ws/control`, persisted
JSON, env `GUJUSUB_SETTINGS`); `/display` is now output-only (no panel, no
localStorage, `f` fullscreen only); the mic page has the full settings panel,
Translate switch (D16: off = no translator work), confidence sliders (inert),
asr_mode (en/auto disabled, run as gu), and a live status strip. Content modes
renamed primary/translation/both. Polish: control snapshot carries `defaults`
(Reset uses it), startup logs settings path + asr_mode/translate, lang-en block
honours the configured font. Real-browser check (headless Chromium via
playwright-core, display ws proxied to inject events) passed: panel, live push,
reloads, `s` inert, `f` fullscreen, empty-translation fallback, lang-en.
Not covered: status strip with a live mic (no audio in headless run).
Open: WS3/4 engines for en/auto; confidence consumption; per-mic status tags.

**Review (WS2):** Codex found 3 mediums, all fixed before landing: numeric
settings now clamped to the panel's ranges (out-of-range saved values revert
to defaults); viewer join is ordered against broadcasts with a per-viewer
backlog and a 2 s bound on the connecting viewer's first send; the control
socket got an 8 KiB message cap, strict shapes, a 64 KiB frame limit, no-op
write suppression, a per-source rate limit (10/s, burst 20) and off-loop
settings writes. Deferred (low): fan-out ordering for overlapping sets,
panel re-`get` after a rejected set, `_joining` cleanup on cancellation.
Tests: 245.


## 2026-10-09 - Investigation: dropping non-Gujarati weights / shrinking the Indic models

No code changed. Question: can the non-Gujarati weights be dropped, and can
the Indic models be optimised further? Measured on the M1 Pro (16 GB), CPU,
`samples/test-guju.m4a`, median of 5 decodes, RSS after warmup.

- **There are almost no Gujarati-specific weights to drop in the 600M ASR.**
  Encoder = 607.7M params fp32 (2.43 GB) shared by all 22 languages; the
  per-language parts are 21 unused `joint_post_net_*` (0.65 MB each), 5376 of
  5633 CTC head rows (~21 MB) and the RNNT sessions we never run (~45 MB).
  The translator transliterates Gujarati to Devanagari before tokenising, so
  its 122.7k source vocab is mostly shared Devanagari; only the Arabic-script,
  Ol Chiki and Meetei rows (~28k, ~57 MB fp32) are provably unused.
- **Numbers** (load s / RSS MB / 5 s decode ms / 8 s decode ms / text):
  600M fp32 (current): 6.5-11 / 2900 / 207 / 267 / `...પાણી પીઉું છું અતાર`.
  600M int8 dynamic (MatMul only, 217 of 289): 3.6 / 1400 / 180 / 228 /
  `...પીઉ છું અત્યારે`. CTC-only sessions of that: 930 MB.
  600M 4-bit MatMulNBits: 1200 MB but 50% slower (313 ms at 5 s), rejected.
  Gujarati-only 120M (`OpenVoiceOS/ai4bharat-indicconformer-gu-onnx`, MIT,
  export of `ai4bharat/indicconformer_stt_gu_hybrid_ctc_rnnt_large`, 80 mel,
  subsampling 4, vocab 257): fp32 460 MB on disk / 2.4 / 550 / 176 / 206 /
  `...પીવ છું અત્યારે`; MatMul int8 (self-made, 177 MB): 0.4 / 555 / 113 / 142 /
  `...પીઉ છું અત્યારે`. Its shipped `model.int8.onnx` uses ConvInteger with
  signed weights and does not load in onnxruntime 1.20 (NOT_IMPLEMENTED).
  CoreML EP for the 600M encoder: 387 ms at 5 s, slower than CPU.
  ORT arena off: no RSS change. Translator: load 0.9 s, +730 MB RSS, 84 ms per
  sentence; `model.bin` is fp32 on disk (808 MB) and quantised on load.
- **Caveat**: one clip. Both candidates got અત્યારે where the current model
  gives અતાર, but accuracy needs real clips before switching (same gap noted
  for confidence/LID calibration).
- **Incident**: writing a quantised graph to a path that was a hardlink of the
  hub blob overwrote the original `assets/encoder.onnx` in both the hub cache
  and `~/.cache/gujusub`. Corrupted copies moved to
  `~/.agents/backups/retired-corrupt-encoder-onnx-2026-10-09/`, the graph
  re-downloaded (2,976,427 bytes, 1903 Constant + 289 MatMul) and the baseline
  re-run with identical output. See gotchas.
- Also now in the hub cache: `OpenVoiceOS/ai4bharat-indicconformer-gu-onnx`
  (~590 MB). Unrelated leftovers there from earlier experiments: nllb-200
  (4.6 GB), prajdabre rotary IndicTrans2 (1.6 GB), kompress, MiniLM, ModernBERT.

Open: pick between 600M int8 (same model, -1.5 GB, ~15% faster) and the
Gujarati-only 120M int8 (-2.3 GB, ~45% faster, 177 MB download) once real
clips exist; either needs a CTC-only loader that bypasses `model_onnx.py`;
pre-quantised IndicTrans2 CT2 on disk would cut 600 MB of download/disk only.

## 2026-10-14 - `--gu-model fast`: Gujarati-only 120M int8 engine

Follow-up to the 2026-10-09 investigation. New `gujusub/engine_gu_fast.py`
(`FastGujaratiEngine`): fetches `OpenVoiceOS/ai4bharat-indicconformer-gu-onnx`
(fp32 graph + weights + vocab, ~460 MB), materialises it, quantises MatMuls to
int8 once (`model.gujusub-int8.onnx`, temp-file + rename), runs the ORT session
with onnx-asr's NumPy NeMo preprocessor and decodes through `ctc_words`
(40 ms frames). `engines.py` gained `GU_MODELS`, `engine_loaders(gu_model)` and a
`gu_model` argument on `load_engines`; `server.py --gu-model full|fast` (default
`full`); `tools/replay.py --gu-model`; `tools/prefetch_models.py --gu-model
full|fast|both` (download and `--verify`). README, architecture and decisions
updated; launcher comments mention the flag.
Verified: `prefetch_models.py --verify --gu-model fast` transcribes the sample's
first 3 s as `મારું નામ ભવ્ય છે`; replay full vs fast: decode p50 211 vs 106 ms,
max lag 0.74 vs 0.20 s, RTF 0.44 vs 0.19, final text differs on one word
(`પીઉ` vs `પીવ`). Tests: 439 (new `tests/test_engine_gu_fast.py`,
registry and flag tests).
Open: accuracy on real clips before making `fast` the default; the installers
still prefetch only the full model (pass `--gu-model fast` or `both` by hand);
the fast engine ignores `--device` (CPU only, like Parakeet).

## 2026-10-16 — How sabha-stt.vercel.app is fast (investigation, no code changes)

User asked how https://sabha-stt.vercel.app/ gets near-instant captions. Read
its Next.js bundles (no login needed). Answer: it does no local inference at all.

- ASR + translation are **Soniox** cloud (`SonioxClient`, model `stt-rt-v5`,
  `enableLanguageIdentification`, `enableEndpointDetection`, `languageHints
  ["en","gu","hi"]`, one-way translation `gu→en` or `en→gu`). The browser
  streams the mic/Dante feed straight to Soniox over WebSocket using a
  temporary key from `/api/soniox-token`. Tokens arrive with `is_final`,
  `language`, `translation_status`, and an `<end>` endpoint token.
- The `/dictionary` page feeds Soniox's `context` field: `general` rules
  (domain, devotional terminology, jay naad), `terms` (custom vocabulary) and
  `translation_terms` (source→target pairs), trimmed to a 9.6 KB JSON budget.
  Contributions go through `/api/dictionary` with a review page.
- Audio input: `getUserMedia`, or a local "Sabha Network Receiver" /
  `dante-rx` on `ws://127.0.0.1:8770/audio` played through an AudioWorklet
  ring buffer (100 ms target fill, underruns filled with silence).
- Client-side post-processing: jay-naad chant rewriting, a devotional-term
  correction pass on English, a profanity filter, and segment tracking keyed
  on `<end>`.

Implication for guju-sub: their speed is the vendor's streaming model plus
~zero client compute, not a trick we can port. Our equivalent levers are
`--gu-model fast` (shipped 2026-10-14) and the context/glossary idea
(we already have `glossary_baps.json`). A Soniox engine would mean a paid
cloud dependency and audio leaving the machine.

## 2026-10-16 — Cloud API backends + Sabha-style display (implementation)

Request: "add an option for API based inference", key entry with the key
hidden but for its last characters, no local models when a cloud engine is
selected, "ElevenLabs and other popular providers", and "change our display
to how they do it".

Done (orchestrated: 6 research agents, then 9 implementation agents in
parallel, each owning disjoint files):
- `gujusub/cloud/` package: `base.py` (CloudSession protocol, SegmentTracker,
  WebSocketSession with reconnects), `keys.py` (0600 key file + env fallback,
  masking), `registry.py` (catalogue, `asr_backend` values), and six providers:
  `soniox.py`, `elevenlabs.py`, `deepgram.py`, `openai.py`, `sarvam.py`,
  `azure.py` (SDK-based, optional dependency). Each has a fake-socket test
  file. AssemblyAI researched and dropped (no Gujarati streaming).
- `gujusub/cloud_pipeline.py` (CloudPipeline: raw PCM16 -> provider, events
  -> the usual publish path), `gujusub/local_models.py` (one-shot background
  loader for switching back to local at runtime), `server.py` `--backend`,
  control messages `providers` / `set_key` / `clear_key` / `keys`, status
  fields `backend`, `cloud`, `models`; `settings.asr_backend`.
- Mic page: "Speech engine" section with provider notes, masked key status,
  password field, Save/Remove, Azure region; status strip shows the engine.
- Display rebuilt after the Sabha overlay: per-language row engine (measureText
  wrapping, phrase breaks on final/punctuation/word cap/900 ms pause), rows
  grow upward, 100 ms word fade-in, Gujarati block above English, bottom panel
  / per-line chips / shadow / outline, F/G/C/+/-/Esc keys, 20 s fade,
  `?preview=1` placeholders; schema 4 with 12 new display keys.
- README cloud table, wiki architecture/decisions/broadcast-setup updated,
  `start.sh`/`start.cmd` mention `--backend`, requirements note the optional
  Azure SDK.

Not verified live: no provider keys on this machine, so every client was
tested only against fakes. First real session per provider needs a debug pass
(watch the server log for the provider's error message). Open: Soniox and
Sarvam override the base `_reconnect` for "server finished the stream" and
close-code classification — a proper hook in `base.py` would be cleaner.
