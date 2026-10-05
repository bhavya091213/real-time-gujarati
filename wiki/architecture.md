# Architecture

## Pipeline

```
mic (browser page or mic_client.py)
  │  PCM16 mono 16 kHz over WebSocket /ws
  ▼
StreamingTranscriber (streaming.py)
  │  Silero VAD gating; engine registry (engines.py) → engine pinned per
  │  utterance when VAD opens it (asr_mode gu|en, read from settings)
  │  confidence trim (confidence.py) on every decode → LocalAgreement-2 commits
  │  → TranscriptEvent(type, utterance_id, committed, tail, lang)
  ▼
fillers.clean_event (fillers.py)        ← drops um/uh/એટલે… (see decisions.md)
  ▼
glossary (glossary.py)                  ← lang=="en" text: BAPS house spellings
  ▼
Translator.translate (translator.py)    ← gu events only; IndicTrans2 on CTranslate2
  │  pass-through guard: Gujarati-looking output → ""; glossary on the output
  ▼
server.py sends JSON back on /ws AND broadcasts to every /ws/view viewer
  │  (per connection: receiver → coalescing worker → latest-only translator;
  │   see "Server pipeline" below)
  ▼
static/display.html (broadcast output page, /display)
```

## Files

Layout since 2026-09-03: code is the `gujusub/` package, run with `python -m gujusub.server`.

| File | Role |
|------|------|
| `gujusub/server.py` | FastAPI app. Routes: `/` (mic page), `/display` (output page), `/ws` (audio in, events out), `/ws/view` (receive-only: settings snapshot then events), `/ws/control` (JSON both ways: settings panel, status). Flags: `--device cpu\|mps`, `--engines gu,en` (`--lang` is a deprecated no-op), `--port 8765`, `--no-translate`, `--no-filter`. |
| `gujusub/settings.py` | `Settings` dataclass (22 display keys + `asr_mode`, `translate`, `conf_word_min`, `conf_utt_min`, `schema`), `validate`, `SettingsStore` (atomic JSON persist), `display_payload`/`control_payload`/`snapshot_payload`. |
| `gujusub/asr_engine.py` | Wraps `ai4bharat/indic-conformer-600m-multilingual` (CTC). Lock-serialized; not thread-safe otherwise. |
| `gujusub/model_dir.py` | Materialises a HF snapshot into real files (hardlinks) under `~/.cache/gujusub`; required by onnxruntime >= 1.24 (see gotchas). `GUJUSUB_MODEL_DIR` overrides the root. |
| `gujusub/streaming.py` | VAD-gated window, re-decodes every `max(480 ms, 1.2 x last decode)`, commits common prefix of last two hypotheses. Beyond `max_window_s` (5 s), aligned committed words move to a prefix only when retained alignment can distinguish seam residue from repeated speech; otherwise trimming is deferred. 600 ms silence finalizes. 12 s hard cap on total utterance audio. |
| `gujusub/engine.py` | Engine protocol: frozen `Word(text, start_s, end_s, confidence)`, `Transcript(text, words)`, `ASREngine` (`transcribe`, `transcribe_detailed`, `warmup`). Engines with only `transcribe()` still work. |
| `gujusub/engines.py` | `EngineRegistry` (lang → engine), `parse_engine_list`, `load_engines` for `--engines`. The server asks it for the engine at each utterance start; falls back to `gu` with an `asr_warning` if the `en` engine is not loaded. |
| `gujusub/engine_parakeet.py` | Parakeet-TDT 0.6B v2 (int8 ONNX via onnx-asr) as an `ASREngine` with word confidence (`exp(min token logprob)`) and word timestamps. |
| `gujusub/confidence.py` | `trim_words`, `utterance_ok`, `apply_confidence`: per-word and per-utterance gates run after every decode, before LocalAgreement. Thresholds come from settings (`conf_word_min`, `conf_utt_min`). |
| `gujusub/glossary.py` | BAPS glossary matcher with hot-reloaded operator override (`GUJUSUB_GLOSSARY`). |
| `gujusub/data/glossary_baps.json` | Packaged glossary (667 entries, 195 verified). |
| `gujusub/threads.py` | Thread-count policy for ORT and CT2: performance cores, spin-waiting off; `GUJUSUB_THREADS` overrides. |
| `gujusub/fillers.py` | Two-tier filler suppression (ALWAYS / CONTEXTUAL / PHRASES). Pure text transform. |
| `gujusub/translator.py` | CT2 model load + `translate()`. `looks_untranslated()` guard. |
| `gujusub/it2_compat.py` | Shim so IndicTransToolkit imports under transformers 5.x. Must be imported before IndicTransToolkit. |
| `gujusub/static/index.html` | Mic page: settings panel (`/ws/control`), live status strip, device selector (persisted), AudioWorklet → PCM16 → `/ws`, shows Gujarati + English. |
| `gujusub/static/display.html` | Output page. Single file, no deps. See [broadcast-setup.md](broadcast-setup.md). |
| `tools/mic_client.py` | Terminal mic client (sounddevice → `/ws`), prints captions. Debug tool. |
| `tools/transcribe_file.py` | One-shot file transcription (`samples/test-guju.m4a`), optional `--translate`. |
| `tools/replay.py` | Offline replay of a file through `StreamingTranscriber` (+ optional translator) with an arrival-time lag model; prints decodes/s, decode p50/p95/max, translate calls, max/end lag, RTF. Flags: `--loop-to S`, `--no-endpoint`, `--translate`, `--rtf 1`, `--json`. Serial model, so lag with `--translate` is an upper bound (the server translates off the ASR path). |
| `tools/calibrate_confidence.py` | Sweeps `conf_word_min` x `conf_utt_min` over clips on one or both engines (`--recommend W,U`); used for the defaults in decisions.md. |
| `tools/prefetch_models.py` | Downloads + materialises both models; `--verify` loads them and runs a tiny inference. Called by the installers. |
| `tests/` | pytest suite (347 tests, no models loaded). Run: `.venv/bin/python -m pytest` |
| `samples/` | Short Gujarati test clip. |
| `pyproject.toml` | pytest + ruff config. `requirements.txt` runtime, `requirements-dev.txt` adds pytest/httpx/ruff. |
| `install-osx.sh`, `install-windows.cmd` | One-shot installers: find Python >= 3.11, create `.venv`, pip install, verify imports, pre-download models. `--dev` adds dev deps + runs tests; `--skip-models` defers the download. Idempotent. |

## WebSocket event shape

```json
{"type": "partial" | "final",
 "utterance_id": 7,
 "committed": "stable gujarati text",
 "tail": "unstable gujarati text",
 "translation": "english (may be empty if translation failed/disabled)"}
```

`committed` only grows within an utterance; `tail` may change. `final` has an
empty tail and the full re-decoded text. Events whose text is entirely fillers
are not sent at all. `translation` is absent with `--no-translate`.

Translation attachment (since 2026-10-04): a `partial` is sent immediately with
the last translation known for its utterance (`""` until the first one lands;
it may describe an older committed prefix). When a background translation
finishes, the latest partial is **re-sent** with the same `type`,
`utterance_id`, `committed` and `tail` and the new `translation`. Clients
must treat repeated partials as idempotent updates. A `final` is sent exactly
once, after its own translation, and nothing from that utterance follows it.

Event field `lang` (`"gu"` | `"en"`, optional; missing means `gu`) is the language
of the engine that decoded the utterance. It picks the primary-line typeface on
the display, and the server skips translation (`translation: ""`) for `en`.

## Control/view messages (since WS2)

- `/ws/view` first message: `{"type":"settings", <22 display keys>, "schema":3}`;
  later settings changes arrive as further display-only `settings` messages.
- `/ws/control` on connect: `{"type":"settings", <all keys>, "defaults":{...}}`
  then `{"type":"status", ...}` (asr, translate, lid, decode_ms, avg_decode_ms,
  window_s, decodes, lag_s, backlog_s, utterances, asr_warning (string or null,
  e.g. the `en` engine is not loaded so Gujarati is used instead); once per second per mic
  pipeline, idle with `asr:null` otherwise). In: `{"type":"get"}`,
  `{"type":"set","settings":{...}}`. Failures reply `{"type":"error","message":...}`
  and change nothing.

## Server pipeline (per /ws connection, `server._Pipeline`)

- **Receiver**: only `receive_bytes` → int16→float32 → `asyncio.Queue`.
- **Worker**: waits for audio, drains everything queued, calls
  `transcriber.feed()` once on the concatenation (one executor hop per batch,
  not per 8 ms message). Of the batch's events only the last partial per
  utterance is kept (`latest_events`), and all finals are kept. Finals are
  translated inline, then sent. On disconnect the worker drains the queue,
  then `flush()` runs (never concurrently with `feed()`). If the worker
  crashes it logs and closes the socket with 1011.
- **Translator task**: a latest-only slot. Partial text is queued only when
  `committed` changed. At most one partial translation starts per
  `PARTIAL_TRANSLATE_GAP_S` (1 s). Calls still go through `run_in_executor`
  and the Translator lock.
- Every event sent to the mic client is also broadcast to `/ws/view`.
- Note: `feed()` itself still decodes once per interval of audio inside a
  large batch. Coalescing saves hops and stale sends, not decodes.

## Running

`./start.sh --open` (macOS) or `start.cmd --open` (Windows) installs on first
run (via `install-osx.sh` / `install-windows.cmd`), starts the server and opens
both pages. Flags: see the README table. Equivalent manual commands:

```bash
.venv/bin/python -m gujusub.server        # loads ASR + translator (~30 s), port 8765
open http://localhost:8765/               # mic page — pick input, Start mic
open http://localhost:8765/display        # broadcast page — put this in OBS
```

Run from the repo root so the `gujusub` package is importable. Models are
cached by huggingface_hub after the first download.

## Git

Repository initialised 2026-09-03 (`.gitignore` excludes `.venv`, caches,
model weights, ad-hoc recordings). History before that exists only in
[sessions.md](sessions.md).
