# Architecture

## Pipeline

```
mic (browser page or mic_client.py)
  │  PCM16 mono 16 kHz over WebSocket /ws
  ▼
StreamingTranscriber (streaming.py)
  │  Silero VAD gating + LocalAgreement-2 commits
  │  → TranscriptEvent(type, utterance_id, committed, tail)
  ▼
fillers.clean_event (fillers.py)        ← drops um/uh/એટલે… (see decisions.md)
  ▼
Translator.translate (translator.py)    ← IndicTrans2 dist-200M on CTranslate2
  │  pass-through guard: Gujarati-looking output → ""
  ▼
server.py sends JSON back on /ws AND broadcasts to every /ws/view viewer
  ▼
static/display.html (broadcast output page, /display)
```

## Files

Layout since 2026-09-03: code is the `gujusub/` package, run with `python -m gujusub.server`.

| File | Role |
|------|------|
| `gujusub/server.py` | FastAPI app. Routes: `/` (mic page), `/display` (output page), `/ws` (audio in, events out), `/ws/view` (receive-only event broadcast). Flags: `--device cpu\|mps`, `--lang gu`, `--port 8765`, `--no-translate`, `--no-filter`. |
| `gujusub/asr_engine.py` | Wraps `ai4bharat/indic-conformer-600m-multilingual` (CTC). Lock-serialized; not thread-safe otherwise. |
| `gujusub/streaming.py` | VAD-gated growing window, re-decodes every 480 ms, commits common prefix of last two hypotheses. 600 ms silence finalizes. 12 s hard cap. |
| `gujusub/fillers.py` | Two-tier filler suppression (ALWAYS / CONTEXTUAL / PHRASES). Pure text transform. |
| `gujusub/translator.py` | CT2 model load + `translate()`. `looks_untranslated()` guard. |
| `gujusub/it2_compat.py` | Shim so IndicTransToolkit imports under transformers 5.x. Must be imported before IndicTransToolkit. |
| `gujusub/static/index.html` | Mic page: device selector (persisted), AudioWorklet → PCM16 → `/ws`, shows Gujarati + English. |
| `gujusub/static/display.html` | Output page. Single file, no deps. See [broadcast-setup.md](broadcast-setup.md). |
| `tools/mic_client.py` | Terminal mic client (sounddevice → `/ws`), prints captions. Debug tool. |
| `tools/transcribe_file.py` | One-shot file transcription (`samples/test-guju.m4a`), optional `--translate`. |
| `tests/` | pytest suite (34 tests, no models loaded). Run: `.venv/bin/python -m pytest` |
| `samples/` | Short Gujarati test clip. |
| `pyproject.toml` | pytest + ruff config. `requirements.txt` runtime, `requirements-dev.txt` adds pytest/httpx/ruff. |

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
are not sent at all.

## Running

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
