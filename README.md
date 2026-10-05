# guju-sub

Live Gujarati speech → on-screen Gujarati captions + English translation,
built for keying onto a broadcast or projector feed (OBS / ProPresenter → ATEM).
Everything runs locally on one machine; no cloud services.

## How it works

1. A browser mic page (or `tools/mic_client.py`) streams 16 kHz PCM to the server.
2. Silero VAD gates speech; the utterance is re-decoded every ~0.5 s with
   [IndicConformer](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
   and stable text is committed with LocalAgreement-2.
3. Filler words (um, uh, અં, એટલે…) are stripped by position-aware rules.
4. [IndicTrans2](https://huggingface.co/adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M)
   (CTranslate2, int8) translates to English.
5. Events are broadcast to a self-contained display page you drop into OBS.

English speech is recognised with NVIDIA
[Parakeet-TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
(CC-BY-4.0, NVIDIA; int8 ONNX export by
[istupakov](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx), run via
[onnx-asr](https://github.com/istupakov/onnx-asr)).

See [`wiki/`](wiki/README.md) for architecture, design decisions, gotchas, and
the broadcast setup guide.

## Quick start

Requires Python 3.11 or newer (the installer finds it, and on macOS installs
`python@3.12` with Homebrew if needed).

```bash
git clone <repo-url> guju-sub && cd guju-sub
./start.sh --open            # macOS
start.cmd --open             :: Windows (Command Prompt)
```

On first run the start script calls the installer: it creates `.venv/`,
installs the dependencies, pre-downloads the ~1 GB of models, and loads them
once to verify. It then starts the server and, once it is ready, opens the mic
page (`/`) and the broadcast page (`/display`) in your browser. Later runs skip
the install and start in ~10 s. Ctrl-C stops the server. The start script stops
anything already listening on the port first (use `--keep-port` to skip).

Windows prerequisites: Python 3.11+ from
[python.org](https://www.python.org/downloads/windows/) (tick **Add python.exe
to PATH**) and the
[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
("Desktop development with C++"), because `IndicTransToolkit` ships no Windows
wheel and pip compiles it.

The installers can also be run directly (`./install-osx.sh` /
`install-windows.cmd`; `--dev` adds pytest/ruff and runs the tests,
`--skip-models` defers the download). They are safe to re-run.

Manual install:

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m gujusub.server
```

## Flags

`start.sh` / `start.cmd` take their own flags and pass everything else to the
server unchanged. Run from any directory; relative paths (e.g. `--replay FILE`)
resolve from the repo root.

| Flag | Where | Default | Meaning |
|---|---|---|---|
| `--setup` | start script | off | Run the installer first even if `.venv` exists |
| `--skip-models` | start script | off | Pass to the installer (defer the model download) |
| `--open` | start script | off | Open the mic page and `/display` in the browser once the server is ready |
| `--display-only` | start script | off | Open only `/display` (e.g. a check on the OBS machine) |
| `--threads N` | start script | perf cores | Set `GUJUSUB_THREADS=N` |
| `--model-dir PATH` | start script | `~/.cache/gujusub` | Set `GUJUSUB_MODEL_DIR=PATH` |
| `--replay FILE [flags]` | start script | | Run `tools/replay.py FILE ...` instead of the server |
| `--verify` | start script | | Load both models once and exit |
| `--keep-port` | start script | off | Do not stop an existing listener on the server port before starting |
| `--test` | start script | | Run pytest and exit |
| `--help` | start script | | Usage plus the live server flag list |
| `--device cpu\|mps` | server | `cpu` | ASR device; `mps` is Apple Silicon only (Windows uses `cpu`) |
| `--lang CODE` | server | `gu` | ASR language |
| `--port N` | server | `8765` | HTTP/WebSocket port |
| `--no-translate` | server | off | Skip Gujarati to English translation |
| `--no-filter` | server | off | Show filler words instead of hiding them |

## Configuration

The settings panel on the mic page (`/`) is the way to configure speech language, translation, and every caption style. Settings are stored on the server and pushed live to `/display`. Content modes are `primary` (ASR text), `translation`, and `both`.

| Environment variable | Default | Meaning |
|---|---|---|
| `GUJUSUB_THREADS` | performance-core count (min 2) | ONNX Runtime / CTranslate2 thread count |
| `GUJUSUB_SETTINGS` | `~/.cache/gujusub/settings.json` | Where the server persists panel settings |
| `GUJUSUB_MODEL_DIR` | `~/.cache/gujusub` | Where the symlink-free ASR model copy lives |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face hub cache (downloaded models) |

- **Models.** About 1 GB total: the IndicConformer ASR model and the
  IndicTrans2 CTranslate2 translator, downloaded once into the hub cache. The
  ASR model is also hardlinked into `GUJUSUB_MODEL_DIR` (no extra disk) because
  recent onnxruntime refuses to load external weights through the cache's
  symlinks.
- **Audio input.** Open the mic page, choose the input in the device selector,
  click **Start mic**.
- **Display page** (`/display`, add as an OBS browser source): output only,
  no panel; `f` for fullscreen. Old URL-parameter links still work as a
  fallback but server settings win. See
  [`wiki/broadcast-setup.md`](wiki/broadcast-setup.md) for the OBS recipe.
- **Performance.** Threads default to the performance cores. On a Mac mini
  running OBS on the same machine, lower `--threads` if OBS drops frames.
  Measure latency with
  `./start.sh --replay samples/test-guju.m4a --loop-to 60 --no-endpoint --translate`
  (see `tools/replay.py --help`).

## Layout

```
gujusub/            Python package
  server.py         FastAPI app: /, /display, /ws (audio in), /ws/view (broadcast out)
  streaming.py      VAD-gated streaming transcriber with LocalAgreement commits
  asr_engine.py     IndicConformer wrapper
  engine.py         ASR engine types/interface
  threads.py        Thread-count policy (GUJUSUB_THREADS)
  fillers.py        Filler-word suppression
  translator.py     IndicTrans2 on CTranslate2 + pass-through guard
  it2_compat.py     Shims for IndicTransToolkit under transformers 5.x
  static/           index.html (mic page), display.html (broadcast page)
tools/              mic_client.py (terminal client), transcribe_file.py (offline check), replay.py (latency benchmark)
tests/              pytest suite
samples/            Short Gujarati test clip
wiki/               Project knowledge base (read this before changing things)
start.sh            macOS launcher (installs on first run)
start.cmd           Windows launcher
install-osx.sh      macOS installer
install-windows.cmd Windows installer
```

## Development

```bash
pip install -r requirements-dev.txt
pytest                         # 128 tests, no models loaded
ruff check .                   # lint
python tools/transcribe_file.py samples/test-guju.m4a --translate
python tools/replay.py samples/test-guju.m4a --loop-to 60 --no-endpoint --translate   # latency/lag benchmark
```
