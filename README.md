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

See [`wiki/`](wiki/README.md) for architecture, design decisions, gotchas, and
the broadcast setup guide.

## Install

Requires Python 3.11 or newer. The installers create `.venv/`, install the
dependencies, verify the imports, and pre-download the ~1 GB of models into the
Hugging Face cache. They are safe to re-run.

### macOS

```bash
./install-osx.sh            # add --dev to also install pytest/ruff and run the tests
```

If no suitable Python is found and Homebrew is present, the script installs
`python@3.12` for you.

### Windows

1. Install Python 3.11+ from [python.org](https://www.python.org/downloads/windows/)
   and tick **Add python.exe to PATH**.
2. Install the
   [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   with the **Desktop development with C++** workload. `IndicTransToolkit`
   ships no Windows wheel, so pip compiles it from source.
3. From a Command Prompt in the repo folder:

```bat
install-windows.cmd         # add --dev to also install pytest/ruff and run the tests
```

Both scripts accept `--skip-models` to defer the model download to first run.
After downloading, the installers load both models once (`tools/prefetch_models.py --verify`)
so version problems surface at install time rather than on show day.

The ASR model is copied out of the Hugging Face cache into `~/.cache/gujusub`
(hardlinks, no extra disk) because recent onnxruntime refuses to load external
weights through the cache's symlinks. Set `GUJUSUB_MODEL_DIR` to move it.

### Manual

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python -m gujusub.server            # macOS
.venv\Scripts\python -m gujusub.server        # Windows
```

Startup takes ~10 s with cached models (longer on the first run if the
installer skipped the download). Then open:

- `http://localhost:8765/` — mic page: pick an input, click **Start mic**
- `http://localhost:8765/display` — broadcast page: add as an OBS browser source
  (press `s` for settings, `f` for fullscreen; **Copy URL** bakes settings into the link)

Flags: `--device cpu|mps` (`mps` is Apple Silicon only; Windows uses `cpu`),
`--lang gu`, `--port 8765`, `--no-translate`, `--no-filter`.

Always run from the repo root so the `gujusub` package is importable.

## Layout

```
gujusub/            Python package
  server.py         FastAPI app: /, /display, /ws (audio in), /ws/view (broadcast out)
  streaming.py      VAD-gated streaming transcriber with LocalAgreement commits
  asr_engine.py     IndicConformer wrapper
  fillers.py        Filler-word suppression
  translator.py     IndicTrans2 on CTranslate2 + pass-through guard
  it2_compat.py     Shims for IndicTransToolkit under transformers 5.x
  static/           index.html (mic page), display.html (broadcast page)
tools/              mic_client.py (terminal client), transcribe_file.py (offline check)
tests/              pytest suite
samples/            Short Gujarati test clip
wiki/               Project knowledge base (read this before changing things)
install-osx.sh      macOS installer
install-windows.cmd Windows installer
```

## Development

```bash
pip install -r requirements-dev.txt
pytest                         # 34 tests, no models loaded
ruff check .                   # lint
python tools/transcribe_file.py samples/test-guju.m4a --translate
```
