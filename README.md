# guju-sub

Live Gujarati speech → on-screen Gujarati captions + English translation,
built for keying onto a broadcast or projector feed (OBS / ProPresenter → ATEM).
Everything runs locally on one Mac; no cloud services.

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

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m gujusub.server          # first run downloads ~1 GB of models
```

Then open:

- `http://localhost:8765/` — mic page: pick an input, click **Start mic**
- `http://localhost:8765/display` — broadcast page: add as an OBS browser source
  (press `s` for settings, `f` for fullscreen; **Copy URL** bakes settings into the link)

Flags: `--device cpu|mps`, `--lang gu`, `--port 8765`, `--no-translate`, `--no-filter`.

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
```

## Development

```bash
pip install -r requirements-dev.txt
pytest                         # 34 tests, no models loaded
ruff check .                   # lint
python tools/transcribe_file.py samples/test-guju.m4a --translate
```
