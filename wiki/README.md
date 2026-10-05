# guju-sub LLM Wiki

Knowledge base for humans and LLM agents working on this project. Read this
page first, then follow links as needed. Keep pages short and factual; update
them when behaviour changes, and append to `sessions.md` at the end of a
working session.

## What this project is

Live Gujarati speech → Gujarati captions + English translation, for keying
onto a broadcast/projector feed (OBS / ProPresenter → ATEM). Local models,
runs entirely on one Mac, no cloud.

## Pages

| Page | What it covers |
|------|----------------|
| [architecture.md](architecture.md) | Pipeline, files, WebSocket protocol, how to run |
| [decisions.md](decisions.md) | Why things are the way they are (streaming algorithm, filler rules, display choices) |
| [gotchas.md](gotchas.md) | Things that bit us: Parakeet quirks, saved confidence settings, Gujarati regex, translator pass-through, saved-settings vs defaults |
| [broadcast-setup.md](broadcast-setup.md) | Getting captions into OBS/ProPresenter and keyed on an ATEM |
| [sessions.md](sessions.md) | Chronological log of work sessions and open threads |

## Conventions for editing this wiki

- One fact in one place; link rather than duplicate.
- Record *why*, not just *what* — the code already shows what.
- Dates absolute (YYYY-MM-DD), never "yesterday".
- When something in `gotchas.md` gets fixed for good, move it to `decisions.md`
  or delete it.
