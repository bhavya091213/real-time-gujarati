# guju-sub

Live Gujarati captions + English translation for broadcast keying.

Before working here, read `wiki/README.md` — it links architecture, design
decisions, gotchas, broadcast setup, and the session log. At the end of a
session, append to `wiki/sessions.md` and update any page whose facts changed.

Layout: code in `gujusub/` (package, static pages under `gujusub/static/`),
tests in `tests/`, CLI helpers in `tools/`, sample audio in `samples/`.

Run server: `.venv/bin/python -m gujusub.server`
Tests: `.venv/bin/python -m pytest` (config in `pyproject.toml`, no models loaded)
