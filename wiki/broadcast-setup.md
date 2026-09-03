# Broadcast setup

Goal: captions from `/display` keyed over program video on an ATEM.

## Signal chain options

1. **OBS browser source → OBS Virtual Camera / NDI / HDMI out → ATEM**
   - Add a Browser source, URL from "Copy URL" (e.g.
     `http://localhost:8765/display?mode=en&bg=%23000000&size=72&…`),
     width/height matching your output (1920×1080).
   - For an **alpha** overlay inside OBS itself, set `bg=transparent`.
   - For the ATEM, output opaque black background + white text and **luma key**,
     or green (`bg=%2300ff00`) and **chroma key**.
2. **ProPresenter web/browser input** — point it at the same URL; ProPresenter
   handles the key on its own output, or pass through to the ATEM as above.
3. **Plain browser, fullscreen (`f`), HDMI to ATEM** — simplest; still works
   because the page hides its UI after 3 s.

## Recommended display settings for keying

| Setting | Value | Why |
|--------|-------|-----|
| Content | English only | Keyed output is for the English-reading audience |
| Background | Black (`#000000`) | Luma key on ATEM; cleanest edges with white text |
| Text color | White | |
| Outline | 0 for luma key; 3–6 px black if over a bright shot in chroma mode | |
| Size / Min size | 72 / 40 at 1080p | Auto-fit handles long sentences |
| Align | Left (default) | Stable reading position while words append |
| Caption bar | On, white, 8 px | Marks the caption zone; fades with text |
| Clear after | 6 s | Stale caption disappears during long pauses |
| Bottom margin | 8–10 % | Stays inside action-safe |
| Stable text only | Off normally; On if flicker is distracting | |

Use **Copy URL** once dialed in and paste that into OBS/ProPresenter so the
configuration is baked into the URL (no localStorage dependence on the
capture machine).

## Audio into the transcriber

On the mic page (`/`) pick the input feeding the presenter's mic (mixer aux,
ATEM USB audio, or a dedicated mic). The status line shows which device is
live. Echo cancellation / noise suppression are currently always on in the
browser capture; if a clean line feed sounds degraded, that is the first
thing to toggle (not yet exposed in the UI — see sessions.md open threads).

## Keyboard on the display page

`s` settings panel · `f` fullscreen · `Esc` close panel · double-click stage =
fullscreen. Shortcuts are ignored while typing in a panel field.
