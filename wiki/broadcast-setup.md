# Broadcast setup

Goal: captions from `/display` keyed over program video on an ATEM.

## Operator flow (since WS2, 2026-10-05)

1. On the operator machine open the mic page (`/`). Its **Settings** panel is
   the only place to configure captions: speech language, **Translate** switch,
   confidence sliders, and every display setting (content, font, size, colours,
   background, alignment, margins, caption bar, `aspect`). A status strip shows
   the running pipeline (ASR, translate on/off, decode ms, lag).
2. Add `http://<host>:8765/display` as the OBS browser source (no URL
   parameters needed), width/height matching your output (1920x1080).
3. Changes in the panel reach every open `/display` within about 1 s.

Settings live on the **server** and persist across restarts in
`~/.cache/gujusub/settings.json` (override with env `GUJUSUB_SETTINGS`). The
display page has no panel; it applies whatever the server pushes. Old
"Copy URL" links (`?mode=en&bg=...&size=...`) still work as a fallback while
the server is unreachable, but server settings win as soon as they arrive.

## Signal chain options

1. **OBS browser source -> OBS Virtual Camera / NDI / HDMI out -> ATEM**
   - Browser source URL `/display` (see above).
   - For an **alpha** overlay inside OBS itself, set background to transparent
     in the panel.
   - For the ATEM, output opaque black background + white text and **luma key**,
     or green (`#00ff00`) and **chroma key**.
2. **ProPresenter web/browser input** - point it at the same URL; ProPresenter
   handles the key on its own output, or pass through to the ATEM as above.
3. **Plain browser, fullscreen (`f`), HDMI to ATEM** - simplest; still works
   because the page hides its UI after 3 s.

## Recommended display settings for keying

| Setting | Value | Why |
|--------|-------|-----|
| Content | Translation (English) only | Keyed output is for the English-reading audience |
| Background | Black (`#000000`) | Luma key on ATEM; cleanest edges with white text |
| Text color | White | |
| Outline | 0 for luma key; 3–6 px black if over a bright shot in chroma mode | |
| Size / Min size | 72 / 40 at 1080p | Auto-fit handles long sentences |
| Align | Left (default) | Stable reading position while words append |
| Caption bar | On, white, 8 px | Marks the caption zone; fades with text |
| Clear after | 6 s | Stale caption disappears during long pauses |
| Bottom margin | 8–10 % | Stays inside action-safe |
| Stable text only | Off normally; On if flicker is distracting | |

The `aspect` setting (16:9, 4:3, 21:9, 1:1, 9:16) sets the stage ratio; match
it to the OBS source size.

## Audio into the transcriber

On the mic page (`/`) pick the input feeding the presenter's mic (mixer aux,
ATEM USB audio, or a dedicated mic). The status line shows which device is
live. Echo cancellation / noise suppression are currently always on in the
browser capture; if a clean line feed sounds degraded, that is the first
thing to toggle (not yet exposed in the UI — see sessions.md open threads).

## Keyboard on the display page

`f` fullscreen, double-click stage = fullscreen. There is no settings panel on
the display page (`s` does nothing); configure from the mic page.
