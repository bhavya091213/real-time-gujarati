# Broadcast setup

Goal: captions from `/display` keyed over program video on an ATEM.

## Operator flow (since WS2, 2026-10-05)

1. On the operator machine open the mic page (`/`). Its **Settings** panel is
   the only place to configure captions: speech language, **Translate** switch,
   confidence sliders, and every display setting (content, font, size, colours,
   background, position, rows, backing, `aspect`). A status strip shows
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

## The output page (`/display`, schema 4, since 2026-10-16)

Modelled on sabha-stt.vercel.app. Two blocks, each anchored at its
bottom-centre: the **translation** (English, Inter) block's bottom edge sits at
`y` % of the stage height, and the **primary** (ASR; Gujarati in Noto Sans
Gujarati, or English when the utterance is `lang:"en"`) block sits `gap` %
higher (pushed up further if it would overlap). Each block shows its last
`rows_primary` / `rows_translation` rows; text grows upward from the bottom
edge and old rows drop off the top. A new row starts for each utterance, after
strong punctuation (`. ! ? … । ॥`), after a 900 ms pause in the text, or at the
per-phrase word cap (English 16 / Gujarati 14). Words are wrapped with canvas
measurement, so the browser never re-wraps them. New words fade in over
100 ms (`wordfade`; off under reduced motion). After `clear` s without new
text the captions fade (700 ms) and are cleared.

Content mode `translation` shows only the English block, with the ASR text in
its place until a translation arrives; `primary` only the ASR block; `both`
both.

Backing (`panel`): `bottom` = full-width bottom panel, at least `panel_h` %
tall (it grows to cover the lowest block), `lines` = one chip per row, `none`.
Opacity `panel_opacity`. `shadow` = soft character shadow; `ow`/`oc` = text
outline. `size` is px at a 1080-px-high stage (it scales with the stage) and
`scale` multiplies it. Webfonts load from Google Fonts when online; offline
the system fallbacks are used and the layout is re-measured, so it still works.

Legacy keys `minsize`, `bottom`, `bar`, `barw`, `barc` are still accepted and
ignored (a `?bottom=` URL parameter maps to `y = 100 - bottom`).

`/display?preview=1` shows placeholder text while idle, which is handy for
placing the blocks. Without it the page stays blank when idle, as an OBS
source should.

## Recommended display settings for keying

| Setting | Value | Why |
|--------|-------|-----|
| Content | Translation (English) only | Keyed output is for the English-reading audience |
| Background | Black (`#000000`) | Luma key on ATEM; cleanest edges with white text |
| Backing | Bottom panel over transparent / chroma; None for luma key | A black panel keys out on luma anyway |
| Text color | White | |
| Outline / shadow | 0 for luma key; soft shadow or 2–4 px outline over bright video | |
| Size / scale | 56 / 1 at 1080p (default) | ≈ 5.2 % of stage height |
| Rows | Gujarati 2, English 3 | |
| Align | Center (default) | |
| Bottom edge (`y`) | 88 (default) | Stays inside action-safe |
| Clear after | 20 s | Stale caption disappears during long pauses |
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

`F` fullscreen (double-click the stage too), `Esc` leave fullscreen, `G` cycle
content (both → translation → primary), `C` clear captions, `+`/`-` text scale
±0.1 (0.6–1.8). `G` and `+`/`-` are local and last until the server pushes the
next settings change. The controls and the status dot (grey idle, amber
connecting, green live, red error) hide after 3 s without mouse movement.
There is no settings panel on the display page; configure from the mic page.
