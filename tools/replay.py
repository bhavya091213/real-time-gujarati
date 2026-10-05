"""Replay an audio file through StreamingTranscriber as the server would.

Reports latency/throughput (decode times, translate times, lag) and the captions.
Lag uses an arrival-time model: each chunk "arrives" at its audio time; processing
is serial, so lag = processing clock - arrival time. The real server translates off
the ASR path (latest-only worker thread), so with --translate the lag reported here
is an upper bound: translation time is charged to the serial processing clock.

Run:  python tools/replay.py samples/test-guju.m4a
      python tools/replay.py samples/test-guju.m4a --loop-to 60 --no-endpoint --translate
"""

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gujusub.fillers import clean_event  # noqa: E402
from gujusub.server import PARTIAL_TRANSLATE_GAP_S, get_glossary  # noqa: E402
from gujusub.streaming import (  # noqa: E402
    SAMPLE_RATE,
    StreamingConfig,
    StreamingTranscriber,
    TranscriptEvent,
)

NO_ENDPOINT_MS = 10_000_000  # "never": utterances run to the hard cap


def _load_gu():
    from gujusub.asr_engine import ASREngine

    return ASREngine(lang="gu", device="cpu")


def _load_en():
    from gujusub.engine_parakeet import ParakeetEngine

    return ParakeetEngine()


ENGINES: dict[str, Callable[[], object]] = {"gu": _load_gu, "en": _load_en}  # WS4 adds auto


def load_audio(path: Path) -> np.ndarray:
    from transcribe_file import load_mono_16k  # sibling tool; same loading path

    return load_mono_16k(path).numpy().astype(np.float32)


def make_vad():
    from gujusub.streaming import SileroVAD

    return SileroVAD()


def loop_to(audio: np.ndarray, seconds: float) -> np.ndarray:
    """Repeat audio until it reaches `seconds` (0 = unchanged)."""
    target = int(seconds * SAMPLE_RATE)
    if target <= len(audio):
        return audio
    reps = math.ceil(target / len(audio))
    return np.tile(audio, reps)[:target]


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return float(ordered[rank - 1])


class _TimedEngine:
    """Engine proxy recording (window seconds, decode ms) for every decode."""

    def __init__(self, engine, clock):
        self._engine, self._clock = engine, clock
        self.decodes: list[tuple[float, float]] = []

    def _timed(self, fn, audio):
        t = self._clock()
        out = fn(audio)
        self.decodes.append((len(audio) / SAMPLE_RATE, (self._clock() - t) * 1000))
        return out

    def transcribe(self, audio):
        return self._timed(self._engine.transcribe, audio)

    def __getattr__(self, name):
        # Offer transcribe_detailed only when the wrapped engine has it, so the
        # transcriber's hasattr() probe sees the engine's real capabilities.
        attr = getattr(self._engine, name)
        if name == "transcribe_detailed":
            return lambda audio: self._timed(attr, audio)
        return attr


class _Translations:
    """Translate policy of server.py: finals always; partials only when the committed
    text changed AND >= PARTIAL_TRANSLATE_GAP_S of audio time passed since the last
    partial translation started."""

    def __init__(self, translator, clock):
        self.translator, self.clock = translator, clock
        self.requested: tuple[int, str] | None = None
        self.last_partial_at = -math.inf  # audio time of the last partial translation
        self.times_ms: list[float] = []

    def __call__(self, event: TranscriptEvent, now: float) -> str:
        text = f"{event.committed} {event.tail}".strip()
        if self.translator is None or not text:
            return ""
        key = (event.utterance_id, event.committed)
        if event.type != "final":
            if self.requested == key or now - self.last_partial_at < PARTIAL_TRANSLATE_GAP_S:
                return ""
            self.last_partial_at = now
        self.requested = key
        t = self.clock()
        english = get_glossary().apply(self.translator.translate(text))
        self.times_ms.append((self.clock() - t) * 1000)
        return english


def run_replay(
    audio: np.ndarray,
    engine,
    *,
    translator=None,
    config: StreamingConfig | None = None,
    vad=None,
    chunk_ms: int = 8,
    rtf: int = 0,
    filter_fillers: bool = True,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Feed `audio` in chunk_ms chunks; return {"summary": {...}, "events": [...]}."""
    timed = _TimedEngine(engine, clock)
    st = StreamingTranscriber(timed, config, vad)
    translate = _Translations(translator, clock)
    chunk = max(1, chunk_ms * SAMPLE_RATE // 1000)
    duration = len(audio) / SAMPLE_RATE
    records: list[dict] = []
    lags: list[float] = []
    busy = 0.0  # total processing seconds
    vclock = 0.0  # processing clock on the audio timeline
    wall0 = clock()

    def process(feed_fn, arrival: float) -> None:
        nonlocal vclock, busy
        if rtf:
            wait = arrival - (clock() - wall0)
            if wait > 0:
                sleep(wait)
        start = max(vclock, arrival)
        n0 = len(timed.decodes)
        t0 = clock()
        events = feed_fn()
        for ev in events:
            shown = _display(ev, filter_fillers)
            english = translate(shown, arrival)
            records.append(_record(shown, english, timed.decodes[n0:], n0, arrival))
        elapsed = clock() - t0
        busy += elapsed
        vclock = start + elapsed
        lags.append(vclock - arrival)
        for rec in records[len(records) - len(events):]:
            rec["lag_s"] = vclock - arrival
            rec["t"] = arrival

    for i in range(0, len(audio), chunk):
        part = audio[i : i + chunk]
        process(lambda part=part: st.feed(part), (i + len(part)) / SAMPLE_RATE)
    process(st.flush, duration)

    return {
        "summary": _summarize(duration, timed, translate, lags, busy, records),
        "events": records,
    }


def _display(ev: TranscriptEvent, filter_fillers: bool) -> TranscriptEvent:
    """Same order as server.filtered: fillers, then glossary for English captions."""
    committed, tail = ev.committed, ev.tail
    if filter_fillers:
        committed, tail = clean_event(committed, tail, final=ev.type == "final")
    if getattr(ev, "lang", "gu") == "en":
        glossary = get_glossary()
        committed, tail = glossary.apply(committed), glossary.apply(tail)
    if (committed, tail) == (ev.committed, ev.tail):
        return ev
    return TranscriptEvent(ev.type, ev.utterance_id, committed, tail)


def _record(ev, english, decodes, n0, arrival) -> dict:
    window_s, decode_ms = decodes[-1] if decodes else (0.0, 0.0)
    return {
        "t": arrival,
        "type": ev.type,
        "utterance_id": ev.utterance_id,
        "window_s": window_s,
        "decode_ms": decode_ms,
        "lag_s": 0.0,
        "committed": ev.committed,
        "tail": ev.tail,
        "translation": english,
    }


def _summarize(duration, timed, translate, lags, busy, records) -> dict:
    ms = [d for _, d in timed.decodes]
    finals = [f"{r['committed']} {r['tail']}".strip() for r in records if r["type"] == "final"]
    return {
        "audio_s": duration,
        "decodes": len(ms),
        "decodes_per_s": len(ms) / duration if duration else 0.0,
        "decode_p50_ms": percentile(ms, 50),
        "decode_p95_ms": percentile(ms, 95),
        "decode_max_ms": max(ms, default=0.0),
        "translate_calls": len(translate.times_ms),
        "translate_p50_ms": percentile(translate.times_ms, 50),
        "max_lag_s": max(lags, default=0.0),
        "end_lag_s": lags[-1] if lags else 0.0,
        "rtf": busy / duration if duration else 0.0,
        "final_text": " ".join(t for t in finals if t),
    }


def format_event(r: dict) -> str:
    line = (
        f"{r['t']:7.2f}s {r['type']:7s} u{r['utterance_id']:<3d} win={r['window_s']:5.1f}s "
        f"dec={r['decode_ms']:5.0f}ms lag={r['lag_s']:5.2f}s  {r['committed']} | {r['tail']}"
    )
    return f"{line}  => {r['translation']}" if r["translation"] else line


def format_summary(s: dict) -> str:
    rows = [
        ("audio", f"{s['audio_s']:.1f} s"),
        ("decodes", f"{s['decodes']} ({s['decodes_per_s']:.2f}/s)"),
        ("decode p50/p95/max", f"{s['decode_p50_ms']:.0f} / {s['decode_p95_ms']:.0f} / "
                               f"{s['decode_max_ms']:.0f} ms"),
        ("translate calls", f"{s['translate_calls']} (p50 {s['translate_p50_ms']:.0f} ms)"),
        ("max lag", f"{s['max_lag_s']:.2f} s"),
        ("lag at end", f"{s['end_lag_s']:.2f} s"),
        ("RTF", f"{s['rtf']:.3f}"),
        ("final text", s["final_text"]),
    ]
    return "\n".join(f"{k:20s} {v}" for k, v in rows)


def write_json(result: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("audio", type=Path)
    p.add_argument("--engine", default="gu", choices=sorted(ENGINES))
    p.add_argument("--loop-to", type=float, default=0.0, metavar="SECONDS")
    p.add_argument("--no-endpoint", action="store_true", help="never endpoint on silence")
    p.add_argument("--rtf", type=int, choices=(0, 1), default=0, help="1 = pace in real time")
    p.add_argument("--translate", action="store_true")
    p.add_argument("--json", type=Path, metavar="OUT.json")
    p.add_argument("--chunk-ms", type=int, default=8)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    audio = loop_to(load_audio(args.audio), args.loop_to)
    engine = ENGINES[args.engine]()
    engine.warmup()
    translator = None
    if args.translate:
        from gujusub.translator import Translator

        translator = Translator()
        translator.warmup()
    config = StreamingConfig(endpoint_ms=NO_ENDPOINT_MS) if args.no_endpoint else StreamingConfig()
    result = run_replay(
        audio, engine, translator=translator, config=config, vad=make_vad(),
        chunk_ms=args.chunk_ms, rtf=args.rtf,
    )  # fmt: skip
    if not args.quiet:
        print("\n".join(format_event(r) for r in result["events"]))
    print(format_summary(result["summary"]))
    if args.json:
        write_json(result, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
