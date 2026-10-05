"""Calibrate confidence trimming (conf_word_min, conf_utt_min) on real and garbage audio.

For each input and engine: replay the full streaming pipeline (tools/replay.run_replay)
with trimming OFF and dump every final decode's words + confidences, then sweep
word_min x utt_min through the same pipeline and report, per cell:
  kept% = final words kept vs OFF, drop% = utterances (with a final at OFF) that lost
  their final, ev = events emitted, and WER vs a `<audio>.txt` reference if present
  (else "chg" = WER of the trimmed final text against the OFF final text).
Decodes are cached by window audio, so the sweep costs little beyond the OFF run.

--synthetic adds three garbage inputs (written as WAVs to --scratch):
  noise   5 s white noise at -30 dBFS RMS
  tremolo 5 s 440 Hz tone (+2 harmonics) with 5 Hz tremolo, -12 dBFS peak
  noisy   first audio input + white noise at 0 dB SNR

Run:  python tools/calibrate_confidence.py samples/test-guju.m4a --engine gu --engine en \\
          --synthetic --scratch /tmp/x --out calibration.md --recommend 0.3,0.5
"""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay import ENGINES, load_audio, make_vad, run_replay  # noqa: E402

from gujusub.confidence import apply_confidence  # noqa: E402
from gujusub.streaming import SAMPLE_RATE  # noqa: E402

WORD_MINS = tuple(round(0.1 * i, 1) for i in range(10))
UTT_MINS = (0.0, 0.3, 0.5, 0.7)
SYNTH_S = 5.0


# --- metrics ------------------------------------------------------------------


def wer(ref: list[str], hyp: list[str]) -> float:
    """Word error rate = word edit distance / len(ref) (len(hyp) if ref is empty)."""
    if not ref:
        return float(len(hyp) > 0)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def finals_by_utt(events: list[dict]) -> dict[int, str]:
    return {
        e["utterance_id"]: f"{e['committed']} {e['tail']}".strip()
        for e in events
        if e["type"] == "final" and f"{e['committed']} {e['tail']}".strip()
    }


def cell_metrics(off: dict, run: dict, reference: list[str] | None) -> dict:
    base, got = finals_by_utt(off["events"]), finals_by_utt(run["events"])
    base_words = sum(len(t.split()) for t in base.values())
    got_words = sum(len(t.split()) for t in got.values())
    hyp = run["summary"]["final_text"].split()
    return {
        "kept": got_words / base_words if base_words else 1.0,
        "dropped": sum(u not in got for u in base) / len(base) if base else 0.0,
        "events": len(run["events"]),
        "wer": wer(reference if reference is not None
                   else off["summary"]["final_text"].split(), hyp),
    }


# --- engine/VAD probes ------------------------------------------------------------


class _CountingVAD:
    """VAD proxy counting frames and resets (a reset precedes each final decode)."""

    def __init__(self, vad):
        self._vad, self.frames, self.reset_at = vad, 0, -1

    def speech_prob(self, frame):
        self.frames += 1
        return self._vad.speech_prob(frame)

    def reset(self):
        self.reset_at = self.frames
        self._vad.reset()


class CachingEngine:
    """Engine proxy: caches transcribe_detailed by window audio and, when `probe`
    is set, logs the words of each final decode (first decode after a VAD reset
    with no new frames in between)."""

    def __init__(self, engine):
        self._engine = engine
        self.lang = getattr(engine, "lang", "gu")
        self._cache: dict[bytes, object] = {}
        self.probe: _CountingVAD | None = None
        self.final_words: list[list[tuple[str, float]]] = []
        self._logged_reset = -1

    def transcribe_detailed(self, audio):
        key = hashlib.sha1(audio.tobytes()).digest()
        if key not in self._cache:
            self._cache[key] = self._engine.transcribe_detailed(audio)
        out = self._cache[key]
        p = self.probe
        if p is not None and p.reset_at == p.frames and p.reset_at != self._logged_reset:
            self._logged_reset = p.reset_at
            self.final_words.append([(w.text, round(float(w.conf), 4)) for w in out.words])
        return out

    def transcribe(self, audio):
        return self.transcribe_detailed(audio).text

    def warmup(self):
        self._engine.warmup()


def replay(audio, engine: CachingEngine, thresholds, probe=False) -> dict:
    vad = _CountingVAD(make_vad())
    engine.probe = vad if probe else None
    try:
        return run_replay(audio, engine, vad=vad, thresholds=thresholds)
    finally:
        engine.probe = None


# --- synthetic garbage ------------------------------------------------------------


def synthetic(base: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(SYNTH_S * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    noise = rng.standard_normal(n) * 10 ** (-30 / 20)
    tone = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((440, 1.0), (880, 0.4), (1320, 0.2)))
    tone = tone * (0.6 + 0.4 * np.sin(2 * np.pi * 5 * t))
    tone = tone / np.abs(tone).max() * 10 ** (-12 / 20)
    rms = float(np.sqrt(np.mean(base**2)))
    noisy = base + rng.standard_normal(len(base)) * rms
    return {name: a.astype(np.float32) for name, a in
            (("noise", noise), ("tremolo", tone), ("noisy", noisy))}


# --- report ---------------------------------------------------------------------


def conf_stats(final_words) -> str:
    confs = [c for words in final_words for _, c in words]
    if not confs:
        return "no words"
    q = statistics.quantiles(confs, n=10) if len(confs) > 1 else [confs[0]] * 9
    return (f"{len(confs)} words, min {min(confs):.2f} p10 {q[0]:.2f} "
            f"p50 {statistics.median(confs):.2f} max {max(confs):.2f}")


def direct_decode(name, audio, engine, recommend) -> list[str]:
    """Decode the whole clip with VAD bypassed: does the gate itself reject it?"""
    tr = engine.transcribe_detailed(audio)
    words = " ".join(f"{w.text}({w.conf:.2f})" for w in tr.words) or "(no words)"
    line = f"VAD bypassed, whole-clip decode: {words}"
    if recommend:
        kept = apply_confidence(tr, *recommend).text
        line += f" -> at {recommend}: `{kept or '(dropped)'}`"
    return [line[:600], ""]


def sweep_table(cells: dict) -> list[str]:
    head = "| word_min | " + " | ".join(f"utt {u}" for u in UTT_MINS) + " |"
    rows = [head, "|" + "---|" * (len(UTT_MINS) + 1)]
    for wm in WORD_MINS:
        vals = []
        for um in UTT_MINS:
            c = cells[(wm, um)]
            vals.append(f"{c['kept']:.0%} / {c['dropped']:.0%} / {c['events']} / "
                        f"{c['wer']:.2f}")
        rows.append(f"| {wm} | " + " | ".join(vals) + " |")
    return rows


def calibrate(name, audio, engine, reference, recommend) -> tuple[list[str], dict]:
    off = replay(audio, engine, (0.0, 0.0), probe=True)
    finals = off["summary"]["final_text"]
    cells = {(wm, um): cell_metrics(off, replay(audio, engine, (wm, um)), reference)
             for wm in WORD_MINS for um in UTT_MINS}
    metric = "WER vs ref" if reference is not None else "chg vs OFF"
    lines = [f"### {name} — engine {engine.lang}", "",
             f"OFF: {len(off['events'])} events; final: `{finals or '(none)'}`  ",
             f"final-decode word conf: {conf_stats(engine.final_words)}  ",
             "words: " + " ".join(f"{t}({c:.2f})" for ws in engine.final_words
                                  for t, c in ws)[:600], "",
             f"cell = kept% / utt dropped% / events / {metric}", ""]
    lines += sweep_table(cells)
    if name.startswith("synth-"):
        lines += [""] + direct_decode(name, audio, engine, recommend)
    rec = cells.get(recommend) if recommend else None
    summary = {"input": name, "engine": engine.lang, "off_events": len(off["events"]),
               "off_text": finals, "final_words": engine.final_words,
               "recommended": rec}
    if rec is not None:
        run = replay(audio, engine, recommend)
        summary["recommended_text"] = run["summary"]["final_text"]
        lines += ["", f"At recommended {recommend}: final `"
                  f"{summary['recommended_text'] or '(none)'}`"]
    return lines + [""], summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("audio", type=Path, nargs="+")
    p.add_argument("--engine", action="append", choices=sorted(ENGINES))
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--scratch", type=Path, default=Path("."))
    p.add_argument("--out", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument("--recommend", type=str, default="", metavar="WORD_MIN,UTT_MIN")
    args = p.parse_args(argv)
    recommend = tuple(float(x) for x in args.recommend.split(",")) if args.recommend else None

    inputs = {path.name: load_audio(path) for path in args.audio}
    refs = {path.name: path.with_suffix(".txt") for path in args.audio}
    if args.synthetic:
        args.scratch.mkdir(parents=True, exist_ok=True)
        import soundfile as sf

        for name, a in synthetic(next(iter(inputs.values()))).items():
            sf.write(args.scratch / f"synth-{name}.wav", a, SAMPLE_RATE)
            inputs[f"synth-{name}"] = a

    lines, summaries = ["# Confidence calibration", ""], []
    for lang in args.engine or ["gu"]:
        raw = ENGINES[lang]()
        raw.warmup()
        for name, audio in inputs.items():
            ref_path = refs.get(name)
            reference = (ref_path.read_text().split()
                         if ref_path is not None and ref_path.exists() else None)
            section, summary = calibrate(name, audio, CachingEngine(raw), reference, recommend)
            print("\n".join(section), flush=True)
            lines += section
            summaries.append(summary)
    if args.out:
        args.out.write_text("\n".join(lines))
    if args.json:
        args.json.write_text(json.dumps(summaries, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
