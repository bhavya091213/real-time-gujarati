"""Node-run tests for the pure JS in gujusub/static/display.html.

The page marks testable sections with `/* @test-begin NAME */ ... /* @test-end NAME */`;
each test extracts the sections it needs and runs them under node with stubs.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from gujusub.settings import Settings, display_payload

DISPLAY_HTML = Path(__file__).parents[1] / "gujusub" / "static" / "display.html"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required to run display.html JS"
)


def section(name: str) -> str:
    html = DISPLAY_HTML.read_text()
    match = re.search(
        rf"/\* @test-begin {name} \*/(.*?)/\* @test-end {name} \*/", html, re.DOTALL
    )
    assert match is not None, f"missing @test-begin/@test-end {name} markers"
    return match.group(1)


def run_node(body: str, *sections: str) -> None:
    code = "\n".join(section(s) for s in sections)
    script = f'const assert = require("node:assert/strict");\n{code}\n{body}'
    subprocess.run(["node", "-e", script], check=True)


# Fake measure: every character is 10 px wide (so " " is 10 px).
HELPERS = """
const measure = (t) => t.length * 10;
const text = (rows) => rows.map((r) => r.map((w) => w.t).join(" "));
"""


def test_explicit_empty_translation_clears_same_utterance():
    run_node(
        """
let state = { uid: null, committed: "", tail: "", translation: "" };
let fed = 0;
function scheduleRender() {}
function feedEngines() { fed++; }

handleEvent({ type: "partial", utterance_id: 7, committed: "ગુજરાતી", tail: "",
              translation: "English" });
handleEvent({ type: "partial", utterance_id: 7, committed: "ગુજરાતી text", tail: "" });
assert.equal(state.translation, "English", "an omitted translation should preserve state");

handleEvent({ type: "final", utterance_id: 7, committed: "ગુજરાતી text", tail: "",
              translation: "" });
assert.equal(state.translation, "", "an explicit empty translation should clear state");

handleEvent({ type: "partial", utterance_id: 8, committed: "new", tail: "t" });
assert.equal(state.committed, "new");
assert.equal(state.tail, "t");
assert.equal(state.translation, "", "a new utterance resets translation");
handleEvent({ type: "final", utterance_id: 8, committed: "new t", tail: "ignored" });
assert.equal(state.tail, "", "finals have no tail");
handleEvent({ type: "settings" });
assert.equal(fed, 5, "non-caption messages are ignored");
""",
        "handleEvent",
    )


def test_wrapping_uses_measure_and_pads_rows_above():
    run_node(
        HELPERS
        + """
const e = new RowEngine(3, 16, measure);
e.update(1, "aaaa bbbb cccc dddd", "", false, 0);
// width 100: "aaaa bbbb" = 90 fits, adding " cccc" = 140 does not.
assert.deepEqual(text(e.rows(100)), ["", "aaaa bbbb", "cccc dddd"]);
assert.deepEqual(e.rows(100)[0], [], "padding rows are empty and sit above");
// A wider block re-wraps the same words.
assert.deepEqual(text(e.rows(1000)), ["", "", "aaaa bbbb cccc dddd"]);
// An over-long word still gets its own row.
assert.deepEqual(text(new RowEngine(2, 16, measure).rows(10)), ["", ""]);
const big = new RowEngine(2, 16, measure);
big.update(1, "supercalifragilistic x", "", false, 0);
assert.deepEqual(text(big.rows(50)), ["supercalifragilistic", "x"]);
""",
        "words",
        "RowEngine",
    )


def test_pending_follows_final_on_same_row():
    run_node(
        HELPERS
        + """
const e = new RowEngine(2, 16, measure);
e.update(1, "one two", "three", false, 0);
const rows = e.rows(1000);
assert.deepEqual(text(rows), ["", "one two three"]);
assert.deepEqual(rows[1].map((w) => w.final), [true, true, false]);
e.update(1, "one two three four", "", true, 10);
assert.deepEqual(e.rows(1000)[1].map((w) => w.final), [true, true, true, true]);
""",
        "words",
        "RowEngine",
    )


def test_phrase_breaks_punctuation_final_and_word_cap():
    run_node(
        HELPERS
        + """
const e = new RowEngine(6, 16, measure);
e.update(1, "Hello there. How are you?\\u201d fine", "", false, 0);
assert.deepEqual(text(e.rows(1000)).filter(Boolean), ["Hello there.", "How are you?", "fine"],
                 "strong punctuation (and quote stripping) starts a new row");
const g = new RowEngine(4, 16, measure);
g.update(1, "કેમ છો।", "મજામાં", false, 0);
assert.deepEqual(text(g.rows(1000)).filter(Boolean), ["કેમ છો।", "મજામાં"], "danda breaks");
// A new utterance (after a final) always starts a new row.
e.clear();
e.update(1, "first", "", true, 0);
e.update(2, "second", "", false, 5);
assert.deepEqual(text(e.rows(1000)).filter(Boolean), ["first", "second"]);
// Word cap.
const c = new RowEngine(3, 3, measure);
c.update(1, "a b c d e f g", "", false, 0);
assert.deepEqual(text(c.rows(1000)), ["a b c", "d e f", "g"]);
// Only the last N rows are kept.
const n = new RowEngine(2, 1, measure);
n.update(1, "x y z", "", false, 0);
assert.deepEqual(text(n.rows(1000)), ["y", "z"]);
""",
        "words",
        "RowEngine",
    )


def test_pause_starts_new_row_but_rewrites_do_not():
    run_node(
        HELPERS
        + """
const e = new RowEngine(3, 16, measure);
e.update(1, "alpha beta", "", false, 0);
e.update(1, "alpha beta gamma", "", false, 500);       // no pause: same row
assert.deepEqual(text(e.rows(1000)), ["", "", "alpha beta gamma"]);
e.update(1, "alpha beta gamma delta", "", false, 1500); // 1000 ms pause
assert.deepEqual(text(e.rows(1000)), ["", "alpha beta gamma", "delta"]);
// Repeated identical text (e.g. translation-only partials) is not a change.
assert.equal(e.update(1, "alpha beta gamma delta", "", false, 9000), false);
// A rewrite (translation replacing fallback text) drops pause breaks.
const t = new RowEngine(3, 16, measure);
t.update(1, "", "ગુજરાતી લખાણ", false, 0);
t.update(1, "", "Gujarati text here", false, 2000);
assert.deepEqual(text(t.rows(1000)), ["", "", "Gujarati text here"]);
""",
        "words",
        "RowEngine",
    )


def test_word_diff_prefix3_stability():
    run_node(
        """
let n = 100;
const newId = () => ++n;
const first = diffWords([], ["the", "quick", "brown"], newId);
assert.deepEqual(first.map((w) => w.fresh), [true, true, true]);
// "brow" -> "browne" (same first 3 chars) keeps its id; "fox" is new.
const second = diffWords(first, ["The", "quick", "browne", "fox"], newId);
assert.deepEqual(second.map((w) => w.id).slice(0, 3), first.map((w) => w.id));
assert.deepEqual(second.map((w) => w.fresh), [false, false, false, true]);
assert.equal(second[2].t, "browne", "text updates in place");
// Dropping the first word (row scrolled off) keeps the rest stable.
const third = diffWords(second, ["quick", "browne", "fox"], newId);
assert.deepEqual(third.map((w) => w.id), second.slice(1).map((w) => w.id));
// A revised word that differs in its first 3 chars is fresh.
const fourth = diffWords(third, ["quick", "green", "fox"], newId);
assert.deepEqual(fourth.map((w) => w.fresh), [false, true, false]);
assert.equal(new Set(fourth.map((w) => w.id)).size, 3, "ids stay unique");
assert.deepEqual(normalizeWords(' "Hi"   don\\'t  ‘x’ '), ["Hi", "don't", "x"]);
""",
        "words",
    )


def test_page_defaults_equal_server_display_payload():
    expected = json.dumps(display_payload(Settings()))
    run_node(
        f"""
const expected = {expected};
assert.equal(SCHEMA, expected.schema);
const {{ schema, ...keys }} = expected;
assert.deepEqual(DEFAULTS, keys);
""",
        "DEFAULTS",
    )
