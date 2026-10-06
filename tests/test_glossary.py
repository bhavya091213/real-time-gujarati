import json
import os
import time
from pathlib import Path

import pytest

from gujusub.glossary import ENV_VAR, Glossary, load_glossary

DATA = Path(__file__).resolve().parents[1] / "gujusub" / "data" / "glossary_baps.json"

ENTRIES = [
    {"canonical": "Pramukh Swami Maharaj", "variants": ["pramukh swami maharaj", "pramuk swami"]},
    {"canonical": "Pramukh Swami Maharaj Ni Jay", "variants": ["pramukh swami maharaj ni jay"]},
    {"canonical": "Swami", "variants": ["swami"]},
    {"canonical": "Swami Ni Vato", "variants": ["swami ni vato"]},
    {"canonical": "Prasad", "variants": ["prasad", "prashad"]},
    {"canonical": "Beta", "variants": ["beta", "betaa"], "safe": False},
    {"canonical": "Ben", "variants": ["behen", "ben"], "safe": False},
]


@pytest.fixture
def g():
    return Glossary(ENTRIES)


def test_longest_match_wins(g):
    assert g.apply("pramukh swami maharaj ni jay") == "Pramukh Swami Maharaj Ni Jay"
    assert g.apply("pramukh swami maharaj ki") == "Pramukh Swami Maharaj ki"


def test_overlap_prefers_longer_then_continues(g):
    assert g.apply("swami ni vato swami") == "Swami Ni Vato Swami"


def test_case_insensitive_and_canonical_casing(g):
    assert g.apply("PRAMUK SWAMI spoke") == "Pramukh Swami Maharaj spoke"
    assert g.apply("prasad") == "Prasad"


def test_punctuation_attached_to_edges(g):
    assert g.apply("take prasad.") == "take Prasad."
    assert g.apply('"prashad," he said') == '"Prasad," he said'
    assert g.apply("(pramuk swami!)") == "(Pramukh Swami Maharaj!)"


def test_interior_punctuation_breaks_phrase(g):
    assert g.apply("pramuk, swami") == "pramuk, Swami"


def test_unsafe_entry_untouched_when_only_case_differs(g):
    assert g.apply("beta version") == "beta version"
    assert g.apply("Beta version") == "Beta version"
    assert g.apply("ben said") == "ben said"


def test_unsafe_entry_rewritten_when_spelling_differs(g):
    assert g.apply("betaa came") == "Beta came"
    assert g.apply("my behen.") == "my Ben."


def test_gujarati_tokens_untouched(g):
    assert g.apply("પ્રસાદ prasad સ્વામી") == "પ્રસાદ Prasad સ્વામી"
    assert g.apply("prasadજી") == "prasadજી"


def test_no_match_returns_input_unchanged(g):
    assert g.apply("hello   world") == "hello   world"
    assert g.apply("") == ""


def test_idempotent(g):
    for s in ["pramuk swami prasad.", "betaa behen", "pramukh swami maharaj ni jay swami"]:
        once = g.apply(s)
        assert g.apply(once) == once


def test_packaged_glossary_idempotent_on_every_variant():
    gl = load_glossary()
    entries = json.loads(DATA.read_text(encoding="utf-8"))["entries"]
    assert len(entries) > 500
    for e in entries:
        for v in e["variants"]:
            once = gl.apply(f"we said {v} today.")
            assert gl.apply(once) == once, v


def test_packaged_glossary_rewrites_known_term():
    assert "Swaminarayan" in load_glossary().apply("jai swaminarayan.")


def test_speed_50_word_line():
    gl = load_glossary()
    base = "we offered prasad to pramukh swami maharaj and the beta came today ".split()
    line = " ".join((base * 5)[:50])
    gl.apply(line)
    t0 = time.perf_counter()
    for _ in range(100):
        gl.apply(line)
    assert (time.perf_counter() - t0) * 1000 / 100 < 5


def _write(path, entries):
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")


@pytest.mark.parametrize(
    "entries",
    [
        {},
        [None],
        [{"canonical": 123, "variants": ["foo"]}],
        [{"canonical": " ", "variants": ["foo"]}],
        [{"canonical": "Foo", "variants": "foo"}],
        [{"canonical": "Foo", "variants": [123]}],
        [{"canonical": "Foo", "variants": [" "]}],
        [{"canonical": "Foo", "variants": ["foo"], "safe": "false"}],
    ],
    ids=[
        "entries-not-list",
        "entry-not-object",
        "canonical-not-string",
        "canonical-empty",
        "variants-not-list",
        "variant-not-string",
        "variant-empty",
        "safe-not-boolean",
    ],
)
def test_malformed_override_schema_falls_back_to_packaged(tmp_path, entries):
    f = tmp_path / "g.json"
    _write(f, entries)

    gl = load_glossary(f)

    assert len(gl) > 500
    assert "Swaminarayan" in gl.apply("jai swaminarayan.")


@pytest.mark.parametrize(
    "entries",
    [
        [{"canonical": 123, "variants": ["bar"]}],
        [{"canonical": "Bar", "variants": [123]}],
    ],
    ids=["canonical-not-string", "variant-not-string"],
)
def test_malformed_override_reload_keeps_previous(tmp_path, entries):
    f = tmp_path / "g.json"
    _write(f, [{"canonical": "Foo", "variants": ["fu"]}])
    gl = load_glossary(f)
    _write(f, entries)
    os.utime(f, (time.time() + 10, time.time() + 10))
    gl._checked -= 3

    gl.maybe_reload()

    assert gl.apply("fu") == "Foo"


def test_env_override_and_hot_reload(tmp_path, monkeypatch):
    f = tmp_path / "g.json"
    _write(f, [{"canonical": "Foo", "variants": ["fu"]}])
    monkeypatch.setenv(ENV_VAR, str(f))
    gl = load_glossary()
    assert gl.apply("fu") == "Foo"

    _write(f, [{"canonical": "Bar", "variants": ["bar"]}])
    os.utime(f, (time.time() + 10, time.time() + 10))
    gl.maybe_reload()
    assert gl.apply("bar") == "bar"  # throttled: checked < 2 s ago
    gl._checked -= 3
    gl.maybe_reload()
    assert gl.apply("bar") == "Bar"
    assert gl.apply("fu") == "fu"


def test_missing_override_falls_back_to_packaged(tmp_path, monkeypatch, caplog):
    f = tmp_path / "nope.json"
    monkeypatch.setenv(ENV_VAR, str(f))
    gl = load_glossary()
    assert len(gl) > 500
    assert "unusable" in caplog.text
    _write(f, [{"canonical": "Zed", "variants": ["zed"]}])  # created later: hot-loads
    gl._checked -= 3
    gl.maybe_reload()
    assert gl.apply("zed") == "Zed"


def test_broken_reload_keeps_previous(tmp_path):
    f = tmp_path / "g.json"
    _write(f, [{"canonical": "Foo", "variants": ["fu"]}])
    gl = load_glossary(f)
    f.write_text("{not json", encoding="utf-8")
    os.utime(f, (time.time() + 10, time.time() + 10))
    gl._checked -= 3
    gl.maybe_reload()
    assert gl.apply("fu") == "Foo"
