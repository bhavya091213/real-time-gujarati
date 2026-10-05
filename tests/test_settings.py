import dataclasses
import json

import pytest

from gujusub import settings as S
from gujusub.settings import Settings, SettingsStore, validate

DISPLAY_DEFAULTS = {  # copied from static/display.html DEFAULTS
    "aspect": "16:9", "mode": "translation", "stable": False, "clear": 6,
    "font": 'system-ui, "Noto Sans Gujarati", sans-serif', "size": 64,
    "minsize": 36, "weight": 700, "lh": 1.3, "ls": 0, "tt": "none",
    "color": "#ffffff", "ow": 0, "oc": "#000000", "shadow": False,
    "bg": "#000000", "align": "left", "pad": 5, "bottom": 8, "bar": True,
    "barw": 8, "barc": "#ffffff",
}

NUMERIC_LIMITS = [
    ("clear", 0, 600),
    ("size", 24, 160),
    ("minsize", 16, 160),
    ("weight", 1, 1000),
    ("lh", 1.0, 2.2),
    ("ls", -2, 20),
    ("ow", 0, 12),
    ("pad", 0, 25),
    ("bottom", 0, 50),
    ("barw", 0, 24),
]


def test_defaults_match_display_page():
    s = Settings()
    assert len(DISPLAY_DEFAULTS) == 22
    assert {k: getattr(s, k) for k in DISPLAY_DEFAULTS} == DISPLAY_DEFAULTS
    assert (s.asr_mode, s.translate, s.conf_word_min, s.conf_utt_min, s.schema) == (
        "gu", True, 0.5, 0.7, 3)


def test_payloads():
    s = Settings()
    disp = S.display_payload(s)
    assert disp == {**DISPLAY_DEFAULTS, "schema": 3}
    ctrl = S.control_payload(s)
    assert ctrl["asr_mode"] == "gu" and ctrl["translate"] is True
    assert set(ctrl) == set(disp) | {"asr_mode", "translate", "conf_word_min",
                                     "conf_utt_min"}


@pytest.mark.parametrize("partial", [
    {"size": 48, "color": "rgba(0, 0, 0, 0.5)", "bg": "transparent"},
    {"lh": 1.5, "ls": -1, "tt": "uppercase", "align": "center", "weight": 900},
    {"asr_mode": "auto", "translate": False, "conf_word_min": 0.4},
    {"aspect": "9:16", "mode": "both", "clear": 0, "barc": "#abc"},
    {},
])
def test_validate_good_partial_round_trips(partial):
    assert validate(partial) == partial


@pytest.mark.parametrize(("key", "low", "high"), NUMERIC_LIMITS)
def test_validate_accepts_numeric_boundaries(key, low, high):
    assert validate({key: low}) == {key: low}
    assert validate({key: high}) == {key: high}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        *((key, low - 0.01) for key, low, _ in NUMERIC_LIMITS),
        *((key, high + 0.01) for key, _, high in NUMERIC_LIMITS),
        ("clear", 1e100),
        ("size", 1e100),
        ("ls", -1e100),
        ("ow", 1e100),
    ],
)
def test_validate_rejects_values_outside_numeric_boundaries(key, value):
    with pytest.raises(ValueError):
        validate({key: value})


@pytest.mark.parametrize("partial", [
    {"nope": 1},                      # unknown key
    {"schema": 4},                    # not settable
    {"size": "64"},                   # wrong type
    {"size": True},                   # bool is not a number
    {"stable": 1},                    # int is not a bool
    {"size": 0},                      # must be positive
    {"ow": -1},                       # must be >= 0
    {"size": float("nan")},           # not finite
    {"mode": "fr"},                   # bad enum
    {"asr_mode": "hi"},
    {"align": "justify"},
    {"color": "red; background:url(x)"},  # not a colour
    {"font": "x</style>"},            # CSS/HTML breakout chars
    {"conf_utt_min": 1.5},            # out of [0, 1]
    {"aspect": "wide"},
])
def test_validate_rejects(partial):
    with pytest.raises(ValueError):
        validate(partial)


def test_validate_rejects_non_dict():
    with pytest.raises(ValueError):
        validate(["size", 1])


def test_with_updates_is_immutable():
    s = Settings()
    t = s.with_updates({"size": 40})
    assert (s.size, t.size) == (64, 40)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.size = 1
    with pytest.raises(ValueError):
        s.with_updates({"size": -3})


def test_store_missing_file_gives_defaults(tmp_path):
    store = SettingsStore(tmp_path / "none" / "settings.json")
    assert store.load() == Settings()
    assert store.get() == Settings()


def test_store_corrupt_file_gives_defaults(tmp_path, caplog):
    p = tmp_path / "settings.json"
    p.write_text("{not json")
    assert SettingsStore(p).load() == Settings()
    assert "settings" in caplog.text.lower()


def test_store_skips_bad_keys_keeps_good(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"size": 50, "mode": "klingon", "zzz": 1, "schema": 2}))
    s = SettingsStore(p).load()
    assert s.size == 50 and s.mode == "translation" and s.schema == 3


def test_store_skips_out_of_range_fields_and_keeps_valid_boundaries(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({
        "size": 160,
        "minsize": 16,
        "clear": 600.01,
        "ls": -2.01,
        "ow": 1e100,
        "schema": 2,
    }))

    s = SettingsStore(p).load()

    assert (s.size, s.minsize) == (160, 16)
    assert (s.clear, s.ls, s.ow) == (Settings().clear, Settings().ls, Settings().ow)
    assert s.schema == 3


def test_store_update_persists_atomically(tmp_path):
    p = tmp_path / "sub" / "settings.json"
    store = SettingsStore(p)
    store.load()
    new = store.update({"size": 42, "translate": False})
    assert new.size == 42 and store.get() is new
    assert SettingsStore(p).load() == new
    assert [f.name for f in p.parent.iterdir()] == ["settings.json"]  # no .tmp left
    assert json.loads(p.read_text())["schema"] == 3


def test_store_unchanged_update_returns_current_without_saving(
        tmp_path, monkeypatch):
    store = SettingsStore(tmp_path / "settings.json")
    current = store.load()
    saves = []
    monkeypatch.setattr(store, "save", lambda settings: saves.append(settings))

    result = store.update({"size": current.size})

    assert result is current
    assert store.get() is current
    assert saves == []
    assert not store.path.exists()


def test_store_invalid_update_changes_nothing(tmp_path):
    p = tmp_path / "settings.json"
    store = SettingsStore(p)
    store.load()
    with pytest.raises(ValueError):
        store.update({"size": 10, "mode": "bad"})
    assert store.get() == Settings()
    assert not p.exists()


def test_default_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GUJUSUB_SETTINGS", str(tmp_path / "x.json"))
    assert S.default_path() == tmp_path / "x.json"
    monkeypatch.delenv("GUJUSUB_SETTINGS")
    assert S.default_path().name == "settings.json"
    assert S.default_path().parent.name == "gujusub"


@pytest.mark.parametrize("raw,expected", [
    ("primary", "primary"), ("translation", "translation"), ("both", "both"),
    ("gu", "primary"), ("en", "translation"),
])
def test_mode_accepts_new_and_legacy_values_stores_new_style(raw, expected):
    assert validate({"mode": raw}) == {"mode": expected}
    assert Settings().with_updates({"mode": raw}).mode == expected


def test_store_migrates_legacy_mode_on_load(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"mode": "gu"}))
    assert SettingsStore(p).load().mode == "primary"
