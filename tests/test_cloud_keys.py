import json
import stat

import pytest

from gujusub.cloud import keys as K
from gujusub.cloud.keys import KeyStore, mask, validate_key, validate_region
from gujusub.cloud.registry import BY_ID, PROVIDERS

SECRET = "sk-live-SECRET-abcd1234"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for info in PROVIDERS:
        monkeypatch.delenv(info.key_env, raising=False)
        monkeypatch.delenv(K._region_env_name(info), raising=False)
    monkeypatch.delenv("GUJUSUB_KEYS", raising=False)


@pytest.fixture
def store(tmp_path):
    s = KeyStore(tmp_path / "sub" / "keys.json")
    s.load()
    return s


def test_set_writes_0600_file_and_round_trips(store):
    store.set("soniox", f"  {SECRET}  ")
    mode = stat.S_IMODE(store.path.stat().st_mode)
    assert mode == 0o600
    again = KeyStore(store.path)
    again.load()
    assert again.get(BY_ID["soniox"]).key == SECRET  # stripped
    assert not store.path.with_name("keys.json.tmp").exists()


def test_clear_removes_saved_key(store):
    store.set("soniox", SECRET)
    store.clear("soniox")
    store.clear("soniox")  # idempotent
    assert store.get(BY_ID["soniox"]) is None
    assert json.loads(store.path.read_text()) == {}


def test_env_fallback_with_region_env(store, monkeypatch):
    azure = BY_ID["azure"]
    assert store.get(azure) is None
    monkeypatch.setenv("AZURE_SPEECH_KEY", SECRET)
    monkeypatch.setenv("AZURE_SPEECH_REGION", "westeurope")
    creds = store.get(azure)
    assert (creds.key, creds.region) == (SECRET, "westeurope")
    assert store.source(azure) == "env"


def test_saved_key_wins_over_env(store, monkeypatch):
    monkeypatch.setenv("SONIOX_API_KEY", "env-key-0000zzzz")
    store.set("soniox", SECRET)
    assert store.get(BY_ID["soniox"]).key == SECRET
    assert store.source(BY_ID["soniox"]) == "saved"


def test_mask_shows_only_last_four():
    assert mask(SECRET) == "••••1234"
    assert mask("short") == "••••"


@pytest.mark.parametrize("bad", [None, 5, "", "short", "has space in it", "x" * 600,
                                 "abcdefgh\x00"])
def test_validate_key_rejects(bad):
    with pytest.raises(ValueError):
        validate_key(bad)


def test_validate_region():
    assert validate_region(None) is None and validate_region("") is None
    assert validate_region(" EastUS ") == "eastus"
    for bad in ("a b", "x", 3, "east_us"):
        with pytest.raises(ValueError):
            validate_region(bad)


def test_status_never_contains_the_key(store, monkeypatch):
    store.set("soniox", SECRET)
    store.set("azure", "azure-key-9999wxyz", "eastus")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-key-5555qrst")
    st = store.status(list(PROVIDERS))
    assert st["soniox"] == {"set": True, "hint": "••••1234", "source": "saved",
                            "region": None, "env": "SONIOX_API_KEY"}
    assert st["azure"]["region"] == "eastus"
    assert st["deepgram"]["source"] == "env" and st["deepgram"]["hint"] == "••••qrst"
    assert st["openai"] == {"set": False, "hint": "", "source": None, "region": None,
                            "env": "OPENAI_API_KEY"}
    text = json.dumps(st)
    for secret in (SECRET, "azure-key-9999wxyz", "dg-key-5555qrst"):
        assert secret not in text


def test_load_ignores_corrupt_file_and_bad_entries(tmp_path, caplog):
    path = tmp_path / "keys.json"
    path.write_text("{nope")
    s = KeyStore(path)
    s.load()
    assert s.get(BY_ID["soniox"]) is None
    path.write_text(json.dumps({"soniox": {"key": SECRET}, "deepgram": {"key": "bad"},
                                "openai": "nope"}))
    s.load()
    assert s.get(BY_ID["soniox"]).key == SECRET
    assert s.get(BY_ID["deepgram"]) is None and s.get(BY_ID["openai"]) is None
    assert SECRET not in caplog.text


def test_default_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GUJUSUB_KEYS", str(tmp_path / "k.json"))
    assert KeyStore().path == tmp_path / "k.json"
