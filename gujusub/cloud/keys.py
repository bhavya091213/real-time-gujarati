"""API keys for cloud providers: saved in a 0600 JSON file, or read from the environment.

The full key never leaves the server; the UI only sees ``mask(key)`` (the last
four characters) and where it came from. Default file
``~/.cache/gujusub/keys.json`` (override with ``GUJUSUB_KEYS``).
"""

import json
import logging
import os
import re
from pathlib import Path

from gujusub.cloud.base import Credentials, ProviderInfo

logger = logging.getLogger("cloud.keys")

MIN_KEY_LEN = 8
MAX_KEY_LEN = 512
MAX_REGION_LEN = 64
_REGION = re.compile(r"[a-z0-9-]{2,64}")
MASK = "••••"
REGION_ENV_SUFFIX = "_REGION"  # e.g. AZURE_SPEECH_KEY -> AZURE_SPEECH_REGION


def default_path() -> Path:
    env = os.environ.get("GUJUSUB_KEYS")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "gujusub" / "keys.json"


def mask(key: str) -> str:
    """'sk-abc...wxyz' -> '••••wxyz'; short keys are fully hidden."""
    tail = key[-4:] if len(key) >= MIN_KEY_LEN else ""
    return MASK + tail


def validate_key(key: object) -> str:
    if not isinstance(key, str):
        raise ValueError("key must be a string")
    key = key.strip()
    if not MIN_KEY_LEN <= len(key) <= MAX_KEY_LEN:
        raise ValueError(f"key must be {MIN_KEY_LEN}-{MAX_KEY_LEN} characters")
    if any(c.isspace() for c in key) or not key.isprintable():
        raise ValueError("key contains whitespace or control characters")
    return key


def validate_region(region: object) -> str | None:
    if region is None or region == "":
        return None
    if not isinstance(region, str):
        raise ValueError("region must be a string")
    region = region.strip().lower()
    if not _REGION.fullmatch(region):
        raise ValueError("region must look like 'eastus' or 'westeurope'")
    return region


def _env_name(info: ProviderInfo) -> str:
    return info.key_env


def _region_env_name(info: ProviderInfo) -> str:
    base = info.key_env[:-4] if info.key_env.endswith("_KEY") else info.key_env
    return base + REGION_ENV_SUFFIX


class KeyStore:
    """Saved credentials per provider id, with environment fallback."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else default_path()
        self._saved: dict[str, dict] = {}

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._saved = {}
            return
        except (OSError, ValueError) as e:
            logger.warning("keys file %s unreadable (%s); ignoring it", self.path, e)
            self._saved = {}
            return
        good: dict[str, dict] = {}
        if isinstance(raw, dict):
            for pid, entry in raw.items():
                if not isinstance(entry, dict):
                    continue
                try:
                    good[pid] = {"key": validate_key(entry.get("key")),
                                 "region": validate_region(entry.get("region"))}
                except ValueError as e:
                    logger.warning("ignoring saved key for %s: %s", pid, e)
        self._saved = good

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self._saved, f, indent=2)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def set(self, provider: str, key: str, region: str | None = None) -> None:
        self._saved[provider] = {"key": validate_key(key), "region": validate_region(region)}
        self._write()

    def clear(self, provider: str) -> None:
        if self._saved.pop(provider, None) is not None:
            self._write()

    def get(self, info: ProviderInfo) -> Credentials | None:
        """Saved key first, then the environment; None when neither is set."""
        saved = self._saved.get(info.id)
        if saved:
            return Credentials(saved["key"], saved.get("region"))
        env_key = os.environ.get(_env_name(info), "").strip()
        if not env_key:
            return None
        region = os.environ.get(_region_env_name(info), "").strip() or None
        return Credentials(env_key, region)

    def source(self, info: ProviderInfo) -> str | None:
        if info.id in self._saved:
            return "saved"
        if os.environ.get(_env_name(info), "").strip():
            return "env"
        return None

    def status(self, infos: list[ProviderInfo]) -> dict[str, dict]:
        """What the UI may see: never the key itself."""
        out = {}
        for info in infos:
            creds = self.get(info)
            out[info.id] = {
                "set": creds is not None,
                "hint": mask(creds.key) if creds else "",
                "source": self.source(info),
                "region": creds.region if creds else None,
                "env": info.key_env,
            }
        return out
