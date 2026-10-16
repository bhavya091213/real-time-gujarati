"""Catalogue of cloud ASR providers and the ``asr_backend`` values.

``asr_backend`` is ``"local"`` (the bundled models) or a provider id below.
Provider modules are imported only when a session is opened, so listing the
catalogue costs nothing at startup.
"""

import importlib

from gujusub.cloud.base import CloudSession, Credentials, ProviderInfo, SessionConfig

LOCAL = "local"

PROVIDERS: tuple[ProviderInfo, ...] = (
    ProviderInfo(
        id="soniox", label="Soniox", key_env="SONIOX_API_KEY",
        key_url="https://console.soniox.com/",
        gujarati="yes", translates=True, language_id=True,
        note="stt-rt-v5: Gujarati + English, built-in translation and language ID "
             "(what sabha-stt uses).",
    ),
    ProviderInfo(
        id="elevenlabs", label="ElevenLabs", key_env="ELEVENLABS_API_KEY",
        key_url="https://elevenlabs.io/app/settings/api-keys",
        gujarati="yes", translates=False, language_id=True,
        note="Scribe v2 Realtime: Gujarati supported; translation stays local.",
    ),
    ProviderInfo(
        id="deepgram", label="Deepgram", key_env="DEEPGRAM_API_KEY",
        key_url="https://console.deepgram.com/",
        gujarati="yes", translates=False, language_id=False,
        note="Nova-3 with language=gu (added 2026-05); no Gujarati/English auto-detect; "
             "translation stays local.",
    ),
    ProviderInfo(
        id="openai", label="OpenAI", key_env="OPENAI_API_KEY",
        key_url="https://platform.openai.com/api-keys",
        gujarati="partial", translates=False, language_id=False,
        note="Realtime transcription (gpt-4o-transcribe): Gujarati via Whisper-class "
             "models; text arrives per turn, not word by word.",
    ),
    ProviderInfo(
        id="sarvam", label="Sarvam AI", key_env="SARVAM_API_KEY",
        key_url="https://dashboard.sarvam.ai/",
        gujarati="yes", translates=False, language_id=True,
        note="Saaras v3 realtime, gu-IN (language_code auto for Gujarati/English); "
             "Indian-language specialist; translation stays local.",
    ),
    ProviderInfo(
        id="azure", label="Azure AI Speech", key_env="AZURE_SPEECH_KEY",
        key_url="https://portal.azure.com/",
        gujarati="yes", translates=False, language_id=True, needs_region=True,
        note="gu-IN continuous recognition via the Azure Speech SDK "
             "(pip install azure-cognitiveservices-speech); needs key + region.",
    ),
)

BY_ID: dict[str, ProviderInfo] = {p.id: p for p in PROVIDERS}
BACKENDS: tuple[str, ...] = (LOCAL, *BY_ID)


def provider_info(provider_id: str) -> ProviderInfo:
    try:
        return BY_ID[provider_id]
    except KeyError:
        raise ValueError(f"unknown provider {provider_id!r}; choose from {list(BY_ID)}") from None


def is_cloud(backend: str) -> bool:
    return backend != LOCAL and backend in BY_ID


def catalog() -> list[dict]:
    """JSON-safe provider list for the mic page (no secrets)."""
    return [
        {"id": p.id, "label": p.label, "key_env": p.key_env, "key_url": p.key_url,
         "gujarati": p.gujarati, "translates": p.translates, "language_id": p.language_id,
         "needs_region": p.needs_region, "note": p.note}
        for p in PROVIDERS
    ]


def open_session(provider_id: str, creds: Credentials, config: SessionConfig) -> CloudSession:
    """Instantiate (not start) the provider's Session."""
    info = provider_info(provider_id)
    module = importlib.import_module(f"gujusub.cloud.{info.id}")
    return module.Session(creds, config)
