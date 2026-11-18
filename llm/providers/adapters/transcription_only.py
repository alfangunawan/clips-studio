"""For a provider that only transcribes (AssemblyAI).

It has a key to check but nothing to pick clips with, so the text functions
refuse in plain words instead of sending a request that cannot work. The
transcription itself is in transcription/cloud.py, like every provider's.
"""

from llm.providers.adapters.chat_completions import check_key_generic
from llm.providers.base import LLMError, ModelInfo, ProviderSpec


def _refuse(spec: ProviderSpec) -> LLMError:
    return LLMError("not_configured", f"{spec.label} only transcribes. Choose another provider "
                                      "for picking clips in Settings → AI.")


def generate(spec: ProviderSpec, key: str, model: str, prompt: str, *, json_mode=False, schema=None) -> str:
    raise _refuse(spec)


def chat(spec: ProviderSpec, key: str, model: str, messages, tools=None):
    raise _refuse(spec)


def list_models(spec: ProviderSpec, key: str) -> list[ModelInfo]:
    raise _refuse(spec)


def check_key(spec: ProviderSpec, key: str) -> str:
    return check_key_generic(spec, key, list_models)
