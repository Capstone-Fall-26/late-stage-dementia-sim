"""Environment-variable resolution for LLM backend configuration.

Keeps the "which env var maps to which backend" knowledge in one place so
the CLI's `chat` and `test-connection` commands (and anything else that
needs to build an `LLMProvider`) stay in sync instead of duplicating
per-backend env-var lookups.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_OLLAMA_DEFAULT_MODEL = "llama3.2:3b"
_GEMINI_DEFAULT_MODEL = "gemini-flash-lite-latest"
_AZURE_DEFAULT_API_VERSION = "2024-10-21"

_SUPPORTED_BACKENDS = ("ollama", "gemini", "azure", "mock")


@dataclass(frozen=True)
class LLMConfig:
    """Resolved model name and credentials, ready to pass to `get_llm_provider`."""

    backend: str
    model: str
    api_key: str | None = None
    host: str | None = None
    endpoint: str | None = None
    api_version: str | None = None


def resolve_llm_config(backend: str, model: str | None = None) -> LLMConfig:
    """Resolve the model name and credentials for `backend` from the environment.

    An explicit `model` always wins over the backend's env-var default, so a
    `--model` CLI flag can still override `.env`.
    """
    if backend == "ollama":
        resolved_model = (
            model if model is not None else os.environ.get("OLLAMA_MODEL", _OLLAMA_DEFAULT_MODEL)
        )
        return LLMConfig(backend=backend, model=resolved_model, host=os.environ.get("OLLAMA_HOST"))
    if backend == "gemini":
        resolved_model = (
            model if model is not None else os.environ.get("GEMINI_MODEL", _GEMINI_DEFAULT_MODEL)
        )
        return LLMConfig(
            backend=backend, model=resolved_model, api_key=os.environ.get("GEMINI_API_KEY")
        )
    if backend == "azure":
        resolved_model = (
            model if model is not None else os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        )
        return LLMConfig(
            backend=backend,
            model=resolved_model,
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", _AZURE_DEFAULT_API_VERSION),
        )
    if backend == "mock":
        return LLMConfig(backend=backend, model=model if model is not None else "mock")
    raise ValueError(f"Unknown LLM backend: {backend!r}. Expected one of {_SUPPORTED_BACKENDS}.")
