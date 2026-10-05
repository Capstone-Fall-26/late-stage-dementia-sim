from __future__ import annotations

import pytest

from core.llm_config import resolve_llm_config


def test_ollama_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    cfg = resolve_llm_config("ollama")
    assert cfg.model == "llama3.2:3b"
    assert cfg.host is None


def test_ollama_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    monkeypatch.setenv("OLLAMA_HOST", "http://remote:11434")
    cfg = resolve_llm_config("ollama")
    assert cfg.model == "mistral"
    assert cfg.host == "http://remote:11434"


def test_explicit_model_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    cfg = resolve_llm_config("ollama", model="llama3.2:3b")
    assert cfg.model == "llama3.2:3b"


def test_gemini_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")  # pragma: allowlist secret
    monkeypatch.setenv("GEMINI_MODEL", "gemini-pro")
    cfg = resolve_llm_config("gemini")
    assert cfg.api_key == "fake-key"  # pragma: allowlist secret
    assert cfg.model == "gemini-pro"


def test_azure_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")  # pragma: allowlist secret
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    cfg = resolve_llm_config("azure")
    assert cfg.api_key == "fake-key"  # pragma: allowlist secret
    assert cfg.endpoint == "https://x.openai.azure.com"
    assert cfg.model == "my-deployment"
    assert cfg.api_version == "2024-10-21"


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        resolve_llm_config("not-a-backend")
