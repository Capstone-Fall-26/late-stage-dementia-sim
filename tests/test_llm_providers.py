from __future__ import annotations

import pytest

from core.llm_providers import AzureOpenAIProvider, MockProvider, get_llm_provider


def test_azure_requires_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        AzureOpenAIProvider(model="my-deployment", api_key=None, endpoint="https://x.openai.azure.com")


def test_azure_requires_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        AzureOpenAIProvider(model="my-deployment", api_key="fake-key", endpoint=None)


def test_azure_requires_model() -> None:
    with pytest.raises(ValueError, match="deployment name"):
        AzureOpenAIProvider(model="", api_key="fake-key", endpoint="https://x.openai.azure.com")


def test_get_llm_provider_routes_to_azure() -> None:
    provider = get_llm_provider(
        backend="azure",
        model="my-deployment",
        api_key="fake-key",
        endpoint="https://x.openai.azure.com",
    )
    assert isinstance(provider, AzureOpenAIProvider)
    assert provider.model == "my-deployment"


def test_get_llm_provider_routes_to_mock() -> None:
    assert isinstance(get_llm_provider(backend="mock", model="unused"), MockProvider)


def test_get_llm_provider_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        get_llm_provider(backend="not-a-backend", model="unused")
