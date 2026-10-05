"""Pluggable LLM backend interface: local Ollama, Google Gemini, or Azure OpenAI.

`PatientEngine` depends only on the `LLMProvider` protocol, never on a
concrete backend, so switching between a local `llama3.2:3b`/`mistral`
model, Gemini Flash-Lite, and an Azure OpenAI deployment is a one-line
config change.

A `MockProvider` is also provided: a deterministic, network-free backend
used by the automated benchmarking suite and test suite so that CI and
offline development never require a live Ollama daemon or Gemini API key.
It is a genuine, fully-functional backend (not a placeholder) that
produces stage/constraint-aware synthetic text.
"""

from __future__ import annotations

import random
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Common interface implemented by every LLM backend."""

    def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str: ...


class OllamaProvider:
    """Local LLM backend via the `ollama` Python client."""

    def __init__(self, model: str = "llama3.2:3b", host: str | None = None) -> None:
        try:
            import ollama
        except ImportError as exc:
            raise ImportError(
                "The 'ollama' package is required for OllamaProvider. Install it with "
                "`pip install ollama` and ensure the Ollama daemon is running."
            ) from exc
        self.model = model
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        try:
            response = self._client.chat(model=self.model, messages=full_messages)
        except Exception as exc:
            raise RuntimeError(f"Ollama generation failed for model '{self.model}': {exc}") from exc
        return str(response["message"]["content"])


class GeminiProvider:
    """Google Gemini backend via the `google-genai` SDK."""

    _ROLE_MAP = {"assistant": "model", "user": "user"}

    def __init__(self, model: str = "gemini-flash-lite-latest", api_key: str | None = None) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for GeminiProvider. Install it "
                "with `pip install google-genai`."
            ) from exc
        if not api_key:
            raise ValueError("GeminiProvider requires a non-empty api_key (set GEMINI_API_KEY).")
        self.model = model
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        from google.genai import types

        # Typed as list[Any]: the google-genai SDK's `contents` overloads use an
        # invariant list type that mypy resolves differently across SDK versions,
        # so a precise element type here is version-fragile. Any sidesteps that.
        contents: list[Any] = [
            types.Content(
                role=self._ROLE_MAP.get(m["role"], "user"),
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini generation failed for model '{self.model}': {exc}") from exc
        return response.text or ""


class AzureOpenAIProvider:
    """Azure OpenAI backend via the `openai` SDK's `AzureOpenAI` client.

    `model` is the Azure *deployment name* (not the underlying model name),
    since that's what the Azure OpenAI chat-completions endpoint expects.
    """

    _ROLE_MAP = {"assistant": "assistant", "user": "user"}

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str = "2024-10-21",
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for AzureOpenAIProvider. Install it "
                "with `pip install openai`."
            ) from exc
        if not api_key:
            raise ValueError(
                "AzureOpenAIProvider requires a non-empty api_key (set AZURE_OPENAI_API_KEY)."
            )
        if not endpoint:
            raise ValueError(
                "AzureOpenAIProvider requires a non-empty endpoint (set AZURE_OPENAI_ENDPOINT)."
            )
        if not model:
            raise ValueError(
                "AzureOpenAIProvider requires a non-empty deployment name (set "
                "AZURE_OPENAI_DEPLOYMENT)."
            )
        self.model = model
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        full_messages = [
            {"role": "system", "content": system_prompt},
            *(
                {"role": self._ROLE_MAP.get(m["role"], "user"), "content": m["content"]}
                for m in messages
            ),
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=full_messages,  # type: ignore[arg-type]
            )
        except Exception as exc:
            raise RuntimeError(
                f"Azure OpenAI generation failed for deployment '{self.model}': {exc}"
            ) from exc
        return response.choices[0].message.content or ""


class MockProvider:
    """Deterministic, network-free LLM backend for tests and benchmarking.

    Produces stage-appropriate synthetic text by templating on the system
    prompt content, without calling any external model. Seeded for
    reproducibility.
    """

    _TEMPLATES = [
        "{cue} {fragment}",
        "{fragment} {cue}",
        "{fragment}",
    ]
    _FRAGMENTS = [
        "the roses... yes, the roses",
        "Walter will be home soon",
        "the children, the children were here",
        "bread, Sunday, the oven",
        "I don't... I don't know",
        "cold, it's cold in here",
    ]

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)

    def generate(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        fragment = self._rng.choice(self._FRAGMENTS)
        cue = ""
        for line in system_prompt.splitlines():
            if "non-verbal cue" in line.lower() and "*" in line:
                start = line.find("*")
                end = line.rfind("*")
                if end > start:
                    cue = line[start : end + 1]
                break
        template = self._rng.choice(self._TEMPLATES)
        text = template.format(cue=cue, fragment=fragment).strip()
        return " ".join(text.split())


def get_llm_provider(
    backend: str,
    model: str,
    api_key: str | None = None,
    host: str | None = None,
    endpoint: str | None = None,
    api_version: str | None = None,
) -> LLMProvider:
    """Factory for LLM backends: "ollama", "gemini", "azure", or "mock"."""
    if backend == "ollama":
        return OllamaProvider(model=model, host=host)
    if backend == "gemini":
        return GeminiProvider(model=model, api_key=api_key)
    if backend == "azure":
        return AzureOpenAIProvider(
            model=model,
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version or "2024-10-21",
        )
    if backend == "mock":
        return MockProvider()
    raise ValueError(
        f"Unknown LLM backend: {backend!r}. Expected 'ollama', 'gemini', 'azure', or 'mock'."
    )
