"""Local, offline-friendly embedding interface for patient memory vectors.

Two backends are supported:

- ``hashing`` (default): a deterministic, dependency-free bag-of-n-grams
  hashing embedding. No model download, no network access, fully
  reproducible -- ideal for CI, tests, and offline research environments.
- ``sentence-transformer``: wraps a local `sentence-transformers` model
  (installed via the ``embeddings`` extra) for higher-quality semantic
  embeddings when available.

Both backends implement the same `Embedder` protocol and produce
L2-normalized float32 vectors of dimension `EMBED_DIM`, so the vector store
schema never has to change when swapping backends.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

EMBED_DIM = 384

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Common interface implemented by every embedding backend."""

    dim: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _char_ngrams(token: str, n: int = 3) -> list[str]:
    if len(token) <= n:
        return [token]
    return [token[i : i + n] for i in range(len(token) - n + 1)]


class HashingEmbedder:
    """Deterministic hashing-trick embedding over word + character n-grams.

    This is not semantically as rich as a trained sentence encoder, but it is
    reproducible, requires no model weights, and is more than sufficient to
    (a) retrieve topically related episodic memories and (b) exercise the
    full retrieval-decay/noise-injection pipeline in tests without network
    access.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    def _hash_index(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def _sign(self, feature: str) -> float:
        digest = hashlib.blake2b(feature.encode("utf-8") + b"sign", digest_size=1).digest()
        return 1.0 if digest[0] % 2 == 0 else -1.0

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = _tokenize(text)
        features: list[str] = list(tokens)
        for token in tokens:
            features.extend(_char_ngrams(token))
        if not features:
            return vector
        for feature in features:
            idx = self._hash_index(feature)
            vector[idx] += self._sign(feature)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dim))


class SentenceTransformerEmbedder:
    """Wraps a local `sentence-transformers` model, projected to `EMBED_DIM`.

    Requires the optional `embeddings` extra (`pip install .[embeddings]`).
    Raises `ImportError` at construction time if unavailable, so callers
    should catch that and fall back to `HashingEmbedder`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = EMBED_DIM) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Install with "
                "`pip install .[embeddings]` or use EMBEDDING_BACKEND=hashing."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = dim
        native_dim = self._model.get_sentence_embedding_dimension()
        self._projection: np.ndarray | None = None
        if native_dim != dim:
            rng = np.random.default_rng(seed=42)
            self._projection = rng.standard_normal((native_dim, dim)).astype(np.float32)

    def _project(self, embeddings: np.ndarray) -> np.ndarray:
        if self._projection is not None:
            embeddings = embeddings @ self._projection
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embeddings / norms).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        raw = self._model.encode([text], convert_to_numpy=True)
        return self._project(raw)[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim))
        raw = self._model.encode(texts, convert_to_numpy=True)
        return self._project(raw)


def get_embedder(backend: str = "hashing", model_name: str | None = None) -> Embedder:
    """Factory for embedding backends.

    Args:
        backend: "hashing" or "sentence-transformer".
        model_name: only used for "sentence-transformer".

    Falls back to `HashingEmbedder` if `sentence-transformer` is requested
    but the optional dependency is not installed.
    """
    if backend == "sentence-transformer":
        try:
            return SentenceTransformerEmbedder(model_name=model_name or "all-MiniLM-L6-v2")
        except ImportError:
            return HashingEmbedder()
    return HashingEmbedder()
