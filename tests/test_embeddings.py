from __future__ import annotations

import numpy as np

from memory.embeddings import EMBED_DIM, HashingEmbedder, get_embedder


def test_hashing_embedder_deterministic() -> None:
    embedder = HashingEmbedder()
    v1 = embedder.embed("Margaret loved growing roses in her garden.")
    v2 = embedder.embed("Margaret loved growing roses in her garden.")
    assert np.allclose(v1, v2)
    assert v1.shape == (EMBED_DIM,)


def test_hashing_embedder_unit_norm() -> None:
    embedder = HashingEmbedder()
    vector = embedder.embed("Walter and Margaret were married for fifty-one years.")
    norm = np.linalg.norm(vector)
    assert 0.99 <= norm <= 1.01


def test_hashing_embedder_distinguishes_different_text() -> None:
    embedder = HashingEmbedder()
    v1 = embedder.embed("She taught second grade for decades.")
    v2 = embedder.embed("The roses bloomed early that June.")
    similarity = float(np.dot(v1, v2))
    assert similarity < 0.9


def test_get_embedder_defaults_to_hashing() -> None:
    embedder = get_embedder("hashing")
    assert isinstance(embedder, HashingEmbedder)


def test_get_embedder_falls_back_when_sentence_transformer_missing() -> None:
    embedder = get_embedder("sentence-transformer", model_name="all-MiniLM-L6-v2")
    assert embedder.dim == EMBED_DIM
