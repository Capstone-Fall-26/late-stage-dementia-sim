"""Milvus Lite client wrapper for patient episodic and biographical memory.

Uses `pymilvus.MilvusClient` against a local Milvus Lite file
(`./milvus_patient_memory.db` by default) -- no separate Milvus server
required. Two collections are auto-created on first use:

- `episodic_memory`: free-text autobiographical episodes with an emotional
  valence tag and era label, used to simulate the patient recalling past
  experiences.
- `biographical_facts`: discrete factual statements (career, family, etc.)
  used both for grounding responses and for consistency checking in the
  evaluation suite.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from pymilvus import MilvusClient

from core.models import PatientProfile
from memory.embeddings import EMBED_DIM, Embedder

EPISODIC_COLLECTION = "episodic_memory"
FACTS_COLLECTION = "biographical_facts"


class PatientMemoryClient:
    """Thin, typed wrapper around a Milvus Lite database for one patient."""

    def __init__(
        self, db_path: str | Path = "./milvus_patient_memory.db", dim: int = EMBED_DIM
    ) -> None:
        self.db_path = str(db_path)
        self.dim = dim
        self._client = MilvusClient(uri=self.db_path)
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        for name in (EPISODIC_COLLECTION, FACTS_COLLECTION):
            if not self._client.has_collection(collection_name=name):
                self._client.create_collection(
                    collection_name=name,
                    dimension=self.dim,
                    auto_id=True,
                    enable_dynamic_field=True,
                    metric_type="COSINE",
                )
            # Collections created by a previous process/connection persist on
            # disk but are not automatically loaded into memory for search on
            # a fresh connection -- load explicitly every time.
            self._client.load_collection(collection_name=name)

    # ------------------------------------------------------------------ #
    # Inserts
    # ------------------------------------------------------------------ #

    def insert_episodic_memory(
        self,
        text: str,
        embedding: np.ndarray,
        emotional_valence: str,
        era: str,
        timestamp: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "vector": embedding.tolist(),
            "text": text,
            "emotional_valence": emotional_valence,
            "era": era,
            "timestamp": timestamp if timestamp is not None else time.time(),
        }
        if extra:
            record.update(extra)
        self._client.insert(collection_name=EPISODIC_COLLECTION, data=[record])

    def insert_biographical_fact(
        self,
        fact: str,
        embedding: np.ndarray,
        category: str,
        confidence: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "vector": embedding.tolist(),
            "fact": fact,
            "category": category,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        if extra:
            record.update(extra)
        self._client.insert(collection_name=FACTS_COLLECTION, data=[record])

    def seed_from_profile(self, profile: PatientProfile, embedder: Embedder) -> dict[str, int]:
        """Idempotently seed both collections from a PatientProfile.

        Skips seeding a collection if it already contains data (checked via
        row count), so calling this repeatedly across CLI sessions is safe.
        """
        inserted = {"episodic_memory": 0, "biographical_facts": 0}

        if self.count(EPISODIC_COLLECTION) == 0:
            for memory in profile.episodic_memories:
                vector = embedder.embed(memory.text)
                self.insert_episodic_memory(
                    text=memory.text,
                    embedding=vector,
                    emotional_valence=memory.emotional_valence,
                    era=memory.era,
                )
                inserted["episodic_memory"] += 1

        if self.count(FACTS_COLLECTION) == 0:
            for fact in profile.biographical_facts:
                vector = embedder.embed(fact.fact)
                self.insert_biographical_fact(
                    fact=fact.fact,
                    embedding=vector,
                    category=fact.category,
                    confidence=fact.confidence,
                )
                inserted["biographical_facts"] += 1

        return inserted

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Raw vector search, returning a flat list of hit dicts.

        Each hit dict contains at least `id`, `distance` (cosine similarity,
        higher is more similar), plus any requested `output_fields`.
        """
        fields = output_fields or [
            "text",
            "fact",
            "emotional_valence",
            "era",
            "category",
            "confidence",
        ]
        raw = self._client.search(
            collection_name=collection_name,
            data=[query_vector.tolist()],
            limit=top_k,
            output_fields=fields,
        )
        hits: list[dict[str, Any]] = []
        if not raw:
            return hits
        for hit in raw[0]:
            entity = hit.get("entity", hit)
            merged = {"id": hit.get("id"), "distance": hit.get("distance", 0.0)}
            for field in fields:
                if field in entity:
                    merged[field] = entity[field]
                elif field in hit:
                    merged[field] = hit[field]
            hits.append(merged)
        return hits

    def count(self, collection_name: str) -> int:
        try:
            stats = self._client.get_collection_stats(collection_name=collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            rows = self._client.query(
                collection_name=collection_name, filter="id >= 0", output_fields=["id"]
            )
            return len(rows)

    def close(self) -> None:
        self._client.close()
