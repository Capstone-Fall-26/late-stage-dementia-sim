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

Records carry dynamic fields beyond the vector: `source` ("seed" or
"conversation"), `access_count`/`last_accessed` (rehearsal), and, once a
record has been degraded, `corrupted`, `corruption_level` and
`original_text`/`original_fact` (the pre-corruption ground truth, kept so
evaluation can tell what the patient *should* have remembered).
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
            else:
                self._check_dimension(name)
            # Collections created by a previous process/connection persist on
            # disk but are not automatically loaded into memory for search on
            # a fresh connection -- load explicitly every time.
            self._client.load_collection(collection_name=name)

    def _check_dimension(self, name: str) -> None:
        """Fail loudly if an existing collection was built with another embedder."""
        for field in self._client.describe_collection(collection_name=name)["fields"]:
            if field["name"] == "vector":
                existing = int(field["params"]["dim"])
                if existing != self.dim:
                    raise ValueError(
                        f"Collection '{name}' in {self.db_path} has dimension {existing}, but "
                        f"the current embedder produces {self.dim}. Re-seed with "
                        "`reset-memory`, or point --db-path at a new file."
                    )

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
    ) -> int:
        record: dict[str, Any] = {
            "vector": embedding.tolist(),
            "text": text,
            "emotional_valence": emotional_valence,
            "era": era,
            "timestamp": timestamp if timestamp is not None else time.time(),
            "source": "seed",
            "access_count": 0,
        }
        if extra:
            record.update(extra)
        return self._insert(EPISODIC_COLLECTION, record)

    def insert_biographical_fact(
        self,
        fact: str,
        embedding: np.ndarray,
        category: str,
        confidence: float,
        extra: dict[str, Any] | None = None,
    ) -> int:
        record: dict[str, Any] = {
            "vector": embedding.tolist(),
            "fact": fact,
            "category": category,
            "confidence": confidence,
            "timestamp": time.time(),
            "source": "seed",
            "access_count": 0,
        }
        if extra:
            record.update(extra)
        return self._insert(FACTS_COLLECTION, record)

    def _insert(self, collection_name: str, record: dict[str, Any]) -> int:
        result = self._client.insert(collection_name=collection_name, data=[record])
        return int(list(result["ids"])[0])

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
            "timestamp",
            "source",
            "access_count",
            "corrupted",
            "corruption_level",
            "original_text",
            "original_fact",
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

    def get_record(self, collection_name: str, record_id: int) -> dict[str, Any] | None:
        """Fetch one full record (including its vector), or None if absent."""
        rows = self._client.get(
            collection_name=collection_name, ids=[record_id], output_fields=["*"]
        )
        return dict(rows[0]) if rows else None

    def all_records(
        self, collection_name: str, include_vectors: bool = False
    ) -> list[dict[str, Any]]:
        """Return every record in a collection (patient memories are small)."""
        rows = self._client.query(
            collection_name=collection_name, filter="id >= 0", output_fields=["*"], limit=16384
        )
        out = [dict(r) for r in rows]
        if not include_vectors:
            for row in out:
                row.pop("vector", None)
        return out

    def update_record(
        self,
        collection_name: str,
        record_id: int,
        changes: dict[str, Any],
        vector: np.ndarray | None = None,
    ) -> None:
        """Merge `changes` (and optionally a new vector) into a stored record.

        Implemented as an upsert on the same primary key, so the id is stable.
        """
        record = self.get_record(collection_name, record_id)
        if record is None:
            raise KeyError(f"No record with id {record_id} in '{collection_name}'")
        record.update(changes)
        if vector is not None:
            record["vector"] = vector.tolist()
        self._client.upsert(collection_name=collection_name, data=[record])

    def touch(self, collection_name: str, record_ids: list[int]) -> None:
        """Record a successful retrieval (rehearsal) for each id."""
        now = time.time()
        for record_id in record_ids:
            record = self.get_record(collection_name, record_id)
            if record is None:
                continue
            record["access_count"] = int(record.get("access_count", 0)) + 1
            record["last_accessed"] = now
            self._client.upsert(collection_name=collection_name, data=[record])

    def reset(self) -> None:
        """Drop and recreate both collections, discarding all memories."""
        for name in (EPISODIC_COLLECTION, FACTS_COLLECTION):
            if self._client.has_collection(collection_name=name):
                self._client.drop_collection(collection_name=name)
        self._ensure_collections()

    def stats(self) -> dict[str, dict[str, int]]:
        """Per-collection row, conversation-sourced and corrupted counts."""
        out: dict[str, dict[str, int]] = {}
        for name in (EPISODIC_COLLECTION, FACTS_COLLECTION):
            rows = self.all_records(name)
            out[name] = {
                "rows": len(rows),
                "from_conversation": sum(r.get("source") == "conversation" for r in rows),
                "corrupted": sum(bool(r.get("corrupted")) for r in rows),
            }
        return out

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
