"""Cognitively-degraded vector recall for patient memory.

Wraps a raw Milvus vector search with two effects intended to simulate
memory impairment in moderate-to-severe dementia:

1. **Distance attenuation** - retrieved similarity scores are attenuated by
   (a) an era-based "retrograde preservation" weight, reflecting the
   well-documented clinical pattern (Ribot's law) that remote autobiographical
   memories are relatively spared while recently-formed memories are the
   most vulnerable, and (b) a session-level time-decay factor whose rate
   scales with disease stage severity.
2. **Noise injection** - with a stage-dependent probability, the top-ranked
   memory is swapped for an "adjacent" (lower-ranked or unrelated) record,
   simulating the patient confidently retrieving the wrong memory or fact
   (confabulation / misattribution).

This module intentionally does not decide *how* a retrieval result should
be phrased -- that is the job of `behavior.symptom_sampler` and
`core.patient_engine`. It only decides *what* comes back from memory, and
flags whether that result was corrupted.
"""

from __future__ import annotations

import time

import numpy as np

from core.models import MemoryHit, Stage, StatisticalPriors
from memory.client import EPISODIC_COLLECTION, FACTS_COLLECTION, PatientMemoryClient
from memory.embeddings import Embedder

# Approximate age (in years) associated with each episodic-memory "era" tag,
# used purely to compute the retrograde-preservation weighting curve. These
# are simulation conveniences, not clinical measurements.
_ERA_AGE_YEARS: dict[str, float] = {
    "childhood": 70.0,
    "young_adult": 55.0,
    "adulthood": 35.0,
    "career": 20.0,
    "late_adulthood": 5.0,
}
_DEFAULT_ERA_AGE_YEARS = 30.0

# Retention floor for the *most recent* memories, by stage -- lower means
# more severe impairment of recently-formed memories. Remote-memory
# retention converges toward `_OLD_CEILING` regardless of stage.
_RECENT_FLOOR: dict[Stage, float] = {
    Stage.FIVE: 0.55,
    Stage.SIX: 0.35,
    Stage.SEVEN: 0.15,
}
_OLD_CEILING = 0.95
_ERA_GRADIENT_K = 0.05

# Session-level time-decay rate (per second of `elapsed_seconds`), scaling
# with stage severity.
_TIME_DECAY_LAMBDA: dict[Stage, float] = {
    Stage.FIVE: 1.0e-6,
    Stage.SIX: 3.0e-6,
    Stage.SEVEN: 8.0e-6,
}


def _era_preservation_weight(era: str | None, stage: Stage) -> float:
    if era is None:
        return 1.0
    age = _ERA_AGE_YEARS.get(era, _DEFAULT_ERA_AGE_YEARS)
    floor = _RECENT_FLOOR[stage]
    return floor + (_OLD_CEILING - floor) * (1.0 - np.exp(-_ERA_GRADIENT_K * age))


def _time_decay(elapsed_seconds: float, stage: Stage) -> float:
    if elapsed_seconds <= 0:
        return 1.0
    return float(np.exp(-_TIME_DECAY_LAMBDA[stage] * elapsed_seconds))


class MemoryRecall:
    """Cognitively-degraded retrieval over a `PatientMemoryClient`."""

    def __init__(
        self,
        client: PatientMemoryClient,
        embedder: Embedder,
        priors: StatisticalPriors,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.priors = priors
        self.rng = rng or np.random.default_rng()

    def _noise_probability(self, collection: str, stage: Stage) -> float:
        profile = self.priors.get_stage_profile(stage)
        if collection == FACTS_COLLECTION:
            return profile.factual_inversion_rate.value
        return profile.anomia_rate.value * 0.5

    def recall(
        self,
        query: str,
        collection: str,
        stage: Stage,
        top_k: int = 5,
        elapsed_seconds: float = 0.0,
    ) -> list[MemoryHit]:
        """Run a cognitively-degraded vector search.

        Args:
            query: the natural-language query (typically the trainee's last
                utterance, or a topic keyword).
            collection: `memory.client.EPISODIC_COLLECTION` or
                `memory.client.FACTS_COLLECTION`.
            stage: current patient GDS/FAST stage, controlling decay/noise
                severity.
            top_k: number of results to return.
            elapsed_seconds: simulated time since the memory was last
                reinforced; larger values increase decay.

        Returns:
            A list of `MemoryHit`, degraded and possibly noise-injected.
        """
        query_vector = self.embedder.embed(query)
        fetch_n = max(top_k * 3, top_k + 2)
        candidates = self.client.search(collection, query_vector, top_k=fetch_n)

        scored: list[dict] = []
        for cand in candidates:
            era = cand.get("era")
            weight = _era_preservation_weight(era, stage) * _time_decay(elapsed_seconds, stage)
            attenuated_score = float(cand.get("distance", 0.0)) * weight
            scored.append({**cand, "attenuated_score": attenuated_score})

        scored.sort(key=lambda c: c["attenuated_score"], reverse=True)
        ranked = scored[:top_k]

        confused_index: int | None = None
        noise_prob = self._noise_probability(collection, stage)
        if ranked and self.rng.random() < noise_prob:
            if len(ranked) >= 2:
                swap_idx = int(self.rng.integers(1, len(ranked)))
                ranked[0], ranked[swap_idx] = ranked[swap_idx], ranked[0]
                confused_index = 0
            else:
                random_vector = self.rng.standard_normal(self.embedder.dim).astype(np.float32)
                random_vector /= max(float(np.linalg.norm(random_vector)), 1e-8)
                fallback = self.client.search(collection, random_vector, top_k=1)
                if fallback:
                    ranked = [{**fallback[0], "attenuated_score": fallback[0].get("distance", 0.0)}]
                    confused_index = 0

        text_field = "text" if collection == EPISODIC_COLLECTION else "fact"
        hits: list[MemoryHit] = []
        for i, item in enumerate(ranked):
            text = item.get(text_field, item.get("fact", item.get("text", "")))
            hits.append(
                MemoryHit(
                    text=text,
                    score=float(item.get("attenuated_score", 0.0)),
                    collection=collection,
                    confused=(i == confused_index),
                    metadata={
                        k: v for k, v in item.items() if k not in {text_field, "attenuated_score"}
                    },
                )
            )
        return hits

    def recall_episodic(self, query: str, stage: Stage, top_k: int = 5) -> list[MemoryHit]:
        return self.recall(query, EPISODIC_COLLECTION, stage, top_k=top_k)

    def recall_facts(self, query: str, stage: Stage, top_k: int = 5) -> list[MemoryHit]:
        return self.recall(query, FACTS_COLLECTION, stage, top_k=top_k)


def now() -> float:
    """Convenience wrapper (kept here so callers need not import `time` directly)."""
    return time.time()
