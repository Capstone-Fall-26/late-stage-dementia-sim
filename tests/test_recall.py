from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.models import Stage, StatisticalPriors
from memory.client import EPISODIC_COLLECTION, FACTS_COLLECTION, PatientMemoryClient
from memory.embeddings import HashingEmbedder
from memory.recall import MemoryRecall, _era_preservation_weight


@pytest.fixture
def seeded_client(tmp_path: Path) -> PatientMemoryClient:
    client = PatientMemoryClient(db_path=tmp_path / "test_memory.db")
    embedder = HashingEmbedder()

    episodes = [
        (
            "Reading a picture book aloud to second graders on a Friday afternoon.",
            "positive",
            "career",
        ),
        ("Cutting the first roses of the season in the garden in June.", "positive", "adulthood"),
        ("Her mother too ill to get out of bed during childhood.", "negative", "childhood"),
        ("The hospital room where Walter died, machines humming.", "negative", "late_adulthood"),
    ]
    for text, valence, era in episodes:
        client.insert_episodic_memory(text, embedder.embed(text), valence, era)

    facts = [
        ("Margaret taught second grade for 28 years.", "career", 0.95),
        ("Margaret grew up in Cleveland, Ohio.", "childhood", 0.9),
    ]
    for fact, category, confidence in facts:
        client.insert_biographical_fact(fact, embedder.embed(fact), category, confidence)

    return client


def test_search_returns_relevant_results(seeded_client: PatientMemoryClient) -> None:
    embedder = HashingEmbedder()
    query_vector = embedder.embed("Tell me about teaching school children.")
    hits = seeded_client.search(EPISODIC_COLLECTION, query_vector, top_k=2)
    assert len(hits) <= 2
    assert all("text" in hit for hit in hits)


def test_recall_returns_memory_hits(
    seeded_client: PatientMemoryClient, priors: StatisticalPriors
) -> None:
    embedder = HashingEmbedder()
    recall = MemoryRecall(seeded_client, embedder, priors, rng=np.random.default_rng(99))
    hits = recall.recall_episodic("What did you do at school?", Stage.FIVE, top_k=2)
    assert len(hits) > 0
    for hit in hits:
        assert hit.collection == EPISODIC_COLLECTION
        assert isinstance(hit.score, float)


def test_recall_facts_uses_facts_collection(
    seeded_client: PatientMemoryClient, priors: StatisticalPriors
) -> None:
    embedder = HashingEmbedder()
    recall = MemoryRecall(seeded_client, embedder, priors, rng=np.random.default_rng(7))
    hits = recall.recall_facts("Where did you grow up?", Stage.SEVEN, top_k=2)
    assert all(hit.collection == FACTS_COLLECTION for hit in hits)


def test_era_preservation_weight_favors_remote_memories() -> None:
    remote_weight = _era_preservation_weight("childhood", Stage.SEVEN)
    recent_weight = _era_preservation_weight("late_adulthood", Stage.SEVEN)
    assert remote_weight > recent_weight


def test_era_preservation_gap_widens_with_stage_severity() -> None:
    gap_stage5 = _era_preservation_weight("childhood", Stage.FIVE) - _era_preservation_weight(
        "late_adulthood", Stage.FIVE
    )
    gap_stage7 = _era_preservation_weight("childhood", Stage.SEVEN) - _era_preservation_weight(
        "late_adulthood", Stage.SEVEN
    )
    assert gap_stage7 > gap_stage5
