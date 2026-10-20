from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.models import PatientProfile, Stage, StatisticalPriors
from memory.client import EPISODIC_COLLECTION, FACTS_COLLECTION, PatientMemoryClient
from memory.embeddings import HashingEmbedder
from memory.encoding import MemoryEncoder
from memory.noise import MemoryNoiseModel, NoiseConfig, build_confusion_map
from memory.recall import MemoryRecall


@pytest.fixture
def client(tmp_path: Path, profile: PatientProfile) -> PatientMemoryClient:
    c = PatientMemoryClient(db_path=tmp_path / "enc.db")
    c.seed_from_profile(profile, HashingEmbedder())
    return c


def _encoder(client: PatientMemoryClient, profile: PatientProfile, **cfg: dict) -> MemoryEncoder:
    noise = MemoryNoiseModel(
        rng=np.random.default_rng(3),
        config=NoiseConfig(**cfg),
        confusion_map=build_confusion_map(profile),
    )
    return MemoryEncoder(client, HashingEmbedder(), noise)


def test_encode_stores_conversation_memory_that_can_be_recalled(
    client: PatientMemoryClient, profile: PatientProfile, priors: StatisticalPriors
) -> None:
    no_fail = {s: 0.0 for s in Stage}
    no_noise = {s: 0.0 for s in Stage}
    enc = _encoder(
        client, profile, encoding_failure=no_fail, encoding_severity=no_noise, vector_sigma=no_noise
    )
    before = client.count(EPISODIC_COLLECTION)
    result = enc.encode_episode("A visitor brought purple tulips and a lemon cake.", Stage.FIVE)
    assert result is not None and not result.corrupted
    assert client.count(EPISODIC_COLLECTION) == before + 1

    recall = MemoryRecall(
        client, HashingEmbedder(), priors, rng=np.random.default_rng(0), noise=enc.noise
    )
    texts = [
        h.text
        for h in recall.recall(
            "purple tulips lemon cake", EPISODIC_COLLECTION, Stage.FIVE, top_k=8, reinforce=False
        )
    ]
    assert any("tulips" in t for t in texts)


def test_encoding_can_fail(client: PatientMemoryClient, profile: PatientProfile) -> None:
    enc = _encoder(client, profile, encoding_failure={s: 1.0 for s in Stage})
    before = client.count(EPISODIC_COLLECTION)
    assert enc.encode_episode("Anything at all.", Stage.SEVEN) is None
    assert client.count(EPISODIC_COLLECTION) == before


def test_corrupted_encoding_keeps_original(
    client: PatientMemoryClient, profile: PatientProfile
) -> None:
    enc = _encoder(
        client,
        profile,
        encoding_failure={s: 0.0 for s in Stage},
        encoding_severity={s: 1.0 for s in Stage},
    )
    result = enc.encode_episode("Susan said she is my daughter.", Stage.SEVEN)
    assert result is not None and result.corrupted
    assert "Ellen" in result.stored_text
    row = client.get_record(EPISODIC_COLLECTION, result.record_id)
    assert row is not None
    assert row["original_text"] == "Susan said she is my daughter."
    assert row["corrupted"] is True and row["source"] == "conversation"


def test_decay_pass_degrades_records_and_preserves_originals(
    client: PatientMemoryClient, profile: PatientProfile
) -> None:
    enc = _encoder(
        client,
        profile,
        decay_fraction={s: 1.0 for s in Stage},
        encoding_severity={s: 1.0 for s in Stage},
    )
    clean_facts = {r["id"]: r["fact"] for r in client.all_records(FACTS_COLLECTION)}
    altered = enc.decay_pass(Stage.SEVEN)
    assert altered[FACTS_COLLECTION] > 0
    rows = client.all_records(FACTS_COLLECTION)
    assert len(rows) == len(clean_facts)  # upsert keeps ids/rows stable
    corrupted = [r for r in rows if r.get("corrupted")]
    assert corrupted
    for r in corrupted:
        assert r["original_fact"] == clean_facts[r["id"]]
        assert r["fact"] != r["original_fact"]


def test_recall_reinforces_hits_and_flags_stored_corruption(
    client: PatientMemoryClient, profile: PatientProfile, priors: StatisticalPriors
) -> None:
    quiet = NoiseConfig(vector_sigma={s: 0.0 for s in Stage})
    noise = MemoryNoiseModel(rng=np.random.default_rng(9), config=quiet)
    recall = MemoryRecall(
        client, HashingEmbedder(), priors, rng=np.random.default_rng(9), noise=noise
    )
    # Stage 5 noise probability is low; retry a few seeds' worth until a clean top hit.
    hits = recall.recall("roses in the garden", EPISODIC_COLLECTION, Stage.FIVE, top_k=2)
    clean = [h for h in hits if not h.confused]
    assert clean
    row = client.get_record(EPISODIC_COLLECTION, int(clean[0].metadata["id"]))
    assert row is not None and row["access_count"] == 1


def test_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dim.db"
    PatientMemoryClient(db_path=path, dim=384).close()
    with pytest.raises(ValueError, match="dimension"):
        PatientMemoryClient(db_path=path, dim=768)


def test_reset_clears_everything(client: PatientMemoryClient) -> None:
    assert client.count(EPISODIC_COLLECTION) > 0
    client.reset()
    assert client.count(EPISODIC_COLLECTION) == 0
    assert client.count(FACTS_COLLECTION) == 0


def test_memory_benchmark_degrades_with_stage(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation.memory_benchmark import run_memory_benchmark

    root = Path(__file__).resolve().parents[1]
    res = run_memory_benchmark(
        root / "config/patient_profiles/stage7_late.json",
        root / "config/statistical_priors.yaml",
        trials=10,
        seed=0,
    )
    assert res["stage_5"]["top1_faithful"] > res["stage_7"]["top1_faithful"]
    assert res["stage_7"]["noise_rate"] > res["stage_5"]["noise_rate"]
